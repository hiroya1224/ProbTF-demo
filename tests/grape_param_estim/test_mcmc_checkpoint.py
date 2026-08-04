from pathlib import Path
import tempfile
import unittest

import numpy as np

from grape_param_estim.posterior.checkpoint import (
    MCMC_CHECKPOINT_SCHEMA,
    McmcChainCheckpoint,
    McmcCheckpointError,
    load_mcmc_checkpoint,
    save_mcmc_checkpoint,
)
from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    PosteriorPoint,
    ProposalMixture,
    QuadraticSurrogate,
    SymmetricProposalKernel,
    TargetEvaluation,
)
from grape_param_estim.posterior.mcmc import (
    McmcCancelled,
    McmcChainSettings,
    run_mcmc_chain,
)


class RandomWalkKernel(SymmetricProposalKernel):
    def __init__(self, kernel_name="walk"):
        self.kernel_name = kernel_name

    @property
    def name(self):
        return self.kernel_name

    def propose(self, point, random_state):
        vector = point.vector.copy()
        vector[0] += 0.17 * float(random_state.normal(size=1)[0])
        return PosteriorPoint.from_vector(vector)


def _exact_evaluation(point, warm_start):
    objective = 0.5 * float(
        point.static_coordinate @ point.static_coordinate
    )
    log_determinant = 0.2 + 0.01 * float(point.static_coordinate[0] ** 2)
    delay_log_prior = -0.5 * float(point.delay ** 2)
    return TargetEvaluation(
        point=point,
        log_density=(
            delay_log_prior - objective - 0.5 * log_determinant
        ),
        successful=True,
        failure_reason="",
        inner_iterations=3,
        warm_start=point.static_coordinate.copy(),
        graph_objective=objective,
        local_log_determinant=log_determinant,
        delay_log_prior=delay_log_prior,
    )


def _sampler(kernel_name="walk"):
    center = PosteriorPoint(np.zeros(18), 0.02)
    surrogate = QuadraticSurrogate(
        center,
        _exact_evaluation(center, None).log_density,
        np.zeros((19, 19)),
    )
    proposal = ProposalMixture(
        (RandomWalkKernel(kernel_name),), np.ones(1)
    )
    return (
        DelayedAcceptanceSampler(
            surrogate,
            proposal,
            (0.0, 0.05),
            _exact_evaluation,
        ),
        _exact_evaluation(center, None),
    )


def _assert_chain_results_equal(test_case, first, second):
    scalar_fields = (
        "chain_id",
        "mode_id",
        "warmup_steps",
        "thinning",
        "total_transitions",
    )
    for name in scalar_fields:
        test_case.assertEqual(getattr(first, name), getattr(second, name))
    array_fields = (
        "sample_id",
        "draw_index",
        "static_coordinate",
        "delay",
        "log_density",
        "attempted_kernel",
        "accepted_kernel",
        "accepted",
        "stage_one_accepted",
        "stage_two_attempted",
        "full_target_cache_hit",
        "inner_solve_failed",
        "inner_iterations",
        "graph_objective",
        "local_log_determinant",
        "delay_log_prior",
    )
    for name in array_fields:
        np.testing.assert_array_equal(
            getattr(first, name), getattr(second, name), err_msg=name
        )
    test_case.assertEqual(first.kernel_summaries, second.kernel_summaries)


class McmcCheckpointTests(unittest.TestCase):
    def _cancelled_checkpoint(self, split_transition=5):
        sampler, initial = _sampler()
        settings = McmcChainSettings(
            "chain-resume", "mode-map", 3, 5, thinning=2
        )
        completed = [0]
        callbacks = []

        def progress(completed_transition, _total, _step):
            completed[0] = completed_transition

        with self.assertRaises(McmcCancelled) as context:
            run_mcmc_chain(
                sampler,
                initial,
                settings,
                np.random.RandomState(819),
                cancellation_requested=lambda: completed[0]
                >= split_transition,
                progress=progress,
                checkpoint_callback=callbacks.append,
            )
        self.assertEqual(context.exception.completed_transitions, split_transition)
        self.assertIsInstance(
            context.exception.checkpoint, McmcChainCheckpoint
        )
        self.assertEqual(
            tuple(value.completed_transition for value in callbacks),
            tuple(range(1, split_transition + 1)),
        )
        return settings, initial, context.exception.checkpoint

    def test_saved_resume_is_bit_identical_to_uninterrupted_chain(self):
        settings, initial, checkpoint = self._cancelled_checkpoint()
        uninterrupted_sampler, uninterrupted_initial = _sampler()
        uninterrupted = run_mcmc_chain(
            uninterrupted_sampler,
            uninterrupted_initial,
            settings,
            np.random.RandomState(819),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chain-checkpoint.npz"
            save_mcmc_checkpoint(str(path), checkpoint)
            loaded = load_mcmc_checkpoint(str(path))
            self.assertIsNone(loaded.current.warm_start)
            self.assertTrue(loaded.current.has_density_components)
            resumed_sampler, _unused_initial = _sampler()
            resumed = run_mcmc_chain(
                resumed_sampler,
                initial,
                settings,
                np.random.RandomState(999999),
                resume_checkpoint=loaded,
            )
        _assert_chain_results_equal(self, uninterrupted, resumed)

    def test_round_trip_is_pickle_free_and_atomic(self):
        _settings, _initial, checkpoint = self._cancelled_checkpoint(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            save_mcmc_checkpoint(str(path), checkpoint)
            self.assertTrue(path.is_file())
            self.assertEqual(tuple(path.parent.glob(".checkpoint.npz.*.tmp")), tuple())
            with np.load(str(path), allow_pickle=False) as archive:
                self.assertEqual(str(archive["schema"][0]), MCMC_CHECKPOINT_SCHEMA)
                self.assertTrue(
                    all(not archive[name].dtype.hasobject for name in archive.files)
                )
            loaded = load_mcmc_checkpoint(str(path))
        self.assertEqual(loaded.completed_transition, 5)
        self.assertEqual(loaded.kernel_names, ("walk",))
        self.assertEqual(loaded.random_state.has_gauss, 1)
        self.assertEqual(
            loaded.kernel_counters["walk"].attempts, 5
        )
        self.assertEqual(len(loaded.retained), 1)
        np.testing.assert_array_equal(
            loaded.current.point.vector, checkpoint.current.point.vector
        )
        self.assertEqual(
            loaded.current.log_density, checkpoint.current.log_density
        )

    def test_loader_rejects_schema_unknown_nonfinite_and_object_dtype(self):
        _settings, _initial, checkpoint = self._cancelled_checkpoint(5)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            save_mcmc_checkpoint(str(source), checkpoint)
            with np.load(str(source), allow_pickle=False) as archive:
                valid = {name: archive[name].copy() for name in archive.files}

            corruptions = []
            wrong_schema = dict(valid)
            wrong_schema["schema"] = np.asarray(("unsupported/v99",))
            corruptions.append(wrong_schema)
            unknown = dict(valid)
            unknown["unexpected"] = np.asarray((1,), dtype=np.int64)
            corruptions.append(unknown)
            nonfinite = dict(valid)
            nonfinite["current_log_density"] = np.asarray((np.nan,))
            corruptions.append(nonfinite)
            object_dtype = dict(valid)
            object_dtype["current_delay"] = np.asarray((0.02,), dtype=object)
            corruptions.append(object_dtype)

            for index, arrays in enumerate(corruptions):
                path = Path(directory) / "corrupt-{}.npz".format(index)
                np.savez_compressed(str(path), **arrays)
                with self.assertRaises(McmcCheckpointError):
                    load_mcmc_checkpoint(str(path))

    def test_resume_rejects_changed_settings_and_kernel_names(self):
        settings, initial, checkpoint = self._cancelled_checkpoint(5)
        sampler, _ = _sampler()
        with self.assertRaises(ValueError):
            run_mcmc_chain(
                sampler,
                initial,
                McmcChainSettings(
                    settings.chain_id,
                    settings.mode_id,
                    settings.warmup_steps,
                    settings.retained_draws + 1,
                    settings.thinning,
                ),
                np.random.RandomState(1),
                resume_checkpoint=checkpoint,
            )
        changed_sampler, _ = _sampler("different-kernel")
        with self.assertRaises(ValueError):
            run_mcmc_chain(
                changed_sampler,
                initial,
                settings,
                np.random.RandomState(1),
                resume_checkpoint=checkpoint,
            )

    def test_cancellation_exposes_checkpoint_without_save_callback(self):
        sampler, initial = _sampler()
        settings = McmcChainSettings("chain-a", "mode-map", 0, 3)
        checks = [False, True]
        with self.assertRaises(McmcCancelled) as context:
            run_mcmc_chain(
                sampler,
                initial,
                settings,
                np.random.RandomState(4),
                cancellation_requested=lambda: checks.pop(0),
            )
        checkpoint = context.exception.checkpoint
        self.assertIsInstance(checkpoint, McmcChainCheckpoint)
        self.assertEqual(checkpoint.completed_transition, 1)
        self.assertEqual(checkpoint.kernel_counters["walk"].attempts, 1)
        self.assertEqual(len(checkpoint.retained), 1)


if __name__ == "__main__":
    unittest.main()

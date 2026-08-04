import unittest

import numpy as np

from grape_param_estim.posterior.delayed_acceptance import (
    DelayedAcceptanceSampler,
    ExactTargetCache,
    GaussianSubspaceKernel,
    PosteriorPoint,
    ProposalMixture,
    QuadraticSurrogate,
    SymmetricProposalKernel,
    TargetEvaluation,
    build_ridge_aware_proposal,
    delayed_acceptance_log_probabilities,
)


class FixedKernel(SymmetricProposalKernel):
    def __init__(self, candidate):
        self.candidate = candidate

    @property
    def name(self):
        return "fixed"

    def propose(self, point, random_state):
        return self.candidate


class FixedRandom:
    def __init__(self, uniforms):
        self.uniforms = iter(uniforms)

    def choice(self, count, p):
        return 0

    def uniform(self):
        return next(self.uniforms)


def _point(first=0.0, delay=0.02):
    coordinate = np.zeros(18)
    coordinate[0] = first
    return PosteriorPoint(coordinate, delay)


def _successful(point, log_density, warm_start=None, iterations=2):
    return TargetEvaluation(
        point=point,
        log_density=log_density,
        successful=True,
        failure_reason="",
        inner_iterations=iterations,
        warm_start=warm_start,
    )


class DelayedAcceptanceTests(unittest.TestCase):
    def test_point_and_cache_use_exact_bits_without_rounding(self):
        cache = ExactTargetCache()
        first = _point(first=0.1)
        nearby = _point(first=np.nextafter(0.1, 1.0))
        calls = []

        def evaluate(point, warm_start):
            calls.append((point, warm_start))
            return _successful(point, -1.0, warm_start="solved")

        evaluation, hit = cache.evaluate(first, evaluate, "initial")
        self.assertFalse(hit)
        repeated, hit = cache.evaluate(first, evaluate, "ignored")
        self.assertTrue(hit)
        self.assertIs(repeated, evaluation)
        _, hit = cache.evaluate(nearby, evaluate, "nearby")
        self.assertFalse(hit)
        self.assertEqual(len(calls), 2)

    def test_exact_cache_rejects_inconsistent_target_decompositions(self):
        cache = ExactTargetCache()
        point = _point()
        first = TargetEvaluation(
            point=point,
            log_density=-2.0,
            successful=True,
            failure_reason="",
            inner_iterations=1,
            graph_objective=1.0,
            local_log_determinant=2.0,
            delay_log_prior=0.0,
        )
        inconsistent = TargetEvaluation(
            point=point,
            log_density=-2.0,
            successful=True,
            failure_reason="",
            inner_iterations=1,
            graph_objective=2.0,
            local_log_determinant=0.0,
            delay_log_prior=0.0,
        )
        cache.store(first)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            cache.store(inconsistent)

    def test_quadratic_surrogate_preserves_exact_zero_information_ridge(self):
        center = _point()
        information = np.eye(19)
        information[0, 0] = 0.0
        surrogate = QuadraticSurrogate(center, -3.0, information)
        self.assertEqual(surrogate.log_density(_point(first=7.0)), -3.0)
        shifted_delay = _point(delay=0.12)
        self.assertAlmostEqual(
            surrogate.log_density(shifted_delay),
            -3.0 - 0.5 * 0.1**2,
        )

    def test_builds_all_five_ridge_aware_symmetric_kernels(self):
        information = np.diag(np.linspace(0.0, 17.0, 18))
        ridge = np.zeros(18)
        ridge[0] = 1.0
        proposal = build_ridge_aware_proposal(
            information,
            ridge,
            delay_scale=0.005,
        )
        self.assertEqual(
            tuple(kernel.name for kernel in proposal.kernels),
            (
                "local_gaussian",
                "exact_ridge",
                "near_ridge",
                "identified_subspace",
                "delay_only",
            ),
        )
        self.assertAlmostEqual(float(np.sum(proposal.weights)), 1.0)
        random = np.random.RandomState(9)
        for kernel in proposal.kernels:
            candidate = kernel.propose(_point(), random)
            self.assertEqual(candidate.vector.shape, (19,))

    def test_second_stage_formula_removes_surrogate_change(self):
        stage_one, stage_two = delayed_acceptance_log_probabilities(
            current_exact_log_density=-5.0,
            candidate_exact_log_density=-5.4,
            current_surrogate_log_density=-2.0,
            candidate_surrogate_log_density=-2.1,
        )
        self.assertAlmostEqual(stage_one, -0.1)
        self.assertAlmostEqual(stage_two, -0.3)
        reverse_one, reverse_two = delayed_acceptance_log_probabilities(
            -5.4,
            -5.0,
            -2.1,
            -2.0,
        )
        self.assertEqual(reverse_one, 0.0)
        self.assertEqual(reverse_two, 0.0)

    def test_stage_two_exact_correction_is_never_skipped(self):
        current_point = _point(first=0.0)
        candidate_point = _point(first=1.0)
        current = _successful(current_point, -1.0, warm_start="current")
        surrogate = QuadraticSurrogate(
            current_point,
            0.0,
            np.zeros((19, 19)),
        )
        proposal = ProposalMixture(
            (FixedKernel(candidate_point),), np.ones(1)
        )
        warm_starts = []

        def target(point, warm_start):
            warm_starts.append(warm_start)
            return _successful(point, -3.0, warm_start="candidate")

        sampler = DelayedAcceptanceSampler(
            surrogate,
            proposal,
            (-0.1, 0.1),
            target,
        )
        result = sampler.step(current, FixedRandom((0.5, 0.9)))
        self.assertTrue(result.stage_one_accepted)
        self.assertTrue(result.stage_two_attempted)
        self.assertFalse(result.stage_two_accepted)
        self.assertFalse(result.accepted)
        self.assertIs(result.current, current)
        self.assertEqual(warm_starts, ["current"])

    def test_matching_surrogate_accepts_at_second_stage(self):
        current_point = _point(first=0.0)
        candidate_point = _point(first=0.2)
        information = np.zeros((19, 19))
        information[0, 0] = 2.0
        surrogate = QuadraticSurrogate(current_point, -1.0, information)
        current = _successful(current_point, surrogate.log_density(current_point))
        proposal = ProposalMixture(
            (FixedKernel(candidate_point),), np.ones(1)
        )

        def target(point, warm_start):
            return _successful(point, surrogate.log_density(point))

        sampler = DelayedAcceptanceSampler(
            surrogate,
            proposal,
            (-0.1, 0.1),
            target,
        )
        result = sampler.step(current, FixedRandom((0.1, 0.999)))
        self.assertTrue(result.stage_one_accepted)
        self.assertTrue(result.stage_two_attempted)
        self.assertTrue(result.stage_two_accepted)
        self.assertTrue(result.accepted)
        self.assertEqual(result.current.point.exact_cache_key, candidate_point.exact_cache_key)

    def test_out_of_bounds_delay_rejects_without_full_target(self):
        current_point = _point(delay=0.02)
        candidate_point = _point(delay=0.2)
        proposal = ProposalMixture(
            (FixedKernel(candidate_point),), np.ones(1)
        )
        calls = []

        def target(point, warm_start):
            calls.append(point)
            return _successful(point, 0.0)

        sampler = DelayedAcceptanceSampler(
            QuadraticSurrogate(current_point, 0.0, np.eye(19)),
            proposal,
            (0.0, 0.05),
            target,
        )
        result = sampler.step(
            _successful(current_point, 0.0), FixedRandom(())
        )
        self.assertFalse(result.stage_one_accepted)
        self.assertFalse(result.stage_two_attempted)
        self.assertEqual(calls, [])

    def test_inner_failure_is_explicit_second_stage_rejection(self):
        current_point = _point()
        candidate_point = _point(first=0.1)
        proposal = ProposalMixture(
            (FixedKernel(candidate_point),), np.ones(1)
        )

        def target(point, warm_start):
            return TargetEvaluation.failure(point, "lm_nonconverged", 7)

        sampler = DelayedAcceptanceSampler(
            QuadraticSurrogate(
                current_point, 0.0, np.zeros((19, 19))
            ),
            proposal,
            (0.0, 0.05),
            target,
        )
        result = sampler.step(
            _successful(current_point, 0.0), FixedRandom((0.4,))
        )
        self.assertTrue(result.stage_one_accepted)
        self.assertTrue(result.stage_two_attempted)
        self.assertTrue(result.inner_solve_failed)
        self.assertFalse(result.accepted)
        self.assertEqual(
            result.candidate_evaluation.failure_reason,
            "lm_nonconverged",
        )

    def test_gaussian_kernel_rejects_nonorthonormal_basis(self):
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            GaussianSubspaceKernel(
                "bad",
                np.ones((19, 2)),
                np.ones(2),
            )


if __name__ == "__main__":
    unittest.main()

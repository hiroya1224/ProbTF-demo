from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    PC_EXACT_ORACLE_CAPABILITIES,
    ExactOracleIdentity,
    ExactOracleReplayOutput,
    PersistentSubprocessExactControllerOracle,
)
from grape_param_estim.controller import ControllerCoreState
from grape_param_estim.controller.external_oracle import (
    BatchingExactControllerOracle,
    StatefulExactOracleControllerBackend,
    batched_exact_controller_backend_factory,
)
from grape_param_estim.inference.tempered_smc import (
    BatchPlantInference,
    TemperedSmcConfig,
)
from grape_param_estim.plant_assimilation import (
    _parallelism_config,
    load_assimilation_config,
)
from grape_param_estim.inference.likelihood import (
    MultipleEpisodeLikelihood,
)


def _identity():
    return ExactOracleIdentity(
        protocol=EXACT_ORACLE_PROTOCOL,
        backend_id="test/batched_exact_controller",
        implementation_language="c++",
        source_commit="12345678",
        artifact_sha256="a" * 64,
        capabilities=PC_EXACT_ORACLE_CAPABILITIES,
        fidelity="pc_exact",
    )


def _request(request_id, snapshot_id="shared"):
    return {
        "snapshot": {"snapshot_id": snapshot_id},
        "jobs": [
            {
                "ticks": [
                    {"request_id": int(request_id), "tick": 0}
                ]
            },
            {
                "ticks": [
                    {"request_id": int(request_id), "tick": 1}
                ]
            },
        ],
    }


class _RecordingExactOracle:
    is_exact = True
    transport_is_persistent = True

    def __init__(self):
        self.identity = _identity()
        self.requests = []
        self._lock = threading.Lock()

    def replay(self, payload):
        with self._lock:
            self.requests.append(copy.deepcopy(payload))
        values = []
        job_ticks = []
        events = []
        final_states = []
        for job_index, job in enumerate(payload["jobs"]):
            request_id = int(job["ticks"][0]["request_id"])
            for tick_index, tick in enumerate(job["ticks"]):
                value = 10 * request_id + int(tick["tick"])
                values.append([float(value)])
                job_ticks.append(
                    [float(job_index), float(tick_index)]
                )
                events.append(value)
            final_states.append(
                {
                    "request_id": request_id,
                    "job_index": job_index,
                }
            )
        return ExactOracleReplayOutput(
            identity=self.identity,
            continuous={
                "value": np.asarray(values, dtype=float),
                "job_tick": np.asarray(job_ticks, dtype=float),
            },
            events=np.asarray(events, dtype=int),
            final_states=tuple(final_states),
        )


class ExactOracleBatchingTests(unittest.TestCase):
    def test_concurrent_requests_are_batched_and_split_in_order(self):
        oracle = _RecordingExactOracle()
        batching = BatchingExactControllerOracle(
            oracle, max_batch_size=4, batch_wait_s=0.05
        )
        barrier = threading.Barrier(4)

        def invoke(request_id):
            barrier.wait()
            return batching.replay(_request(request_id))

        with ThreadPoolExecutor(max_workers=4) as executor:
            replies = tuple(executor.map(invoke, range(4)))

        self.assertEqual(len(oracle.requests), 1)
        self.assertEqual(len(oracle.requests[0]["jobs"]), 8)
        for request_id, reply in enumerate(replies):
            np.testing.assert_array_equal(
                reply.continuous["value"],
                [[10.0 * request_id], [10.0 * request_id + 1.0]],
            )
            np.testing.assert_array_equal(
                reply.continuous["job_tick"],
                [[0.0, 0.0], [1.0, 0.0]],
            )
            np.testing.assert_array_equal(
                reply.events,
                [10 * request_id, 10 * request_id + 1],
            )
            self.assertEqual(
                tuple(
                    int(item["request_id"])
                    for item in reply.final_states
                ),
                (request_id, request_id),
            )

    def test_different_headers_are_never_combined(self):
        oracle = _RecordingExactOracle()
        batching = BatchingExactControllerOracle(
            oracle, max_batch_size=4, batch_wait_s=0.02
        )
        barrier = threading.Barrier(2)

        def invoke(values):
            barrier.wait()
            return batching.replay(_request(*values))

        with ThreadPoolExecutor(max_workers=2) as executor:
            tuple(
                executor.map(
                    invoke,
                    ((0, "snapshot-a"), (1, "snapshot-b")),
                )
            )

        self.assertEqual(len(oracle.requests), 2)
        self.assertEqual(
            {
                item["snapshot"]["snapshot_id"]
                for item in oracle.requests
            },
            {"snapshot-a", "snapshot-b"},
        )

    def test_backend_factory_shares_one_batching_coordinator(self):
        oracle = _RecordingExactOracle()

        def base_factory():
            return StatefulExactOracleControllerBackend(oracle)

        factory = batched_exact_controller_backend_factory(
            base_factory,
            max_batch_size=8,
            batch_wait_s=0.001,
        )
        first = factory()
        second = factory()

        self.assertIsNot(first, second)
        self.assertIs(first.transport, second.transport)
        self.assertIsInstance(
            first.transport, BatchingExactControllerOracle
        )
        self.assertIs(first.transport.underlying, oracle)

    def test_actual_cpp_batches_two_stateful_adapters(self):
        from test_stateful_exact_oracle import (
            _core_input,
            _snapshot,
        )

        repository = Path(__file__).resolve().parents[2]
        executable = (
            repository.parents[1]
            / "devel/.private/gimbalrotor/lib/gimbalrotor"
            / "gimbalrotor_controller_replay"
        )
        if not executable.is_file():
            self.skipTest(
                "built C++ controller replay executable unavailable"
            )
        digest = hashlib.sha256(
            executable.read_bytes()
        ).hexdigest()
        command = [
            str(executable),
            "--artifact-sha256",
            digest,
        ]
        handshake = subprocess.run(
            command,
            input=json.dumps(
                {
                    "protocol": EXACT_ORACLE_PROTOCOL,
                    "operation": "handshake",
                    "payload": {},
                }
            )
            + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=True,
        )
        identity = ExactOracleIdentity.from_mapping(
            json.loads(handshake.stdout)["identity"]
        )
        with PersistentSubprocessExactControllerOracle(
            command, identity, timeout_s=5.0
        ) as oracle:
            replay_calls = []
            call_lock = threading.Lock()
            underlying_replay = oracle.replay

            def counted_replay(payload):
                with call_lock:
                    replay_calls.append(len(payload["jobs"]))
                return underlying_replay(payload)

            oracle.replay = counted_replay
            batching = BatchingExactControllerOracle(
                oracle,
                max_batch_size=2,
                batch_wait_s=0.05,
            )
            backends = (
                StatefulExactOracleControllerBackend(batching),
                StatefulExactOracleControllerBackend(batching),
            )
            for backend in backends:
                backend.reset(
                    _snapshot(identity),
                    ControllerCoreState(
                        target_gimbal_angles=(
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        )
                    ),
                )
            barrier = threading.Barrier(2)

            def step(backend):
                barrier.wait()
                return backend.step(_core_input(0.0))

            with ThreadPoolExecutor(max_workers=2) as executor:
                outputs = tuple(executor.map(step, backends))

        self.assertEqual(replay_calls, [2])
        self.assertEqual(
            tuple(output.stamp for output in outputs), (0.0, 0.0)
        )
        self.assertEqual(
            tuple(backend.state.previous_stamp for backend in backends),
            (0.0, 0.0),
        )


class MultipleEpisodeParallelismTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        observations = (
            SimpleNamespace(
                episode_id="episode-a",
                role="inference_failure",
                source_bag_sha256="a" * 64,
                normalized_episode_sha256="b" * 64,
            ),
            SimpleNamespace(
                episode_id="episode-b",
                role="inference_failure",
                source_bag_sha256="c" * 64,
                normalized_episode_sha256="d" * 64,
            ),
        )
        nuisance = {
            item.episode_id: (
                SimpleNamespace(
                    weight=0.25,
                    state_sample_id="{}-0".format(
                        item.episode_id
                    ),
                    value=0.0,
                ),
                SimpleNamespace(
                    weight=0.75,
                    state_sample_id="{}-1".format(
                        item.episode_id
                    ),
                    value=1.0,
                ),
            )
            for item in observations
        }
        return observations, nuisance

    @staticmethod
    def _rollout(activity):
        activity_lock = threading.Lock()

        def rollout(particle, episode, nuisance):
            with activity_lock:
                activity["current"] += 1
                activity["maximum"] = max(
                    activity["maximum"],
                    activity["current"],
                )
            time.sleep(0.01)
            with activity_lock:
                activity["current"] -= 1
            episode_offset = (
                10.0 if episode.episode_id == "episode-b" else 0.0
            )
            return (
                100.0 * float(particle[0])
                + episode_offset
                + float(nuisance.value)
            )

        return rollout

    def test_parallel_jobs_match_serial_order_and_diagnostics(self):
        observations, nuisance = self._inputs()
        evaluator = SimpleNamespace(
            evaluate=lambda rollout, episode: SimpleNamespace(
                total=float(rollout)
            )
        )
        particles = np.asarray([[2.0], [1.0], [3.0]])
        serial_activity = {"current": 0, "maximum": 0}
        parallel_activity = {"current": 0, "maximum": 0}
        serial = MultipleEpisodeLikelihood(
            observations,
            nuisance,
            self._rollout(serial_activity),
            evaluator,
            worker_count=1,
        )
        parallel = MultipleEpisodeLikelihood(
            observations,
            nuisance,
            self._rollout(parallel_activity),
            evaluator,
            worker_count=4,
        )

        serial_values = serial(particles)
        parallel_values = parallel(particles)

        np.testing.assert_array_equal(
            parallel_values, serial_values
        )
        self.assertEqual(
            tuple(parallel.last_components),
            tuple(serial.last_components),
        )
        self.assertEqual(
            tuple(parallel.last_components),
            tuple(
                (
                    particle_index,
                    episode.episode_id,
                    sample_index,
                )
                for particle_index in range(3)
                for episode in observations
                for sample_index in range(2)
            ),
        )
        self.assertEqual(serial_activity["maximum"], 1)
        self.assertGreaterEqual(
            parallel_activity["maximum"], 2
        )

    def test_duplicate_cache_keys_are_serialized(self):
        observation = SimpleNamespace(
            episode_id="episode-a",
            role="inference_failure",
            source_bag_sha256="a" * 64,
            normalized_episode_sha256="b" * 64,
        )
        nuisance = SimpleNamespace(
            weight=1.0,
            state_sample_id="shared-state",
        )
        cache = {}
        counters = {"hits": 0, "misses": 0}
        cache_lock = threading.Lock()

        def rollout(particle, episode, sample):
            key = (
                float(particle[0]),
                episode.episode_id,
                sample.state_sample_id,
            )
            with cache_lock:
                if key in cache:
                    counters["hits"] += 1
                    return cache[key]
                counters["misses"] += 1
            time.sleep(0.02)
            value = float(particle[0])
            with cache_lock:
                cache[key] = value
            return value

        likelihood = MultipleEpisodeLikelihood(
            (observation,),
            {"episode-a": (nuisance,)},
            rollout,
            SimpleNamespace(
                evaluate=lambda value, episode: SimpleNamespace(
                    total=value
                )
            ),
            worker_count=3,
        )
        output = likelihood(np.asarray([[1.0], [2.0], [1.0]]))

        np.testing.assert_array_equal(output, [1.0, 2.0, 1.0])
        self.assertEqual(counters, {"hits": 1, "misses": 2})

    def test_concurrent_likelihood_calls_singleflight_shared_cache_key(
        self,
    ):
        observation = SimpleNamespace(
            episode_id="episode-a",
            role="inference_failure",
            source_bag_sha256="a" * 64,
            normalized_episode_sha256="b" * 64,
        )
        nuisance = SimpleNamespace(
            weight=1.0,
            state_sample_id="shared-state",
        )
        cache = {}
        counters = {"hits": 0, "misses": 0}
        cache_lock = threading.Lock()

        def rollout(particle, episode, sample):
            key = float(particle[0])
            with cache_lock:
                if key in cache:
                    counters["hits"] += 1
                    return cache[key]
                counters["misses"] += 1
            time.sleep(0.03)
            with cache_lock:
                cache[key] = key
            return key

        likelihood = MultipleEpisodeLikelihood(
            (observation,),
            {"episode-a": (nuisance,)},
            rollout,
            SimpleNamespace(
                evaluate=lambda value, episode: SimpleNamespace(
                    total=value
                )
            ),
            worker_count=2,
        )
        barrier = threading.Barrier(2)

        def invoke(_):
            barrier.wait()
            return likelihood(np.asarray([[1.0]]))

        with ThreadPoolExecutor(max_workers=2) as executor:
            outputs = tuple(executor.map(invoke, range(2)))

        for output in outputs:
            np.testing.assert_array_equal(output, [1.0])
        self.assertEqual(counters, {"hits": 1, "misses": 1})

    def test_likelihood_worker_count_must_be_positive(self):
        observations, nuisance = self._inputs()
        with self.assertRaisesRegex(ValueError, "worker_count"):
            MultipleEpisodeLikelihood(
                observations,
                nuisance,
                lambda particle, episode, sample: None,
                SimpleNamespace(evaluate=lambda value, episode: None),
                worker_count=0,
            )


class SmcChainParallelismTests(unittest.TestCase):
    def test_chain_threads_preserve_seed_order(self):
        activity_lock = threading.Lock()
        activity = {"current": 0, "maximum": 0}

        class FakeSmc:
            def __init__(self, prior, transform, config):
                self.config = config

            def run(self, log_likelihood):
                with activity_lock:
                    activity["current"] += 1
                    activity["maximum"] = max(
                        activity["maximum"], activity["current"]
                    )
                time.sleep(0.03)
                with activity_lock:
                    activity["current"] -= 1
                return SimpleNamespace(seed=self.config.seed)

        inference = BatchPlantInference(
            prior=object(),
            transform=object(),
            hypothesis_builder=lambda values: values,
            log_likelihood=lambda values: values,
            config=TemperedSmcConfig(seed=41),
            chain_count=4,
            chain_worker_count=3,
        )
        with patch(
            "grape_param_estim.inference.tempered_smc."
            "TemperedResampleMoveSmc",
            FakeSmc,
        ):
            chains = inference.run_chains()

        self.assertEqual(
            tuple(item.seed for item in chains), (41, 42, 43, 44)
        )
        self.assertGreaterEqual(activity["maximum"], 2)

    def test_chain_worker_count_must_be_positive(self):
        with self.assertRaisesRegex(
            ValueError, "chain_worker_count"
        ):
            BatchPlantInference(
                prior=object(),
                transform=object(),
                hypothesis_builder=lambda values: values,
                log_likelihood=lambda values: values,
                chain_worker_count=0,
            )


class ParallelismConfigTests(unittest.TestCase):
    def test_production_config_enables_bounded_parallelism(self):
        repository = Path(__file__).resolve().parents[2]
        config, _ = load_assimilation_config(
            repository
            / "ros/examples/grape-param-estim/config"
            / "plant_assimilation.yaml"
        )
        parallelism = _parallelism_config(config)

        self.assertGreater(parallelism["worker_count"], 1)
        self.assertGreater(
            parallelism["exact_controller_batch_size"], 1
        )
        self.assertEqual(parallelism["chain_worker_count"], 1)
        self.assertGreater(
            parallelism["exact_controller_batch_wait_s"], 0.0
        )

    def test_parallelism_config_rejects_coercions_and_bad_ranges(self):
        valid = {
            "inference": {
                "worker_count": 4,
                "chain_worker_count": 1,
                "exact_controller_batch_size": 4,
                "exact_controller_batch_wait_s": 0.001,
            }
        }
        invalid_cases = (
            ("worker_count", True, TypeError),
            ("worker_count", 1.5, TypeError),
            ("chain_worker_count", 0, ValueError),
            (
                "exact_controller_batch_size",
                -1,
                ValueError,
            ),
            (
                "exact_controller_batch_wait_s",
                "0.1",
                TypeError,
            ),
            (
                "exact_controller_batch_wait_s",
                np.inf,
                ValueError,
            ),
        )
        for name, value, exception in invalid_cases:
            with self.subTest(name=name, value=value):
                config = copy.deepcopy(valid)
                config["inference"][name] = value
                with self.assertRaises(exception):
                    _parallelism_config(config)


if __name__ == "__main__":
    unittest.main()

import copy
import unittest

from grape_param_estim_gui.workflow import (
    ArtifactRef,
    AttemptStatus,
    StageStatus,
    UpstreamRef,
    WorkflowError,
    WorkflowMode,
    WorkflowStage,
    WorkflowState,
    WorkflowTransitionError,
    artifact_content_fingerprint,
    canonical_fingerprint,
    completion_fingerprint,
    stage_input_fingerprint,
)


ROOT_A = "sha256:" + "a" * 64
ROOT_B = "sha256:" + "b" * 64
REQUEST_A = "sha256:" + "c" * 64
REQUEST_B = "sha256:" + "d" * 64
CREATED = "2026-08-04T10:00:00+09:00"
STARTED = "2026-08-04T10:00:01+09:00"
FINISHED = "2026-08-04T10:00:02+09:00"


def _workflow(mode=WorkflowMode.STEP):
    return WorkflowState.create(
        workflow_id="workflow-a",
        definition_id="noise-then-parameters-v1",
        mode=mode,
        stages=(
            WorkflowStage("noise", "em-diagonal-q-v1"),
            WorkflowStage(
                "parameters", "static-parameters-v1", depends_on=("noise",)
            ),
        ),
    )


def _stage_input(state, stage_id, root, settings, upstream=()):
    stage = state.stage(stage_id)
    return stage_input_fingerprint(
        definition_fingerprint=state.definition_fingerprint,
        stage_id=stage.stage_id,
        algorithm_version=stage.algorithm_version,
        root_input_fingerprint=root,
        stage_settings=settings,
        upstream=upstream,
    )


def _artifact(attempt, artifact_id, marker):
    content = artifact_content_fingerprint(
        {"schema": "test/artifact/v1", "status": "complete"},
        {"payload.npz": marker * 64},
    )
    completion = completion_fingerprint(
        stage_input=attempt.stage_input_fingerprint,
        request_fingerprint=attempt.request_fingerprint,
        artifact_schema="test/artifact/v1",
        artifact_content=content,
    )
    return ArtifactRef(
        schema="test/artifact/v1",
        artifact_id=artifact_id,
        relative_path=attempt.output_path,
        content_fingerprint=content,
        completion_fingerprint=completion,
    )


def _complete_noise(state, root=ROOT_A, attempt_id="attempt-noise-1", marker="1"):
    fingerprint = _stage_input(state, "noise", root, {"q": "diagonal"})
    state = state.begin_attempt(
        stage_id="noise",
        attempt_id=attempt_id,
        request_path="workflow/noise/{}/request.json".format(attempt_id),
        output_path="workflow/noise/{}/bundle".format(attempt_id),
        root_input_fingerprint=root,
        stage_input=fingerprint,
        request_fingerprint=REQUEST_A,
        created_at=CREATED,
    )
    state = state.mark_running(attempt_id, started_at=STARTED)
    state = state.mark_complete(
        attempt_id,
        _artifact(state.attempt(attempt_id), "noise-output-1", marker),
        finished_at=FINISHED,
    )
    return state, fingerprint


class WorkflowFingerprintTest(unittest.TestCase):
    def test_canonical_fingerprints_ignore_mapping_and_file_order(self):
        self.assertEqual(
            canonical_fingerprint({"b": 2, "a": {"y": 4, "x": 3}}),
            canonical_fingerprint({"a": {"x": 3, "y": 4}, "b": 2}),
        )
        manifest = {"status": "complete", "schema": "test/v1"}
        first = artifact_content_fingerprint(
            manifest,
            {"z.npz": "1" * 64, "a.json": "2" * 64},
        )
        second = artifact_content_fingerprint(
            {"schema": "test/v1", "status": "complete"},
            {"a.json": "sha256:" + "2" * 64, "z.npz": "1" * 64},
        )
        self.assertEqual(first, second)
        self.assertNotEqual(
            first,
            artifact_content_fingerprint(
                manifest,
                {"z.npz": "3" * 64, "a.json": "2" * 64},
            ),
        )

    def test_stage_input_binds_scientific_inputs_but_not_mode(self):
        step = _workflow(WorkflowMode.STEP)
        all_at_once = step.with_mode(WorkflowMode.ALL)
        first = _stage_input(step, "noise", ROOT_A, {"q": [1.0, 2.0]})
        self.assertEqual(
            first,
            _stage_input(all_at_once, "noise", ROOT_A, {"q": [1.0, 2.0]}),
        )
        self.assertNotEqual(
            first, _stage_input(step, "noise", ROOT_B, {"q": [1.0, 2.0]})
        )
        self.assertNotEqual(
            first, _stage_input(step, "noise", ROOT_A, {"q": [1.0, 2.1]})
        )

    def test_completion_binds_stage_request_schema_and_content(self):
        content = canonical_fingerprint({"payload": "a"})
        first = completion_fingerprint(
            stage_input=ROOT_A,
            request_fingerprint=REQUEST_A,
            artifact_schema="test/v1",
            artifact_content=content,
        )
        self.assertNotEqual(
            first,
            completion_fingerprint(
                stage_input=ROOT_A,
                request_fingerprint=REQUEST_B,
                artifact_schema="test/v1",
                artifact_content=content,
            ),
        )


class WorkflowLifecycleTest(unittest.TestCase):
    def test_step_all_and_stage_status_follow_one_complete_boundary(self):
        state = _workflow()
        noise_input = _stage_input(
            state, "noise", ROOT_A, {"q": "diagonal"}
        )
        parameter_placeholder = canonical_fingerprint({"not": "ready"})
        self.assertEqual(
            state.stage_status("noise", noise_input), StageStatus.READY
        )
        self.assertEqual(
            state.stage_status("parameters", parameter_placeholder),
            StageStatus.BLOCKED,
        )

        state, noise_input = _complete_noise(state)
        self.assertEqual(
            state.stage_status("noise", noise_input), StageStatus.COMPLETE
        )
        upstream = state.completion_ref("noise", noise_input)
        self.assertIsInstance(upstream, UpstreamRef)
        parameter_input = _stage_input(
            state,
            "parameters",
            ROOT_A,
            {"estimate_delay": True},
            (upstream,),
        )
        self.assertEqual(
            state.stage_status("parameters", parameter_input, (upstream,)),
            StageStatus.READY,
        )
        state = state.begin_attempt(
            stage_id="parameters",
            attempt_id="attempt-parameters-active",
            request_path="workflow/parameters/active/request.json",
            output_path="workflow/parameters/active/bundle",
            root_input_fingerprint=ROOT_A,
            stage_input=parameter_input,
            request_fingerprint=REQUEST_B,
            upstream=(upstream,),
            created_at=CREATED,
        )
        self.assertEqual(
            state.stage_status("noise", noise_input), StageStatus.COMPLETE
        )
        self.assertEqual(
            state.stage_status("parameters", parameter_input, (upstream,)),
            StageStatus.RUNNING,
        )
        self.assertEqual(state.mode, WorkflowMode.STEP)
        self.assertEqual(state.with_mode("ALL").mode, WorkflowMode.ALL)

    def test_only_one_attempt_can_be_active_and_transitions_are_terminal(self):
        state = _workflow()
        noise_input = _stage_input(state, "noise", ROOT_A, {})
        state = state.begin_attempt(
            stage_id="noise",
            attempt_id="attempt-1",
            request_path="workflow/noise/attempt-1/request.json",
            output_path="workflow/noise/attempt-1/bundle",
            root_input_fingerprint=ROOT_A,
            stage_input=noise_input,
            request_fingerprint=REQUEST_A,
            created_at=CREATED,
        )
        self.assertEqual(
            state.stage_status("noise", noise_input), StageStatus.RUNNING
        )
        with self.assertRaises(WorkflowTransitionError):
            state.begin_attempt(
                stage_id="noise",
                attempt_id="attempt-2",
                request_path="workflow/noise/attempt-2/request.json",
                output_path="workflow/noise/attempt-2/bundle",
                root_input_fingerprint=ROOT_A,
                stage_input=noise_input,
                request_fingerprint=REQUEST_A,
                created_at=CREATED,
            )
        state = state.mark_running("attempt-1", started_at=STARTED)
        state = state.mark_cancelled(
            "attempt-1", "user_requested", finished_at=FINISHED
        )
        self.assertEqual(state.attempt("attempt-1").status, AttemptStatus.CANCELLED)
        self.assertEqual(
            state.stage_status("noise", noise_input), StageStatus.RETRY
        )
        with self.assertRaises(WorkflowTransitionError):
            state.mark_running("attempt-1", started_at=STARTED)

    def test_retry_appends_a_new_attempt_and_preserves_history(self):
        state = _workflow()
        noise_input = _stage_input(state, "noise", ROOT_A, {})
        common = dict(
            stage_id="noise",
            request_path="workflow/noise/attempt-1/request.json",
            output_path="workflow/noise/attempt-1/bundle",
            root_input_fingerprint=ROOT_A,
            stage_input=noise_input,
            request_fingerprint=REQUEST_A,
            created_at=CREATED,
        )
        state = state.begin_attempt(attempt_id="attempt-1", **common)
        state = state.mark_failed(
            "attempt-1", "worker_failed", finished_at=FINISHED
        )
        state = state.retry_attempt(
            stage_id="noise",
            attempt_id="attempt-2",
            request_path="workflow/noise/attempt-2/request.json",
            output_path="workflow/noise/attempt-2/bundle",
            root_input_fingerprint=ROOT_A,
            stage_input=noise_input,
            request_fingerprint=REQUEST_B,
            created_at="2026-08-04T10:01:00+09:00",
        )
        attempts = state.stage("noise").attempts
        self.assertEqual([value.number for value in attempts], [1, 2])
        self.assertEqual(attempts[0].status, AttemptStatus.FAILED)
        self.assertEqual(attempts[1].retry_of, "attempt-1")
        self.assertEqual(attempts[1].status, AttemptStatus.QUEUED)

    def test_changed_root_and_upstream_make_completed_stages_stale(self):
        state, noise_a = _complete_noise(_workflow())
        upstream_a = state.completion_ref("noise", noise_a)
        parameters_a = _stage_input(
            state, "parameters", ROOT_A, {}, (upstream_a,)
        )
        state = state.begin_attempt(
            stage_id="parameters",
            attempt_id="attempt-parameters-1",
            request_path="workflow/parameters/attempt-1/request.json",
            output_path="workflow/parameters/attempt-1/bundle",
            root_input_fingerprint=ROOT_A,
            stage_input=parameters_a,
            request_fingerprint=REQUEST_B,
            upstream=(upstream_a,),
            created_at=CREATED,
        )
        state = state.mark_complete(
            "attempt-parameters-1",
            _artifact(
                state.attempt("attempt-parameters-1"),
                "parameter-output-1",
                "2",
            ),
            finished_at=FINISHED,
        )
        noise_b = _stage_input(state, "noise", ROOT_B, {"q": "diagonal"})
        self.assertEqual(
            state.stage_status("noise", noise_b), StageStatus.STALE
        )
        state = state.begin_attempt(
            stage_id="noise",
            attempt_id="attempt-noise-2",
            request_path="workflow/noise/attempt-2/request.json",
            output_path="workflow/noise/attempt-2/bundle",
            root_input_fingerprint=ROOT_B,
            stage_input=noise_b,
            request_fingerprint=canonical_fingerprint({"request": 2}),
            created_at="2026-08-04T10:02:00+09:00",
        )
        state = state.mark_complete(
            "attempt-noise-2",
            _artifact(state.attempt("attempt-noise-2"), "noise-output-2", "3"),
            finished_at="2026-08-04T10:02:01+09:00",
        )
        upstream_b = state.completion_ref("noise", noise_b)
        parameters_b = _stage_input(
            state, "parameters", ROOT_B, {}, (upstream_b,)
        )
        self.assertEqual(
            state.stage_status("parameters", parameters_b, (upstream_b,)),
            StageStatus.STALE,
        )


class WorkflowValidationTest(unittest.TestCase):
    def test_round_trip_is_strict_and_preserves_immutable_history(self):
        state, _fingerprint = _complete_noise(_workflow(WorkflowMode.ALL))
        payload = state.to_dict()
        self.assertEqual(WorkflowState.from_dict(payload), state)

        extra = copy.deepcopy(payload)
        extra["unexpected"] = True
        with self.assertRaises(WorkflowError):
            WorkflowState.from_dict(extra)

        changed_definition = copy.deepcopy(payload)
        changed_definition["stages"][0]["algorithm_version"] = "changed"
        with self.assertRaisesRegex(WorkflowError, "definition fingerprint"):
            WorkflowState.from_dict(changed_definition)

    def test_safe_relative_paths_and_aware_timestamps_are_required(self):
        state = _workflow()
        fingerprint = _stage_input(state, "noise", ROOT_A, {})
        for path in ("../escape", "/absolute", "C:/drive", "a\\b"):
            with self.subTest(path=path):
                with self.assertRaises(WorkflowError):
                    state.begin_attempt(
                        stage_id="noise",
                        attempt_id="attempt-1",
                        request_path=path,
                        output_path="workflow/noise/attempt-1/bundle",
                        root_input_fingerprint=ROOT_A,
                        stage_input=fingerprint,
                        request_fingerprint=REQUEST_A,
                        created_at=CREATED,
                    )
        with self.assertRaisesRegex(WorkflowError, "UTC offset"):
            state.begin_attempt(
                stage_id="noise",
                attempt_id="attempt-1",
                request_path="workflow/noise/attempt-1/request.json",
                output_path="workflow/noise/attempt-1/bundle",
                root_input_fingerprint=ROOT_A,
                stage_input=fingerprint,
                request_fingerprint=REQUEST_A,
                created_at="2026-08-04T10:00:00",
            )

    def test_upstream_must_reference_a_matching_complete_attempt(self):
        state, noise_input = _complete_noise(_workflow())
        parent = state.completion_ref("noise", noise_input)
        bad = UpstreamRef(
            stage_id="noise",
            attempt_id=parent.attempt_id,
            completion_fingerprint="sha256:" + "f" * 64,
        )
        parameter_input = _stage_input(
            state, "parameters", ROOT_A, {}, (bad,)
        )
        self.assertEqual(
            state.stage_status("parameters", parameter_input, (bad,)),
            StageStatus.BLOCKED,
        )
        with self.assertRaises(WorkflowTransitionError):
            state.begin_attempt(
                stage_id="parameters",
                attempt_id="attempt-parameters-1",
                request_path="workflow/parameters/attempt-1/request.json",
                output_path="workflow/parameters/attempt-1/bundle",
                root_input_fingerprint=ROOT_A,
                stage_input=parameter_input,
                request_fingerprint=REQUEST_B,
                upstream=(bad,),
                created_at=CREATED,
            )


if __name__ == "__main__":
    unittest.main()

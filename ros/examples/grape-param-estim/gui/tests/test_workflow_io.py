import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from grape_param_estim_gui.workflow import (
    ArtifactRef,
    AttemptStatus,
    StageStatus,
    WorkflowMode,
    WorkflowStage,
    WorkflowState,
    artifact_content_fingerprint,
    canonical_fingerprint,
    completion_fingerprint,
    stage_input_fingerprint,
)
from grape_param_estim_gui.workflow_io import (
    DEFAULT_DEFINITION_ID,
    DIAGONAL_Q_ALGORITHM_VERSION,
    WORKFLOW_FILE_NAME,
    WorkflowIoError,
    create_default_workflow,
    default_workflow_stages,
    load_workflow,
    recover_interrupted_attempt,
    save_workflow,
)


ROOT_INPUT = "sha256:" + "a" * 64
REQUEST = "sha256:" + "b" * 64
CREATED = "2026-08-04T01:00:00+00:00"
STARTED = "2026-08-04T01:00:01+00:00"
RECOVERED = "2026-08-04T01:00:02+00:00"
LEGACY_DIAGONAL_Q_ALGORITHM_VERSION = "diagonal-q-em-v1"


def _begin_diagonal_q(state: WorkflowState, attempt_id: str = "q-attempt-1"):
    stage = state.stage("diagonal_q")
    stage_input = stage_input_fingerprint(
        definition_fingerprint=state.definition_fingerprint,
        stage_id=stage.stage_id,
        algorithm_version=stage.algorithm_version,
        root_input_fingerprint=ROOT_INPUT,
        stage_settings={"q_structure": "diagonal"},
    )
    return state.begin_attempt(
        stage_id=stage.stage_id,
        attempt_id=attempt_id,
        request_path="workflow/diagonal_q/{}/request.json".format(attempt_id),
        output_path="workflow/diagonal_q/{}/bundle".format(attempt_id),
        root_input_fingerprint=ROOT_INPUT,
        stage_input=stage_input,
        request_fingerprint=REQUEST,
        created_at=CREATED,
    )


def _complete_diagonal_q(state: WorkflowState):
    state = _begin_diagonal_q(state)
    attempt = state.attempt("q-attempt-1")
    content = artifact_content_fingerprint(
        {"schema": "grape-param-estim/diagonal-q/v1", "status": "complete"},
        {"posterior.npz": "c" * 64},
    )
    artifact = ArtifactRef(
        schema="grape-param-estim/diagonal-q/v1",
        artifact_id="diagonal-q-output-1",
        relative_path=attempt.output_path,
        content_fingerprint=content,
        completion_fingerprint=completion_fingerprint(
            stage_input=attempt.stage_input_fingerprint,
            request_fingerprint=attempt.request_fingerprint,
            artifact_schema="grape-param-estim/diagonal-q/v1",
            artifact_content=content,
        ),
    )
    state = state.mark_complete("q-attempt-1", artifact, finished_at=RECOVERED)
    return state, state.completion_ref(
        "diagonal_q", attempt.stage_input_fingerprint
    )


def _legacy_workflow() -> WorkflowState:
    return WorkflowState.create(
        workflow_id="project-a",
        definition_id=DEFAULT_DEFINITION_ID,
        mode=WorkflowMode.STEP,
        stages=(
            WorkflowStage(
                "diagonal_q", LEGACY_DIAGONAL_Q_ALGORITHM_VERSION
            ),
            WorkflowStage(
                "static_parameters",
                "augmented-static-enkf-v1",
                depends_on=("diagonal_q",),
            ),
        ),
    )


def _current_diagonal_q_input(state: WorkflowState) -> str:
    stage = state.stage("diagonal_q")
    return stage_input_fingerprint(
        definition_fingerprint=state.definition_fingerprint,
        stage_id=stage.stage_id,
        algorithm_version=stage.algorithm_version,
        root_input_fingerprint=ROOT_INPUT,
        stage_settings={"q_structure": "diagonal"},
    )


class WorkflowIoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_file_creates_step_state_with_exact_default_definition(self):
        state = load_workflow(self.root, "project-a")

        self.assertFalse((self.root / WORKFLOW_FILE_NAME).exists())
        self.assertEqual(state.workflow_id, "project-a")
        self.assertEqual(state.mode, WorkflowMode.STEP)
        self.assertEqual(state.definition_id, DEFAULT_DEFINITION_ID)
        self.assertEqual(state.stages, default_workflow_stages())
        self.assertEqual(
            [stage.stage_id for stage in state.stages],
            ["diagonal_q", "static_parameters"],
        )
        self.assertEqual(
            [stage.algorithm_version for stage in state.stages],
            [DIAGONAL_Q_ALGORITHM_VERSION, "augmented-static-enkf-v1"],
        )
        self.assertEqual(state.stages[1].depends_on, ("diagonal_q",))

    def test_round_trip_preserves_state_and_uses_project_local_file(self):
        state = _begin_diagonal_q(
            create_default_workflow("project-a", mode=WorkflowMode.ALL)
        )

        destination = save_workflow(self.root, "project-a", state)

        self.assertEqual(destination, self.root / WORKFLOW_FILE_NAME)
        self.assertEqual(load_workflow(self.root, "project-a"), state)
        self.assertTrue(destination.read_bytes().endswith(b"\n"))

    def test_atomic_replace_failure_preserves_old_state_and_removes_temp(self):
        original = create_default_workflow("project-a")
        save_workflow(self.root, "project-a", original)

        with mock.patch(
            "grape_param_estim_gui.workflow_io.os.replace",
            side_effect=OSError("replace failed"),
        ):
            with self.assertRaisesRegex(WorkflowIoError, "replace failed"):
                save_workflow(
                    self.root,
                    "project-a",
                    original.with_mode(WorkflowMode.ALL),
                )

        self.assertEqual(load_workflow(self.root, "project-a"), original)
        self.assertEqual(list(self.root.glob(".workflow.json.*.tmp")), [])

    def test_read_rejects_malformed_duplicate_nonfinite_and_extra_json(self):
        valid = create_default_workflow("project-a").to_dict()
        cases = {
            "malformed": "{not-json\n",
            "duplicate": '{"schema":"first","schema":"second"}\n',
            "nonfinite": '{"value":NaN}\n',
            "overflow": '{"value":1e999}\n',
            "extra": json.dumps({**valid, "unexpected": True}),
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                (self.root / WORKFLOW_FILE_NAME).write_text(
                    payload, encoding="utf-8"
                )
                with self.assertRaises(WorkflowIoError):
                    load_workflow(self.root, "project-a")

    def test_valid_but_different_definition_is_rejected(self):
        other = WorkflowState.create(
            workflow_id="project-a",
            definition_id="different-definition-v1",
            mode=WorkflowMode.STEP,
            stages=(WorkflowStage("single", "different-algorithm-v1"),),
        )
        (self.root / WORKFLOW_FILE_NAME).write_text(
            json.dumps(other.to_dict()), encoding="utf-8"
        )

        with self.assertRaisesRegex(WorkflowIoError, "definition"):
            load_workflow(self.root, "project-a")
        with self.assertRaisesRegex(WorkflowIoError, "definition"):
            save_workflow(self.root, "project-a", other)

    def test_known_v1_complete_history_is_reconciled_as_stale(self):
        legacy, _upstream = _complete_diagonal_q(_legacy_workflow())
        legacy_attempt = legacy.attempt("q-attempt-1")
        (self.root / WORKFLOW_FILE_NAME).write_text(
            json.dumps(legacy.to_dict()), encoding="utf-8"
        )

        state = load_workflow(self.root, "project-a")

        self.assertEqual(
            state.stage("diagonal_q").algorithm_version,
            DIAGONAL_Q_ALGORITHM_VERSION,
        )
        self.assertEqual(state.attempt("q-attempt-1"), legacy_attempt)
        self.assertNotEqual(
            state.definition_fingerprint, legacy.definition_fingerprint
        )
        self.assertEqual(
            state.stage_status(
                "diagonal_q", _current_diagonal_q_input(state)
            ),
            StageStatus.STALE,
        )
        save_workflow(self.root, "project-a", state)
        self.assertEqual(load_workflow(self.root, "project-a"), state)

    def test_known_v1_failed_history_is_reconciled_as_ready(self):
        legacy = _begin_diagonal_q(_legacy_workflow())
        legacy = legacy.mark_failed(
            "q-attempt-1", "numerical_failure", finished_at=RECOVERED
        )
        legacy_attempt = legacy.attempt("q-attempt-1")
        (self.root / WORKFLOW_FILE_NAME).write_text(
            json.dumps(legacy.to_dict()), encoding="utf-8"
        )

        state = load_workflow(self.root, "project-a")

        self.assertEqual(state.attempt("q-attempt-1"), legacy_attempt)
        self.assertEqual(
            state.stage_status(
                "diagonal_q", _current_diagonal_q_input(state)
            ),
            StageStatus.READY,
        )

    def test_unknown_algorithm_change_is_not_reconciled(self):
        unsupported = WorkflowState.create(
            workflow_id="project-a",
            definition_id=DEFAULT_DEFINITION_ID,
            mode=WorkflowMode.STEP,
            stages=(
                WorkflowStage("diagonal_q", "diagonal-q-unknown-v9"),
                WorkflowStage(
                    "static_parameters",
                    "augmented-static-enkf-v1",
                    depends_on=("diagonal_q",),
                ),
            ),
        )
        (self.root / WORKFLOW_FILE_NAME).write_text(
            json.dumps(unsupported.to_dict()), encoding="utf-8"
        )

        with self.assertRaisesRegex(WorkflowIoError, "definition"):
            load_workflow(self.root, "project-a")

    def test_tampered_algorithm_with_known_fingerprint_is_rejected(self):
        for label, original in (
            ("v1", _legacy_workflow()),
            ("v2", create_default_workflow("project-a")),
        ):
            with self.subTest(label=label):
                payload = original.to_dict()
                known_fingerprint = payload["definition_fingerprint"]
                payload["stages"][0]["algorithm_version"] = (
                    "diagonal-q-tampered-v99"
                )
                self.assertEqual(
                    payload["definition_fingerprint"], known_fingerprint
                )
                (self.root / WORKFLOW_FILE_NAME).write_text(
                    json.dumps(payload), encoding="utf-8"
                )

                with self.assertRaisesRegex(
                    WorkflowIoError, "definition fingerprint"
                ):
                    load_workflow(self.root, "project-a")

    def test_missing_or_reordered_stages_are_cleanly_rejected(self):
        malformed_definitions = {
            "missing": WorkflowState.create(
                workflow_id="project-a",
                definition_id=DEFAULT_DEFINITION_ID,
                mode=WorkflowMode.STEP,
                stages=(
                    WorkflowStage(
                        "diagonal_q",
                        LEGACY_DIAGONAL_Q_ALGORITHM_VERSION,
                    ),
                ),
            ),
            "reordered": WorkflowState.create(
                workflow_id="project-a",
                definition_id=DEFAULT_DEFINITION_ID,
                mode=WorkflowMode.STEP,
                stages=(
                    WorkflowStage(
                        "static_parameters", "augmented-static-enkf-v1"
                    ),
                    WorkflowStage(
                        "diagonal_q",
                        LEGACY_DIAGONAL_Q_ALGORITHM_VERSION,
                        depends_on=("static_parameters",),
                    ),
                ),
            ),
        }
        for label, malformed in malformed_definitions.items():
            with self.subTest(label=label):
                (self.root / WORKFLOW_FILE_NAME).write_text(
                    json.dumps(malformed.to_dict()), encoding="utf-8"
                )

                with self.assertRaisesRegex(WorkflowIoError, "definition"):
                    load_workflow(self.root, "project-a")

    def test_project_and_workflow_ids_are_safe_and_bound_on_load_and_save(self):
        with self.assertRaisesRegex(WorkflowIoError, "project_id"):
            load_workflow(self.root, "../project-a")
        with self.assertRaisesRegex(WorkflowIoError, "workflow_id"):
            create_default_workflow("project-a", workflow_id="bad/id")

        another = create_default_workflow("another-project")
        with self.assertRaisesRegex(WorkflowIoError, "workflow ID"):
            save_workflow(self.root, "project-a", another)
        (self.root / WORKFLOW_FILE_NAME).write_text(
            json.dumps(another.to_dict()), encoding="utf-8"
        )
        with self.assertRaisesRegex(WorkflowIoError, "workflow ID"):
            load_workflow(self.root, "project-a")

    def test_symlink_workflow_entry_cannot_read_or_replace_outside_file(self):
        outside = self.root.parent / (self.root.name + "-outside.json")
        outside.write_text("outside remains unchanged\n", encoding="utf-8")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.root / WORKFLOW_FILE_NAME).symlink_to(outside)

        with self.assertRaisesRegex(WorkflowIoError, "regular file"):
            load_workflow(self.root, "project-a")
        with self.assertRaisesRegex(WorkflowIoError, "regular file"):
            save_workflow(
                self.root, "project-a", create_default_workflow("project-a")
            )
        self.assertEqual(
            outside.read_text(encoding="utf-8"),
            "outside remains unchanged\n",
        )

    def test_recovery_interrupts_queued_and_running_attempts(self):
        for running in (False, True):
            with self.subTest(running=running):
                state = _begin_diagonal_q(create_default_workflow("project-a"))
                if running:
                    state = state.mark_running(
                        "q-attempt-1", started_at=STARTED
                    )

                recovered = recover_interrupted_attempt(
                    state, finished_at=RECOVERED
                )
                attempt = recovered.attempt("q-attempt-1")
                self.assertEqual(attempt.status, AttemptStatus.INTERRUPTED)
                self.assertEqual(attempt.failure, "application_restart")
                self.assertEqual(attempt.finished_at, RECOVERED)
                self.assertEqual(attempt.started_at, STARTED if running else None)
                self.assertIsNone(recovered.active_attempt)

    def test_recovery_preserves_complete_history_and_requires_utc_time(self):
        state, upstream = _complete_diagonal_q(
            create_default_workflow("project-a")
        )
        completed = state.attempt("q-attempt-1")
        stage = state.stage("static_parameters")
        stage_input = stage_input_fingerprint(
            definition_fingerprint=state.definition_fingerprint,
            stage_id=stage.stage_id,
            algorithm_version=stage.algorithm_version,
            root_input_fingerprint=ROOT_INPUT,
            stage_settings={"estimate_delay": True},
            upstream=(upstream,),
        )
        state = state.begin_attempt(
            stage_id="static_parameters",
            attempt_id="parameters-attempt-1",
            request_path="workflow/static_parameters/attempt-1/request.json",
            output_path="workflow/static_parameters/attempt-1/bundle",
            root_input_fingerprint=ROOT_INPUT,
            stage_input=stage_input,
            request_fingerprint=canonical_fingerprint({"request": 2}),
            upstream=(upstream,),
            created_at=CREATED,
        )

        recovered = recover_interrupted_attempt(state, finished_at=RECOVERED)

        self.assertEqual(recovered.attempt("q-attempt-1"), completed)
        self.assertIs(recovered.attempt("q-attempt-1"), completed)
        self.assertEqual(
            recovered.attempt("parameters-attempt-1").status,
            AttemptStatus.INTERRUPTED,
        )
        with self.assertRaisesRegex(WorkflowIoError, "aware UTC"):
            recover_interrupted_attempt(
                state, finished_at="2026-08-04T10:00:00"
            )
        with self.assertRaisesRegex(WorkflowIoError, "aware UTC"):
            recover_interrupted_attempt(
                state, finished_at="2026-08-04T10:00:00+09:00"
            )

    def test_recovery_without_active_attempt_returns_same_state(self):
        state = create_default_workflow("project-a")
        self.assertIs(
            recover_interrupted_attempt(state, finished_at=RECOVERED), state
        )


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from grape_param_estim_gui.workflow import (
    AttemptStatus,
    WorkflowMode,
    WorkflowStage,
    WorkflowState,
    canonical_fingerprint,
    stage_input_fingerprint,
)
from grape_param_estim_gui.workflow_io import (
    BATCH_ESTIMATION_ALGORITHM_VERSION,
    BATCH_ESTIMATION_STAGE_ID,
    DEFAULT_DEFINITION_ID,
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
FINISHED = "2026-08-04T01:00:02+00:00"


def _stage_input(state, settings=None):
    stage = state.stage(BATCH_ESTIMATION_STAGE_ID)
    return stage_input_fingerprint(
        definition_fingerprint=state.definition_fingerprint,
        stage_id=stage.stage_id,
        algorithm_version=stage.algorithm_version,
        root_input_fingerprint=ROOT_INPUT,
        stage_settings=settings or {"run_mode": "estimate_only"},
    )


def _begin(state, attempt_id="attempt-1", *, resume=False, settings=None):
    return state.begin_attempt(
        stage_id=BATCH_ESTIMATION_STAGE_ID,
        attempt_id=attempt_id,
        request_path="runs/{}/request.json".format(attempt_id),
        output_path="runs/run-a/estimation_run",
        root_input_fingerprint=ROOT_INPUT,
        stage_input=_stage_input(state, settings),
        request_fingerprint=REQUEST,
        resume=resume,
        created_at=CREATED,
    )


class WorkflowIoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_default_is_one_batch_stage_and_preserves_step_all_choice(self):
        state = load_workflow(self.root, "project-a")
        self.assertEqual(state.mode, WorkflowMode.STEP)
        self.assertEqual(state.definition_id, DEFAULT_DEFINITION_ID)
        self.assertEqual(state.stages, default_workflow_stages())
        self.assertEqual(len(state.stages), 1)
        self.assertEqual(state.stages[0].stage_id, BATCH_ESTIMATION_STAGE_ID)
        self.assertEqual(
            state.stages[0].algorithm_version,
            BATCH_ESTIMATION_ALGORITHM_VERSION,
        )
        self.assertEqual(state.with_mode(WorkflowMode.ALL).mode, WorkflowMode.ALL)

    def test_round_trip_preserves_attempt_and_resume_flag(self):
        state = _begin(create_default_workflow("project-a"))
        state = state.mark_cancelled(
            "attempt-1", "user_cancelled", finished_at=FINISHED
        )
        state = state.resume_attempt(
            stage_id=BATCH_ESTIMATION_STAGE_ID,
            attempt_id="attempt-2",
            request_path="runs/attempt-2/request.json",
            output_path="runs/run-a/estimation_run",
            root_input_fingerprint=ROOT_INPUT,
            stage_input=_stage_input(state),
            request_fingerprint=REQUEST,
            created_at=CREATED,
        )

        destination = save_workflow(self.root, "project-a", state)
        loaded = load_workflow(self.root, "project-a")

        self.assertEqual(destination, self.root / WORKFLOW_FILE_NAME)
        self.assertEqual(loaded, state)
        resumed = loaded.attempt("attempt-2")
        self.assertTrue(resumed.resume)
        self.assertEqual(resumed.retry_of, "attempt-1")
        self.assertEqual(resumed.output_path, loaded.attempt("attempt-1").output_path)

    def test_resume_requires_prior_terminal_attempt_and_same_output(self):
        state = create_default_workflow("project-a")
        with self.assertRaisesRegex(Exception, "earlier terminal"):
            _begin(state, resume=True)

        state = _begin(state)
        state = state.mark_failed("attempt-1", "failed", finished_at=FINISHED)
        resumed = _begin(state, "attempt-2", resume=True)
        payload = resumed.to_dict()
        payload["stages"][0]["attempts"][1]["output_path"] = "runs/other"
        with self.assertRaisesRegex(Exception, "reuse its prior output"):
            WorkflowState.from_dict(payload)

    def test_restart_marks_active_attempt_interrupted_for_resume(self):
        state = _begin(create_default_workflow("project-a"))
        state = state.mark_running("attempt-1", started_at=STARTED)
        recovered = recover_interrupted_attempt(state, finished_at=FINISHED)
        attempt = recovered.attempt("attempt-1")
        self.assertEqual(attempt.status, AttemptStatus.INTERRUPTED)
        self.assertEqual(attempt.failure, "application_restart")
        self.assertIsNone(recovered.active_attempt)
        self.assertTrue(_begin(recovered, "attempt-2", resume=True).attempt("attempt-2").resume)

    def test_old_two_stage_workflow_is_rejected_without_migration(self):
        old = WorkflowState.create(
            workflow_id="project-a",
            definition_id="diagonal-q-then-static-parameters-v1",
            mode=WorkflowMode.STEP,
            stages=(
                WorkflowStage("diagonal_q", "diagonal-q-generalized-em-v2"),
                WorkflowStage(
                    "static_parameters",
                    "augmented-static-enkf-v1",
                    depends_on=("diagonal_q",),
                ),
            ),
        )
        (self.root / WORKFLOW_FILE_NAME).write_text(
            json.dumps(old.to_dict()), encoding="utf-8"
        )
        with self.assertRaises(WorkflowIoError):
            load_workflow(self.root, "project-a")

    def test_malformed_duplicate_nonfinite_and_unknown_definition_fail(self):
        valid = create_default_workflow("project-a").to_dict()
        cases = (
            "{not-json\n",
            '{"schema":"first","schema":"second"}\n',
            '{"value":NaN}\n',
            json.dumps({**valid, "unexpected": True}),
        )
        for payload in cases:
            with self.subTest(payload=payload[:20]):
                (self.root / WORKFLOW_FILE_NAME).write_text(payload, encoding="utf-8")
                with self.assertRaises(WorkflowIoError):
                    load_workflow(self.root, "project-a")

    def test_atomic_replace_failure_preserves_old_state(self):
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


if __name__ == "__main__":
    unittest.main()

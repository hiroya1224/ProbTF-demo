"""QProcess boundary for estimator workers using strict JSON Lines progress."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QTimer,
    Signal,
)

from .process_control import finalize_cancelled_bundle, send_cooperative_interrupt

try:
    from grape_param_estim.progress import ProgressEvent
except ImportError:  # GUI can still give a useful startup error through the runner.
    ProgressEvent = None  # type: ignore[assignment,misc]


class EstimatorProcessRunner(QObject):
    """Run one worker without a shell and enforce its stdout wire protocol."""

    progress = Signal(object)
    stderrLog = Signal(str)
    started = Signal()
    finished = Signal(str)
    cancelled = Signal()
    failed = Signal(str)
    runningChanged = Signal(bool)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        terminate_after_ms: int = 5_000,
        kill_after_ms: int = 5_000,
    ) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.SeparateChannels)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._terminate_timer = QTimer(self)
        self._terminate_timer.setSingleShot(True)
        self._terminate_timer.timeout.connect(self._terminate)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill)
        self._terminate_after_ms = max(int(terminate_after_ms), 1)
        self._kill_after_ms = max(int(kill_after_ms), 1)
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._last_fraction = 0.0
        self._run_id: str | None = None
        self._output_directory: Path | None = None
        self._cancel_requested = False
        self._cancel_reason = "user_requested"
        self._protocol_error: str | None = None
        self._terminal_emitted = False

    @property
    def running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    @property
    def process(self) -> QProcess:
        return self._process

    def start(
        self,
        program: str | Path,
        arguments: Sequence[str | Path],
        *,
        output_directory: str | Path,
        run_id: str | None = None,
        working_directory: str | Path | None = None,
    ) -> None:
        if self.running:
            raise RuntimeError("an estimator process is already running")
        executable = str(program)
        if not executable:
            raise ValueError("worker program cannot be empty")
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._last_fraction = 0.0
        self._run_id = run_id
        self._output_directory = Path(output_directory).expanduser().resolve()
        self._cancel_requested = False
        self._cancel_reason = "user_requested"
        self._protocol_error = None
        self._terminal_emitted = False
        if working_directory is not None:
            self._process.setWorkingDirectory(str(Path(working_directory).resolve()))
        environment = QProcessEnvironment.systemEnvironment()
        for variable in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ):
            if not environment.contains(variable):
                environment.insert(variable, "1")
        self._process.setProcessEnvironment(environment)
        self._process.setProgram(executable)
        self._process.setArguments([str(value) for value in arguments])
        self._process.start(QProcess.ReadWrite)

    def request_cancel(self, reason: str = "user_requested") -> None:
        """Interrupt the worker now, then escalate to terminate and kill."""

        if not self.running or self._cancel_requested:
            return
        self._cancel_requested = True
        self._cancel_reason = str(reason) or "user_requested"
        process_id = int(self._process.processId())
        try:
            interrupted = send_cooperative_interrupt(process_id)
        except OSError:
            interrupted = False
        if interrupted:
            # Existing worker SIGINT handlers set their CancellationToken;
            # forecast-boundary checks then mark the bundle cancelled.
            self._terminate_timer.start(self._terminate_after_ms)
            return
        # QProcess has no portable interrupt primitive on non-POSIX systems.
        # Termination is therefore the explicit platform fallback.
        self._begin_termination_fallback()

    def _begin_termination_fallback(self) -> None:
        if self.running:
            self._process.terminate()
            self._kill_timer.start(self._kill_after_ms)

    def _on_started(self) -> None:
        self.runningChanged.emit(True)
        self.started.emit()

    def _read_stdout(self) -> None:
        try:
            raw = bytes(self._process.readAllStandardOutput()).decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            self._fail_protocol("worker stdout is not UTF-8: {}".format(error))
            return
        self._stdout_buffer += raw
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            if line.endswith("\r"):
                line = line[:-1]
            self._accept_progress_line(line)

    def _read_stderr(self) -> None:
        raw = bytes(self._process.readAllStandardError()).decode("utf-8", "replace")
        self._stderr_buffer += raw
        while "\n" in self._stderr_buffer:
            line, self._stderr_buffer = self._stderr_buffer.split("\n", 1)
            self.stderrLog.emit(line.rstrip("\r"))

    def _accept_progress_line(self, line: str) -> None:
        if self._protocol_error is not None:
            return
        if ProgressEvent is None:
            self._fail_protocol("progress parser is unavailable in the GUI environment")
            return
        try:
            event = ProgressEvent.from_json(line)
        except Exception as error:
            self._fail_protocol("worker stdout is not valid progress JSONL: {}".format(error))
            return
        if self._run_id is not None and event.run_id != self._run_id:
            self._fail_protocol("worker progress run_id does not match the request")
            return
        if event.fraction + 5.0e-13 < self._last_fraction:
            self._fail_protocol("worker progress fraction decreased")
            return
        if event.eta_seconds is not None and (
            not math.isfinite(event.eta_seconds) or event.eta_seconds < 0.0
        ):
            self._fail_protocol("worker progress ETA is not finite and non-negative")
            return
        self._last_fraction = max(self._last_fraction, float(event.fraction))
        self.progress.emit(event)

    def _fail_protocol(self, message: str) -> None:
        self._protocol_error = message
        if self.running:
            self._process.kill()

    def _terminate(self) -> None:
        self._begin_termination_fallback()

    def _kill(self) -> None:
        if self.running:
            self._process.kill()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.Crashed and self._cancel_requested:
            return
        if error == QProcess.FailedToStart:
            self._emit_failed("worker failed to start: {}".format(self._process.errorString()))

    def _on_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ) -> None:
        self._read_stdout()
        self._read_stderr()
        if self._stdout_buffer:
            self._accept_progress_line(self._stdout_buffer.rstrip("\r"))
            self._stdout_buffer = ""
        if self._stderr_buffer:
            self.stderrLog.emit(self._stderr_buffer.rstrip("\r"))
            self._stderr_buffer = ""
        self._terminate_timer.stop()
        self._kill_timer.stop()
        self.runningChanged.emit(False)
        if self._terminal_emitted:
            return
        if self._protocol_error is not None:
            self._emit_failed(self._protocol_error)
        elif self._cancel_requested:
            if self._output_directory is not None:
                try:
                    finalised = finalize_cancelled_bundle(
                        self._output_directory, self._cancel_reason
                    )
                except Exception as error:  # preserve the cancelled terminal state
                    self.stderrLog.emit(
                        "could not finalise cancelled bundle manifest: {}".format(
                            error
                        )
                    )
                else:
                    if not finalised:
                        self.stderrLog.emit(
                            "cancelled worker left no writing bundle manifest"
                        )
            self._terminal_emitted = True
            self.cancelled.emit()
        elif exit_status != QProcess.NormalExit or exit_code != 0:
            self._emit_failed("worker exited with code {}".format(exit_code))
        elif self._output_directory is None:
            self._emit_failed("worker output directory was not configured")
        else:
            self._terminal_emitted = True
            self.finished.emit(str(self._output_directory))

    def _emit_failed(self, message: str) -> None:
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        self.failed.emit(message)


__all__ = ["EstimatorProcessRunner"]

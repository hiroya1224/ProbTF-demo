"""Wall/CPU stage telemetry; never a criterion for accepting numerical results."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


class RuntimeMonitor:
    """One writer per process role, with a visible entry before long work starts.

    JSONL preserves completed and failed intervals. The small atomic current
    file lets an independent observer locate work that has not yet returned.
    Only explicit scalar metadata belongs here, never candidate densities.
    """

    def __init__(self, output, role):
        directory = Path(output)/'runtime'
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/(role+'_progress.jsonl')
        self.current_path = directory/(role+'_current.json')
        self.role = role
        self.stack = []
        self.sequence = 0
        self.mark('monitor_started')

    def _write(self, event, **metadata):
        self.sequence += 1
        record = dict(role=self.role, pid=os.getpid(), sequence=self.sequence,
            event=event, wall_time_s=time.time(), monotonic_s=time.monotonic(),
            process_cpu_s=time.process_time(), **metadata)
        with self.path.open('a') as stream:
            stream.write(json.dumps(record, separators=(',', ':'), allow_nan=False)+'\n')
        state = dict(record, active_stage=None if not self.stack else self.stack[-1])
        temporary = self.current_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, separators=(',', ':'), allow_nan=False))
        temporary.replace(self.current_path)

    def mark(self, name, **context):
        self._write('mark', name=name, context=context)

    @contextmanager
    def stage(self, name, **context):
        entry = dict(name=name, context=context, start_monotonic_s=time.monotonic(),
                     start_wall_time_s=time.time(), start_cpu_s=time.process_time())
        self.stack.append(entry)
        self._write('stage_started', stage=name, context=context)
        outcome, reason = 'completed', None
        try:
            yield
        except BaseException as error:
            outcome, reason = 'failed', type(error).__name__+': '+str(error)
            raise
        finally:
            elapsed = time.monotonic()-entry['start_monotonic_s']
            cpu = time.process_time()-entry['start_cpu_s']
            self.stack.pop()
            self._write('stage_finished', stage=name, context=context,
                duration_s=elapsed, cpu_s=cpu, outcome=outcome, error=reason)

"""Latest supported SG pair for a live estimator, independent of ROS.

The source's overlap midpoint and nearest-centred covering SG window are
retained. Raw samples support polynomial fits; they are not a queue of future
posterior updates. A busy estimator jumps to the newest common frontier.
"""

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass, replace
import math
import json
from threading import RLock

from .imu_pair_pipeline import ImuRelativePosePipeline, _SgWindow


@dataclass(frozen=True)
class RawSample:
    stamp_ns: int
    frame_id: str
    payload: object


@dataclass(frozen=True)
class SgSelection:
    stamp_ns: int
    frontier_ns: int
    windows: dict


def seconds(stamp_ns):
    return stamp_ns // 1000000000 + (stamp_ns % 1000000000) * 1e-9


class LatestImuPairBuffer:
    window_type = _SgWindow

    def __init__(self, frames, window_size=11, fit_history_size=10, capacity=16384,
                 max_sample_gap_s=None):
        self.frames = tuple(frames)
        self.window_size = int(window_size)
        self.fit_history_size = int(fit_history_size)
        self.capacity = int(capacity)
        if max_sample_gap_s is not None and (not math.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0):
            raise ValueError("max_sample_gap_s must be positive or None")
        self.max_sample_gap_ns = None if max_sample_gap_s is None else max_sample_gap_s * 1e9
        minimum = self.window_size + self.fit_history_size - 1
        if len(self.frames) != 2 or len(set(self.frames)) != 2:
            raise ValueError("Two distinct IMU selectors are required")
        if self.window_size < 3 or self.window_size != window_size:
            raise ValueError("window_size must be an integer >= 3")
        if self.fit_history_size < 1 or self.fit_history_size != fit_history_size:
            raise ValueError("fit_history_size must be a positive integer")
        if self.capacity != capacity or self.capacity < minimum:
            raise ValueError("Raw history capacity must hold the SG window and fit history")
        self.lock = RLock()
        self.samples = {frame: deque(maxlen=self.capacity) for frame in self.frames}
        self.bound_frames = {}
        self.input_stamps = {}
        self.last_support = None
        self.last_selection = None
        self.waiting_reason = "Waiting for both IMUs"
        self.error = ""
        self.received = 0

    def select_frame(self, frame_id):
        frame_id = str(frame_id).strip().lstrip("/")
        for selector in self.frames:
            if frame_id == selector or ("__" not in selector and frame_id.split("__", 1)[0] == selector):
                return selector
        return None

    def reset(self):
        with self.lock:
            for raw in self.samples.values():
                raw.clear()
            self.bound_frames.clear()
            self.input_stamps.clear()
            self.last_support = self.last_selection = None
            self.waiting_reason = "Waiting for both IMUs"
            self.error = ""
            self.received = 0

    def add_sample(self, frame_id, stamp_ns, payload):
        frame_id = str(frame_id).strip().lstrip("/")
        selector = self.select_frame(frame_id)
        if selector is None:
            return False
        if isinstance(stamp_ns, bool) or int(stamp_ns) != stamp_ns or stamp_ns < 0:
            raise ValueError("Acquisition time must be nonnegative integer nanoseconds")
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            previous = self.bound_frames.get(selector)
            if previous is not None and previous != frame_id:
                self.error = "Ambiguous IMU {}: {} and {}; reset after fixing the names".format(selector, previous, frame_id)
                raise ValueError(self.error)
            if selector in self.input_stamps and stamp_ns <= self.input_stamps[selector]:
                raise ValueError("IMU acquisition timestamps must be strictly increasing")
            self.samples[selector].append(RawSample(int(stamp_ns), frame_id, payload))
            self.bound_frames[selector] = frame_id
            self.input_stamps[selector] = int(stamp_ns)
            self.received += 1
        return True

    def select_latest(self, after_ns=None):
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            if len(self.input_stamps) != 2:
                self.waiting_reason = "Waiting for both IMUs"
                return None
            frontier = min(self.input_stamps.values())
            if self.last_selection is not None and frontier <= self.last_selection.frontier_ns:
                self.waiting_reason = "Waiting for the common acquisition frontier to advance"
                return None
            # Reconstruct only the source's last ten SG windows at the newest
            # common frontier; never replay intermediate acquisition events.
            histories = {}
            for frame in self.frames:
                raw = tuple(self.samples[frame])
                end = bisect_right([sample.stamp_ns for sample in raw], frontier)
                if end < self.window_size:
                    self.waiting_reason = "No complete SG window at the common frontier"
                    return None
                start = max(0, end - self.window_size - self.fit_history_size + 1)
                histories[frame] = raw[start:end]
            lower = max(history[-self.window_size].stamp_ns for history in histories.values())
            upper = min(history[-1].stamp_ns for history in histories.values())
            if lower > upper:
                self.waiting_reason = "No common measured SG support; extrapolation forbidden"
                return None
            common = lower + (upper - lower) // 2
            if after_ns is not None and common <= after_ns:
                self.waiting_reason = "Waiting for a newer supported SG evaluation time"
                return None
            windows = {}
            for frame, history in histories.items():
                candidates = [history[start:start + self.window_size]
                              for start in range(len(history) - self.window_size + 1)]
                candidates = [window for window in candidates
                              if window[0].stamp_ns <= common <= window[-1].stamp_ns]
                if not candidates:
                    self.waiting_reason = "No covering SG fit; extrapolation forbidden"
                    return None
                windows[frame] = min(candidates, key=lambda window:
                    abs(window[(self.window_size - 1) // 2].stamp_ns - common))
                if self.max_sample_gap_ns is not None and any(
                        b.stamp_ns - a.stamp_ns > self.max_sample_gap_ns
                        for a, b in zip(windows[frame], windows[frame][1:])):
                    self.waiting_reason = "Measured acquisition gap in SG support; waiting for a complete window"
                    return None
            selection = SgSelection(common, frontier, windows)
            self.last_selection = selection
            self.waiting_reason = "Ready"
            # Older raw data can no longer support a future selected window.
            # Ahead-of-frontier data remain until the other IMU catches up.
            keep = self.window_size + self.fit_history_size - 1
            for frame in self.frames:
                raw = self.samples[frame]
                before = sum(sample.stamp_ns <= frontier for sample in raw)
                for _ in range(max(0, before - keep)):
                    raw.popleft()
            return selection

    def diagnostics(self):
        with self.lock:
            return dict(raw_history_counts={frame: len(raw) for frame, raw in self.samples.items()},
                        received_samples=self.received, latest_input_ns=dict(self.input_stamps),
                        common_frontier_ns=min(self.input_stamps.values()) if len(self.input_stamps) == 2 else None,
                        selected_stamp_ns=None if self.last_selection is None else self.last_selection.stamp_ns,
                        selected_frontier_ns=None if self.last_selection is None else self.last_selection.frontier_ns,
                        waiting_reason=self.waiting_reason, bound_frames=dict(self.bound_frames))


class LatestImuRelativePosePipeline(ImuRelativePosePipeline):
    def __init__(self, parent_frame_id, child_frame_id, window_size=11, polynomial_order=7,
                 fit_history_size=10, raw_history_capacity=16384, max_sample_gap_s=None, **options):
        super().__init__(parent_frame_id, child_frame_id, window_size, polynomial_order,
                         fit_history_size, **options)
        self.polynomial_order = int(polynomial_order)
        self.preprocessor = LatestImuPairBuffer(self.preprocessor.frames, window_size,
                                              fit_history_size, raw_history_capacity, max_sample_gap_s)
        self.last_attempt_ns = None
        self.last_selection = None

    def reset(self):
        super().reset()
        self.last_attempt_ns = None
        self.last_selection = None

    def ingest(self, frame_id, stamp_ns, payload):
        return self.preprocessor.add_sample(frame_id, stamp_ns, payload)

    def update_latest(self, decode=lambda sample: sample.payload):
        selection = self.preprocessor.select_latest(self.last_attempt_ns)
        if selection is None:
            return None
        self.last_attempt_ns = selection.stamp_ns
        pair = []
        support = {}
        for frame in self.preprocessor.frames:
            window = selection.windows[frame]
            # Subtract integer nanoseconds BEFORE conversion to float. Epoch
            # timestamps cannot erase the sub-millisecond SG time differences.
            samples = tuple(((sample.stamp_ns - selection.stamp_ns) * 1e-9,) + tuple(decode(sample))
                            for sample in window)
            fit = self.preprocessor.window_type.fit(samples, self.polynomial_order)
            observation = fit.evaluate(frame, 0.0)
            pair.append(replace(observation, stamp=seconds(selection.stamp_ns)))
            support[frame] = (window[-1].frame_id, seconds(window[0].stamp_ns), seconds(window[-1].stamp_ns))
        self.preprocessor.last_support = support
        self.last_pair = tuple(pair)
        self.last_selection = selection
        record = self._estimate_pair(self.last_pair)
        detail = record.provenance.detail + " Latest SG selection: " + json.dumps(dict(
            stamp_ns=selection.stamp_ns, frontier_ns=selection.frontier_ns,
            window_ns={frame: [sample.stamp_ns for sample in window]
                       for frame, window in selection.windows.items()}), sort_keys=True)
        return replace(record, provenance=replace(record.provenance, detail=detail))

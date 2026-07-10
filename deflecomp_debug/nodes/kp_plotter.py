#!/usr/bin/env python3
from collections import deque
import threading
from typing import Deque, List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray


Sample = Tuple[float, np.ndarray]


class PlotSeries:
    def __init__(self, topic: str, title: str, ax, max_points: int) -> None:
        self.topic = topic
        self.title = title
        self.ax = ax
        self.samples: Deque[Sample] = deque(maxlen=max(2, max_points))
        self.start_stamp: Optional[float] = None
        self.n_series: Optional[int] = None
        self.needs_line_reset = False
        self.logged_first_sample = False
        self.lines = []
        self.wait_text = None


class KpPlotter:
    def __init__(self) -> None:
        self.topic = rospy.get_param("~topic", "/deflecomp/kp_hat")
        self.topics = self._parse_labels(rospy.get_param("~topics", [])) or [self.topic]
        self.titles = self._parse_labels(rospy.get_param("~titles", []))
        self.window_s = float(rospy.get_param("~window_s", 60.0))
        self.max_points = int(rospy.get_param("~max_points", 5000))
        self.update_hz = float(rospy.get_param("~update_hz", 10.0))
        self.labels = self._parse_labels(rospy.get_param("~labels", []))
        self.raise_window = self._as_bool(rospy.get_param("~raise_window", False))

        self.lock = threading.RLock()
        self.series: List[PlotSeries] = []

        try:
            import matplotlib
            matplotlib.rcParams["figure.raise_window"] = bool(self.raise_window)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("kp_plotter requires matplotlib. Install python3-matplotlib.") from exc

        self.plt = plt
        self.plt.ion()
        self.fig, axes = self.plt.subplots(
            1,
            len(self.topics),
            num="deflecomp stiffness Kp",
            sharey=True,
            squeeze=False,
        )
        for idx, topic in enumerate(self.topics):
            title = self.titles[idx] if idx < len(self.titles) else self._default_title(topic)
            series = PlotSeries(topic=topic, title=title, ax=axes[0][idx], max_points=self.max_points)
            self._init_axis(series)
            self.series.append(series)
        self.fig.tight_layout()
        self._configure_window_behavior()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

        self.subs = [
            rospy.Subscriber(series.topic, Float64MultiArray, self._make_cb(series), queue_size=20)
            for series in self.series
        ]
        rospy.loginfo("kp_plotter: subscribed to %s", ", ".join(self.topics))

    @staticmethod
    def _parse_labels(value) -> List[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        return [str(item).strip() for item in list(value) if str(item).strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in ("0", "false", "no", "off")
        return bool(value)

    @staticmethod
    def _default_title(topic: str) -> str:
        name = str(topic).strip().rstrip("/").split("/")[-1]
        return name or "Kp"

    def _init_axis(self, series: PlotSeries) -> None:
        series.ax.set_title(series.title)
        series.ax.set_xlabel("time [s]")
        series.ax.set_ylabel("Kp")
        series.ax.grid(True, alpha=0.3)
        series.ax.set_xlim(0.0, max(self.window_s, 1.0))
        series.ax.set_ylim(0.0, 1.0)
        series.wait_text = series.ax.text(
            0.5,
            0.5,
            f"waiting for\n{series.topic}",
            transform=series.ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )

    def _configure_window_behavior(self) -> None:
        if self.raise_window:
            return
        try:
            manager = self.plt.get_current_fig_manager()
            window = getattr(manager, "window", None)
            if window is None:
                return
            try:
                from matplotlib.backends.qt_compat import QtCore
                window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
                window.setAttribute(QtCore.Qt.WA_ShowWithoutActivating, True)
            except Exception:
                pass
            try:
                window.attributes("-topmost", False)
            except Exception:
                pass
        except Exception as exc:
            rospy.logdebug("kp_plotter: could not adjust window raise behavior: %s", exc)

    def _make_cb(self, series: PlotSeries):
        def cb(msg: Float64MultiArray) -> None:
            stamp = rospy.Time.now().to_sec()
            if series.start_stamp is None:
                series.start_stamp = stamp
            data = np.asarray(msg.data, dtype=float)
            if data.size == 0:
                return
            with self.lock:
                if series.n_series is None or series.n_series != data.size:
                    series.n_series = int(data.size)
                    series.needs_line_reset = True
                series.samples.append((stamp - series.start_stamp, data.copy()))
                self._trim_window_locked(series)
            if not series.logged_first_sample:
                series.logged_first_sample = True
                rospy.loginfo("kp_plotter: %s received first sample with %d values", series.topic, data.size)

        return cb

    def _reset_lines(self, series: PlotSeries) -> None:
        series.ax.cla()
        series.ax.set_title(series.title)
        series.ax.set_xlabel("time [s]")
        series.ax.set_ylabel("Kp")
        series.ax.grid(True, alpha=0.3)
        series.wait_text = None
        series.lines = []
        n = int(series.n_series or 0)
        labels = self.labels if len(self.labels) == n else [f"Kp[{idx}]" for idx in range(n)]
        for label in labels:
            line, = series.ax.plot([], [], linewidth=1.5, label=label)
            series.lines.append(line)
        series.ax.legend(loc="upper right")
        self.fig.tight_layout()

    def _trim_window_locked(self, series: PlotSeries) -> None:
        if self.window_s <= 0.0 or not series.samples:
            return
        t_latest = series.samples[-1][0]
        while len(series.samples) > 2 and t_latest - series.samples[0][0] > self.window_s:
            series.samples.popleft()

    def spin(self) -> None:
        rate = rospy.Rate(max(self.update_hz, 1e-3))
        while not rospy.is_shutdown():
            self._draw()
            rate.sleep()

    def _draw(self) -> None:
        with self.lock:
            snapshots = []
            for series in self.series:
                needs_line_reset = series.needs_line_reset
                if needs_line_reset:
                    series.needs_line_reset = False
                snapshots.append((series, series.n_series, needs_line_reset, list(series.samples)))

        y_arrays = []
        for series, n_series, needs_line_reset, samples in snapshots:
            if n_series is None or not samples:
                if series.wait_text is not None:
                    self.fig.canvas.draw_idle()
                continue
            if needs_line_reset:
                self._reset_lines(series)

            t = np.array([sample[0] for sample in samples], dtype=float)
            y = np.vstack([sample[1] for sample in samples])
            y_arrays.append(y)
            for idx, line in enumerate(series.lines):
                line.set_data(t, y[:, idx])

            if self.window_s > 0.0:
                t_max = max(self.window_s, float(t[-1]))
                series.ax.set_xlim(max(0.0, t_max - self.window_s), t_max)
            else:
                series.ax.set_xlim(float(t[0]), max(float(t[-1]), float(t[0]) + 1e-3))

        if not y_arrays:
            self.plt.pause(0.001)
            return

        y_all = np.vstack(y_arrays)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        if abs(y_max - y_min) < 1e-9:
            pad = max(1.0, abs(y_max) * 0.05)
        else:
            pad = 0.08 * (y_max - y_min)
        for series in self.series:
            series.ax.set_ylim(y_min - pad, y_max + pad)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def main() -> None:
    rospy.init_node("kp_plotter")
    plotter = KpPlotter()
    plotter.spin()


if __name__ == "__main__":
    main()

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
        self.bands = []
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
        self.cov_topic = str(rospy.get_param("~cov_topic", "/deflecomp/kp_cov_diag")).strip()
        self.cov_topics = set(
            self._parse_labels(rospy.get_param("~cov_topics", "/deflecomp/kp_est,/deflecomp/kp_hat"))
        )
        self.cov_sigma = float(rospy.get_param("~cov_sigma", 2.0))
        self.cov_alpha = float(rospy.get_param("~cov_alpha", 0.14))

        self.lock = threading.RLock()
        self.series: List[PlotSeries] = []
        self.cov_samples: Deque[Sample] = deque(maxlen=max(2, self.max_points))
        self.cov_n_series: Optional[int] = None
        self.logged_first_cov = False

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
        self.sub_cov = None
        if self.cov_topic:
            self.sub_cov = rospy.Subscriber(self.cov_topic, Float64MultiArray, self._cb_cov, queue_size=20)
        rospy.loginfo("kp_plotter: subscribed to %s", ", ".join(self.topics))
        if self.sub_cov is not None:
            rospy.loginfo(
                "kp_plotter: subscribed to %s for +/- %.3g sigma bands on %s",
                self.cov_topic,
                self.cov_sigma,
                ", ".join(sorted(self.cov_topics)) if self.cov_topics else "(none)",
            )

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
                series.samples.append((stamp, data.copy()))
                self._trim_window_locked(series)
            if not series.logged_first_sample:
                series.logged_first_sample = True
                rospy.loginfo("kp_plotter: %s received first sample with %d values", series.topic, data.size)

        return cb

    def _cb_cov(self, msg: Float64MultiArray) -> None:
        data = np.asarray(msg.data, dtype=float)
        if data.size == 0:
            return
        stamp = rospy.Time.now().to_sec()
        with self.lock:
            self.cov_n_series = int(data.size)
            self.cov_samples.append((stamp, data.copy()))
            self._trim_cov_window_locked()
        if not self.logged_first_cov:
            self.logged_first_cov = True
            rospy.loginfo("kp_plotter: %s received first sample with %d values", self.cov_topic, data.size)

    def _reset_lines(self, series: PlotSeries) -> None:
        series.ax.cla()
        series.ax.set_title(series.title)
        series.ax.set_xlabel("time [s]")
        series.ax.set_ylabel("Kp")
        series.ax.grid(True, alpha=0.3)
        series.wait_text = None
        series.lines = []
        series.bands = []
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

    def _trim_cov_window_locked(self) -> None:
        if self.window_s <= 0.0 or not self.cov_samples:
            return
        t_latest = self.cov_samples[-1][0]
        while len(self.cov_samples) > 2 and t_latest - self.cov_samples[0][0] > self.window_s:
            self.cov_samples.popleft()

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
            cov_snapshot = list(self.cov_samples)
            cov_n_series = self.cov_n_series

        y_arrays = []
        for series, n_series, needs_line_reset, samples in snapshots:
            if n_series is None or not samples:
                if series.wait_text is not None:
                    self.fig.canvas.draw_idle()
                continue
            if needs_line_reset:
                self._reset_lines(series)

            t_abs = np.array([sample[0] for sample in samples], dtype=float)
            t0 = float(series.start_stamp if series.start_stamp is not None else t_abs[0])
            t = t_abs - t0
            y = np.vstack([sample[1] for sample in samples])
            y_arrays.append(y)
            for idx, line in enumerate(series.lines):
                line.set_data(t, y[:, idx])

            bounds = self._bounds_for_series(series, t_abs, y, cov_snapshot, cov_n_series)
            self._draw_bands(series, t, bounds)
            if bounds is not None:
                y_lower, y_upper = bounds
                y_arrays.append(y_lower)
                y_arrays.append(y_upper)

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

    def _bounds_for_series(
        self,
        series: PlotSeries,
        t_abs: np.ndarray,
        y: np.ndarray,
        cov_samples: List[Sample],
        cov_n_series: Optional[int],
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if series.topic not in self.cov_topics or self.cov_sigma <= 0.0:
            return None
        if cov_n_series != y.shape[1] or not cov_samples:
            return None
        cov_t = np.array([sample[0] for sample in cov_samples], dtype=float)
        cov_y = np.vstack([sample[1] for sample in cov_samples])
        if cov_y.shape[1] != y.shape[1]:
            return None
        cov_interp = np.zeros_like(y)
        if cov_t.size == 1:
            cov_interp[:] = cov_y[0]
        else:
            for idx in range(y.shape[1]):
                cov_interp[:, idx] = np.interp(t_abs, cov_t, cov_y[:, idx])

        sigma_log = np.sqrt(np.clip(cov_interp, 0.0, np.inf))
        scale = self.cov_sigma * sigma_log
        y_pos = np.maximum(y, 1e-12)
        return y_pos * np.exp(-scale), y_pos * np.exp(scale)

    def _draw_bands(
        self,
        series: PlotSeries,
        t: np.ndarray,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for band in series.bands:
            try:
                band.remove()
            except ValueError:
                pass
        series.bands = []
        if bounds is None:
            return

        y_lower, y_upper = bounds
        for idx, line in enumerate(series.lines):
            color = line.get_color()
            band = series.ax.fill_between(
                t,
                y_lower[:, idx],
                y_upper[:, idx],
                color=color,
                alpha=self.cov_alpha,
                linewidth=0.0,
            )
            series.bands.append(band)


def main() -> None:
    rospy.init_node("kp_plotter")
    plotter = KpPlotter()
    plotter.spin()


if __name__ == "__main__":
    main()

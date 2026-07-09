#!/usr/bin/env python3
from collections import deque
import threading
from typing import Deque, List, Optional, Tuple

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray


Sample = Tuple[float, np.ndarray]


class KpPlotter:
    def __init__(self) -> None:
        self.topic = rospy.get_param("~topic", "/deflecomp/kp_hat")
        self.window_s = float(rospy.get_param("~window_s", 60.0))
        self.max_points = int(rospy.get_param("~max_points", 5000))
        self.update_hz = float(rospy.get_param("~update_hz", 10.0))
        self.labels = self._parse_labels(rospy.get_param("~labels", []))

        self.samples: Deque[Sample] = deque(maxlen=max(2, self.max_points))
        self.start_stamp: Optional[float] = None
        self.n_series: Optional[int] = None
        self.needs_line_reset = False
        self.logged_first_sample = False
        self.lock = threading.RLock()
        self.lines = []

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("kp_plotter requires matplotlib. Install python3-matplotlib.") from exc

        self.plt = plt
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots(num="deflecomp stiffness Kp")
        self.ax.set_title("Stiffness estimate")
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("Kp")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(0.0, max(self.window_s, 1.0))
        self.ax.set_ylim(0.0, 1.0)
        self.wait_text = self.ax.text(
            0.5,
            0.5,
            f"waiting for\n{self.topic}",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

        self.sub = rospy.Subscriber(self.topic, Float64MultiArray, self._cb_kp, queue_size=20)
        rospy.loginfo("kp_plotter: subscribed to %s", self.topic)

    @staticmethod
    def _parse_labels(value) -> List[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
        return [str(item).strip() for item in list(value) if str(item).strip()]

    def _cb_kp(self, msg: Float64MultiArray) -> None:
        stamp = rospy.Time.now().to_sec()
        if self.start_stamp is None:
            self.start_stamp = stamp
        data = np.asarray(msg.data, dtype=float)
        if data.size == 0:
            return
        with self.lock:
            if self.n_series is None or self.n_series != data.size:
                self.n_series = int(data.size)
                self.needs_line_reset = True
            self.samples.append((stamp - self.start_stamp, data.copy()))
            self._trim_window_locked()
        if not self.logged_first_sample:
            self.logged_first_sample = True
            rospy.loginfo("kp_plotter: received first sample with %d values", data.size)

    def _reset_lines(self) -> None:
        self.ax.cla()
        self.ax.set_title("Stiffness estimate")
        self.ax.set_xlabel("time [s]")
        self.ax.set_ylabel("Kp")
        self.ax.grid(True, alpha=0.3)
        self.wait_text = None
        self.lines = []
        n = int(self.n_series or 0)
        labels = self.labels if len(self.labels) == n else [f"Kp[{idx}]" for idx in range(n)]
        for label in labels:
            line, = self.ax.plot([], [], linewidth=1.5, label=label)
            self.lines.append(line)
        self.ax.legend(loc="upper right")
        self.fig.tight_layout()

    def _trim_window_locked(self) -> None:
        if self.window_s <= 0.0 or not self.samples:
            return
        t_latest = self.samples[-1][0]
        while len(self.samples) > 2 and t_latest - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def spin(self) -> None:
        rate = rospy.Rate(max(self.update_hz, 1e-3))
        while not rospy.is_shutdown():
            self._draw()
            rate.sleep()

    def _draw(self) -> None:
        with self.lock:
            n_series = self.n_series
            needs_line_reset = self.needs_line_reset
            if needs_line_reset:
                self.needs_line_reset = False
            samples = list(self.samples)

        if n_series is None or not samples:
            if self.wait_text is not None:
                self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
            return
        if needs_line_reset:
            self._reset_lines()

        t = np.array([sample[0] for sample in samples], dtype=float)
        y = np.vstack([sample[1] for sample in samples])
        for idx, line in enumerate(self.lines):
            line.set_data(t, y[:, idx])

        if self.window_s > 0.0:
            t_max = max(self.window_s, float(t[-1]))
            self.ax.set_xlim(max(0.0, t_max - self.window_s), t_max)
        else:
            self.ax.set_xlim(float(t[0]), max(float(t[-1]), float(t[0]) + 1e-3))

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if abs(y_max - y_min) < 1e-9:
            pad = max(1.0, abs(y_max) * 0.05)
        else:
            pad = 0.08 * (y_max - y_min)
        self.ax.set_ylim(y_min - pad, y_max + pad)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def main() -> None:
    rospy.init_node("kp_plotter")
    plotter = KpPlotter()
    plotter.spin()


if __name__ == "__main__":
    main()

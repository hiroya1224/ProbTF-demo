"""Exercise real H.264/MP4 timing, including a >32-bit nanosecond duration."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from render import NS, frame_schedule, remux_exact, verify_video


class CompleteBagTimingTest(unittest.TestCase):
    def test_subframe_exact_second_and_long_recording(self):
        for duration in (2_345_678, 20_123_456, NS, 208_999_333_161):
            with self.subTest(duration_ns=duration), tempfile.TemporaryDirectory() as directory:
                pts, durations = frame_schedule(duration)
                source = Path(directory) / "encoded.mp4"
                destination = Path(directory) / "timed.mp4"
                subprocess.run([
                    "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=120",
                    "-frames:v", str(len(pts)), "-c:v", "libx264", "-preset", "ultrafast",
                    "-bf", "0", "-threads", "2", "-video_track_timescale", str(NS),
                    "-use_editlist", "0", str(source),
                ], check=True)
                remux_exact(source, destination, pts, durations)
                checks = verify_video(destination, pts, durations)
                self.assertEqual(checks["first_pts_ns"], 0)
                self.assertEqual(checks["end_ns"], duration)
                self.assertEqual(checks["packet_count"], len(pts))
                self.assertEqual(checks["decode_errors"], 0)


if __name__ == "__main__":
    unittest.main()

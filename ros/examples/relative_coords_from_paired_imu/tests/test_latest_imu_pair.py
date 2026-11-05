"""Raw SG support and latest-result scheduling, including asynchronous bursts."""

import numpy as np
import pytest

from relative_coords_from_paired_imu.latest_imu_pair import (
    LatestImuPairBuffer, LatestImuRelativePosePipeline)


EPOCH = 1800000000000000000
FRAMES = ("parent", "child")


def payload(stamp):
    t = (stamp - EPOCH) * 1e-9
    gyro = np.array([1 + .3*t + .1*t*t, .7 - .2*t, .3 + .05*t])
    return gyro, np.array([.1*t, .2, 9.81]), np.eye(3)*1e-4, np.eye(3)*1e-3


def feed(pipeline, count, start=0):
    for i in range(start, start + count):
        for role, frame in enumerate(FRAMES):
            stamp = EPOCH + i * 500000 + role * 170000
            pipeline.ingest(frame, stamp, payload(stamp))


def test_busy_estimator_uses_only_latest_pair_and_integer_time_sg():
    pipeline = LatestImuRelativePosePipeline(*FRAMES, raw_history_capacity=2048)
    feed(pipeline, 1000)
    assert pipeline.estimator.update_count == 0
    record = pipeline.update_latest()
    assert record is not None
    assert pipeline.estimator.update_count == 1
    selection = pipeline.last_selection
    assert EPOCH + 490000000 < selection.stamp_ns < EPOCH + 500000000
    for frame, observation in zip(FRAMES, pipeline.last_pair):
        window = selection.windows[frame]
        assert len(window) == 11
        assert window[0].stamp_ns <= selection.stamp_ns <= window[-1].stamp_ns
        t = (selection.stamp_ns - EPOCH) * 1e-9
        np.testing.assert_allclose(observation.angular_velocity, payload(selection.stamp_ns)[0], atol=2e-12)
        np.testing.assert_allclose(observation.angular_acceleration, [.3 + .2*t, -.2, .05], atol=2e-9)
    assert pipeline.update_latest() is None
    assert not pipeline._ready_records
    feed(pipeline, 200, start=1000)
    assert pipeline.update_latest().stamp > record.stamp
    assert pipeline.estimator.update_count == 2
    assert pipeline.preprocessor.received == 2400
    assert max(pipeline.preprocessor.diagnostics()["raw_history_counts"].values()) <= 21


def test_arbitrary_cross_sensor_delivery_selects_identical_latest_support():
    messages = []
    for role, frame in enumerate(FRAMES):
        messages.append([(frame, EPOCH+i*1000000+role*370000, None) for i in range(80)])
    sequences = [messages[0]+messages[1], messages[1]+messages[0],
                 [item for pair in zip(*messages) for item in pair]]
    selections = []
    for sequence in sequences:
        buffer = LatestImuPairBuffer(FRAMES, capacity=256)
        for item in sequence:
            buffer.add_sample(*item)
        selections.append(buffer.select_latest())
    assert selections[0] == selections[1] == selections[2]


def test_disjoint_or_expired_support_waits_without_extrapolation_or_fallback():
    buffer = LatestImuPairBuffer(FRAMES, capacity=32)
    for i in range(100):
        buffer.add_sample("parent", EPOCH+i*1000000, None)
    for i in range(20):
        buffer.add_sample("child", EPOCH+i*1000000+100000, None)
    assert buffer.select_latest() is None
    assert buffer.last_selection is None
    for i in range(20, 110):
        buffer.add_sample("child", EPOCH+i*1000000+100000, None)
    selected = buffer.select_latest()
    assert selected is not None
    assert all(window[0].stamp_ns <= selected.stamp_ns <= window[-1].stamp_ns
               for window in selected.windows.values())


def test_no_duplicate_update_for_paused_sensor_and_reset_clears_both_histories():
    pipeline = LatestImuRelativePosePipeline(*FRAMES)
    feed(pipeline, 30)
    assert pipeline.update_latest() is not None
    for i in range(30, 100):
        stamp = EPOCH+i*500000
        pipeline.ingest("parent", stamp, payload(stamp))
    # The last child was 0.17 ms ahead: one final newer common frontier is
    # supported when the parent catches up, even without another child input.
    assert pipeline.update_latest() is not None
    assert pipeline.update_latest() is None
    assert pipeline.estimator.update_count == 2
    pipeline.reset()
    assert pipeline.estimator.update_count == 0
    assert pipeline.update_latest() is None
    assert not pipeline.preprocessor.bound_frames


def test_ambiguous_mac_is_a_sticky_error_and_unrelated_imu_does_not_enter_history():
    buffer = LatestImuPairBuffer(FRAMES)
    assert not buffer.add_sample("unrelated", EPOCH, None)
    buffer.add_sample("parent__aabbcc", EPOCH, None)
    with pytest.raises(ValueError, match="Ambiguous"):
        buffer.add_sample("parent__ddeeff", EPOCH+1, None)
    with pytest.raises(ValueError, match="Ambiguous"):
        buffer.select_latest()
    buffer.reset()
    assert buffer.add_sample("parent__ddeeff", EPOCH+1, None)


def test_a_hole_inside_endpoint_bounds_is_not_silently_interpolated():
    buffer = LatestImuPairBuffer(FRAMES, max_sample_gap_s=.05)
    for frame in FRAMES:
        for ms in (0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 105):
            buffer.add_sample(frame, EPOCH+ms*1000000, None)
    assert buffer.select_latest() is None
    assert "gap" in buffer.waiting_reason
    assert buffer.last_selection is None
    # Wait for genuinely measured replacement windows; no hold/edge fallback.
    for frame in FRAMES:
        for ms in range(106, 121):
            buffer.add_sample(frame, EPOCH+ms*1000000, None)
    selected = buffer.select_latest()
    assert selected is not None
    assert all(window[0].stamp_ns >= EPOCH+100000000 for window in selected.windows.values())

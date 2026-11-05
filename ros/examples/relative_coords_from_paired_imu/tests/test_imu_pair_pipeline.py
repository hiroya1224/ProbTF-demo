import numpy as np
import pytest

from relative_coords_from_paired_imu.imu_pair_pipeline import (
    ImuRelativePosePipeline,
    SynchronizedImuPairPreprocessor,
    imu_measurement_covariance,
)


def sample(processor, frame, time, epoch=0.0):
    relative_time = time - epoch
    gyro = np.array([0.4 + 0.5 * relative_time, -0.2 + relative_time ** 2, 0.7])
    force = np.array([0.1 * relative_time, relative_time ** 2, 9.81])
    covariance = np.array([[0.01, 0.002, 0], [0.002, 0.02, 0.001], [0, 0.001, 0.015]])
    return processor.update(frame, time, gyro, force, covariance, covariance * 2)


def test_async_pair_uses_common_acquisition_stamp_inside_both_windows():
    processor = SynchronizedImuPairPreprocessor("p", "c", window_size=7, polynomial_order=3)
    epoch = 1800000000.0
    pairs = []
    for index in range(30):
        for frame, offset in (("p__aaaaaa", 0.0), ("c__bbbbbb", 0.0006)):
            stamp = epoch + index * 0.002 + offset + 0.0001 * np.sin(index)
            pair = sample(processor, frame, stamp, epoch)
            if pair is None:
                continue
            pairs.append(pair)
            assert pair[0].stamp == pair[1].stamp
            common = pair[0].stamp
            time = common - epoch
            for observation in pair:
                np.testing.assert_allclose(observation.angular_velocity, [0.4 + 0.5 * time, -0.2 + time ** 2, 0.7], atol=1e-8)
                np.testing.assert_allclose(observation.angular_acceleration, [0.5, 2 * time, 0], atol=1e-6)
                np.testing.assert_allclose(observation.specific_force, [0.1 * time, time ** 2, 9.81], atol=1e-8)
                assert np.linalg.eigvalsh(observation.angular_velocity_covariance).min() > 0
                assert observation.angular_velocity_covariance[0, 1] > 0
            for raw_frame, lower, upper in processor.last_support.values():
                assert lower <= common <= upper
                assert "__" in raw_frame
    assert len(pairs) >= 23
    assert all(a[0].stamp < b[0].stamp for a, b in zip(pairs, pairs[1:]))


def test_ahead_input_is_preserved_until_the_other_sensor_catches_up():
    processor = SynchronizedImuPairPreprocessor("p", "c", window_size=5, polynomial_order=2)
    for index in range(40):
        assert sample(processor, "p", 1 + index * 0.01) is None
    assert len(processor.pending["p"]) == 40
    assert not processor.samples["p"]
    emitted = []
    for index in range(40):
        pair = sample(processor, "c", 1.002 + index * 0.01)
        if pair is not None:
            emitted.append(pair)
        emitted.extend(processor.drain())
    assert len(emitted) == 35
    assert not processor.pending["p"]
    for index in range(40, 50):
        assert sample(processor, "p", 1 + index * 0.01) is None
    assert len(processor.pending["p"]) == 10
    for index in range(40, 50):
        pair = sample(processor, "c", 1.002 + index * 0.01)
        assert pair is not None
        assert pair[0].stamp == pair[1].stamp
        for _, lower, upper in processor.last_support.values():
            assert lower <= pair[0].stamp <= upper
    processor.reset()
    assert all(not buffer for buffer in processor.samples.values())
    assert all(not buffer for buffer in processor.fit_history.values())
    assert all(not buffer for buffer in processor.pending.values())
    assert not tuple(processor.drain())


def test_logical_frame_matching_full_mac_selection_and_ambiguity():
    processor = SynchronizedImuPairPreprocessor("p", "c", 5, 2)
    assert sample(processor, "unrelated", 1.0) is None
    assert processor.bound_frames == {}
    sample(processor, "/p__abcdef", 1.0)
    with pytest.raises(ValueError, match="ambiguous"):
        sample(processor, "p__123456", 1.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        sample(processor, "p__abcdef", 1.0)
    exact = SynchronizedImuPairPreprocessor("p__abcdef", "p__123456", 5, 2)
    sample(exact, "p__abcdef", 1.0)
    sample(exact, "p__123456", 1.0)
    assert len(exact.bound_frames) == 2
    with pytest.raises(ValueError, match="overlap"):
        SynchronizedImuPairPreprocessor("p", "p__abcdef")


def test_unknown_covariance_is_configured_noise_and_unavailable_is_rejected():
    np.testing.assert_allclose(imu_measurement_covariance(np.zeros(9), 0.02), np.eye(3) * 0.02)
    with pytest.raises(ValueError, match="unavailable"):
        imu_measurement_covariance([-1] + [0] * 8, 0.02)
    covariance = np.array([[1, 0.2, 0], [0.2, 2, 0], [0, 0, 3]])
    np.testing.assert_array_equal(imu_measurement_covariance(covariance, 1e-5), covariance)
    with pytest.raises(ValueError, match="finite"):
        imu_measurement_covariance(np.full(9, np.nan), 0.02)


def test_raw_stream_returns_same_law_as_estimator_with_physical_source_ids():
    pipeline = ImuRelativePosePipeline("p", "c", window_size=5, polynomial_order=2)
    records = []
    for index in range(25):
        for frame, offset in (("p__abcdef", 0), ("c__123456", 0.002)):
            record = sample(pipeline, frame, 1 + index * 0.01 + offset)
            if record is not None:
                records.append(record)
                estimate = pipeline.estimator.result()
                emitted = record.distribution.components[0]
                original = estimate.distribution.components[0]
                assert emitted.orientation is original.orientation
                assert emitted.translation is original.translation
                assert record.stamp == estimate.stamp == pipeline.last_pair[0].stamp
                assert record.provenance.source_ids == ("imu_extended_leastsq", "p__abcdef", "c__123456")
                assert emitted.provenance.source_ids == record.provenance.source_ids
                assert record.approximation == estimate.approximation
    assert len(records) > 10
    pipeline.reset()
    assert pipeline.estimator.update_count == 0
    assert pipeline.last_pair is None

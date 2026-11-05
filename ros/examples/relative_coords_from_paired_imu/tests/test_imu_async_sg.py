"""Acquisition-time SG regression, including the actual Fire batch pattern."""

from collections import deque
from itertools import zip_longest

import numpy as np
import pytest
from scipy.signal import savgol_coeffs

from relative_coords_from_paired_imu.imu_pair_pipeline import ImuRelativePosePipeline, SynchronizedImuPairPreprocessor


EPOCH = 1800000000.0
OBSERVATION_FIELDS = (
    "angular_velocity", "angular_acceleration", "specific_force",
    "angular_velocity_covariance", "angular_acceleration_covariance", "specific_force_covariance",
)


def streams(parent_count=60, child_count=60, parent_dt=0.002, child_dt=0.002, offset=0.0006):
    result = {}
    covariance = np.array([[1e-4, 2e-5, 0], [2e-5, 1.5e-4, -1e-5], [0, -1e-5, 1.3e-4]])
    for frame, count, delta, shift in (("p", parent_count, parent_dt, 0), ("c", child_count, child_dt, offset)):
        samples = []
        for index in range(count):
            stamp = EPOCH + index * delta + shift + 0.00004 * np.sin(index * 0.7)
            time = (stamp - EPOCH) / 0.1
            # Deterministic measurement perturbations distinguish different
            # fit windows; exact polynomials alone would hide wrong selection.
            phase = index * 0.83 + (0.4 if frame == "c" else 0)
            omega = np.array([0.4 + time, 0.2 + time ** 2, 0.7 - time + 0.01 * time ** 7])
            omega += 0.0005 * np.array([np.sin(phase), np.cos(phase), np.sin(phase * 1.3)])
            force = np.array([0.1 * time, 0.3 + 0.02 * time ** 3, 9.81 - 0.01 * time ** 2])
            samples.append((frame + ("__aabbcc" if frame == "p" else "__ddeeff"), stamp,
                            omega, force, covariance, covariance * 3))
        result[frame] = samples
    return result


def schedule(inputs, pattern):
    if pattern == "interleaved":
        return [event for pair in zip_longest(inputs["p"], inputs["c"]) for event in pair if event is not None]
    if pattern in ("p_ahead", "c_ahead"):
        first = pattern[0]
        return inputs[first] + inputs["c" if first == "p" else "p"]
    if pattern == "arrival_jitter":
        arrivals = []
        for priority, frame in enumerate(("p", "c")):
            previous = -np.inf
            for index, event in enumerate(inputs[frame]):
                delay = 0.04 * (1 + np.sin(index * 0.13 + priority))
                arrival = max(event[1] + delay, previous + 1e-6)
                arrivals.append((arrival, priority, index, event))
                previous = arrival
        return [event for _, _, _, event in sorted(arrivals, key=lambda item: item[:3])]
    block = int(pattern.split("_")[1])
    result = []
    for start in range(0, max(map(len, inputs.values())), block):
        for frame in (("c", "p") if pattern.startswith("child") else ("p", "c")):
            result.extend(inputs[frame][start:start + block])
    return result


def source_sg_oracle(inputs, window_size=11, degree=7, history_size=10):
    """Independent transcription of source window selection and evaluation.

    Process the source's parent-triggered procedure in acquisition order, up
    to the common known frontier. Use scaled least squares and polynomial
    evaluation (not the producer's helpers) for numerical stability.
    """
    frontier = min(values[-1][1] for values in inputs.values())
    events = sorted(inputs["p"] + inputs["c"], key=lambda event: (event[1], event[0].startswith("p")))
    windows = {frame: deque(maxlen=window_size) for frame in inputs}
    fits = {frame: deque(maxlen=history_size) for frame in inputs}
    output = []
    for event in events:
        if event[1] > frontier:
            break
        frame = event[0].split("__")[0]
        windows[frame].append(event)
        if len(windows[frame]) == window_size:
            data = tuple(windows[frame])
            center = data[(window_size - 1) // 2][1]
            times = np.array([sample[1] - center for sample in data])
            scale = max(abs(times))
            design = np.polynomial.polynomial.polyvander(times / scale, degree)
            values = np.array([np.concatenate((sample[2], sample[3])) for sample in data])
            coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
            fits[frame].append((center, data[0][1], data[-1][1], scale, coefficients))
        if frame != "p" or not all(fits.values()):
            continue
        lower = max(values[0][1] for values in windows.values())
        upper = min(values[-1][1] for values in windows.values())
        if lower > upper:
            continue
        stamp = lower + (upper - lower) / 2
        if output and stamp <= output[-1][0]:
            continue
        selected = {}
        for key in windows:
            fit = min((candidate for candidate in fits[key] if candidate[1] <= stamp <= candidate[2]),
                      key=lambda candidate: abs(candidate[0] - stamp))
            center, start, end, scale, coefficients = fit
            argument = (stamp - center) / scale
            value = np.polynomial.polynomial.polyval(argument, coefficients)
            derivative = np.polynomial.polynomial.polyval(
                argument, np.polynomial.polynomial.polyder(coefficients, axis=0)
            ) / scale
            selected[key] = (value, derivative, start, end, fit is not fits[key][-1])
        output.append((stamp, selected))
    return output


def preprocess(inputs, pattern):
    processor = SynchronizedImuPairPreprocessor("p", "c")
    output = []
    for event in schedule(inputs, pattern):
        processor.add_sample(*event)
        for pair in processor.drain():
            support = dict(processor.last_support)
            output.append((pair, support))
    return processor, output


@pytest.mark.parametrize("pattern", ["interleaved", "batch_1", "batch_7", "batch_10", "child_10",
                                     "batch_23", "p_ahead", "c_ahead", "arrival_jitter"])
def test_source_sg_windows_values_and_derivatives_survive_async_delivery(pattern):
    inputs = streams()
    reference = source_sg_oracle(inputs)
    processor, output = preprocess(inputs, pattern)
    assert len(output) == len(reference) == 49
    assert all(len(history) == 10 for history in processor.fit_history.values())
    for (pair, support), (stamp, selected) in zip(output, reference):
        for observation in pair:
            assert observation.stamp == stamp
            value, derivative, start, end, _ = selected[observation.frame_id]
            assert support[observation.frame_id][1:] == (start, end)
            assert start <= stamp <= end
            np.testing.assert_allclose(observation.angular_velocity, value[:3], atol=1e-10)
            np.testing.assert_allclose(observation.angular_acceleration, derivative[:3], atol=1e-7)
            np.testing.assert_allclose(observation.specific_force, value[3:], atol=1e-10)


@pytest.mark.parametrize("pattern", ["batch_10", "child_10", "p_ahead", "c_ahead", "arrival_jitter"])
def test_async_arrival_preserves_every_observation_covariance_exactly(pattern):
    inputs = streams()
    _, baseline = preprocess(inputs, "interleaved")
    _, actual = preprocess(inputs, pattern)
    assert len(actual) == len(baseline)
    for (left_pair, left_support), (right_pair, right_support) in zip(actual, baseline):
        assert left_support == right_support
        for left, right in zip(left_pair, right_pair):
            assert left.stamp == right.stamp
            for field in OBSERVATION_FIELDS:
                np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


def test_equal_timestamp_ties_do_not_depend_on_which_sender_arrives_first():
    inputs = streams(offset=0.0)
    reference = source_sg_oracle(inputs)
    assert len(reference) == 50
    for pattern in ("interleaved", "child_10", "p_ahead", "c_ahead"):
        _, output = preprocess(inputs, pattern)
        assert len(output) == 50
        for (pair, _), (stamp, selected) in zip(output, reference):
            for observation in pair:
                assert observation.stamp == stamp
                np.testing.assert_allclose(observation.angular_velocity,
                                           selected[observation.frame_id][0][:3], atol=1e-10)


@pytest.mark.parametrize("offset", [-0.0006, 0.0, 0.0006])
def test_unequal_sample_rates_and_sensor_gaps_are_delivery_independent(offset):
    inputs = streams(parent_count=90, child_count=58, child_dt=0.0031, offset=offset)
    inputs["c"] = inputs["c"][:23] + inputs["c"][29:]
    _, baseline = preprocess(inputs, "interleaved")
    assert baseline
    for pattern in ("batch_10", "p_ahead", "c_ahead", "arrival_jitter"):
        _, actual = preprocess(inputs, pattern)
        assert len(actual) == len(baseline)
        for (left, support), (right, original_support) in zip(actual, baseline):
            assert support == original_support
            for a, b in zip(left, right):
                assert a.stamp == b.stamp
                assert support[a.frame_id][1] <= a.stamp <= support[a.frame_id][2]
                for field in OBSERVATION_FIELDS:
                    np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


@pytest.mark.parametrize("pattern", ["batch_10", "p_ahead", "c_ahead", "arrival_jitter"])
def test_complete_gaussian_bingham_posterior_is_delivery_independent(pattern):
    inputs = streams()

    def run(order):
        pipeline = ImuRelativePosePipeline("p", "c")
        output = []
        for event in schedule(inputs, order):
            output.extend(pipeline.update_all(*event))
        assert pipeline.estimator.update_count == len(output) == 49
        return output

    for actual, expected in zip(run(pattern), run("interleaved")):
        assert actual.stamp == expected.stamp
        assert actual.provenance == expected.provenance
        assert actual.approximation == expected.approximation
        left, right = actual.distribution.components[0], expected.distribution.components[0]
        assert left.orientation.kind == right.orientation.kind
        assert left.orientation.inverse_concentration == right.orientation.inverse_concentration
        np.testing.assert_array_equal(left.orientation.shape_matrix, right.orientation.shape_matrix)
        np.testing.assert_array_equal(left.translation.mean_at_reference, right.translation.mean_at_reference)
        np.testing.assert_array_equal(left.translation.residual_covariance, right.translation.residual_covariance)


def test_single_result_compatibility_api_retains_all_released_pairs():
    inputs = streams(parent_count=80, child_count=80)
    prefix = schedule({key: values[:20] for key, values in inputs.items()}, "interleaved")
    # A delayed child acquisition frontier releases several parent events
    # in ONE call, even though only one input was delivered at that call.
    burst = inputs["p"][20:] + [inputs["c"][-1]]
    processor = SynchronizedImuPairPreprocessor("p", "c")
    pipeline = ImuRelativePosePipeline("p", "c")
    baseline = ImuRelativePosePipeline("p", "c")
    pairs, records, expected = [], [], []
    for event in prefix + burst:
        pair = processor.update(*event)
        record = pipeline.update(*event)
        if pair is not None:
            pairs.append(pair)
        if record is not None:
            records.append(record)
        expected.extend(baseline.update_all(*event))
    tail = tuple(processor.drain())
    assert len(tail) > 1
    pairs.extend(tail)
    records.extend(pipeline.drain())
    assert len(pairs) == len(records) == len(expected)
    for pair, record, reference in zip(pairs, records, expected):
        assert pair[0].stamp == pair[1].stamp == record.stamp == reference.stamp
        np.testing.assert_array_equal(record.distribution.components[0].translation.mean_at_reference,
                                      reference.distribution.components[0].translation.mean_at_reference)
    assert not tuple(processor.drain())
    assert not tuple(pipeline.drain())


def test_nearest_centered_history_window_is_used_instead_of_latest_window():
    inputs = streams(parent_count=100, child_count=34, child_dt=0.0061)
    reference = source_sg_oracle(inputs)
    assert any(fit[4] for _, selected in reference for fit in selected.values())
    _, output = preprocess(inputs, "batch_10")
    assert len(output) == len(reference)
    for (_, support), (stamp, selected) in zip(output, reference):
        for frame, (_, _, start, end, _) in selected.items():
            assert support[frame][1:] == (start, end)
            assert start <= stamp <= end


def test_default_sg_matches_classic_coefficients_and_correct_time_shift():
    delta = 1 / 512  # Exactly represented, including at a ROS epoch timestamp.
    times = EPOCH + np.arange(-5, 6) * delta
    covariance = np.eye(3) * 1e-4
    expected_value = savgol_coeffs(11, 7, deriv=0, delta=delta, use="dot")
    expected_derivative = savgol_coeffs(11, 7, deriv=1, delta=delta, use="dot")
    for impulse in range(11):
        samples = tuple((stamp, np.array([float(index == impulse), 0, 0]), np.zeros(3), covariance, covariance)
                        for index, stamp in enumerate(times))
        result = SynchronizedImuPairPreprocessor._evaluate("p", samples, EPOCH, 7)
        assert result.angular_velocity[0] == pytest.approx(expected_value[impulse], abs=1e-10)
        assert result.angular_acceleration[0] == pytest.approx(expected_derivative[impulse], abs=1e-8)
        np.testing.assert_allclose(result.angular_velocity_covariance,
                                   covariance * np.dot(expected_value, expected_value), atol=1e-12)
        np.testing.assert_allclose(result.angular_acceleration_covariance,
                                   covariance * np.dot(expected_derivative, expected_derivative), atol=1e-9)
    samples = tuple((stamp, np.array([stamp - EPOCH, (stamp - EPOCH) ** 2, (stamp - EPOCH) ** 3]),
                     np.array([1, 2, 9.81]), covariance, covariance) for stamp in times)
    for shift in (-0.0007, 0.0, 0.0007):
        target = EPOCH + shift
        offset = target - EPOCH
        result = SynchronizedImuPairPreprocessor._evaluate("p", samples, target, 7)
        np.testing.assert_allclose(result.angular_velocity, [offset, offset ** 2, offset ** 3], atol=1e-12)
        np.testing.assert_allclose(result.angular_acceleration, [1, 2 * offset, 3 * offset ** 2], atol=1e-10)
    with pytest.raises(ValueError, match="extrapolate"):
        SynchronizedImuPairPreprocessor._evaluate("p", samples, EPOCH + 0.1, 7)

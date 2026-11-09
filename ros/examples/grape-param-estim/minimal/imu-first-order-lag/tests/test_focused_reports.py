"""Requested plot contents retain the supplied reconstruction and full spread."""
from pathlib import Path
from contextlib import ExitStack
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import focused_reports as reports


def _captured_figures(stack):
    captured = []

    class CapturePdf:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def savefig(self, figure):
            captured.append(figure)

    stack.enter_context(patch.object(reports, "PdfPages", CapturePdf))
    stack.enter_context(patch.object(reports.plt.Figure, "savefig", lambda *args, **kwargs: None))
    return captured


def inputs():
    time = np.array([10., 10.1, 10.35, 10.6])
    count = len(time)
    raw_gyro = np.zeros((count, 3))
    raw_accel = np.tile([0., 0., 9.8], (count, 1))
    data = dict(time=time, gyro=raw_gyro, accel=raw_accel)
    # These are the physical output columns of the caller's actual replay,
    # including an applied nonzero body residual; no fitted sensor curve is
    # available to the report writer as a possible substitute.
    model = dict(time=time.copy(),
        predicted_gyro_body=np.tile([2., -1., .5], (count, 1)),
        predicted_specific_force_body=np.tile([1., 2., 10.8], (count, 1)),
        applied_residual_wrench_body=np.tile([7., 8., 9., .3, -.2, .1], (count, 1)))
    return data, model


def _by_title(figures, text):
    return next(figure for figure in figures if text in figure._suptitle.get_text())


def _check_common_sensor_and_torque_plots(tmp_path, captured_figures):
    data, model = inputs()
    result = reports.write_requested_figures(tmp_path, "case", data, model, [.1, .2, .3])
    assert result["metadata"]["figure_types"] == [
        "gyro_reconstruction", "accel_reconstruction", "applied_residual_torque"]
    assert len(captured_figures) == 3
    assert len(result["files"]) == 7
    elapsed = data["time"] - data["time"][0]
    for sensor, physical_key, tau in (
            ("gyro", "predicted_gyro_body", .2),
            ("accel", "predicted_specific_force_body", .3)):
        target = model[physical_key][0]
        expected = target + (data[sensor][0] - target) * np.exp(-elapsed[:, None] / tau)
        np.testing.assert_allclose(result["arrays"][sensor + "_forward_lag_response"], expected,
                                   rtol=1e-14, atol=1e-14)
        figure = _by_title(captured_figures, sensor + ":")
        for coordinate, axis in enumerate(figure.axes):
            assert len(axis.lines) == 2
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), data[sensor][:, coordinate])
            np.testing.assert_allclose(axis.lines[1].get_ydata(), expected[:, coordinate], atol=1e-14)
            assert not axis.collections  # No unrequested sensor-noise band.
    # A physical residual-induced response cannot silently be replaced by
    # the unexcited raw gyro, or by the unrelated angular-acceleration error.
    assert np.any(result["arrays"]["gyro_forward_lag_response"][1:] != 0.)
    torque = _by_title(captured_figures, "applied residual torque")
    for coordinate, axis in enumerate(torque.axes):
        np.testing.assert_array_equal(axis.lines[0].get_ydata(),
                                      model["applied_residual_wrench_body"][:, coordinate + 3])
        assert "[N m]" in axis.get_ylabel()
        assert not axis.collections


def _check_five_types_full_bands_and_reference(tmp_path, captured_figures):
    data, model = inputs()
    count = len(data["time"])
    q = Rotation.from_euler("xyz", np.tile([.2, -.3, np.deg2rad(179.)], (count, 1))).as_quat()[:, [3, 0, 1, 2]]
    position = np.column_stack((data["time"] - 10., np.zeros(count), np.ones(count)))
    pf = np.tile(np.array([[1000., 2., 3., 4.], [1., 200., 2., 3.],
                          [2., 3., 40., 1.]]), (count, 1, 1))
    rf = np.tile(np.array([[.02, .01, 0., 0.], [0., .03, .01, 0.],
                          [.01, 0., .04, .01]]), (count, 1, 1))
    tracks = {}
    for i, label in enumerate(("gravity-only", "translation-aware")):
        tracks[label] = dict(time=data["time"].copy(), position_fc_world=position + .1*i,
            quaternion_wxyz=q.copy(), covariance_position_fc=pf @ pf.transpose(0, 2, 1),
            position_covariance_factor_fc=pf.copy(),
            covariance_rotation=rf @ rf.transpose(0, 2, 1), rotation_covariance_factor=rf.copy())
    frozen = {label: {key: value.copy() for key, value in branch.items()}
              for label, branch in tracks.items()}
    reference_time = np.array([9.9, 10.05, 10.4, 10.7])
    reference_position = np.arange(12.).reshape(4, 3)
    reference_q = Rotation.from_euler("xyz", np.tile([.2, -.3, np.deg2rad(-179.)], (4, 1))).as_quat()
    result = reports.write_requested_figures(tmp_path, "case", data, model, [.1, .2, .3], tracks,
        reference=dict(time=reference_time, position_world=reference_position,
                       quaternion_xyzw=reference_q), show_model_uncertainty=True)
    assert result["metadata"]["figure_types"] == [
        "gyro_reconstruction", "accel_reconstruction", "position_reconstruction",
        "attitude_reconstruction", "applied_residual_torque"]
    assert len(captured_figures) == 5
    assert len(result["files"]) == 11
    assert result["metadata"]["position_is_upper_bound"]
    arrays = result["arrays"]
    np.testing.assert_array_equal(arrays["reference__time"], reference_time[1:3])
    np.testing.assert_array_equal(arrays["reference__source_index"], [1, 2])
    np.testing.assert_array_equal(arrays["reference__position_world"], reference_position[1:3])

    for label, branch in tracks.items():
        for key, value in branch.items():
            np.testing.assert_array_equal(value, frozen[label][key])
    for index in range(2):
        np.testing.assert_array_equal(arrays[f"trajectory_{index}__quaternion_wxyz"], q)
        np.testing.assert_array_equal(arrays[f"trajectory_{index}__rpy_integer_turn_offset_degrees"],
                                      [0., 0., -360.])
    # Independently transform the full correlated right-angle factor into
    # Euler coordinates. Diagonal-only or untransformed bands would differ.
    sr, cr, sp, cp = np.sin(.2), np.cos(.2), np.sin(-.3), np.cos(-.3)
    e = np.array([[1., 0., -sp], [0., cr, sr*cp], [0., -sr, cr*cp]])
    projected = np.linalg.solve(e, rf[0])
    expected_std = np.rad2deg(np.sqrt(np.sum(projected**2, axis=1)))
    np.testing.assert_allclose(arrays["trajectory_0__rpy_standard_deviation_degrees"],
                               np.tile(expected_std, (count, 1)), atol=1e-13)

    for title, mean_key, std_key in (
            ("FC position:", "position_fc_world", "position_standard_deviation"),
            ("attitude:", "rpy_degrees", "rpy_standard_deviation_degrees")):
        figure = _by_title(captured_figures, title)
        for coordinate in range(3):
            for column in range(2):
                axis = figure.axes[2*coordinate + column]
                reference_line = axis.lines[0]
                assert reference_line.get_color() == "black"
                np.testing.assert_array_equal(reference_line.get_xdata(), reference_time[1:3] - 10.)
                assert len(axis.lines) == 3
                assert len(axis.collections) == (2 if column else 0)
                for index in range(2):
                    mean = arrays[f"trajectory_{index}__{mean_key}"][:, coordinate]
                    std = arrays[f"trajectory_{index}__{std_key}"][:, coordinate]
                    np.testing.assert_array_equal(axis.lines[index + 1].get_ydata(), mean)
                    if column:
                        vertices = axis.collections[index].get_paths()[0].vertices[:, 1]
                        np.testing.assert_allclose(vertices.min(), (mean - 2*std).min())
                        np.testing.assert_allclose(vertices.max(), (mean + 2*std).max())
                        lower, upper = axis.get_ylim()
                        assert lower <= (mean - 2*std).min()
                        assert upper >= (mean + 2*std).max()
        assert any("not a 95% guarantee" in text.get_text() for text in figure.texts)
    assert _by_title(captured_figures, "attitude:").axes[0].get_title() == "Representative attitudes"


def test_common_sensor_and_torque_plots_use_actual_replay_and_forward_lag():
    with ExitStack() as stack, TemporaryDirectory(prefix="focused-common-regression-") as folder:
        _check_common_sensor_and_torque_plots(Path(folder), _captured_figures(stack))


def test_five_types_full_bands_original_reference_and_unchanged_means():
    with ExitStack() as stack, TemporaryDirectory(prefix="focused-band-regression-") as folder:
        _check_five_types_full_bands_and_reference(Path(folder), _captured_figures(stack))


def test_default_omits_model_bands_only_and_retains_all_covariance_data():
    with ExitStack() as stack, TemporaryDirectory(prefix='focused-display-only-') as folder:
        figures = _captured_figures(stack)
        data, model = inputs()
        count = len(data['time'])
        branch = dict(time=data['time'].copy(), position_fc_world=np.zeros((count,3)),
            quaternion_wxyz=np.tile([1.,0.,0.,0.], (count,1)),
            covariance_position_fc=np.tile(1000.*np.eye(3), (count,1,1)),
            covariance_rotation=np.tile(.01*np.eye(3), (count,1,1)))
        report = reports.write_requested_figures(folder, 'case', data, model,
            [.1,.2,.3], {'observer':branch})
        position = _by_title(figures, 'FC position:')
        attitude = _by_title(figures, 'attitude:')
        assert len(position.axes) == 3
        assert all(not axis.collections for axis in position.axes)
        assert len(attitude.axes) == 6
        assert all(len(attitude.axes[2*k+1].collections) == 1 for k in range(3))
        assert not report['metadata']['model_position_uncertainty_shown']
        assert report['metadata']['computed_covariance_data_retained']
        np.testing.assert_array_equal(report['arrays']['trajectory_0__position_standard_deviation'],
                                      np.full((count,3),np.sqrt(1000.)))
        np.testing.assert_allclose(report['arrays']['trajectory_0__rpy_standard_deviation_degrees'],
                                   np.full((count,3),np.rad2deg(.1)))
        np.testing.assert_array_equal(branch['covariance_position_fc'],
                                      np.tile(1000.*np.eye(3), (count,1,1)))

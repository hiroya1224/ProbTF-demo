"""Deterministic analytic checks for mocap-derived body omega and alpha."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


PATH = Path(__file__).resolve().parents[1] / "kinematics.py"
SPEC = importlib.util.spec_from_file_location("mocap_refine_kinematics_tested", PATH)
KIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KIN)


def _irregular_time():
    index = np.arange(2001)
    return .0015 * index + .0002 * np.sin(.79 * index)


def _three_axis_motion(time):
    # Intrinsic XYZ: R = Rx(a) Ry(b) Rz(c), body-to-world.
    a = .3 * np.sin(.7*time)
    b = .2 * np.cos(1.1*time)
    c = .12*time + .1*np.sin(.9*time)
    ad, bd, cd = .21*np.cos(.7*time), -.22*np.sin(1.1*time), .12+.09*np.cos(.9*time)
    add, bdd, cdd = -.147*np.sin(.7*time), -.242*np.cos(1.1*time), -.081*np.sin(.9*time)
    cb, sb, cc, sc = np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    omega = np.column_stack((ad*cb*cc+bd*sc, -ad*cb*sc+bd*cc, ad*sb+cd))
    alpha = np.column_stack((
        add*cb*cc-ad*bd*sb*cc-ad*cd*cb*sc+bdd*sc+bd*cd*cc,
        -add*cb*sc+ad*bd*sb*sc-ad*cd*cb*cc+bdd*cc-bd*cd*sc,
        add*sb+ad*bd*cb+cdd))
    quaternion = Rotation.from_euler("XYZ", np.column_stack((a, b, c))).as_quat()
    return quaternion, omega, alpha


def test_body_omega_and_alpha_for_noncommuting_rotation():
    stamps = _irregular_time()
    query = np.linspace(.2, 2.8, 127)
    q, _, _ = _three_axis_motion(stamps)
    _, omega, alpha = _three_axis_motion(query)
    result = KIN.differentiate_quaternions(stamps, q, query,
                                         window_seconds=.08, degree=5)
    np.testing.assert_allclose(result["omega_body_rad_s"], omega, atol=1e-9, rtol=0)
    np.testing.assert_allclose(result["alpha_body_rad_s2"], alpha, atol=2e-8, rtol=0)


def test_accelerating_single_axis_and_quaternion_sign_invariance():
    stamps = _irregular_time()
    query = np.linspace(.2, 2.8, 41)
    axis = np.array([2., -3., 4.]) / np.sqrt(29.)
    theta = .4*stamps + .15*stamps**2
    quaternion = Rotation.from_rotvec(theta[:, None] * axis).as_quat()
    changed = quaternion.copy()
    changed[::3] *= -1.
    plain = KIN.differentiate_quaternions(stamps, quaternion, query, .1, 5)
    result = KIN.differentiate_quaternions(stamps, changed, query, .1, 5)
    np.testing.assert_allclose(result["omega_body_rad_s"],
                               (.4+.3*query)[:, None] * axis, atol=2e-9, rtol=0)
    np.testing.assert_allclose(result["alpha_body_rad_s2"],
                               np.tile(.3*axis, (len(query), 1)), atol=2e-7, rtol=0)
    np.testing.assert_allclose(np.abs(np.einsum("ni,ni->n", result["quaternion_xyzw"],
                                               plain["quaternion_xyzw"])), 1., atol=4e-16)
    for key in ("omega_body_rad_s", "alpha_body_rad_s2"):
        np.testing.assert_array_equal(result[key], plain[key])


def test_both_normalization_derivative_identities():
    stamps = _irregular_time()
    quaternion, _, _ = _three_axis_motion(stamps)
    result = KIN.differentiate_quaternions(stamps, quaternion,
                                         np.linspace(.2, 2.8, 51), .32, 3)
    np.testing.assert_allclose(np.linalg.norm(result["quaternion_xyzw"], axis=1), 1., atol=3e-16)
    np.testing.assert_allclose(result["quaternion_tangency_residual"], 0., atol=2e-17)
    np.testing.assert_allclose(result["quaternion_second_normalization_residual"], 0., atol=2e-17)


def test_same_first_derivative_as_archived_sg_method():
    old_path = PATH.parents[1] / "gyro_mocap_sg_check" / "sg_compare.py"
    old_spec = importlib.util.spec_from_file_location("archived_mocap_sg_for_test", old_path)
    old = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(old)
    stamps = _irregular_time()
    quaternion, _, _ = _three_axis_motion(stamps)
    query = np.linspace(.2, 2.8, 51)
    before = old.differentiate_quaternions(stamps, quaternion, query, .32, 3)
    after = KIN.differentiate_quaternions(stamps, quaternion, query, .32, 3)
    np.testing.assert_allclose(after["omega_body_rad_s"], before["omega_body"], atol=1e-16, rtol=0)


def test_adapter_preserves_original_fit_support_indices_and_ns():
    stamps = _irregular_time()
    quaternion, _, _ = _three_axis_motion(stamps)
    indices = np.arange(len(stamps), dtype=np.int64) + 700
    header = np.rint(stamps * 1e9).astype(np.int64) + 200000000000
    data = dict(mocap_time=stamps, mocap_quaternion_xyzw=quaternion,
                mocap_source_index=indices, mocap_stamp_ns=header,
                mocap_record_ns=header+300000)
    query = np.array([.4, 1., 2.4])
    result = KIN.prepare_kinematics(data, query)
    for name in ("mocap_source_index", "mocap_stamp_ns", "mocap_record_ns"):
        np.testing.assert_array_equal(result["source_fit_first_"+name],
                                      data[name][result["source_fit_first_row"]])
        np.testing.assert_array_equal(result["source_fit_last_"+name],
                                      data[name][result["source_fit_last_row"]])
    assert result["causal_filter"] is False
    assert result["uniform_time_resampling"] is False


def test_missing_support_and_insufficient_derivative_order_are_errors():
    stamps = _irregular_time()
    quaternion, _, _ = _three_axis_motion(stamps)
    with unittest.TestCase().assertRaisesRegex(ValueError, "full centered-window"):
        KIN.differentiate_quaternions(stamps, quaternion, [0.])
    with unittest.TestCase().assertRaisesRegex(ValueError, "second-derivative"):
        KIN.differentiate_quaternions(stamps, quaternion, [.4], degree=1)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()

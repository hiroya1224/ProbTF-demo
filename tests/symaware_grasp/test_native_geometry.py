import numpy as np
import pytest

from probtf.distributions import BinghamOrientation, OrientationKind
from probtf.geometry import axis_angle_to_quat, quat_mul, quat_to_rotmat
from symaware_grasp.geometry_utils import (
    axially_symmetric_bingham_parameter,
    quaternion_from_approach_and_finger_axes,
)


def test_semantic_axes_define_grasp_rotation_columns():
    quaternion = quaternion_from_approach_and_finger_axes([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    rotation = quat_to_rotmat(quaternion)

    np.testing.assert_allclose(rotation[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize(
    "approach,finger",
    [([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), ([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])],
)
def test_semantic_axes_reject_degenerate_inputs(approach, finger):
    with pytest.raises(ValueError):
        quaternion_from_approach_and_finger_axes(approach, finger)


def test_axial_bingham_keeps_symmetry_axis_as_weak_tangent():
    mode = np.array([1.0, 0.0, 0.0, 0.0])
    parameter = axially_symmetric_bingham_parameter(mode, [0.0, 0.0, 1.0], [500.0, 500.0, 0.1])
    orientation = BinghamOrientation.from_parameter_matrix(parameter, mode)
    symmetry_perturbation = quat_mul(mode, axis_angle_to_quat([0.0, 0.0, 1.0], 0.05))
    constrained_perturbation = quat_mul(mode, axis_angle_to_quat([1.0, 0.0, 0.0], 0.05))

    assert orientation.kind is OrientationKind.FINITE_BINGHAM
    assert float(symmetry_perturbation.T @ parameter @ symmetry_perturbation) > float(
        constrained_perturbation.T @ parameter @ constrained_perturbation
    )

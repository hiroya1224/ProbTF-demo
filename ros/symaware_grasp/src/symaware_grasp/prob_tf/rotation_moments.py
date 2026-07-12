import numpy as np

from symaware_grasp.prob_tf.bingham_moments import bingham_fourth_moment, bingham_second_moment
from symaware_grasp.prob_tf.geometry import quat_to_rotmat


def _rotation_quadratic_forms():
    forms = []

    forms.append(np.diag([1.0, 1.0, -1.0, -1.0]))
    form = np.zeros((4, 4), dtype=float)
    form[0, 3] = form[3, 0] = -1.0
    form[1, 2] = form[2, 1] = 1.0
    forms.append(form)
    form = np.zeros((4, 4), dtype=float)
    form[0, 2] = form[2, 0] = 1.0
    form[1, 3] = form[3, 1] = 1.0
    forms.append(form)

    form = np.zeros((4, 4), dtype=float)
    form[0, 3] = form[3, 0] = 1.0
    form[1, 2] = form[2, 1] = 1.0
    forms.append(form)
    forms.append(np.diag([1.0, -1.0, 1.0, -1.0]))
    form = np.zeros((4, 4), dtype=float)
    form[0, 1] = form[1, 0] = -1.0
    form[2, 3] = form[3, 2] = 1.0
    forms.append(form)

    form = np.zeros((4, 4), dtype=float)
    form[0, 2] = form[2, 0] = -1.0
    form[1, 3] = form[3, 1] = 1.0
    forms.append(form)
    form = np.zeros((4, 4), dtype=float)
    form[0, 1] = form[1, 0] = 1.0
    form[2, 3] = form[3, 2] = 1.0
    forms.append(form)
    forms.append(np.diag([1.0, -1.0, -1.0, 1.0]))

    return forms


ROTATION_ENTRY_FORMS = _rotation_quadratic_forms()
ROTATION_ENTRY_INDEX_ORDER = [
    (0, 0),
    (1, 0),
    (2, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (0, 2),
    (1, 2),
    (2, 2),
]


class RotationMoment:
    def __init__(self, mean_rot, kron_rot):
        self.mean_rot = np.asarray(mean_rot, dtype=float).reshape(3, 3)
        self.kron_rot = np.asarray(kron_rot, dtype=float).reshape(9, 9)

    def apply_second(self, mat):
        vec_mat = np.asarray(mat, dtype=float).reshape(9, order="F")
        vec_out = self.kron_rot @ vec_mat
        return vec_out.reshape((3, 3), order="F")

    def compose(self, other):
        return RotationMoment(self.mean_rot @ other.mean_rot, self.kron_rot @ other.kron_rot)


def identity_rotation_moment():
    return RotationMoment(np.eye(3, dtype=float), np.eye(9, dtype=float))


def deterministic_rotation_moment_from_quaternion(quaternion):
    rotation = quat_to_rotmat(quaternion)
    return RotationMoment(rotation, np.kron(rotation, rotation))


def compute_mean_rot_from_c2(second_moment):
    mean_rotation = np.zeros((3, 3), dtype=float)
    for flat_index, form in enumerate(ROTATION_ENTRY_FORMS):
        row, col = divmod(flat_index, 3)
        mean_rotation[row, col] = float(np.sum(form * second_moment))
    return mean_rotation


def compute_kron_rot_from_c4(fourth_moment):
    kron_rotation = np.zeros((9, 9), dtype=float)
    for row_index, (out_row, out_col) in enumerate(ROTATION_ENTRY_INDEX_ORDER):
        for col_index, (in_row, in_col) in enumerate(ROTATION_ENTRY_INDEX_ORDER):
            form_a = ROTATION_ENTRY_FORMS[out_row * 3 + in_row]
            form_b = ROTATION_ENTRY_FORMS[out_col * 3 + in_col]
            kron_rotation[row_index, col_index] = float(
                np.einsum("ab,cd,abcd->", form_a, form_b, fourth_moment)
            )
    return kron_rotation


def rotation_moment_from_bingham(param_mat, integration_steps=120):
    second_moment = bingham_second_moment(param_mat, integration_steps=integration_steps)
    fourth_moment = bingham_fourth_moment(param_mat, integration_steps=integration_steps)
    mean_rotation = compute_mean_rot_from_c2(second_moment)
    kron_rotation = compute_kron_rot_from_c4(fourth_moment)
    return RotationMoment(mean_rotation, kron_rotation)

"""Finite modal geometry and derivatives in one frozen original body chart.

For every edge, t=t_ref+delta_t and q=q_ref*rvq(delta_r). This evaluates
that geometric embedding; it does not change a body law, covariance or prior,
and adds no first-order distribution-mean correction. Full delta columns are
the geometric extension, including formally fixed coordinates. The solver
must supply delta=L*xi on its frozen prior support, with Dirac rows zero.

The explicit python_reference backend uses homogeneous-matrix product rules.
It is a comparison path, never a silent fallback from the native backend.
"""
import ctypes as ct
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from probtf.geometry import (quat_mul, quat_to_rotmat, rotation_vector_to_quaternion,
                             rotmat_to_quat, skew)
from probtf.spherical_law.first_order import _array, _path, _right_jacobian

from .body_native import _BYTE, _DOUBLE, _INT, _inputs, _library, _native, _pointer


@dataclass(frozen=True)
class ModalPathEvaluation:
    translation: np.ndarray
    rotation_wxyz: np.ndarray
    J_t: np.ndarray
    J_right_rot: np.ndarray
    delta: np.ndarray
    source_state_id: str
    source_stamp: float
    source_command: tuple
    source_version: int
    source_component_id: str
    path: tuple
    backend: str

    def __post_init__(self):
        delta = _array(self.delta)
        if delta.ndim != 1:
            raise ValueError('Finite modal coordinates must be a vector.')
        for name, shape in (('translation', (3,)), ('rotation_wxyz', (4,)),
                            ('J_t', (3, delta.size)), ('J_right_rot', (3, delta.size))):
            object.__setattr__(self, name, _array(getattr(self, name), shape))
        object.__setattr__(self, 'delta', delta)
        object.__setattr__(self, 'source_command', tuple(self.source_command))
        object.__setattr__(self, 'path', tuple(self.path))

    @property
    def reference_state_id(self):
        return self.source_state_id

    def __reduce__(self):
        # Keep the same readonly-array contract after a spawn/pickle boundary.
        return type(self), tuple(getattr(self, name) for name in self.__dataclass_fields__)


@lru_cache(maxsize=1)
def _finite_library():
    library = _library()
    try:
        version = library.probtf_body_finite_modal_abi_version
        function = library.probtf_body_finite_modal_path
    except AttributeError as error:
        raise RuntimeError('Native finite-modal geometry is not built; rebuild '
                           'probtf_observation_fusion_demo.') from error
    version.restype = ct.c_int
    if version() != 1:
        raise RuntimeError('Native finite-modal C ABI is incompatible.')
    function.argtypes = ([ct.c_int, _DOUBLE, _DOUBLE, _DOUBLE, ct.c_int, _INT, _BYTE]
                         + [_DOUBLE]*4 + [ct.c_char_p, ct.c_size_t])
    function.restype = ct.c_int
    return function


def _reference(translations, quaternions, delta, resolved):
    """Independent full homogeneous-matrix product-rule comparison backend."""
    n = delta.size
    total = np.eye(4)
    derivative = np.zeros((n, 4, 4))
    for edge, inverse in resolved:
        offset = 6*edge
        q = quat_mul(quaternions[edge], rotation_vector_to_quaternion(delta[offset+3:offset+6]))
        local = np.eye(4)
        local[:3, :3] = quat_to_rotmat(q)
        local[:3, 3] = translations[edge]+delta[offset:offset+3]
        local_derivative = np.zeros((n, 4, 4))
        Jr = _right_jacobian(delta[offset+3:offset+6])
        for axis in range(3):
            local_derivative[offset+axis, axis, 3] = 1.
            local_derivative[offset+3+axis, :3, :3] = local[:3, :3] @ skew(Jr[:, axis])
        if inverse:
            inv = np.eye(4)
            inv[:3, :3] = local[:3, :3].T
            inv[:3, 3] = -inv[:3, :3] @ local[:3, 3]
            local_derivative = -np.einsum('ij,ajk,kl->ail', inv, local_derivative, inv)
            local = inv
        derivative = (np.einsum('aij,jk->aik', derivative, local)
                      + np.einsum('ij,ajk->aik', total, local_derivative))
        total = total @ local
    right = np.einsum('ij,ajk->aik', total[:3, :3].T, derivative[:, :3, :3])
    Jw = .5*np.array([right[:, 2, 1]-right[:, 1, 2],
                       right[:, 0, 2]-right[:, 2, 0],
                       right[:, 1, 0]-right[:, 0, 1]])
    return total[:3, 3], rotmat_to_quat(total[:3, :3]), derivative[:, :3, 3].T, Jw


def finite_modal_path(state, path, delta):
    """Evaluate finite geometry, d(translation)/d(delta), and right dR.

    ``dR = R * skew(J_right_rot @ ddelta)`` fixes the rotation-Jacobian
    convention. The original source ID, stamp and command remain attached;
    no new distribution, source clock or nominal instruction is constructed.
    """
    translations, quaternions, _, _ = _inputs(state)
    delta = np.ascontiguousarray(delta, dtype=np.float64)
    if delta.shape != state.mean.shape or not np.all(np.isfinite(delta)):
        raise ValueError('Finite modal coordinates must match the source dimension and be finite.')
    resolved = _path(state, path)
    native = _native()
    if native:
        indices = np.ascontiguousarray([edge for edge, _ in resolved], dtype=np.int32)
        inverse = np.ascontiguousarray([value for _, value in resolved], dtype=np.uint8)
        translation, quaternion = np.empty(3), np.empty(4)
        Jt, Jw = np.empty((3, delta.size)), np.empty((3, delta.size))
        error = ct.create_string_buffer(2048)
        status = _finite_library()(len(state.transforms), _pointer(translations),
            _pointer(quaternions), _pointer(delta), len(resolved),
            indices.ctypes.data_as(_INT), inverse.ctypes.data_as(_BYTE),
            *map(_pointer, (translation, quaternion, Jt, Jw)), error, len(error))
        if status:
            raise ValueError(error.value.decode(errors='replace'))
    else:
        translation, quaternion, Jt, Jw = _reference(translations, quaternions, delta, resolved)
    source_path = tuple((state.edge_ids[edge], inverse) for edge, inverse in resolved)
    return ModalPathEvaluation(translation, quaternion, Jt, Jw, delta,
        state.state_id, state.stamp, state.command, state.version,
        state.component_id, source_path, 'native' if native else 'python_reference')

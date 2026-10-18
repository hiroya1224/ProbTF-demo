"""Demo-local native evaluation of the unchanged core first-order body formulas.

The native implementation owns no body law or observation history. It receives
immutable core inputs and returns the existing immutable core state/query types.
A declared python_reference backend remains available for numerical comparisons;
there is no silent fallback after a native failure and no runtime compilation.
"""
import ctypes as ct
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, PointQuery, _array, _path,
    point_query as reference_point_query, pose_query as reference_pose_query,
    direction_query as reference_direction_query,
    reexpress_state as reference_reexpress_state,
    _right_jacobian, bingham_from_rotation_coordinates,
)
from probtf.geometry import quat_mul, rotation_vector_to_quaternion
from probtf.spherical_law.support import support_inverse

_DOUBLE = ct.POINTER(ct.c_double)
_BYTE = ct.POINTER(ct.c_ubyte)
_INT = ct.POINTER(ct.c_int)


def _library_path():
    explicit = os.environ.get('PROBTF_BODY_LIBRARY')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError('Selected native body library does not exist: '+str(path))
        return path
    directories = []
    for name in ('LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'):
        directories.extend(Path(p) if name == 'LD_LIBRARY_PATH' else Path(p)/'lib'
                           for p in os.environ.get(name, '').split(os.pathsep) if p)
    for parent in Path(__file__).resolve().parents:
        if (parent/'devel').is_dir() and (parent/'src').is_dir():
            directories += [parent/'devel/lib', parent/'devel/.private/probtf_observation_fusion_demo/lib']
    for directory in directories:
        path = directory/'libprobtf_body_law.so'
        if path.is_file():
            return path.resolve()
    raise RuntimeError('Native body backend is not built. Run catkin build probtf_observation_fusion_demo and source the workspace setup.')


def _pointer(array):
    return array.ctypes.data_as(_DOUBLE)


@lru_cache(maxsize=1)
def _library():
    lib = ct.CDLL(str(_library_path()))
    lib.probtf_body_abi_version.restype = ct.c_int
    if lib.probtf_body_abi_version() != 1:
        raise RuntimeError('Native body C ABI is incompatible.')
    lib.probtf_body_reexpress.argtypes = [ct.c_int, _DOUBLE, _DOUBLE, _BYTE]+[_DOUBLE]*7+[ct.c_char_p, ct.c_size_t]
    lib.probtf_body_reexpress.restype = ct.c_int
    lib.probtf_body_query.argtypes = [ct.c_int, _DOUBLE, _DOUBLE, _BYTE, _DOUBLE,
        ct.c_int, _INT, _BYTE, _DOUBLE, ct.c_int]+[_DOUBLE]*9+[ct.c_char_p, ct.c_size_t]
    lib.probtf_body_query.restype = ct.c_int
    return lib


@lru_cache(maxsize=1)
def _materialization_library():
    lib = _library()
    if not hasattr(lib, 'probtf_body_materialize_in_chart_abi_version'):
        raise RuntimeError('Native body library lacks the common-chart materialization extension; rebuild the demo.')
    lib.probtf_body_materialize_in_chart_abi_version.restype = ct.c_int
    if lib.probtf_body_materialize_in_chart_abi_version() != 1:
        raise RuntimeError('Native common-chart materialization ABI is incompatible.')
    lib.probtf_body_materialize_in_chart.argtypes = lib.probtf_body_reexpress.argtypes
    lib.probtf_body_materialize_in_chart.restype = ct.c_int
    return lib


def backend_record():
    selected = os.environ.get('PROBTF_BODY_BACKEND', 'native')
    if selected not in ('native', 'python_reference'):
        raise ValueError('PROBTF_BODY_BACKEND must be native or python_reference.')
    return dict(backend=selected, core_equations_changed=False, stochastic_sampling=False,
                common_chart_materialization_supported=True,
                common_chart_materialization_abi=(None if selected == 'python_reference' else
                    _materialization_library().probtf_body_materialize_in_chart_abi_version()),
                strict_reexpression_prior_support_check_preserved=True,
                library_path=None if selected == 'python_reference' else str(_library_path()),
                library_sha256=None if selected == 'python_reference' else hashlib.sha256(_library_path().read_bytes()).hexdigest())


def _native():
    value = os.environ.get('PROBTF_BODY_BACKEND', 'native')
    if value not in ('native', 'python_reference'):
        raise ValueError('PROBTF_BODY_BACKEND must be native or python_reference.')
    return value == 'native'


def _inputs(state):
    if not isinstance(state, JointTransformState):
        raise TypeError('Native body queries require the existing immutable core joint state.')
    translation = np.ascontiguousarray([e.translation for e in state.transforms], dtype=np.float64)
    quaternion = np.ascontiguousarray([e.rotation_wxyz for e in state.transforms], dtype=np.float64)
    random = np.ascontiguousarray([e.bingham is not None for e in state.transforms], dtype=np.uint8)
    covariance = np.ascontiguousarray(state.covariance, dtype=np.float64)
    return translation, quaternion, random, covariance


def _query(state, path, point, kind, query_id, frame):
    translation, quaternion, random, covariance = _inputs(state)
    resolved = _path(state, path)
    indices = np.ascontiguousarray([i for i, _ in resolved], dtype=np.int32)
    inverse = np.ascontiguousarray([v for _, v in resolved], dtype=np.uint8)
    point = np.ascontiguousarray(point, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError('A body query requires a finite three-dimensional point/direction.')
    dimension, n = (6 if kind == 2 else 3), state.mean.size
    mean, mode, correction = [np.empty(dimension) for _ in range(3)]
    jacobian, cross = np.empty((dimension, n)), np.empty((n, dimension))
    output, within, between = [np.empty((dimension, dimension)) for _ in range(3)]
    rotation = np.empty(4)
    error = ct.create_string_buffer(2048)
    status = _library().probtf_body_query(len(state.transforms), _pointer(translation),
        _pointer(quaternion), random.ctypes.data_as(_BYTE), _pointer(covariance),
        len(resolved), indices.ctypes.data_as(_INT), inverse.ctypes.data_as(_BYTE),
        _pointer(point), kind, *map(_pointer, (mean, mode, correction, jacobian,
            output, cross, within, between, rotation)), error, len(error))
    if status:
        raise ValueError(error.value.decode(errors='replace'))
    return PointQuery(mean, output, jacobian, cross, within, between, str(query_id),
        str(frame), state.stamp, state.command, state.version, state.component_id,
        mode, correction, rotation if kind == 2 else None, state_id=state.state_id)


def point_query(state, path, point=(0., 0., 0.), query_id='point', frame='base_link'):
    if not _native():
        return reference_point_query(state, path, point, query_id, frame)
    return _query(state, path, point, 0, query_id, frame)


def direction_query(state, path, direction, query_id='direction', frame='base_link'):
    if not _native():
        return reference_direction_query(state, path, direction, query_id, frame)
    return _query(state, path, direction, 1, query_id, frame)


def pose_query(state, path, query_id='pose', frame='base_link'):
    if not _native():
        return reference_pose_query(state, path, query_id, frame)
    if not state.transforms:
        # The reference orientation Jacobian stacks one column per source
        # coordinate and is undefined for a completely empty source state.
        raise ValueError('A pose query requires at least one transform variable.')
    return _query(state, path, (0., 0., 0.), 2, query_id, frame)


def _state_identity(state):
    identity = hashlib.sha256()
    identity.update(state.covariance.tobytes())
    identity.update(repr((state.version, state.stamp, state.command, state.component_id)).encode('utf-8'))
    for edge in state.transforms:
        identity.update(edge.edge_id.encode('utf-8'))
        identity.update(edge.translation.tobytes())
        identity.update(edge.rotation_wxyz.tobytes())
        if edge.bingham is not None:
            identity.update(edge.bingham.tobytes())
    return identity.hexdigest()


def reexpress_state(state, mean, covariance, version=None, component_id=None):
    if not _native():
        return reference_reexpress_state(state, mean, covariance, version, component_id)
    return _native_materialize(state, mean, covariance, version, component_id,
                               check_source_support=True)


def _reference_materialize_in_chart(state, mean, covariance, version, component_id):
    """The same core equations without an unrelated prior-support condition."""
    if not isinstance(state, JointTransformState):
        raise TypeError('Common-chart materialization needs an immutable core reference geometry.')
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = state.mean.size
    if mean.shape != (n,) or covariance.shape != (n, n):
        raise ValueError('Posterior dimensions differ from the frozen body state.')
    if not np.all(np.isfinite(mean)):
        raise ValueError('Common-chart candidate mean must be finite.')
    support_inverse(covariance, name='common-chart candidate covariance')
    chart = np.eye(n)
    for i, edge in enumerate(state.transforms):
        if edge.bingham is not None:
            block = slice(6*i+3, 6*i+6)
            chart[block, block] = _right_jacobian(mean[block])
    transported = chart @ covariance @ chart.T
    transforms = []
    for i, edge in enumerate(state.transforms):
        quaternion = quat_mul(edge.rotation_wxyz,
            rotation_vector_to_quaternion(mean[6*i+3:6*i+6]))
        rotation_covariance = transported[6*i+3:6*i+6, 6*i+3:6*i+6]
        transforms.append(BinghamTransform(edge.edge_id,
            edge.translation+mean[6*i:6*i+3],
            None if edge.bingham is None else bingham_from_rotation_coordinates(quaternion, rotation_covariance),
            transported[6*i:6*i+3, 6*i:6*i+3],
            quaternion if edge.bingham is None else None))
    return JointTransformState(tuple(transforms), transported,
        state.version+1 if version is None else int(version), state.stamp, state.command,
        state.component_id if component_id is None else str(component_id))


def materialize_state_in_chart(reference_geometry, mean, covariance, version=None,
                               component_id=None):
    """Express another law in this chart; do not condition on its reference P.

    This is NOT an alternate Bayesian posterior update. Its caller owns the
    source-law support/provenance and known-attachment consistency. The
    supplied finite PSD full covariance is preserved as G P G.T, with the
    existing t/q/Bingham parameter equations and actual fixed-rotation checks.
    The reference covariance does not constrain this other law's mean.
    """
    if not _native():
        return _reference_materialize_in_chart(reference_geometry, mean, covariance,
                                                version, component_id)
    return _native_materialize(reference_geometry, mean, covariance, version,
                               component_id, check_source_support=False)


def _native_materialize(state, mean, covariance, version, component_id,
                        *, check_source_support):
    t, q, random, prior = _inputs(state)
    mean = np.ascontiguousarray(mean, dtype=np.float64)
    covariance = np.ascontiguousarray(covariance, dtype=np.float64)
    n, edges = state.mean.size, len(state.transforms)
    if mean.shape != (n,) or covariance.shape != (n, n):
        raise ValueError('Posterior dimensions differ from the frozen body state.')
    translated, rotated = np.empty_like(t), np.empty_like(q)
    parameter, transported = np.empty((edges, 4, 4)), np.empty((n, n))
    error = ct.create_string_buffer(2048)
    function = (_library().probtf_body_reexpress if check_source_support else
                _materialization_library().probtf_body_materialize_in_chart)
    status = function(edges, _pointer(t), _pointer(q),
        random.ctypes.data_as(_BYTE), _pointer(prior), _pointer(mean), _pointer(covariance),
        _pointer(translated), _pointer(rotated), _pointer(parameter), _pointer(transported),
        error, len(error))
    if status:
        raise ValueError(error.value.decode(errors='replace'))
    # Native performs the same support, marginal-consistency and Bingham-gap
    # validation as the existing constructors. Construct their
    # exact immutable types from that validated result, avoiding a duplicate
    # Python eigensystem for every edge and another full support decomposition.
    transforms = []
    for i, old in enumerate(state.transforms):
        edge = object.__new__(BinghamTransform)
        for name, value in dict(edge_id=old.edge_id, translation=_array(translated[i]),
                translation_covariance=_array(transported[6*i:6*i+3, 6*i:6*i+3]),
                rotation_wxyz=_array(rotated[i]),
                bingham=_array(parameter[i]) if random[i] else None).items():
            object.__setattr__(edge, name, value)
        transforms.append(edge)
    result = object.__new__(JointTransformState)
    for name, value in dict(transforms=tuple(transforms), covariance=_array(transported),
            mean=_array(np.zeros(n)), version=state.version+1 if version is None else int(version),
            stamp=state.stamp, command=state.command,
            component_id=state.component_id if component_id is None else str(component_id)).items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, 'state_id', _state_identity(result))
    return result

"""Native analytic reverse derivatives of the existing moment propagation.

The arguments ``mean`` and ``covariance`` describe a Gaussian in the ORIGINAL
reference state's joint chart.  Every evaluation performs the same mean move
and complete covariance transport as ``reexpress_state`` before evaluating the
unchanged high-concentration O(P) query formulas.  A pose query is transported
to one FIXED native observation rotation; a direction/point query may have a
fixed linear projection.  Chart transport, its parameter dependence, O(P)
corrections and all covariance entries are differentiated together.

This module owns neither an optimizer nor a new body law.  The caller owns the
original Gaussian support, PSD parameterization and eventual core-state
construction/domain validation.  It must not interpret this evaluator as a
point posterior, an exact nonlinear push-forward, or a Gaussianity guarantee
outside the existing high-concentration approximation.  The core moment
equations are unchanged; its concentration gates were explicitly removed at
the user's request.  Native errors never silently switch to Python or
numerical differences.
"""
import ctypes as ct
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path

import numpy as np
from probtf.spherical_law.first_order import JointTransformState, _path
from .immutable_body_cache import freeze_json_record


_DOUBLE = ct.POINTER(ct.c_double)
_BYTE = ct.POINTER(ct.c_ubyte)
_INT = ct.POINTER(ct.c_int)


def _library_path():
    explicit = os.environ.get('PROBTF_MOMENT_REVERSE_LIBRARY')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError('Selected moment reverse library does not exist: '+str(path))
        return path
    directories = []
    for variable in ('LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'):
        directories.extend(Path(item) if variable == 'LD_LIBRARY_PATH' else Path(item)/'lib'
                           for item in os.environ.get(variable, '').split(os.pathsep) if item)
    for parent in Path(__file__).resolve().parents:
        if (parent/'devel').is_dir() and (parent/'src').is_dir():
            directories += [parent/'devel/lib', parent/'devel/.private/probtf_observation_fusion_demo/lib']
    for directory in directories:
        path = directory/'libprobtf_moment_reverse.so'
        if path.is_file():
            return path.resolve()
    raise RuntimeError('Native moment reverse backend is not built. Run catkin build '
                       'probtf_observation_fusion_demo and source the workspace setup.')


@lru_cache(maxsize=1)
def _library():
    library = ct.CDLL(str(_library_path()))
    library.probtf_moment_reverse_abi_version.restype = ct.c_int
    if library.probtf_moment_reverse_abi_version() != 1:
        raise RuntimeError('Native moment reverse ABI is incompatible.')
    library.probtf_moment_reverse_evaluate.argtypes = [
        ct.c_int, _DOUBLE, _DOUBLE, _BYTE, _DOUBLE, _DOUBLE,
        ct.c_int, _INT, _BYTE, _DOUBLE, ct.c_int, _DOUBLE,
        ct.c_int, _DOUBLE, ct.c_int,
    ] + [_DOUBLE]*9 + [ct.POINTER(ct.c_longlong), ct.c_char_p, ct.c_size_t]
    library.probtf_moment_reverse_evaluate.restype = ct.c_int
    return library


def _pointer(value):
    return value.ctypes.data_as(_DOUBLE)


def _readonly(value):
    out = np.array(value, dtype=float, copy=True, order='C')
    out.setflags(write=False)
    return out


def _array(value, shape, name):
    out = np.ascontiguousarray(value, dtype=np.float64)
    if out.shape != shape or not np.all(np.isfinite(out)):
        raise ValueError(name+' must have the required shape and be finite.')
    return out


@dataclass(frozen=True)
class MomentReverseEvaluation:
    mean: np.ndarray
    covariance: np.ndarray
    gradient_mean: np.ndarray
    gradient_covariance: np.ndarray
    cost: float
    expected_negative_log_likelihood: object
    expected_negative_log_likelihood_terms: object
    source_state_id: str
    source_stamp: float
    source_command: tuple
    source_version: int
    source_component_id: str
    diagnostics: dict

    def __post_init__(self):
        for name in ('mean', 'covariance', 'gradient_mean', 'gradient_covariance'):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        object.__setattr__(self, 'diagnostics', freeze_json_record(self.diagnostics))

    def __reduce__(self):
        # NumPy pickle reconstructs writable arrays; rerun the dataclass
        # constructor so spawn/IPC preserves the original immutability.
        return type(self), tuple(getattr(self, name) for name in self.__dataclass_fields__)


class MomentQueryEvaluator:
    """Frozen source/path/native chart with reusable native input buffers.

    ``gradient_covariance`` uses the symmetric full-matrix Frobenius
    convention: dF = g_m @ dm + sum(g_P * dP).  An off-diagonal symmetric
    perturbation therefore contributes both entries; it is NOT a packed
    lower-triangular parameter gradient.  The optimizer chains through its
    own fixed-support Cholesky parameterization.
    """
    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError('A moment evaluator source/native chart is immutable.')
        object.__setattr__(self, name, value)

    def __init__(self, state, path, *, kind='pose', point_or_direction=(0., 0., 0.),
                 reference_rotation_wxyz=None, projection_matrix=None):
        if not isinstance(state, JointTransformState):
            raise TypeError('Moment reverse evaluation needs the immutable core joint state.')
        kinds = dict(point=0, direction=1, pose=2)
        if kind not in kinds:
            raise ValueError('Moment query kind must be point, direction, or pose.')
        self.state = state
        self.kind = kind
        self._kind = kinds[kind]
        self._raw_dimension = 6 if kind == 'pose' else 3
        self._n = state.mean.size
        self._translation = np.ascontiguousarray([edge.translation for edge in state.transforms], dtype=float).reshape(-1, 3)
        self._quaternion = np.ascontiguousarray([edge.rotation_wxyz for edge in state.transforms], dtype=float).reshape(-1, 4)
        self._random = np.ascontiguousarray([edge.bingham is not None for edge in state.transforms], dtype=np.uint8)
        resolved = _path(state, path)
        self._indices = np.ascontiguousarray([index for index, unused in resolved], dtype=np.int32)
        self._inverse = np.ascontiguousarray([inverse for unused, inverse in resolved], dtype=np.uint8)
        self.path = tuple((state.edge_ids[index], bool(inverse)) for index, inverse in resolved)
        # pose_query always queries the path-frame origin; its orientation
        # rows and translation/orientation cross share that same origin.
        point = (0., 0., 0.) if kind == 'pose' else point_or_direction
        self._point = _readonly(_array(point, (3,), 'Query point/direction'))
        reference = (1., 0., 0., 0.) if reference_rotation_wxyz is None else reference_rotation_wxyz
        self._reference = _array(reference, (4,), 'Fixed native quaternion')
        norm = float(np.linalg.norm(self._reference))
        if norm == 0.:
            raise ValueError('Fixed native quaternion must be nonzero.')
        self._reference = self._reference/norm
        if projection_matrix is None:
            self._projection = np.empty((0, self._raw_dimension))
            self.dimension = self._raw_dimension
        else:
            projection = np.ascontiguousarray(projection_matrix, dtype=float)
            if (projection.ndim != 2 or projection.shape[1] != self._raw_dimension
                    or projection.shape[0] == 0 or not np.all(np.isfinite(projection))):
                raise ValueError('Fixed projection matrix must have positive rows and the query column dimension.')
            self._projection = _readonly(projection)
            self.dimension = projection.shape[0]
        for name in ('_translation', '_quaternion', '_random', '_indices',
                     '_inverse', '_reference', '_projection'):
            getattr(self, name).setflags(write=False)
        self._frozen = True

    def evaluate(self, mean, covariance, *, likelihood_mean=None, likelihood_precision=None,
                 likelihood_log_normalizer=0., adjoint_mean=None, adjoint_covariance=None):
        mean = _array(mean, (self._n,), 'Original-chart Gaussian mean')
        covariance = _array(covariance, (self._n, self._n), 'Original-chart Gaussian covariance')
        dimension = self.dimension
        likelihood = likelihood_mean is not None or likelihood_precision is not None
        adjoints = adjoint_mean is not None or adjoint_covariance is not None
        if likelihood and adjoints:
            raise ValueError('Choose likelihood cost or explicit moment adjoints for one evaluation.')
        if likelihood and (likelihood_mean is None or likelihood_precision is None):
            raise ValueError('Likelihood mean and precision must be supplied together.')
        target = np.zeros(dimension) if not likelihood else _array(likelihood_mean, (dimension,), 'Native likelihood mean')
        precision = np.zeros((dimension, dimension)) if not likelihood else _array(likelihood_precision, (dimension, dimension), 'Native likelihood precision')
        a = np.zeros(dimension) if adjoint_mean is None else _array(adjoint_mean, (dimension,), 'Mean adjoint')
        B = np.zeros((dimension, dimension)) if adjoint_covariance is None else _array(adjoint_covariance, (dimension, dimension), 'Covariance adjoint')
        normalizer = float(likelihood_log_normalizer)
        if not math.isfinite(normalizer):
            raise ValueError('Native likelihood log normalizer must be finite.')
        output_mean = np.empty(dimension)
        output_covariance = np.empty((dimension, dimension))
        gradient_mean = np.empty(self._n)
        gradient_covariance = np.empty((self._n, self._n))
        cost = np.empty(1)
        nodes = ct.c_longlong()
        error = ct.create_string_buffer(2048)
        status = _library().probtf_moment_reverse_evaluate(
            len(self.state.transforms), _pointer(self._translation), _pointer(self._quaternion),
            self._random.ctypes.data_as(_BYTE), _pointer(mean), _pointer(covariance),
            len(self._indices), self._indices.ctypes.data_as(_INT), self._inverse.ctypes.data_as(_BYTE),
            _pointer(self._point), self._kind, _pointer(self._reference), len(self._projection),
            _pointer(self._projection), 1 if likelihood else (2 if adjoints else 0),
            *map(_pointer, (target, precision, a, B, output_mean, output_covariance,
                           cost, gradient_mean, gradient_covariance)),
            ct.byref(nodes), error, len(error))
        if status:
            raise ValueError(error.value.decode(errors='replace'))
        terms = None
        if likelihood:
            residual = output_mean-target
            quadratic = float(.5*residual@precision@residual)
            trace = float(.5*np.sum(precision.T*output_covariance))
            terms = quadratic, trace, normalizer
        return MomentReverseEvaluation(output_mean, output_covariance, gradient_mean,
            gradient_covariance, float(cost[0]),
            float(cost[0])+normalizer if likelihood else None, terms,
            self.state.state_id, self.state.stamp, self.state.command, self.state.version,
            self.state.component_id, dict(backend='native_analytic_reverse',
                original_reference_chart=True, full_joint_covariance=True,
                O_P_mean_correction=True, all_parameter_dependence_differentiated=True,
                native_chart_fixed=True, kind=self.kind, path=self.path,
                cost_mode='expected_gaussian_negative_log_likelihood' if likelihood else ('moment_adjoint' if adjoints else 'moments_only'),
                covariance_gradient_convention='symmetric full-matrix Frobenius',
                stochastic_sampling=False, numerical_parameter_differences=False,
                core_equations_changed=False, tape_nodes=nodes.value))


def evaluate_moment_query(state, path, mean, covariance, *, kind='pose',
        point_or_direction=(0., 0., 0.), reference_rotation_wxyz=None,
        projection_matrix=None, likelihood_mean=None, likelihood_precision=None,
        likelihood_log_normalizer=0., adjoint_mean=None, adjoint_covariance=None):
    """One-call convenience API; repeated optimization should reuse a context."""
    evaluator = MomentQueryEvaluator(state, path, kind=kind,
        point_or_direction=point_or_direction, reference_rotation_wxyz=reference_rotation_wxyz,
        projection_matrix=projection_matrix)
    return evaluator.evaluate(mean, covariance, likelihood_mean=likelihood_mean,
        likelihood_precision=likelihood_precision, likelihood_log_normalizer=likelihood_log_normalizer,
        adjoint_mean=adjoint_mean, adjoint_covariance=adjoint_covariance)


def backend_record():
    path = _library_path()
    binding = Path(__file__).resolve()
    source = binding.parents[2]/'native'/'moment_reverse.cpp'
    return dict(backend='native_analytic_reverse', library_path=str(path),
                library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                native_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
                binding_sha256=hashlib.sha256(binding.read_bytes()).hexdigest(),
                stochastic_sampling=False, runtime_compilation=False,
                reverse_derivatives=True, core_equations_changed=False)

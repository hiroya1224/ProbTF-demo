"""Demo-local native arithmetic for the existing Bingham moment equations.

The same Chen--Tanaka integral, integration steps, gap parametrization,
L-BFGS-B settings and relative-moment refinement are used. Core is unchanged.
The shared library must be built ahead of time; there is no runtime compiler
or silent fallback to another moment approximation.
"""
import ctypes as ct
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from probtf.bingham import (
    _integration_steps, _symmetric_matrix, _validated_second_moment,
    _eigenvalues_to_parameter_vector, _parameter_vector_to_eigenvalues,
    canonical_bingham_parameter,
)


_POINTER = ct.POINTER(ct.c_double)
_LIBRARY = 'libprobtf_marker_board_math.so'


@lru_cache(maxsize=1)
def _library():
    explicit = os.environ.get('PROBTF_MARKER_BOARD_MATH_LIBRARY')
    candidates = [Path(explicit).expanduser()] if explicit else []
    if not explicit:
        for variable in ('LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'):
            for value in os.environ.get(variable, '').split(os.pathsep):
                if value:
                    candidates.append((Path(value) if variable == 'LD_LIBRARY_PATH'
                                       else Path(value)/'lib')/_LIBRARY)
        for ancestor in Path(__file__).resolve().parents:
            if (ancestor/'devel').is_dir() and (ancestor/'src').is_dir():
                candidates.extend((ancestor/'devel/lib'/_LIBRARY,
                    ancestor/'devel/.private/probtf_observation_fusion_demo/lib'/_LIBRARY))
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError('Native marker-board arithmetic is not built. Run catkin build '
                           'probtf_observation_fusion_demo and source the workspace.')
    library = ct.CDLL(str(path.resolve()))
    library.probtf_backend_path = str(path.resolve())
    library.probtf_marker_board_math_abi_version.restype = ct.c_int
    if library.probtf_marker_board_math_abi_version() != 1:
        raise RuntimeError('Native marker-board arithmetic C ABI is incompatible.')
    library.probtf_marker_board_normalizer.argtypes = [
        _POINTER, ct.c_int, ct.c_int, _POINTER, _POINTER, _POINTER,
        ct.c_char_p, ct.c_size_t]
    library.probtf_marker_board_normalizer.restype = ct.c_int
    return library


@lru_cache(maxsize=2)
def _backend_record(backend):
    root = Path(__file__).resolve().parents[2]
    if backend == 'native':
        library = Path(_library().probtf_backend_path)
        source = root/'native/marker_board_math.cpp'
        return dict(arithmetic_backend='native', library_path=str(library),
            library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            native_source_sha256=(hashlib.sha256(source.read_bytes()).hexdigest()
                                  if source.is_file() else None),
            binding_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime_compilation=False, silent_fallback=False)
    if backend != 'python':
        raise ValueError('Marker-board arithmetic backend must be native or python.')
    import probtf.bingham
    source = Path(probtf.bingham.__file__).resolve()
    return dict(arithmetic_backend='python_reference', reference_source=str(source),
        reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        runtime_compilation=False, silent_fallback=False)


def backend_record(backend=None):
    """A fresh scalar-only record; file hashing is performed once per process."""
    if backend is None:
        backend = os.environ.get('PROBTF_MARKER_BOARD_BACKEND', 'native')
    if backend == 'python_reference':
        backend = 'python'
    return dict(_backend_record(backend))


def normalizer_derivatives(eigenvalues, integration_steps=120, need_second_derivative=False):
    steps = _integration_steps(integration_steps)
    values = np.ascontiguousarray(np.asarray(eigenvalues, float).reshape(4))
    if not np.all(np.isfinite(values)):
        raise ValueError('Bingham eigenvalues must be finite.')
    constant = ct.c_double()
    derivative = np.empty(4)
    hessian = np.empty((4, 4))
    error = ct.create_string_buffer(512)
    code = _library().probtf_marker_board_normalizer(values.ctypes.data_as(_POINTER),
        steps, int(bool(need_second_derivative)), ct.byref(constant),
        derivative.ctypes.data_as(_POINTER), hessian.ctypes.data_as(_POINTER), error, len(error))
    if code:
        raise FloatingPointError(error.value.decode(errors='replace'))
    return constant.value, derivative, hessian if need_second_derivative else None


def bingham_second_moment(parameter_matrix, integration_steps=120):
    parameter = _symmetric_matrix(parameter_matrix, 4, 'parameter_matrix')
    # The scalar gauge is removed exactly as in the core eigendecomposition.
    largest = float(np.linalg.eigvalsh(parameter)[-1])
    canonical = parameter-largest*np.eye(4)
    canonical = .5*(canonical+canonical.T)
    values, vectors = np.linalg.eigh(canonical)
    constant, derivative, _ = normalizer_derivatives(values, integration_steps)
    diagonal = derivative/constant
    diagonal /= float(np.sum(diagonal))
    moment = vectors @ np.diag(diagonal) @ vectors.T
    return .5*(moment+moment.T)


def diagonal_second_moment(eigenvalues, integration_steps=120):
    """Diagonal of the same second moment without eigendecomposing a diagonal.

    This is used only by the existing diagonal-concentration fit residual.
    The integral is symmetric in its four eigenvalue arguments; outputs keep
    their input-coordinate order, including tied eigenvalues.
    """
    values = np.asarray(eigenvalues, float).reshape(4)
    constant, derivative, _ = normalizer_derivatives(values, integration_steps)
    moments = derivative/constant
    return moments/float(np.sum(moments))


def _moment_eigenvalues_and_jacobian(eigenvalues, integration_steps):
    constant, derivative, hessian = normalizer_derivatives(
        eigenvalues, integration_steps, need_second_derivative=True)
    raw = derivative/constant
    raw_jacobian = hessian/constant-np.outer(derivative, derivative)/(constant**2)
    total = float(np.sum(raw))
    return (raw/total,
        raw_jacobian/total-np.outer(raw, np.sum(raw_jacobian, axis=0))/(total**2))


def match_bingham_to_second_moment(second_moment, integration_steps=120,
                                  initial_eigenvalues=None, max_iterations=80):
    target = _validated_second_moment(second_moment)
    values, vectors = np.linalg.eigh(target)
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    initial = [-30., -15., -5., 0.] if initial_eigenvalues is None else initial_eigenvalues
    parameters = _eigenvalues_to_parameter_vector(initial)
    iterations = int(max_iterations)
    if iterations < 1 or iterations != max_iterations:
        raise ValueError('max_iterations must be a positive integer.')

    @lru_cache(maxsize=1)
    def objective(key):
        candidate, gaps = _parameter_vector_to_eigenvalues(key)
        moment, jacobian = _moment_eigenvalues_and_jacobian(candidate, integration_steps)
        error = moment-values
        eigenvalue_jacobian = np.array([
            [-gaps[0], -gaps[1], -gaps[2]], [0., -gaps[1], -gaps[2]],
            [0., 0., -gaps[2]], [0., 0., 0.]])
        return .5*float(error@error), eigenvalue_jacobian.T @ (jacobian.T @ error)

    result = minimize(lambda p: objective(tuple(p))[0], parameters,
        jac=lambda p: objective(tuple(p))[1], method='L-BFGS-B',
        bounds=[(-20., 12.)]*3,
        options={'maxiter': iterations, 'ftol': 1e-14, 'gtol': 1e-10})
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError('Bingham moment matching produced non-finite parameters.')
    fitted, _ = _parameter_vector_to_eigenvalues(result.x)
    return canonical_bingham_parameter(vectors @ np.diag(fitted) @ vectors.T)

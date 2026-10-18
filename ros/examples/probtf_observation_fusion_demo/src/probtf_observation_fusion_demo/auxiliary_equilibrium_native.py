"""Batched native execution of the unchanged auxiliary equilibrium equations.

This backend owns an auxiliary URDF model, never a physical simulator state.
The canonical anchor/path and every returned body-density factor remain the
caller's responsibility. No runtime compilation or silent Python fallback.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path

import numpy as np


_DOUBLE_POINTER = ct.POINTER(ct.c_double)
_BYTE_POINTER = ct.POINTER(ct.c_ubyte)
_LIBRARY_NAME = 'libprobtf_auxiliary_equilibrium.so'


class _Diagnostic(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ('status', 'refinement_termination',
        'refinement_iterations', 'eigenvalues_available', 'convexity_certified',
        'direct_attempted', 'direct_used', 'direct_failed')]+[
        (name, ct.c_uint64) for name in ('requested_stages', 'used_stages',
        'refinement_calls', 'completed_refinements', 'newton_iterations',
        'line_search_evaluations', 'bisections', 'residual_warning_count',
        'nonpositive_curvature_count', 'curvature_audit_failure_count')]+[
        ('last_residual', ct.c_double), ('maximum_residual', ct.c_double),
        ('certificate_ratio_upper', ct.c_double), ('certificate_rounding_margin', ct.c_double),
        ('direct_KKT_residual', ct.c_double), ('eigenvalues', ct.c_double*6),
        ('direct_error', ct.c_char*256), ('error', ct.c_char*512)]


def _library_path(explicit):
    selected = explicit or os.environ.get('PROBTF_AUXILIARY_EQUILIBRIUM_LIBRARY')
    if selected:
        path = Path(selected).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError('Selected native auxiliary library does not exist: '+str(path))
        return path
    directories = []
    for name in ('LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'):
        directories.extend(Path(value) if name == 'LD_LIBRARY_PATH' else Path(value)/'lib'
                           for value in os.environ.get(name, '').split(os.pathsep) if value)
    try:
        import rospkg
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        directories.append(package.parent.parent/'lib')
        for ancestor in package.parents:
            if (ancestor/'devel').is_dir() and (ancestor/'src').is_dir():
                directories.extend((ancestor/'devel/lib', ancestor/'devel/.private/probtf_observation_fusion_demo/lib'))
    except ImportError:
        pass
    except rospkg.ResourceNotFound:
        pass
    for directory in directories:
        if (directory/_LIBRARY_NAME).is_file():
            return (directory/_LIBRARY_NAME).resolve()
    raise RuntimeError('Native auxiliary equilibrium backend is not built. Run catkin build '
                       'probtf_observation_fusion_demo, then source the workspace setup.')


def _pointer(array):
    return array.ctypes.data_as(_DOUBLE_POINTER)


class AuxiliaryEquilibriumNative:
    """Evaluate independent captured contexts and log-K points in one C call.

    All arrays use float64; matrices use row-major order. Default workers=1
    preserves CPU resources for acquisition; explicit parallel rows own
    independent Pinocchio Data and do not change each row's arithmetic.
    Failed rows are explicit and retain NaN output arrays, never zero results.
    """
    def __init__(self, urdf, dimension, spring_types=None, *, library_path=None,
                 certified_fastpath=False):
        self.dimension = int(dimension)
        if not 1 <= self.dimension <= 6:
            raise ValueError('Auxiliary model requires 1..6 scalar revolute joints.')
        path = Path(urdf).expanduser().resolve()
        if not path.is_file():
            raise ValueError('Known auxiliary URDF does not exist.')
        self.urdf_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        types = np.zeros(self.dimension, dtype=np.int32) if spring_types is None else np.asarray(spring_types, dtype=np.int32)
        if types.shape != (self.dimension,) or np.any((types != 0) & (types != 1)):
            raise ValueError('Spring types require one linear(0) or periodic(1) entry per joint.')
        self.spring_types = tuple(int(value) for value in types)
        if not isinstance(certified_fastpath, bool):
            raise ValueError('certified_fastpath must be a boolean.')
        self.certified_fastpath = certified_fastpath
        self.backend_path = str(_library_path(library_path))
        self.backend_sha256 = hashlib.sha256(Path(self.backend_path).read_bytes()).hexdigest()
        self._library = ct.CDLL(self.backend_path)
        self._library.probtf_auxiliary_abi_version.argtypes = []
        self._library.probtf_auxiliary_abi_version.restype = ct.c_int
        if self._library.probtf_auxiliary_abi_version() != 2:
            raise RuntimeError('Native auxiliary backend C ABI is incompatible.')
        self._library.probtf_auxiliary_create.argtypes = [ct.c_char_p, ct.c_int,
            ct.POINTER(ct.c_int), ct.c_char_p, ct.c_size_t]
        self._library.probtf_auxiliary_create.restype = ct.c_void_p
        self._library.probtf_auxiliary_destroy.argtypes = [ct.c_void_p]
        self._library.probtf_auxiliary_destroy.restype = None
        self._library.probtf_auxiliary_gravity_hessian_bound.argtypes = [ct.c_void_p, _DOUBLE_POINTER]
        self._library.probtf_auxiliary_gravity_hessian_bound.restype = ct.c_int
        self._library.probtf_auxiliary_evaluate_batch.argtypes = [ct.c_void_p,
            ct.c_size_t, ct.c_int, ct.c_int]+[_DOUBLE_POINTER]*8+[
            _BYTE_POINTER, _DOUBLE_POINTER, ct.POINTER(_Diagnostic)]
        self._library.probtf_auxiliary_evaluate_batch.restype = ct.c_int
        error = ct.create_string_buffer(2048)
        self._handle = self._library.probtf_auxiliary_create(str(path).encode(), self.dimension,
            types.ctypes.data_as(ct.POINTER(ct.c_int)), error, len(error))
        if not self._handle:
            raise RuntimeError('Native auxiliary initialization failed: '+error.value.decode(errors='replace'))
        self.gravity_hessian_absolute_bound = np.empty((self.dimension, self.dimension))
        if self._library.probtf_auxiliary_gravity_hessian_bound(
                self._handle, _pointer(self.gravity_hessian_absolute_bound)):
            raise RuntimeError('Native auxiliary gravity bound could not be read.')
        self.gravity_hessian_absolute_bound.setflags(write=False)

    def close(self):
        if getattr(self, '_handle', None):
            self._library.probtf_auxiliary_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def _batch(self, target, command, anchor_logK, anchor_theta, *, mode, workers):
        if not getattr(self, '_handle', None):
            raise RuntimeError('Native auxiliary model is closed.')
        arrays = [np.ascontiguousarray(value, dtype=np.float64)
                  for value in (target, command, anchor_logK, anchor_theta)]
        target, command, anchor_logK, anchor_theta = arrays
        if (target.ndim != 2 or target.shape[1] != self.dimension
                or any(value.shape != target.shape for value in arrays[1:])):
            raise ValueError('Batched auxiliary inputs require matching [rows, joints] arrays.')
        if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError('Native auxiliary worker count must be a positive integer.')
        count = len(target)
        theta, residual = np.empty_like(target), np.empty_like(target)
        hessian, parameter_jacobian, derivative = [np.empty((count, self.dimension, self.dimension)) for _ in range(3)]
        active = np.empty(target.shape, dtype=np.uint8)
        diagnostics = (_Diagnostic*count)()
        status = self._library.probtf_auxiliary_evaluate_batch(self._handle, count, mode, int(workers),
            *map(_pointer, arrays), _pointer(theta), _pointer(hessian),
            _pointer(parameter_jacobian), _pointer(derivative),
            active.ctypes.data_as(_BYTE_POINTER), _pointer(residual), diagnostics)
        if status:
            raise RuntimeError('Native auxiliary C call has invalid arguments: '+str(status))
        records = []
        for index, diagnostic in enumerate(diagnostics):
            ok = diagnostic.status == 0
            eigenvalues = (list(diagnostic.eigenvalues)[:self.dimension]
                           if ok and diagnostic.eigenvalues_available else None)
            equilibrium = None if not ok else dict(theta=theta[index].tolist(),
                command=command[index].tolist(), selected_K=(target[index] if mode == 1 else np.exp(target[index])).tolist(),
                KKT_residual_nm=residual[index].tolist(), KKT_residual_inf_nm=diagnostic.last_residual,
                KKT_audit_tolerance_nm=1e-9, KKT_tolerance_satisfied=diagnostic.last_residual <= 1e-9,
                Hessian_eigenvalues=eigenvalues,
                positive_curvature=None if eigenvalues is None else eigenvalues[0] > 0.,
                curvature_audit_error=None if eigenvalues is not None else 'Native diagnostic eigensolver unavailable.',
                refinement_iterations=diagnostic.refinement_iterations,
                refinement_termination=('refinement_iteration_budget', 'refinement_residual_target',
                    'no_further_residual_and_potential_reduction')[diagnostic.refinement_termination],
                refinement_target_nm=1e-11, refinement_converged=diagnostic.last_residual <= 1e-11,
                active_joint_limits=active[index].astype(bool).tolist(), used_finite_auxiliary_result=True,
                audit_is_fatal=False, Hessian_modified=False, physical_stability_guaranteed=False,
                backend='native auxiliary equilibrium')
            distance = None if mode == 1 else float(np.max(np.abs(target[index]-anchor_logK[index])))
            direct_used = bool(diagnostic.direct_used)
            certificate = dict(requested=mode == 2, certified=bool(diagnostic.convexity_certified),
                method='positive witness Bv < diag(K)v with upward-rounded URDF gravity-Hessian entry bounds',
                scope='entire finite joint box; linear joint springs',
                ratio_upper=(float(diagnostic.certificate_ratio_upper) if mode == 2 and
                    np.isfinite(diagnostic.certificate_ratio_upper) else None),
                rounding_margin=float(diagnostic.certificate_rounding_margin),
                direct_attempted=bool(diagnostic.direct_attempted), direct_used=direct_used,
                direct_arithmetic_failed=bool(diagnostic.direct_failed),
                direct_KKT_residual_inf_nm=(float(diagnostic.direct_KKT_residual) if
                    np.isfinite(diagnostic.direct_KKT_residual) else None),
                direct_root_resolution_tolerance_nm=1e-11,
                fallback_reason=diagnostic.direct_error.decode(errors='replace') or None,
                failure_to_certify_is_update_gate=False,
                finite_iterates_claimed_bitwise_equal=False)
            continuation = dict(method=('direct refinement' if mode == 1 else
                    'certified globally strongly convex direct refinement' if direct_used else
                    'same anchor and straight log-K path'),
                requested_stages=None if mode == 1 else int(diagnostic.requested_stages),
                used_stages=None if mode == 1 else int(diagnostic.used_stages),
                nominal_maximum_logK_step=.1,
                used_maximum_logK_step=None if mode == 1 or not diagnostic.used_stages else distance/diagnostic.used_stages,
                work_budget_saturated=not direct_used and diagnostic.requested_stages > diagnostic.used_stages,
                numerical_work_budget=1000, stiffness_candidate_clipped=False, anchor_changed=False,
                audit_is_fatal=False, convexity_certificate=certificate)
            summary = dict(evaluations=int(diagnostic.completed_refinements),
                residual_warning_count=int(diagnostic.residual_warning_count),
                nonpositive_curvature_count=int(diagnostic.nonpositive_curvature_count),
                curvature_audit_failure_count=int(diagnostic.curvature_audit_failure_count),
                maximum_KKT_residual_nm=diagnostic.maximum_residual,
                scope='all completed refinements in this native call')
            records.append(dict(status='ok' if ok else 'failed',
                error=None if ok else diagnostic.error.decode(errors='replace'),
                equilibrium_audit=equilibrium, continuation_audit=continuation,
                equilibrium_summary=summary,
                counts={name: int(getattr(diagnostic, name)) for name in ('refinement_calls',
                    'completed_refinements', 'newton_iterations', 'line_search_evaluations', 'bisections')}))
        return dict(theta=theta, hessian=hessian, parameter_jacobian=parameter_jacobian,
                    derivative=derivative, active=active.astype(bool), residual=residual,
                    diagnostics=records)

    def evaluate_batch(self, logK, command, anchor_logK, anchor_theta, *, workers=1,
                       certified_fastpath=None):
        enabled = self.certified_fastpath if certified_fastpath is None else certified_fastpath
        if not isinstance(enabled, bool):
            raise ValueError('certified_fastpath must be a boolean.')
        return self._batch(logK, command, anchor_logK, anchor_theta,
                           mode=2 if enabled else 0, workers=workers)

    @staticmethod
    def _single(result):
        return dict({key: value[0] for key, value in result.items() if key != 'diagnostics'},
                    **result['diagnostics'][0])

    def continue_from(self, anchor_logK, anchor_theta, x, command, *, certified_fastpath=None):
        return self._single(self.evaluate_batch(np.asarray(x)[None, :],
            np.asarray(command)[None, :], np.asarray(anchor_logK)[None, :], np.asarray(anchor_theta)[None, :],
            certified_fastpath=certified_fastpath))

    def refine(self, theta, command, stiffness):
        stiffness = np.asarray(stiffness)[None, :]
        return self._single(self._batch(stiffness, np.asarray(command)[None, :],
            np.zeros_like(stiffness), np.asarray(theta)[None, :], mode=1, workers=1))


NativeAuxiliaryEquilibrium = AuxiliaryEquilibriumNative

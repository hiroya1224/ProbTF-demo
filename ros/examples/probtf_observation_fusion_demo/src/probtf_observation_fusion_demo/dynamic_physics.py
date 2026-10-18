"""Explicit C ABI binding for the demo's stateful flexible physical plant.

The backend is built by catkin, not during import or execution. Plain float64
arrays cross the boundary; no Python Pinocchio/Boost object ABI is shared.
Only the physical simulator uses this state. Inference's auxiliary equilibrium
model and transform candidate laws remain separate.
"""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import sys

import numpy as np


_LIBRARY_NAME = 'libprobtf_flexible_dynamics.so'
_DOUBLE_NAMES = ('time_s', 'energy_j', 'kinetic_energy_j', 'gravity_energy_j',
    'spring_energy_j', 'damping_loss_j', 'command_work_j', 'impact_loss_j',
    'energy_balance_residual_j', 'max_torque_residual_nm',
    'min_accepted_step_s', 'max_accepted_step_s', 'last_advance_dt_s')
_COUNT_NAMES = ('accepted_substeps', 'newton_iterations', 'line_search_evaluations',
                'step_halvings', 'joint_impacts', 'active_stop_substeps',
                'unconverged_finite_substeps', 'newton_budget_substeps',
                'line_search_stalled_substeps', 'impact_energy_audit_warnings',
                'nominal_work_budget_saturations', 'substep_work_budget_warnings')
_DOUBLE_POINTER = ct.POINTER(ct.c_double)


class _Options(ct.Structure):
    _fields_ = [('max_step_s', ct.c_double), ('min_step_s', ct.c_double),
        ('absolute_residual_tolerance', ct.c_double),
        ('relative_residual_tolerance', ct.c_double),
        ('max_newton_iterations', ct.c_int), ('max_line_search_steps', ct.c_int),
        ('max_step_halvings', ct.c_int)]


def _library_path(explicit):
    explicit = explicit or os.environ.get('PROBTF_DYNAMIC_PHYSICS_LIBRARY')
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError('The explicitly selected native dynamics library does not exist: '+str(path))
        return path
    directories = []
    for name in ('LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'):
        for text in os.environ.get(name, '').split(os.pathsep):
            if text:
                directory = Path(text)
                directories.append(directory if name == 'LD_LIBRARY_PATH' else directory/'lib')
    try:
        import rospkg
        package = Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
        directories.append(package.parent.parent/'lib')  # installed share/package
        for ancestor in package.parents:
            if (ancestor/'devel').is_dir() and (ancestor/'src').is_dir():
                directories.extend((ancestor/'devel/lib', ancestor/
                    'devel/.private/probtf_observation_fusion_demo/lib'))
    except ImportError:
        pass
    except rospkg.ResourceNotFound:
        pass
    for directory in directories:
        path = directory/_LIBRARY_NAME
        if path.is_file():
            return path.resolve()
    raise RuntimeError('Native dynamics backend is not built. Run catkin build '
        'probtf_observation_fusion_demo with /usr/bin/python3, then source the workspace setup. '
        'No runtime compilation or equilibrium fallback is performed.')


def _bind(library):
    signatures = {
        'abi_version': ([], ct.c_int),
        'default_options': ([ct.POINTER(_Options)], None),
        'create': ([ct.c_char_p, ct.c_int, _DOUBLE_POINTER, _DOUBLE_POINTER,
                   ct.c_double, ct.POINTER(_Options), ct.c_char_p, ct.c_size_t], ct.c_void_p),
        'destroy': ([ct.c_void_p], None),
        'last_error': ([ct.c_void_p], ct.c_char_p),
        'reset': ([ct.c_void_p, _DOUBLE_POINTER, _DOUBLE_POINTER, _DOUBLE_POINTER], ct.c_int),
        'advance': ([ct.c_void_p, ct.c_double, _DOUBLE_POINTER, _DOUBLE_POINTER], ct.c_int),
        'trace_size': ([ct.c_void_p], ct.c_size_t),
        'trace_copy': ([ct.c_void_p, ct.c_size_t, _DOUBLE_POINTER, _DOUBLE_POINTER, _DOUBLE_POINTER], ct.c_int),
        'get_state': ([ct.c_void_p, _DOUBLE_POINTER, _DOUBLE_POINTER, _DOUBLE_POINTER], ct.c_int),
        'get_parameters': ([ct.c_void_p, _DOUBLE_POINTER, _DOUBLE_POINTER, _DOUBLE_POINTER], ct.c_int),
        'get_diagnostics': ([ct.c_void_p, _DOUBLE_POINTER, ct.c_size_t,
                             ct.POINTER(ct.c_uint64), ct.c_size_t], ct.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, 'probtf_dynamics_'+name)
        function.argtypes, function.restype = arguments, result
    if library.probtf_dynamics_abi_version() != 2:
        raise RuntimeError('The native dynamics library has an incompatible C ABI version.')


def _pointer(array):
    return array.ctypes.data_as(_DOUBLE_POINTER)


def _readonly(array):
    array.setflags(write=False)
    return array


class NativeDynamics:
    """Stateful implicit midpoint dynamics, with a fixed initial damping law.

    ``advance`` returns relative times including zero and every accepted
    substep endpoint. State outputs are detached read-only arrays. D and the
    reference mass are fixed by the first successful reset, including across
    later resets. ``ctypes.CDLL`` releases the Python GIL during native calls.
    ``damping_ratio`` sets D_i = 2*zeta*sqrt(K_i*M_initial[i, i]); it is a
    diagonal reference convention, not a damping ratio of each coupled mode.
    """
    def __init__(self, urdf, stiffness, *, damping_ratio=2., max_step_s=.001,
                 min_step_s=None, newton_tolerance=None,
                 absolute_residual_tolerance=None, relative_residual_tolerance=None,
                 max_newton_iterations=None, max_line_search_steps=None,
                 max_step_halvings=None, damping=None, library_path=None):
        values = np.asarray(stiffness, dtype=np.float64)
        if (values.ndim != 1 or not 1 <= values.size <= 6
                or not np.all(np.isfinite(values)) or np.any(values <= 0.)):
            raise ValueError('The physical stiffness must contain 1 to 6 finite positive values.')
        self.dimension = values.size
        self._stiffness = np.ascontiguousarray(values)
        self._damping = None if damping is None else self._vector(damping, 'Damping')
        if self._damping is not None and np.any(self._damping < 0.):
            raise ValueError('Fixed physical damping must be nonnegative.')
        self.damping_ratio = float(damping_ratio)
        if not np.isfinite(self.damping_ratio) or self.damping_ratio < 0.:
            raise ValueError('The physical damping ratio must be finite and nonnegative.')
        path = Path(urdf).expanduser().resolve()
        if not path.is_file():
            raise ValueError('The known physical URDF file does not exist.')
        self.backend_path = str(_library_path(library_path))
        self.backend_sha256 = hashlib.sha256(Path(self.backend_path).read_bytes()).hexdigest()
        self._library = ct.CDLL(self.backend_path)
        _bind(self._library)
        self.pinocchio_python_version = getattr(sys.modules.get('pinocchio'), '__version__', None)
        maps = Path('/proc/self/maps')
        self.loaded_native_libraries = (sorted({line.split()[-1] for line in maps.read_text().splitlines()
            if any(name in line for name in ('libpinocchio', 'libeigenpy', 'libboost_python'))
            and line.split()[-1].startswith('/')}) if maps.is_file() else [])
        options = _Options()
        self._library.probtf_dynamics_default_options(ct.byref(options))
        if newton_tolerance is not None:
            if absolute_residual_tolerance is not None:
                raise ValueError('Specify one absolute Newton residual tolerance.')
            absolute_residual_tolerance = newton_tolerance
        updates = dict(max_step_s=max_step_s, min_step_s=min_step_s,
            absolute_residual_tolerance=absolute_residual_tolerance,
            relative_residual_tolerance=relative_residual_tolerance,
            max_newton_iterations=max_newton_iterations,
            max_line_search_steps=max_line_search_steps, max_step_halvings=max_step_halvings)
        for name, value in updates.items():
            if value is None:
                continue
            if name.startswith('max_') and name not in ('max_step_s',):
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError(name+' must be a nonnegative integer.')
                setattr(options, name, int(value))
            else:
                value = float(value)
                allow_zero = name == 'relative_residual_tolerance'
                if not np.isfinite(value) or value < 0. or (value == 0. and not allow_zero):
                    raise ValueError(name+' must be finite and '+('nonnegative.' if allow_zero else 'positive.'))
                setattr(options, name, value)
        self.options = {name: getattr(options, name) for name, _ in _Options._fields_}
        error = ct.create_string_buffer(2048)
        self._handle = self._library.probtf_dynamics_create(str(path).encode(), self.dimension,
            _pointer(self._stiffness), None if self._damping is None else _pointer(self._damping),
            self.damping_ratio, ct.byref(options), error, len(error))
        self._initialized = False
        self._last_advance_failed = False
        self._last_advance_error = None
        self._advance_diagnostics_failed = False
        if not self._handle:
            raise RuntimeError('Native physical dynamics initialization failed: '+error.value.decode(errors='replace'))

    def _vector(self, value, name):
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
            raise ValueError(name+' must be one finite value per physical joint.')
        return np.ascontiguousarray(result)

    def _check(self, result):
        if result != 0:
            error = self._library.probtf_dynamics_last_error(self._handle)
            raise RuntimeError('Native physical dynamics failed: '+
                               (error.decode(errors='replace') if error else 'no diagnostic supplied'))

    def _require_state(self):
        if not getattr(self, '_handle', None):
            raise RuntimeError('Native physical dynamics is closed.')
        if not self._initialized:
            raise RuntimeError('Reset the native physical state before advancing or reading it.')

    def reset(self, position, velocity=None, command=None):
        if not getattr(self, '_handle', None):
            raise RuntimeError('Native physical dynamics is closed.')
        q = self._vector(position, 'Position')
        v = np.zeros(self.dimension) if velocity is None else self._vector(velocity, 'Velocity')
        u = q if command is None else self._vector(command, 'Sent command')
        self._check(self._library.probtf_dynamics_reset(self._handle, _pointer(q), _pointer(v), _pointer(u)))
        self._initialized = True
        self._last_advance_failed = False
        self._last_advance_error = None
        self._advance_diagnostics_failed = False
        return self.state()

    def advance(self, dt, u_start, u_end):
        self._require_state()
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.:
            raise ValueError('A physical advance duration must be finite and nonnegative.')
        start, end = self._vector(u_start, 'Start command'), self._vector(u_end, 'End command')
        try:
            self._check(self._library.probtf_dynamics_advance(self._handle, dt, _pointer(start), _pointer(end)))
        except RuntimeError as error:
            # Native state/trace roll back, while its numerical diagnostics
            # describe the failed attempt. Later successful queries clear the
            # native error buffer, so retain this explicit failure separately.
            self._last_advance_failed = True
            self._last_advance_error = str(error)
            self._advance_diagnostics_failed = True
            raise
        self._last_advance_failed = False
        self._last_advance_error = None
        # A zero-duration native call deliberately preserves diagnostics.
        if dt > 0.:
            self._advance_diagnostics_failed = False
        count = self._library.probtf_dynamics_trace_size(self._handle)
        times = np.empty(count, dtype=np.float64)
        positions = np.empty((count, self.dimension), dtype=np.float64)
        velocities = np.empty_like(positions)
        self._check(self._library.probtf_dynamics_trace_copy(self._handle, count,
            _pointer(times), _pointer(positions), _pointer(velocities)))
        if (count < (1 if dt == 0. else 2) or times[0] != 0.
                or not np.isclose(times[-1], dt, rtol=1e-12, atol=1e-14)
                or np.any(np.diff(times) <= 0.) or not np.all(np.isfinite(positions))
                or not np.all(np.isfinite(velocities))):
            raise RuntimeError('Native physical trace has an invalid accepted-substep history.')
        return dict(times=_readonly(times), positions=_readonly(positions), velocities=_readonly(velocities))

    def diagnostics(self):
        self._require_state()
        values = np.empty(len(_DOUBLE_NAMES), dtype=np.float64)
        counts = np.empty(len(_COUNT_NAMES), dtype=np.uint64)
        self._check(self._library.probtf_dynamics_get_diagnostics(self._handle,
            _pointer(values), values.size, counts.ctypes.data_as(ct.POINTER(ct.c_uint64)), counts.size))
        stiffness, damping = np.empty(self.dimension), np.empty(self.dimension)
        mass = np.empty((self.dimension, self.dimension))
        self._check(self._library.probtf_dynamics_get_parameters(self._handle,
            _pointer(stiffness), _pointer(damping), _pointer(mass)))
        result = dict(zip(_DOUBLE_NAMES, map(float, values)))
        result.update(zip(_COUNT_NAMES, map(int, counts)))
        result.update(stiffness=stiffness.tolist(), damping=damping.tolist(), reference_mass=mass.tolist(),
            damping_ratio=self.damping_ratio, damping_fixed_after_initial_reset=True,
            backend_path=self.backend_path, backend_sha256=self.backend_sha256,
            pinocchio_python_version=self.pinocchio_python_version,
            loaded_native_libraries=list(self.loaded_native_libraries),
            last_advance_failed=self._last_advance_failed,
            last_advance_error=self._last_advance_error,
            advance_diagnostics_scope=('failed_advance_attempt' if self._advance_diagnostics_failed
                else 'accepted_state_and_last_successful_advance'),
            integration_method='implicit midpoint with full rigid-body dynamics, joint springs and fixed damping',
            equilibrium_samples_used=False, synthetic_oscillations_added=False)
        return result

    def state(self):
        self._require_state()
        q, v, a = (np.empty(self.dimension, dtype=np.float64) for _ in range(3))
        self._check(self._library.probtf_dynamics_get_state(self._handle,
            _pointer(q), _pointer(v), _pointer(a)))
        return dict(position=_readonly(q), velocity=_readonly(v), acceleration=_readonly(a),
                    diagnostics=self.diagnostics())

    def close(self):
        if getattr(self, '_handle', None):
            self._library.probtf_dynamics_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

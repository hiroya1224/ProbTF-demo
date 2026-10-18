"""Dynamic physical truth has inertia and damping; candidates remain separate."""
from dataclasses import fields
import ctypes
from pathlib import Path

import numpy as np
import pytest
import yaml

from probtf_observation_fusion_demo.dynamic_physics import NativeDynamics
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene, SensorInputs


def _rotor_urdf(path, *, payload=False):
    # Rotation about world z leaves gravitational potential constant. The
    # active body's inertia about that axis is .08 + 2*.3**2 = .26 kg m².
    # Its fixed tool payload adds .24 + 6*.3**2 = .78 kg m² when requested.
    extra = '''
      <link name="payload"><inertial><origin xyz=".3 0 0"/>
        <mass value="6"/><inertia ixx=".15" iyy=".15" izz=".24"
        ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="payload_mount" type="fixed"><parent link="tool0"/>
        <child link="payload"/><origin xyz="0 0 0"/></joint>
    ''' if payload else ''
    path.write_text('''<robot name="analytic_z_rotor">
      <link name="base_link"/>
      <link name="tool0"><inertial><origin xyz=".3 0 0"/>
        <mass value="2"/><inertia ixx=".05" iyy=".05" izz=".08"
        ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="joint_1" type="revolute"><parent link="base_link"/>
        <child link="tool0"/><axis xyz="0 0 1"/>
        <limit lower="-2" upper="2" effort="100" velocity="10"/>
      </joint>'''+extra+'</robot>')
    return path


def _rotor(path, *, max_step=.001, damping_ratio=.1, payload=False):
    dynamics = NativeDynamics(_rotor_urdf(path, payload=payload), [16.],
        damping_ratio=damping_ratio, max_step_s=max_step)
    dynamics.reset([0.], velocity=[0.], command=[.1])
    return dynamics


def _coupled_rotors(path):
    # Coaxial joints at one origin, with Izz=1 on each moving link, give
    # the constant full mass matrix [[2,1],[1,1]]. Gravity torque is zero.
    path.write_text('''<robot name="coupled_stops"><link name="base_link"/>
      <link name="first"><inertial><mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="tool0"><inertial><mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="joint_1" type="revolute"><parent link="base_link"/><child link="first"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper=".2" effort="100" velocity="10"/></joint>
      <joint name="joint_2" type="revolute"><parent link="first"/><child link="tool0"/>
        <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="100" velocity="10"/></joint>
      </robot>''')
    return path


def _native_trace(dynamics):
    """Read the public C ABI trace to audit failure rollback, without stepping."""
    library, handle = dynamics._library, dynamics._handle
    count = library.probtf_dynamics_trace_size(handle)
    times = np.empty(count)
    q, v = np.empty((count, dynamics.dimension)), np.empty((count, dynamics.dimension))
    pointer = ctypes.POINTER(ctypes.c_double)
    assert library.probtf_dynamics_trace_copy(handle, count,
        times.ctypes.data_as(pointer), q.ctypes.data_as(pointer), v.ctypes.data_as(pointer)) == 0
    return times, q, v


def _analytic_rotor(times, *, command=.1, inertia=.26, stiffness=16., zeta=.1):
    omega = np.sqrt(stiffness/inertia)
    damped = omega*np.sqrt(1-zeta*zeta)
    decay = np.exp(-zeta*omega*times)
    position = command-command*decay*(np.cos(damped*times)
        +zeta/np.sqrt(1-zeta*zeta)*np.sin(damped*times))
    velocity = command*omega/np.sqrt(1-zeta*zeta)*decay*np.sin(damped*times)
    return position, velocity


def test_native_midpoint_matches_under_damped_solution_and_real_ringdown(tmp_path):
    dynamics = _rotor(tmp_path/'rotor.urdf')
    trace = dynamics.advance(3., [.1], [.1])
    times = np.asarray(trace['times'])
    position, velocity = np.asarray(trace['positions'])[:, 0], np.asarray(trace['velocities'])[:, 0]
    assert times[0] == 0. and times[-1] == pytest.approx(3.)
    assert np.all(np.diff(times) > 0.) and np.max(np.diff(times)) <= .001+1e-12
    expected_q, expected_v = _analytic_rotor(times)
    np.testing.assert_allclose(position, expected_q, rtol=0., atol=2e-5)
    np.testing.assert_allclose(velocity, expected_v, rtol=0., atol=2e-4)
    assert np.max(position) > .15  # Actual inertia carries it across static q=.1.
    assert np.any(velocity > .1) and np.any(velocity < -.1)
    energy = .5*.26*velocity**2+.5*16.*(position-.1)**2
    assert np.all(np.isfinite(energy))
    assert energy[-1] < .02*energy[0]
    # Constant M and quadratic V make midpoint's damping-energy identity exact.
    assert np.max(np.diff(energy)) < 3e-12
    final = dynamics.state()
    np.testing.assert_array_equal(final['position'], trace['positions'][-1])
    np.testing.assert_array_equal(final['velocity'], trace['velocities'][-1])
    assert np.all(np.isfinite(final['acceleration']))
    diagnostics = final['diagnostics']
    np.testing.assert_allclose(diagnostics['reference_mass'], [[.26]], atol=1e-13)
    np.testing.assert_allclose(diagnostics['damping'], [2*.1*np.sqrt(16.*.26)], atol=1e-13)
    assert not diagnostics['equilibrium_samples_used']
    assert not diagnostics['synthetic_oscillations_added']
    assert diagnostics['last_advance_dt_s'] == pytest.approx(3.)


def test_native_midpoint_converges_with_smaller_time_step(tmp_path):
    exact_q, exact_v = _analytic_rotor(np.array([1.3]))
    omega = np.sqrt(16./.26)
    errors = []
    for index, step in enumerate((.004, .002, .001)):
        dynamics = _rotor(tmp_path/('rotor_'+str(index)+'.urdf'), max_step=step)
        dynamics.advance(1.3, [.1], [.1])
        state = dynamics.state()
        errors.append(np.linalg.norm([state['position'][0]-exact_q[0],
            (state['velocity'][0]-exact_v[0])/omega]))
    assert errors[2] < .35*errors[1] < .35*.35*errors[0]


def test_fifty_millisecond_delay_advances_all_time_and_matches_partitioned_ramp(tmp_path):
    whole = _rotor(tmp_path/'whole.urdf')
    pieces = _rotor(tmp_path/'pieces.urdf')
    for dynamics in (whole, pieces):
        dynamics.reset([0.], velocity=[0.], command=[0.])
    trace = whole.advance(.05, [0.], [.1])
    assert trace['times'][-1] == pytest.approx(.05)
    assert len(trace['times']) >= 51
    for index in range(5):
        pieces.advance(.01, [.02*index], [.02*(index+1)])
    for key in ('position', 'velocity'):
        np.testing.assert_allclose(whole.state()[key], pieces.state()[key], atol=2e-10, rtol=0.)
    assert np.linalg.norm(whole.state()['velocity']) > .01
    assert whole.state()['position'][0] < .02  # A ramp is not teleported to q_eq=.1.


def test_native_command_work_and_damping_loss_match_independent_midpoint_energy(tmp_path):
    dynamics = _rotor(tmp_path/'work.urdf')
    dynamics.reset([0.], velocity=[0.], command=[0.])
    trace = dynamics.advance(.05, [0.], [.1])
    times = np.asarray(trace['times'])
    q, v = trace['positions'][:, 0], trace['velocities'][:, 0]
    command = .1*times/.05
    mid_q, mid_v = (q[:-1]+q[1:])/2, (v[:-1]+v[1:])/2
    mid_command = (command[:-1]+command[1:])/2
    work = -np.sum(16.*(mid_q-mid_command)*np.diff(command))
    damping = 2*.1*np.sqrt(16.*.26)
    loss = np.sum(damping*mid_v**2*np.diff(times))
    energy = .5*.26*v**2+.5*16.*(q-command)**2
    diagnostics = dynamics.diagnostics()
    assert work > 0. and loss > 0.
    assert diagnostics['command_work_j'] == pytest.approx(work, abs=1e-11)
    assert diagnostics['damping_loss_j'] == pytest.approx(loss, abs=1e-11)
    assert diagnostics['impact_loss_j'] == 0.
    assert energy[-1]-energy[0]-work+loss == pytest.approx(0., abs=1e-10)
    assert diagnostics['energy_balance_residual_j'] == pytest.approx(0., abs=1e-10)


def test_mass_of_a_fixed_attached_tool_changes_the_actual_acceleration(tmp_path):
    light = _rotor(tmp_path/'light.urdf', damping_ratio=0.)
    heavy = _rotor(tmp_path/'heavy.urdf', damping_ratio=0., payload=True)
    light.advance(.0001, [.1], [.1]); heavy.advance(.0001, [.1], [.1])
    ratio = light.state()['velocity'][0]/heavy.state()['velocity'][0]
    assert ratio == pytest.approx(4., rel=1e-5)


def test_full_robot_static_equilibrium_is_a_fixed_point_of_dynamic_truth(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    initial = scene.step(0., 1000., render_camera=False)
    stiffness = np.asarray(scene.config['plant']['stiffness'])
    # The existing startup optimizer returns a small nonzero torque residual
    # (~2e-8 Nm). Refine the mathematical oracle, rather than calling that
    # physical transient a failure of the time integrator or loosening limits.
    equilibrium = np.array(initial.joints, copy=True)
    residual = scene.plant.robot.tau_gravity(equilibrium).copy()+stiffness*(equilibrium-initial.command)
    hessian = scene.plant.robot.d_tau_gravity(equilibrium).copy()+np.diag(stiffness)
    equilibrium -= np.linalg.solve(hessian, residual)
    residual = scene.plant.robot.tau_gravity(equilibrium).copy()+stiffness*(equilibrium-initial.command)
    assert np.max(np.abs(residual)) < 1e-12
    dynamics = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'],
        relative_residual_tolerance=0.)
    dynamics.reset(equilibrium, velocity=np.zeros(6), command=initial.command)
    trace = dynamics.advance(1., initial.command, initial.command)
    assert np.all(np.isfinite(trace['positions'])) and np.all(np.isfinite(trace['velocities']))
    np.testing.assert_allclose(trace['positions'], np.broadcast_to(equilibrium, trace['positions'].shape),
        atol=2e-9, rtol=0.)
    np.testing.assert_allclose(trace['velocities'], 0., atol=2e-8, rtol=0.)


def test_full_robot_trace_satisfies_coupled_RNEA_midpoint_equations(tmp_path):
    import pinocchio as pin
    scene = MovingSwitchScene(tmp_path)
    frame = scene.step(0., 1000., render_camera=False)
    dynamics = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'])
    velocity = np.array([.01, -.02, .015, -.01, .02, -.015])
    dynamics.reset(frame.joints, velocity=velocity, command=frame.command)
    delta = np.array([.002, -.001, .001, .002, -.002, .001])
    trace = dynamics.advance(.03, frame.command, frame.command+delta)
    model = scene.plant.robot.model
    data = model.createData()
    upper = np.triu(np.asarray(pin.crba(model, data, frame.joints)))
    mass = upper+np.triu(upper, 1).T
    diagnostics = dynamics.diagnostics()
    np.testing.assert_allclose(diagnostics['reference_mass'], mass, atol=1e-12, rtol=1e-12)
    damping, stiffness = np.asarray(diagnostics['damping']), np.asarray(diagnostics['stiffness'])
    residuals = []
    times, positions, velocities = trace['times'], trace['positions'], trace['velocities']
    for index, h in enumerate(np.diff(times)):
        if h < 1e-12:
            continue
        q0, q1, v0, v1 = positions[index], positions[index+1], velocities[index], velocities[index+1]
        np.testing.assert_allclose(q1-q0, .5*h*(v0+v1), atol=5e-15, rtol=1e-11)
        qm, vm = .5*(q0+q1), .5*(v0+v1)
        um = frame.command+(.5*(times[index]+times[index+1])/.03)*delta
        residual = np.asarray(pin.rnea(model, data, qm, vm, (v1-v0)/h)).copy()
        residual += damping*vm+stiffness*(qm-um)
        residuals.append(np.max(np.abs(residual)))
    assert residuals and max(residuals) < 1e-7


def test_zero_duration_preserves_state_and_negative_duration_is_rejected(tmp_path):
    dynamics = _rotor(tmp_path/'zero.urdf')
    dynamics.advance(.01, [.1], [.1])
    before = dynamics.state()
    trace = dynamics.advance(0., [.1], [.1])
    np.testing.assert_array_equal(trace['times'], [0.])
    for key, trace_key in (('position', 'positions'), ('velocity', 'velocities')):
        np.testing.assert_array_equal(trace[trace_key], [before[key]])
        np.testing.assert_array_equal(dynamics.state()[key], before[key])
    assert dynamics.diagnostics() == before['diagnostics']
    with pytest.raises(ValueError):
        dynamics.advance(-.01, [.1], [.1])
    np.testing.assert_array_equal(dynamics.state()['position'], before['position'])
    with pytest.raises((RuntimeError, ValueError)):
        dynamics.advance(0., [.1], [.2])
    np.testing.assert_array_equal(dynamics.state()['position'], before['position'])
    with pytest.raises(ValueError):
        NativeDynamics(tmp_path/'zero.urdf', [16.], relative_residual_tolerance=-1.)


def test_joint_stop_impulse_preserves_free_momentum_with_the_full_mass(tmp_path):
    dynamics = NativeDynamics(_coupled_rotors(tmp_path/'stops.urdf'), [1e-8, 1e-8], damping_ratio=0.)
    command = np.array([.1999, 0.])
    dynamics.reset(command, velocity=[1., 0.], command=command)
    trace = dynamics.advance(.001, command, command)
    state = dynamics.state()
    mass = np.array([[2., 1.], [1., 1.]])
    np.testing.assert_allclose(state['diagnostics']['reference_mass'], mass, atol=1e-13)
    assert np.max(trace['positions'][:, 0]) <= .2
    assert state['position'][0] == pytest.approx(.2, abs=3e-11)
    np.testing.assert_allclose(state['velocity'], [0., 1.], atol=2e-8, rtol=0.)
    # Componentwise clipping would give [0,0], violating free-joint momentum.
    assert (mass@state['velocity'])[1] == pytest.approx((mass@np.array([1., 0.]))[1], abs=2e-8)
    assert state['diagnostics']['joint_impacts'] >= 1
    assert state['diagnostics']['impact_loss_j'] == pytest.approx(.5, abs=2e-8)


def test_joint_stop_releases_when_the_effective_torque_points_inward(tmp_path):
    dynamics = NativeDynamics(_coupled_rotors(tmp_path/'release.urdf'), [10., 10.], damping_ratio=0.)
    dynamics.reset([.2, 0.], velocity=[0., 0.], command=[.2, 0.])
    dynamics.advance(.01, [.2, 0.], [.1, 0.])
    state = dynamics.state()
    assert state['position'][0] < .2-1e-6
    assert state['velocity'][0] < -1e-4 and state['velocity'][1] > 1e-4
    assert state['diagnostics']['impact_loss_j'] == 0.


def test_invalid_native_inputs_preserve_the_existing_state_and_command(tmp_path):
    dynamics = _rotor(tmp_path/'invalid.urdf')
    dynamics.advance(.01, [.1], [.1])
    before, trace = dynamics.state(), _native_trace(dynamics)
    invalid = (
        lambda: dynamics.reset([3.], velocity=[0.], command=[.1]),
        lambda: dynamics.reset([2.], velocity=[1.], command=[.1]),
        lambda: dynamics.reset([np.nan], velocity=[0.], command=[.1]),
        lambda: dynamics.advance(.01, [.2], [.2]),
        lambda: dynamics.advance(.01, [.1], [3.]),
        lambda: dynamics.advance(1e308, [.1], [.1]),
    )
    for operation in invalid:
        with pytest.raises((RuntimeError, ValueError)):
            operation()
        after = dynamics.state()
        for key in ('position', 'velocity', 'acceleration'):
            np.testing.assert_array_equal(after[key], before[key])
        assert after['diagnostics']['time_s'] == before['diagnostics']['time_s']
        for actual, expected in zip(_native_trace(dynamics), trace):
            np.testing.assert_array_equal(actual, expected)
    dynamics.advance(.001, [.1], [.1])
    assert not dynamics.diagnostics()['last_advance_failed']
    assert dynamics.diagnostics()['last_advance_error'] is None
    assert dynamics.diagnostics()['advance_diagnostics_scope'] == 'accepted_state_and_last_successful_advance'


def test_finite_overflow_during_reset_rolls_back_all_committed_quantities(tmp_path):
    dynamics = NativeDynamics(_rotor_urdf(tmp_path/'overflow.urdf'), [1e308], damping=[0.])
    dynamics.reset([0.], velocity=[0.], command=[0.])
    before, trace = dynamics.state(), _native_trace(dynamics)
    with pytest.raises(RuntimeError):
        dynamics.reset([1.], velocity=[0.], command=[-1.])
    after = dynamics.state()
    for key in ('position', 'velocity', 'acceleration'):
        np.testing.assert_array_equal(after[key], before[key])
    assert after['diagnostics'] == before['diagnostics']
    for actual, expected in zip(_native_trace(dynamics), trace):
        np.testing.assert_array_equal(actual, expected)
    dynamics.advance(.001, [0.], [0.])
    np.testing.assert_array_equal(dynamics.state()['position'], [0.])


def test_subnormal_positive_duration_never_returns_nonfinite_work_diagnostics(tmp_path):
    dynamics = _rotor(tmp_path/'subnormal.urdf', damping_ratio=0.)
    dynamics.reset([.05], velocity=[0.], command=[0.])
    before = dynamics.state()
    try:
        trace = dynamics.advance(1e-320, [0.], [.1])
    except RuntimeError:
        # A numerical inability to resolve a subnormal interval is permitted;
        # silent NaN work/energy or a partially advanced state is not.
        after = dynamics.state()
        for key in ('position', 'velocity', 'acceleration'):
            np.testing.assert_array_equal(after[key], before[key])
        assert after['diagnostics']['time_s'] == before['diagnostics']['time_s']
    else:
        assert trace['times'][-1] == 1e-320
        assert all(np.isfinite(value) for value in dynamics.diagnostics().values()
                   if isinstance(value, float))


def test_finite_nonlinear_Newton_budget_miss_advances_with_honest_audit(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    frame = scene.step(0., 1000., render_camera=False)
    dynamics = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'],
        max_step_s=.02, absolute_residual_tolerance=1e-14,
        relative_residual_tolerance=1e-14, max_newton_iterations=1, max_step_halvings=0)
    dynamics.reset(frame.joints, velocity=[.4, -.3, .2, -.2, .3, -.4], command=frame.command)
    before = dynamics.state()
    trace = dynamics.advance(.05, frame.command,
        frame.command+np.array([.1, -.1, .1, .1, -.1, .1]))
    after = dynamics.state()
    for key in ('position', 'velocity', 'acceleration'):
        assert np.all(np.isfinite(after[key]))
    assert not np.array_equal(after['position'], before['position'])
    assert trace['times'][-1] == .05
    assert after['diagnostics']['time_s'] == .05
    assert after['diagnostics']['newton_iterations'] >= 1
    assert after['diagnostics']['unconverged_finite_substeps'] > 0
    assert after['diagnostics']['newton_budget_substeps'] > 0
    assert after['diagnostics']['max_torque_residual_nm'] > 1e-10
    assert not after['diagnostics']['last_advance_failed']
    assert after['diagnostics']['last_advance_error'] is None
    dynamics.reset(frame.joints, velocity=np.zeros(6), command=frame.command)
    assert not dynamics.diagnostics()['last_advance_failed']
    assert dynamics.diagnostics()['last_advance_error'] is None
    assert dynamics.diagnostics()['advance_diagnostics_scope'] == 'accepted_state_and_last_successful_advance'


@pytest.mark.parametrize('budgets,warning', [
    ({'max_newton_iterations': 0}, 'newton_budget_substeps'),
    ({'max_line_search_steps': 0}, 'line_search_stalled_substeps'),
])
def test_finite_physics_iteration_or_line_search_audit_never_becomes_a_pipeline_stop(tmp_path, budgets, warning):
    scene = MovingSwitchScene(tmp_path)
    frame = scene.step(0., 1000., render_camera=False)
    dynamics = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'],
        absolute_residual_tolerance=1e-30, relative_residual_tolerance=0., **budgets)
    dynamics.reset(frame.joints, velocity=[.02, -.02, .02, -.02, .02, -.02], command=frame.command)
    for _ in range(2):
        trace = dynamics.advance(.001, frame.command, frame.command)
        assert trace['times'][-1] == .001
        assert np.all(np.isfinite(trace['positions']))
        diagnostics = dynamics.diagnostics()
        assert diagnostics['unconverged_finite_substeps'] > 0
        assert diagnostics[warning] > 0
        assert not diagnostics['last_advance_failed']
    assert dynamics.diagnostics()['time_s'] == pytest.approx(.002)


def test_scene_motion_is_continuous_and_keeps_actual_velocity_when_command_is_held(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    first = scene.step(0., 1000., render_camera=False)
    command = first.command.copy(); command[3] += .02
    moved = scene.step(.001, 1000.001, command=command, render_camera=False)
    static = scene.plant.equilibrium(command)
    assert np.linalg.norm(moved.joints-first.joints) < .001
    assert np.linalg.norm(moved.joints-static) > 1e-3
    assert np.linalg.norm(moved.joint_velocities) > 1e-5
    held = scene.step(.002, 1000.002, render_camera=False)
    np.testing.assert_array_equal(held.command, moved.command)
    assert np.linalg.norm(held.joints-moved.joints) > 1e-8
    assert np.all(np.isfinite(held.joint_accelerations))
    np.testing.assert_array_equal(held.link_poses['rod_tip'][:3, 3], held.tip)
    np.testing.assert_array_equal(held.link_poses['camera_optical_frame'], held.camera_pose)


def test_camera_history_contains_actual_substeps_for_the_entire_exposure(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    scene.step(0., 1000., render_camera=False)
    command = scene.initial_command.copy(); command[3:5] += [.03, -.02]
    frame = scene.step(.2, 1000.2, command=command, render_camera=False)
    times = np.array([item[0] for item in scene.camera_history])
    exposure_start = .2-scene.config['camera']['exposure_time']
    assert len(times) > 32
    assert np.all(np.diff(times) > 0.)
    assert np.max(np.diff(times)) <= .001+1e-12
    assert times[0] <= exposure_start+1e-12 and times[1] >= exposure_start-1e-12
    assert times[-1] == pytest.approx(.2)
    np.testing.assert_array_equal(scene.camera_history[-1][1], frame.camera_pose)
    assert scene.config['camera']['motion_blur_samples'] == 7
    assert scene.config['camera']['exposure_time'] == .09


def test_collision_during_a_native_substep_is_not_hidden_by_a_clear_final_frame(tmp_path, monkeypatch):
    settings = yaml.safe_load((Path(__file__).resolve().parents[1]/
        'config/switch_press_scene.yaml').read_text())
    settings['plant']['collision_checking'] = True
    config = tmp_path/'collision_enabled.yaml'
    config.write_text(yaml.safe_dump(settings))
    scene = MovingSwitchScene(tmp_path/'world', config_path=config)
    scene.step(0., 1000., render_camera=False)
    calls = []
    original = scene._clearance
    class SubstepClearance:
        def displacement_bound(self, *args):
            return original.displacement_bound(*args)
        def __call__(self, boxes):
            calls.append(np.array(boxes[-1][1], copy=True))
            return (-.001 if len(calls) == 2 else .1), 'audited_substep/obstacle'
    monkeypatch.setattr(scene, '_clearance', SubstepClearance())
    command = scene.initial_command.copy(); command[3] += .02
    frame = scene.step(.005, 1000.005, command=command, render_camera=False)
    assert len(calls) >= 5
    assert not np.array_equal(calls[0], calls[1])
    assert frame.clearance == .1
    assert frame.collided and not frame.reached
    assert scene.monitor.minimum_clearance == -.001
    assert frame.physics_diagnostics['collision_checking_enabled']
    assert frame.physics_diagnostics['collision_status'] == 'checked'


def test_default_collision_checks_are_skipped_and_explicitly_unverified(tmp_path, monkeypatch):
    scene = MovingSwitchScene(tmp_path)
    assert scene.config['plant']['collision_checking'] is False
    assert not scene.collision_checking and scene._clearance is None
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Disabled physical collisions must not evaluate geometry or clearance.')
    monkeypatch.setattr(scene.plant, 'boxes', forbidden)
    monkeypatch.setattr(scene, '_monitor_segment', forbidden)
    monkeypatch.setattr(scene, '_clearance', forbidden)
    first = scene.step(0., 1000., render_camera=False)
    command = first.command.copy(); command[3] += .02
    moved = scene.step(.02, 1000.02, command=command, render_camera=False)
    assert np.linalg.norm(moved.joints-first.joints) > 1e-8
    assert np.linalg.norm(moved.joint_velocities) > 1e-8
    for frame in (first, moved):
        assert frame.clearance is None and frame.collided is None
        assert frame.closest_pair == 'unchecked'
        assert scene.monitor.minimum_clearance is None
        assert frame.physics_diagnostics['collision_checking_enabled'] is False
        assert frame.physics_diagnostics['collision_status'] == 'unchecked'
        assert frame.physics_diagnostics['reach_condition'] == 'distance_only_collision_unchecked'
        record = scene.physical_record(frame)
        assert record['clearance_m'] is None and record['collided'] is None
        assert record['collision_checking_enabled'] is False
        assert record['reach_condition'] == 'distance_only_collision_unchecked'
        np.testing.assert_allclose(np.linalg.norm(frame.gravity_in_sensor), 1., atol=1e-14)
    np.testing.assert_array_equal(moved.link_poses['rod_tip'][:3, 3], moved.tip)
    np.testing.assert_array_equal(scene.camera_history[-1][1], moved.camera_pose)


def test_dynamic_truth_does_not_enlarge_the_estimator_sensor_boundary(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    frame = scene.step(0., 1000., render_camera=False)
    assert {item.name for item in fields(SensorInputs)} == {
        'time', 'stamp', 'command', 'image', 'gravity_in_sensor',
        'imu_target_link', 'imu_sigma_rad'}
    sensor = scene.sensor_inputs(frame)
    for name in ('joints', 'joint_velocities', 'joint_accelerations', 'physics_diagnostics',
                 'link_poses', 'stiffness', 'damping', 'mass'):
        assert not hasattr(sensor, name)
    assert sensor.gravity_in_sensor is frame.gravity_in_sensor
    np.testing.assert_allclose(np.linalg.norm(sensor.gravity_in_sensor), 1., atol=1e-14)
    assert not frame.joint_velocities.flags.writeable
    assert not frame.joint_accelerations.flags.writeable


def test_default_scene_uses_the_requested_large_fixed_physical_damping(tmp_path):
    scene = MovingSwitchScene(tmp_path)
    frame = scene.step(0., 1000., render_camera=False)
    diagnostics = frame.physics_diagnostics
    assert scene.config['plant']['dynamics']['damping_ratio'] == 2.
    assert diagnostics['damping_ratio'] == 2.
    mass = np.asarray(diagnostics['reference_mass'])
    stiffness = np.asarray(scene.config['plant']['stiffness'])
    expected = 2.*2.*np.sqrt(stiffness*np.diag(mass))
    np.testing.assert_allclose(diagnostics['damping'], expected, rtol=1e-13, atol=1e-13)
    assert diagnostics['damping_fixed_after_initial_reset']
    assert not diagnostics['synthetic_oscillations_added']

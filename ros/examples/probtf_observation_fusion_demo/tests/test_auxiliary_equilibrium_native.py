"""Native auxiliary batching retains the captured mechanical equations."""
from pathlib import Path
import numpy as np
import pytest
import rospkg
import yaml

from deflecomp_core.model.equilibrium import EquilibriumSolver
from deflecomp_core.model.sensitivity import SensitivityCalculator
from deflecomp_core.model.spring import LinearSpringModel, PeriodicSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from probtf_switch_press_demo.scene import write_robot
from probtf_observation_fusion_demo.auxiliary_equilibrium_native import AuxiliaryEquilibriumNative
from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier


PACKAGE = Path(__file__).resolve().parents[1]


def _reference(urdf, spring=None):
    identifier = StiffnessIdentifier.__new__(StiffnessIdentifier)
    identifier.robot = RobotArm(str(urdf), tip_link='tool0', base_link='base_link')
    identifier.spring = spring or LinearSpringModel()
    identifier.sensitivity = SensitivityCalculator(identifier.robot, identifier.spring,
        active_set_tol=0., kkt_tol=0.)
    identifier.solver = EquilibriumSolver(identifier.robot, identifier.spring)
    return identifier


@pytest.fixture(scope='module')
def arm(tmp_path_factory):
    output = tmp_path_factory.mktemp('native_auxiliary_arm')
    config = yaml.safe_load((PACKAGE/'config/switch_press_scene.yaml').read_text())
    source = Path(rospkg.RosPack().get_path('symaware_grasp'))/'urdf/simple_six_dof_arm.urdf'
    urdf = write_robot(source, output/'robot.urdf', config)
    identifier = _reference(urdf)
    command = np.asarray(config['initial_command'])
    anchor = np.log(np.full(6, 150.))
    theta = identifier.solver.solve(command, np.exp(anchor), theta_init=command)
    theta = identifier._refine_equilibrium(theta, command, np.exp(anchor))[0]
    return urdf, identifier, command, anchor, theta


def _continuation(identifier, command, start, anchor_theta, x):
    count = min(max(1, int(np.ceil(np.max(np.abs(x-start))/.1))), 1000)
    theta = anchor_theta.copy()
    for stage in range(1, count+1):
        point = start+(stage/count)*(x-start)
        theta, H, J, active = identifier._refine_equilibrium(theta, command, np.exp(point))
    return theta, H, J, active


def _pendulum(path):
    path.write_text('''<robot name="auxiliary_pendulum"><link name="base_link"/>
      <link name="tool0"><inertial><origin xyz="1 0 0"/>
        <mass value="1"/><inertia ixx=".1" iyy=".1" izz=".1" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <joint name="joint_1" type="revolute"><parent link="base_link"/><child link="tool0"/>
        <axis xyz="0 1 0"/><limit lower="-2" upper="2" effort="100" velocity="10"/>
      </joint></robot>''')
    return path


@pytest.mark.parametrize('offset', [np.zeros(6), np.array([-.5, -.2, .3, -2., .7, 1.1])])
def test_native_same_anchor_path_matches_original_equilibrium_and_sensitivity(arm, offset):
    urdf, reference, command, anchor, theta = arm
    expected = _continuation(reference, command, anchor, theta, anchor+offset)
    backend = AuxiliaryEquilibriumNative(urdf, 6)
    actual = backend.continue_from(anchor, theta, anchor+offset, command, certified_fastpath=False)
    assert actual['status'] == 'ok', actual['error']
    for key, value in zip(('theta', 'hessian', 'parameter_jacobian', 'active'), expected):
        np.testing.assert_allclose(actual[key], value, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(actual['derivative'], -np.linalg.solve(expected[1], expected[2]), rtol=2e-12, atol=2e-11)
    diagnostic = actual['continuation_audit']
    assert diagnostic['used_stages'] == min(max(1, int(np.ceil(np.max(abs(offset))/.1))), 1000)
    assert actual['equilibrium_summary']['evaluations'] == actual['counts']['completed_refinements']
    assert not actual['equilibrium_audit']['Hessian_modified']
    backend.close()


def test_native_batched_contexts_are_independent_and_parallel_order_is_bitwise_reproducible(arm):
    urdf, _, command, anchor, theta = arm
    backend = AuxiliaryEquilibriumNative(urdf, 6)
    targets = anchor+np.array([[0., .1, -.2, .3, -.4, .5], [.2, -.3, .1, -.4, .2, -.1], [0., .1, -.2, .3, -.4, .5]])
    commands = np.tile(command, (3, 1)); starts = np.tile(anchor, (3, 1)); thetas = np.tile(theta, (3, 1))
    serial = backend.evaluate_batch(targets, commands, starts, thetas)
    parallel = backend.evaluate_batch(targets, commands, starts, thetas, workers=2)
    for key in ('theta', 'hessian', 'parameter_jacobian', 'derivative', 'active', 'residual'):
        np.testing.assert_array_equal(serial[key], parallel[key])
        np.testing.assert_array_equal(serial[key][0], serial[key][2])
    assert all(row['status'] == 'ok' for row in serial['diagnostics'])
    assert serial['diagnostics'] == parallel['diagnostics']
    backend.close()


def test_native_implicit_derivative_matches_command_frozen_equilibrium_difference(arm):
    urdf, _, command, anchor, theta = arm
    backend = AuxiliaryEquilibriumNative(urdf, 6)
    x = anchor+np.array([-.5, -.2, .3, -.7, .2, .4])
    actual = backend.continue_from(anchor, theta, x, command)
    step = 1e-5
    targets = np.vstack([x+step*axis for axis in np.eye(6)]+[x-step*axis for axis in np.eye(6)])
    differences = backend.evaluate_batch(targets, np.tile(command, (12, 1)),
        np.tile(anchor, (12, 1)), np.tile(theta, (12, 1)))
    assert all(row['status'] == 'ok' for row in differences['diagnostics'])
    numerical = (differences['theta'][:6]-differences['theta'][6:]).T/(2*step)
    np.testing.assert_allclose(actual['derivative'], numerical, rtol=2e-7, atol=2e-9)
    backend.close()


def test_finite_negative_curvature_and_stalled_refinement_are_returned_honestly(tmp_path):
    urdf = _pendulum(tmp_path/'pendulum.urdf')
    backend = AuxiliaryEquilibriumNative(urdf, 1)
    theta = np.array([-np.pi/2]); command = theta-.1; K = np.array([1.])
    expected = _reference(urdf)._refine_equilibrium(theta, command, K)
    actual = backend.refine(theta, command, K)
    assert actual['status'] == 'ok'
    np.testing.assert_array_equal(actual['theta'], expected[0])
    np.testing.assert_allclose(actual['hessian'], expected[1], atol=1e-13)
    assert actual['equilibrium_audit']['positive_curvature'] is False
    assert not actual['equilibrium_audit']['refinement_converged']
    assert actual['equilibrium_audit']['refinement_termination'] == 'no_further_residual_and_potential_reduction'
    assert actual['equilibrium_summary']['residual_warning_count'] == 1
    assert np.all(np.isfinite(actual['derivative']))
    backend.close()


@pytest.mark.parametrize('kind', [0, 1])
def test_active_joint_limit_keeps_same_KKT_rows_and_spring_law(tmp_path, kind):
    urdf = _pendulum(tmp_path/'pendulum.urdf')
    spring = PeriodicSpringModel() if kind else LinearSpringModel()
    reference = _reference(urdf, spring)
    backend = AuxiliaryEquilibriumNative(urdf, 1, spring_types=[kind])
    theta, command, K = np.array([-2.]), np.array([-2.]), np.array([.1])
    expected = reference._refine_equilibrium(theta, command, K)
    actual = backend.refine(theta, command, K)
    assert actual['status'] == 'ok'
    for key, value in zip(('theta', 'hessian', 'parameter_jacobian', 'active'), expected):
        np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)
    assert actual['active'][0]
    np.testing.assert_array_equal(actual['derivative'], [[0.]])
    backend.close()


def test_converged_true_interior_root_near_stop_keeps_its_nonzero_K_derivative(tmp_path):
    urdf = _pendulum(tmp_path/'interior_near_stop.urdf')
    reference = _reference(urdf)
    backend = AuxiliaryEquilibriumNative(urdf, 1)
    theta = np.array([2.-8.331e-9])
    stiffness = np.array([50.])
    command = theta+reference.robot.tau_gravity(theta)/stiffness
    raw = reference.robot.tau_gravity(theta)+reference.spring.torque(theta, command, stiffness)
    assert np.max(np.abs(raw)) < 1e-11
    actual = backend.refine(theta, command, stiffness)
    assert actual['status'] == 'ok'
    assert not actual['active'][0]
    assert not reference.sensitivity.equilibrium_active_set(theta, command, stiffness)[0]
    H, J = reference.sensitivity.equilibrium_jacobians(theta, command, stiffness)
    np.testing.assert_allclose(actual['derivative'], -np.linalg.solve(H, J), rtol=1e-12, atol=1e-12)
    assert abs(actual['derivative'][0, 0]) > .01
    # Both perturbations remain strictly inside the true box. A tolerance
    # band would falsely freeze this derivative despite the exact root.
    x, step = np.log(stiffness), 1e-8
    plus = backend.continue_from(x, theta, x+step, command)
    minus = backend.continue_from(x, theta, x-step, command)
    assert not plus['active'][0] and not minus['active'][0]
    np.testing.assert_allclose((plus['theta']-minus['theta'])/(2.*step),
        actual['derivative'][:, 0], rtol=2e-6, atol=2e-8)
    backend.close()


def test_periodic_spring_matches_existing_periodic_model(tmp_path):
    urdf = _pendulum(tmp_path/'pendulum.urdf')
    reference = _reference(urdf, PeriodicSpringModel())
    backend = AuxiliaryEquilibriumNative(urdf, 1, spring_types=[1])
    theta, command, K = np.array([.3]), np.array([-.2]), np.array([50.])
    expected = reference._refine_equilibrium(theta, command, K)
    actual = backend.refine(theta, command, K)
    assert actual['status'] == 'ok'
    for key, value in zip(('theta', 'hessian', 'parameter_jacobian', 'active'), expected):
        np.testing.assert_allclose(actual[key], value, rtol=2e-12, atol=2e-11)
    backend.close()


def test_actual_nonfinite_row_fails_explicitly_without_poisoning_other_batch_rows(arm):
    urdf, _, command, anchor, theta = arm
    backend = AuxiliaryEquilibriumNative(urdf, 6)
    targets = np.vstack([anchor, anchor.copy(), anchor]); targets[1, 2] = float('nan')
    actual = backend.evaluate_batch(targets, np.tile(command, (3, 1)), np.tile(anchor, (3, 1)), np.tile(theta, (3, 1)))
    assert [row['status'] for row in actual['diagnostics']] == ['ok', 'failed', 'ok']
    assert 'NaN or infinity' in actual['diagnostics'][1]['error']
    assert np.all(np.isnan(actual['theta'][1]))
    np.testing.assert_array_equal(actual['theta'][0], actual['theta'][2])
    assert actual['diagnostics'][1]['equilibrium_audit'] is None
    backend.close()


def test_actual_singular_required_sensitivity_is_not_regularized(tmp_path):
    urdf = _pendulum(tmp_path/'pendulum.urdf')
    backend = AuxiliaryEquilibriumNative(urdf, 1)
    theta = np.array([-np.pi/2])
    reference = _reference(urdf)
    K = -reference.robot.d_tau_gravity(theta).diagonal()
    actual = backend.refine(theta, theta, K)
    assert actual['status'] == 'failed'
    assert 'LAPACK dgesv' in actual['error']
    assert np.all(np.isnan(actual['hessian']))
    backend.close()


def test_global_gravity_bound_covers_deterministic_box_points_and_uses_com_geometry(arm):
    urdf, reference, _, _, _ = arm
    backend = AuxiliaryEquilibriumNative(urdf, 6)
    bound = backend.gravity_hessian_absolute_bound
    np.testing.assert_array_equal(bound, bound.T)
    assert np.all(bound >= 0.)
    lower, upper = reference.robot.model.lowerPositionLimit, reference.robot.model.upperPositionLimit
    for fraction in (0., .19, .5, .83, 1.):
        theta = lower+fraction*(upper-lower)
        calculated = reference.robot.d_tau_gravity(theta)
        # The exact root-yaw symmetry is zero; Pinocchio's floating
        # derivative contains cancellation residuals around 1e-16.
        np.testing.assert_allclose(calculated[bound == 0.], 0., atol=1e-13)
        assert np.all(np.abs(calculated[bound > 0.]) <= bound[bound > 0.])
    backend.close()


def test_certified_direct_root_matches_canonical_while_removing_continuation_stages(arm):
    urdf, _, command, anchor, theta = arm
    backend = AuxiliaryEquilibriumNative(urdf, 6, certified_fastpath=True)
    target = np.log(np.full(6, 300.))
    canonical = backend.continue_from(anchor, theta, target, command, certified_fastpath=False)
    direct = backend.continue_from(anchor, theta, target, command)
    proof = direct['continuation_audit']['convexity_certificate']
    assert proof['certified'] and proof['direct_used']
    assert proof['ratio_upper'] < 1.-proof['rounding_margin']
    assert proof['direct_KKT_residual_inf_nm'] <= 1e-11
    assert direct['counts']['completed_refinements'] == 1
    assert canonical['counts']['completed_refinements'] > 1
    assert not direct['continuation_audit']['work_budget_saturated']
    for key in ('theta', 'hessian', 'derivative', 'parameter_jacobian'):
        np.testing.assert_allclose(direct[key], canonical[key], rtol=2e-11, atol=3e-11)
    assert not proof['finite_iterates_claimed_bitwise_equal']
    backend.close()


def test_uncertified_point_retains_canonical_calculation_bitwise(tmp_path):
    urdf = _pendulum(tmp_path/'pendulum.urdf')
    backend = AuxiliaryEquilibriumNative(urdf, 1, certified_fastpath=True)
    command, anchor_theta = np.array([.1]), np.array([.1])
    start, target = np.log([50.]), np.log([1.])
    old = backend.continue_from(start, anchor_theta, target, command, certified_fastpath=False)
    new = backend.continue_from(start, anchor_theta, target, command)
    proof = new['continuation_audit']['convexity_certificate']
    assert not proof['certified'] and not proof['direct_attempted']
    assert new['status'] == old['status'] == 'ok'
    for key in ('theta', 'hessian', 'parameter_jacobian', 'derivative', 'active'):
        np.testing.assert_array_equal(new[key], old[key])
    assert new['counts'] == old['counts']
    backend.close()


def test_direct_finite_residual_miss_falls_back_without_stopping_update(tmp_path):
    urdf = _pendulum(tmp_path/'pendulum.urdf')
    backend = AuxiliaryEquilibriumNative(urdf, 1, certified_fastpath=True)
    reference = _reference(urdf)
    theta, stiffness = np.array([.7]), np.array([1e18])
    # The needed deflection is below floating-point position resolution.
    # A uniqueness certificate cannot manufacture a resolved numeric root.
    command = theta.copy()
    x = np.log(stiffness)
    old = backend.continue_from(x, theta, x, command, certified_fastpath=False)
    new = backend.continue_from(x, theta, x, command)
    proof = new['continuation_audit']['convexity_certificate']
    assert proof['certified'] and proof['direct_attempted'] and not proof['direct_used']
    assert proof['direct_KKT_residual_inf_nm'] > 1e-11
    assert 'canonical continuation retained' in proof['fallback_reason']
    assert new['status'] == old['status'] == 'ok'
    np.testing.assert_array_equal(new['theta'], old['theta'])
    np.testing.assert_array_equal(new['derivative'], old['derivative'])
    backend.close()

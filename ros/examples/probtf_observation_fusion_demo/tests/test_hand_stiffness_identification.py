"""The auxiliary K law fits propagated tool poses at the original commands.

These tests supply estimated Gaussian laws and the declared URDF only.  The
physical scene is used to locate the shared model; no realized pose or true
spring parameter is supplied to the identifier.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from probtf.geometry import quat_to_rotmat
from probtf.spherical_law.fusion import (
    FusedMixture, Gaussian, PosteriorComponent, condition_gaussian,
)
from probtf_observation_fusion_demo.body_native import pose_query, reexpress_state
from probtf_observation_fusion_demo.hand_stiffness import HandPoseDensityRatio
from probtf_observation_fusion_demo.model import FrozenBodyBranch, Stage, WeightedState
from probtf_observation_fusion_demo.moving_inference import MovingInference
from probtf_observation_fusion_demo.stiffness_identification import (
    FrozenTimeIndexedStiffnessFactor, StiffnessIdentifier,
)
from legacy_marker import legacy_marker_scene


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]


@pytest.fixture(scope='module')
def hand_model(tmp_path_factory):
    scene = legacy_marker_scene(tmp_path_factory.mktemp('hand_stiffness_model'))
    inference = MovingInference(scene.robot_description, (),
        config_path=scene.legacy_inference_config)
    # This command is the estimator's declared initial family argument.  It is
    # independent of any simulated realized joint configuration.
    source = inference.prior_at(inference.reference_command, 12., 0)
    covariance = source.covariance.copy()
    covariance[4, 10] = covariance[10, 4] = (
        .2*np.sqrt(covariance[4, 4]*covariance[10, 10]))
    source = reexpress_state(source, source.mean, covariance)
    settings = dict(prior_config_path=str(REPO/
        'ros/examples/deflecomp/deflecomp_ros/config/estimator.yaml'),
        # Keep the cumulative continuous density while avoiding a separate
        # Laplace-evidence test in each endpoint-geometry regression.
        bayes_factor_surprise=dict(enabled=False))
    return scene.urdf, inference, source, settings


def make_identifier(model, **settings):
    urdf, inference, _, base_settings = model
    return StiffnessIdentifier(urdf, inference.body_ids,
        dict(base_settings, **settings))


def observed_result(model, source=None, event_id='hand:observation:0'):
    _, inference, initial, _ = model
    source = initial if source is None else source
    prior = Gaussian(source.mean, source.covariance)
    observation = np.zeros((1, source.mean.size))
    observation[0, 10] = 1.
    after = condition_gaussian(prior, np.array([.004]),
        np.array([[.0007]]), observation)[0]
    child = PosteriorComponent(event_id+':posterior', after)
    displayed = reexpress_state(source, after.mean, after.covariance,
        version=source.version+1, component_id=child.label)
    branch = FrozenBodyBranch(source.component_id, 1., source, prior, (child,))
    return SimpleNamespace(prior=source, hand_path=inference.paths['hand'],
        frozen_body_prior=prior,
        frozen_body_posterior=FusedMixture((child,), (), (event_id,),
            source.version, source.stamp, source.command),
        frozen_body_branches=(branch,), projected_body_numerator=branch,
        stages=(Stage('test_estimated_posterior',
            (WeightedState(child.label, 1., displayed),)),),
        records=dict(observation_ids=[event_id]))


def updated_identifier(model, result=None):
    identifier = make_identifier(model)
    result = observed_result(model) if result is None else result
    record = identifier.update(result)
    assert record['status'] == 'updated_density_factor', record.get('failure_reason')
    return identifier, result, record


def test_default_updates_the_complete_tool_pose_not_intermediate_link_densities(hand_model):
    identifier, result, record = updated_identifier(hand_model)
    assert identifier.matching_target == 'hand_pose'
    factor = identifier.factors[0]
    assert isinstance(factor.ratio, HandPoseDensityRatio)
    assert factor.ratio.hand_path == result.hand_path
    assert factor.ratio.hand_path[-2:] == (('press_rod', False), ('rod_tip', False))
    assert factor.ratio.marginals[0][0].mean.shape == (6,)
    assert factor.ratio.marginals[0][0].covariance.shape == (6, 6)
    description = factor.ratio.record()
    assert description['factor_domain'] == 'root_to_hand_pose'
    assert not description['intermediate_link_density_scored']
    assert not description['physical_truth_received']
    # Independently propagate the actual displayed law, including every
    # upstream rotation/translation covariance and the full tool lever arm.
    actual = factor.ratio.marginals[0][1][0][1]
    expected = pose_query(result.stages[-1].components[0].state, result.hand_path)
    np.testing.assert_allclose(actual.mean, expected.mean, atol=1e-13)
    np.testing.assert_allclose(actual.covariance, expected.covariance, atol=1e-13)
    assert np.linalg.norm(actual.covariance[:3, 3:]) > 0.
    assert record['body_posterior_state_ids'] == [result.stages[-1].components[0].state.state_id]
    saved = description['branches'][0]
    assert saved['source_state_id'] == result.prior.state_id
    assert saved['components'][0]['source_stamp'] == result.prior.stamp
    assert saved['components'][0]['source_command'] == list(result.prior.command)
    np.testing.assert_array_equal(saved['components'][0]['covariance'], actual.covariance)
    assert record['source_command'] == list(result.prior.command)


def test_factor_equals_independently_evaluated_six_dimensional_pose_density_ratio(hand_model):
    identifier, result, _ = updated_identifier(hand_model)
    factor = identifier.factors[0]
    x = identifier.initial_components[0].distribution.mean
    saved = identifier._branch_anchor
    try:
        identifier._branch_anchor = factor.branch_anchor
        physical = identifier._physical_prediction(x, result.prior)
        position, rotation, _, _ = identifier._hand_prediction(
            physical, result.prior, result.hand_path)
    finally:
        identifier._branch_anchor = saved
    states = (result.prior, result.stages[-1].components[0].state)
    densities = []
    for state in states:
        query = pose_query(state, result.hand_path)
        delta = Rotation.from_matrix(quat_to_rotmat(query.rotation_wxyz).T @ rotation).as_rotvec()
        angle = np.linalg.norm(delta)
        log_volume = (0. if angle == 0. else
            2.*np.log(np.sin(angle/2.)/(angle/2.)))
        densities.append(multivariate_normal.logpdf(np.r_[position, delta],
            mean=query.mean, cov=query.covariance)-log_volume)
    value, _ = identifier._factor_evaluation(factor, x)
    assert value == pytest.approx(densities[1]-densities[0], rel=2e-12, abs=2e-11)


@pytest.mark.parametrize('offset', [np.zeros(6), np.array([0., -.08, .06, .03, -.04, .02])])
def test_auxiliary_tool_pose_and_derivatives_match_independent_pinocchio_frame_queries(hand_model, offset):
    identifier, result, _ = updated_identifier(hand_model)
    factor = identifier.factors[0]
    x = identifier.initial_components[0].distribution.mean+offset
    saved = identifier._branch_anchor
    try:
        identifier._branch_anchor = factor.branch_anchor
        physical = identifier._physical_prediction(x, result.prior)
    finally:
        identifier._branch_anchor = saved
    position, rotation, dp, omega = identifier._hand_prediction(
        physical, result.prior, result.hand_path)
    # Pinocchio evaluates the URDF's actual rod_tip frame directly.  This
    # reference uses neither the candidate hand-path composition nor its
    # derivative recursion, and includes both fixed rod attachments.
    robot = identifier.robot
    model, data = robot.model, robot.model.createData()
    tip = model.getFrameId('rod_tip')
    pin.computeJointJacobians(model, data, physical['theta'])
    pin.updateFramePlacements(model, data)
    base_to_tip = data.oMf[robot.base_fid].inverse()*data.oMf[tip]
    tip_jacobian = pin.getFrameJacobian(model, data, tip, pin.ReferenceFrame.LOCAL)
    base_jacobian = pin.getFrameJacobian(model, data, robot.base_fid, pin.ReferenceFrame.WORLD)
    np.testing.assert_array_equal(base_jacobian, 0.)
    expected_dp = base_to_tip.rotation @ tip_jacobian[:3] @ physical['derivative']
    expected_omega = tip_jacobian[3:] @ physical['derivative']
    np.testing.assert_allclose(position, base_to_tip.translation, atol=2e-14)
    np.testing.assert_allclose(rotation, base_to_tip.rotation, atol=2e-14)
    np.testing.assert_allclose(dp, expected_dp, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(omega, expected_omega, rtol=2e-13, atol=2e-14)


def test_missing_hand_path_reports_failure_without_falling_back_to_all_link_matching(hand_model):
    identifier = make_identifier(hand_model)
    result = observed_result(hand_model)
    del result.hand_path
    record = identifier.update(result)
    assert record['status'] == 'failed_density_factor'
    assert not record['update_applied']
    assert 'hand_path' in record['failure_reason']
    assert identifier.matching_target == 'hand_pose'
    assert not identifier.factors


def test_unknown_matching_target_is_rejected_instead_of_silently_selecting_a_model(hand_model):
    with pytest.raises(ValueError, match='matching_target'):
        make_identifier(hand_model, matching_target='any_available_link')


@pytest.mark.parametrize('offset', [np.zeros(6), np.array([0., -.08, .06, .03, -.04, .02])])
def test_endpoint_factor_score_matches_actual_equilibrium_finite_differences(hand_model, offset):
    identifier, _, _ = updated_identifier(hand_model)
    factor = identifier.factors[0]
    x = identifier.initial_components[0].distribution.mean+offset
    value, score = identifier._factor_evaluation(factor, x)
    assert np.isfinite(value) and np.linalg.norm(score[1:]) > 1e-7
    expected = []
    for column in np.eye(6):
        plus = identifier._factor_evaluation(factor, x+1e-5*column)[0]
        minus = identifier._factor_evaluation(factor, x-1e-5*column)[0]
        expected.append((plus-minus)/2e-5)
    np.testing.assert_allclose(score, expected, rtol=3e-4, atol=3e-5)
    # Gravity leaves the vertical root-yaw stiffness unidentifiable here;
    # endpoint-only inference must retain that zero score rather than invent
    # information to select all six spring parameters.
    assert score[0] == pytest.approx(0., abs=1e-10)


def test_upstream_cross_covariance_changes_hand_factor_without_modifying_input_law(hand_model):
    _, inference, source, _ = hand_model
    before_mean = source.mean.tobytes()
    before_covariance = source.covariance.tobytes()
    full, _, _ = updated_identifier(hand_model)
    diagonal = source.covariance.copy()
    diagonal[4, 10] = diagonal[10, 4] = 0.
    changed = reexpress_state(source, source.mean, diagonal,
        component_id='body_without_one_upstream_cross')
    independent, _, _ = updated_identifier(hand_model, observed_result(hand_model, changed))
    full_prior = full.factors[0].ratio.marginals[0][0]
    independent_prior = independent.factors[0].ratio.marginals[0][0]
    assert not np.allclose(full_prior.covariance, independent_prior.covariance, rtol=1e-6, atol=1e-12)
    x = full.initial_components[0].distribution.mean
    assert full._factor_evaluation(full.factors[0], x)[0] != pytest.approx(
        independent._factor_evaluation(independent.factors[0], x)[0], rel=1e-7, abs=1e-9)
    assert source.mean.tobytes() == before_mean
    assert source.covariance.tobytes() == before_covariance
    assert inference.paths['hand'][-1][0] == 'rod_tip'


def test_hand_score_remains_nonzero_and_differentiable_just_inside_a_joint_stop(hand_model):
    identifier = make_identifier(hand_model)
    _, inference, initial, _ = hand_model
    # Construct an exact auxiliary equilibrium one microradian inside the
    # actual lower bound.  This uses declared model gravity, not simulator K
    # or a realized physical pose, and keeps the sent command inside its box.
    x = np.full(6, np.log(500.))
    theta = np.asarray(initial.command).copy()
    theta[1] = identifier.robot.model.lowerPositionLimit[1]+1e-6
    command = theta+identifier.robot.tau_gravity(theta)/np.exp(x)
    assert np.all(command >= identifier.robot.model.lowerPositionLimit)
    assert np.all(command <= identifier.robot.model.upperPositionLimit)
    source = inference.prior_at(command, 15., 1)
    record = identifier.update(observed_result(hand_model, source, 'hand:near_stop'))
    assert record['update_applied'], record.get('failure_reason')
    factor = identifier.factors[0]
    saved = identifier._branch_anchor
    try:
        identifier._branch_anchor = factor.branch_anchor
        physical = identifier._physical_prediction(x, source)
    finally:
        identifier._branch_anchor = saved
    np.testing.assert_allclose(physical['theta'], theta, atol=2e-10)
    assert not physical['active'][1]
    assert abs(physical['derivative'][1, 1]) > 1e-3
    _, score = identifier._factor_evaluation(factor, x)
    direction = np.eye(6)[1]
    numeric = (identifier._factor_evaluation(factor, x+1e-6*direction)[0]
        -identifier._factor_evaluation(factor, x-1e-6*direction)[0])/2e-6
    assert abs(score[1]) > 1e-7
    assert score[1] == pytest.approx(numeric, rel=3e-4, abs=3e-5)


def test_later_observation_retains_old_command_clock_factor_and_continuous_information(hand_model):
    identifier, first, _ = updated_identifier(hand_model)
    first_factor = identifier.factors[0]
    x = identifier.initial_components[0].distribution.mean
    original = identifier._factor_evaluation(first_factor, x)
    stamp = first.prior.stamp+1.2
    command = tuple(np.asarray(first.prior.command)+[0., .03, -.02, 0., 0., 0.])
    _, inference, _, _ = hand_model
    source = inference.prior_at(command, stamp, 1)
    second = observed_result(hand_model, source, 'hand:observation:1')
    record = identifier.update(second)
    assert record['update_applied'], record.get('failure_reason')
    assert len(identifier.factors) == 2
    assert identifier.factors[0] is first_factor
    assert first_factor.states[0].stamp == first.prior.stamp
    assert first_factor.states[0].command == first.prior.command
    assert identifier.factors[1].states[0].stamp == stamp
    assert identifier.factors[1].states[0].command == command
    repeated = identifier._factor_evaluation(first_factor, x)
    assert repeated[0] == original[0]
    np.testing.assert_array_equal(repeated[1], original[1])
    assert identifier.observation_ids == frozenset(('hand:observation:0', 'hand:observation:1'))
    # No reset approximation is enabled in this test: the full K density is
    # exactly its original parameter prior times both retained endpoint ratios.
    initial = multivariate_normal.logpdf(x,
        mean=identifier.initial_components[0].distribution.mean,
        cov=identifier.initial_components[0].distribution.covariance)
    contributions = [identifier._factor_evaluation(factor, x) for factor in identifier.factors]
    actual_value, actual_score = identifier.log_density(x)
    assert actual_value == pytest.approx(initial+sum(value for value, _ in contributions), abs=1e-10)
    np.testing.assert_allclose(actual_score, sum(score for _, score in contributions), atol=1e-10)


def test_pending_observations_each_requery_their_original_command_before_factor_product(hand_model):
    identifier = make_identifier(hand_model)
    _, inference, initial, _ = hand_model
    result_parts, events, specifications = [], [], []
    for index, displacement in enumerate((.0, .13)):
        command = tuple(np.asarray(initial.command)+[0., displacement, -.5*displacement, 0., 0., 0.])
        source = inference.prior_at(command, 20.+index, index)
        identity = 'hand:queued:'+str(index)
        result = observed_result(hand_model, source, identity)
        result_parts.append(result)
        events.append(dict(event_id=identity, kind='imu', stamp=source.stamp,
            sequence=index, command=source.command, payload_hash=identity+':raw'))
        specifications.append(dict(observation_id=identity,
            source_stamp=source.stamp, source_command=list(source.command),
            branches=result.frozen_body_branches,
            posterior_branch=result.projected_body_numerator,
            provenance=dict(raw_hash=events[-1]['payload_hash'])))
    terminal = result_parts[-1]
    terminal.records['observation_ids'] = [item['event_id'] for item in events]
    terminal.records['joint_batch'] = dict(batch_id='hand:queued')
    ledger = dict(kind='actual test command ledger', source_stamp=events[0]['stamp'],
        target_stamp=events[-1]['stamp'], steps=[])
    record = identifier.update_batch(terminal, events=events, command_ledger=ledger,
        batch_id='hand:queued', event_body_ratios=specifications)
    assert record['update_applied'], record.get('failure_reason')
    product = identifier.factors[0]
    assert isinstance(product, FrozenTimeIndexedStiffnessFactor)
    assert len(product.parts) == 2
    assert all(isinstance(part.ratio, HandPoseDensityRatio) for part in product.parts)
    assert [part.states[0].stamp for part in product.parts] == [20., 21.]
    assert [part.states[0].command for part in product.parts] == [event['command'] for event in events]
    x = identifier.initial_components[0].distribution.mean
    individual = [identifier._factor_evaluation(part, x) for part in product.parts]
    value, score = identifier._factor_evaluation(product, x)
    assert value == pytest.approx(sum(item[0] for item in individual), abs=1e-12)
    np.testing.assert_allclose(score, sum(item[1] for item in individual), atol=1e-12)
    assert not np.isclose(individual[0][0], individual[1][0], rtol=1e-7, atol=1e-9)
    assert not record['stiffness_batch_factorization']['endpoint_body_ratio_used_as_K_factor']

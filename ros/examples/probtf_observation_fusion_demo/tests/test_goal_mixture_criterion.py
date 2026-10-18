"""Goal modes stay a mixture even when every branch has the same body law.

The controller uses the joint-law relative criterion, not a product of
independent body/goal marginals. These explicit synthetic input laws isolate
goal modes without changing the production model or introducing a reduction.
"""

from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from probtf.bingham import bingham_log_normalizer
from probtf.geometry import axis_angle_to_quat, quat_mul, rotation_vector_to_quaternion
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, bingham_from_rotation_coordinates,
)
from probtf_observation_fusion_demo.body_native import point_query, pose_query
from probtf_observation_fusion_demo.display import (
    make_markers, serialized_marker_array, snapshot_record, verify_wire_identity,
)
from probtf_observation_fusion_demo.model import Stage, WeightedState, load_inputs
from probtf_observation_fusion_demo.moving_control import MovingController, command_state
from probtf_observation_fusion_demo.node import build_snapshot


CONFIG = Path(__file__).resolve().parents[1]/'config'
GOAL_OFFSET = np.array([.3, 0., 0.])
GOAL_ORIENTATION = np.array([1., 0., 0., 0.])
MOTION = {'max_joint_velocity': .2, 'max_joint_acceleration': .6, 'damping': .025}
URDF = ('<robot name="one"><joint name="one" type="revolute">'
        '<limit lower="-3" upper="3"/></joint></robot>')


def _problem(*, weights=(.25, .75), goal_angles=(-.22, .22), underflow=False):
    command = (.08,)
    body = BinghamTransform('body', [0., 0., 0.],
        bingham_from_rotation_coordinates(axis_angle_to_quat([0., 0., 1.], command[0]),
                                          .002*np.eye(3)), .0002*np.eye(3))
    tip = BinghamTransform('tip', GOAL_OFFSET, rotation_wxyz=[1., 0., 0., 0.])
    states = []
    for index, angle in enumerate(goal_angles):
        location = [2., 0., 0.] if underflow and index == 0 else [0., 0., 0.]
        marker = BinghamTransform('marker', location,
            bingham_from_rotation_coordinates(axis_angle_to_quat([0., 0., 1.], angle),
                                              .003*np.eye(3)), .0003*np.eye(3))
        states.append(JointTransformState.from_independent(
            [body, tip, marker], command=command, stamp=1234.25, version=9,
            component_id='goal_mode_'+str(index)))
    logs = (0., -1000.) if underflow else tuple(np.log(weights))
    weights = (1., 0.) if underflow else weights
    components = tuple(WeightedState('goal_mode_'+str(index), weight, state, log_weight)
                       for index, (weight, state, log_weight) in enumerate(zip(weights, states, logs)))
    stage = Stage('fused', components)
    inference = SimpleNamespace(body_ids=('body',),
        actions=({'joint_index': 0, 'axis': [0., 0., 1.]},),
        paths={'hand': (('body', False), ('tip', False)),
               'marker': (('marker', False),)}, settings={'root_frame': 'base'})
    result = SimpleNamespace(prior=states[0], stages=(stage,), records={'observation_ids': []},
        root_frame='base', link_paths={'body': (('body', False),)},
        hand_path=inference.paths['hand'], camera_path=(('body', False),),
        marker_path=inference.paths['marker'], local_estimates=(), sensor_posteriors=())
    # The two goal modes must not secretly encode different bodies. Their
    # body and tool transforms are the very same immutable input objects.
    assert states[0].transforms[0] is states[1].transforms[0] is body
    assert states[0].transforms[1] is states[1].transforms[1] is tip
    np.testing.assert_array_equal(states[0].covariance[:12, :12], states[1].covariance[:12, :12])
    np.testing.assert_array_equal(states[0].mean[:12], states[1].mean[:12])
    selector = MovingController(inference, URDF, GOAL_OFFSET, GOAL_ORIENTATION, MOTION)
    return inference, result, selector


def _oracle(inference, result, selector, command):
    """Independent normalized SciPy Gaussian and core Bingham density sum."""
    densities, terms = [], []
    for component in result.stages[-1].components:
        state = command_state(component.state, command, inference.body_ids, inference.actions)
        marker_index = state.edge_ids.index('marker')
        # Future q moves the body. Every original goal mode stays frozen.
        assert state.transforms[marker_index] is component.state.transforms[marker_index]
        np.testing.assert_array_equal(state.covariance[6*marker_index:6*marker_index+6,
                                                     6*marker_index:6*marker_index+6],
                                      component.state.covariance[6*marker_index:6*marker_index+6,
                                                                 6*marker_index:6*marker_index+6])
        query = pose_query(state, selector.relative_path)
        quaternion = quat_mul(query.rotation_wxyz,
            rotation_vector_to_quaternion(query.correction[3:]))
        matrix = bingham_from_rotation_coordinates(quaternion, query.covariance[3:, 3:])
        log_density = (multivariate_normal.logpdf(
            GOAL_OFFSET, mean=query.mean[:3], cov=query.covariance[:3, :3])
            + GOAL_ORIENTATION @ matrix @ GOAL_ORIENTATION
            - bingham_log_normalizer(matrix, integration_steps=selector.integration_steps))
        densities.append(float(log_density))
        terms.append(component.log_weight + log_density)
    return float(logsumexp(terms)), np.array(densities), np.array(terms)


def test_goal_mixture_uses_normalized_sum_instead_of_mean_log_or_largest_mode():
    inference, result, selector = _problem()
    selected, diagnostics = selector.step(result, .05)
    expected, densities, _ = _oracle(inference, result, selector, selected)
    np.testing.assert_allclose(diagnostics['log_expected_density'], expected, atol=1e-10)
    assert abs(expected-np.array([.25, .75]) @ densities) > 1e-3
    assert abs(expected-densities[1]) > 1e-3
    assert diagnostics['source_weights'] == [.25, .75]
    assert diagnostics['source_log_weights'] == list(np.log([.25, .75]))
    assert len(diagnostics['selected_branches']) == 2
    for component, branch in zip(result.stages[-1].components, diagnostics['selected_branches']):
        assert branch['parent_state_id'] == component.state.state_id
        assert branch['log_weight'] == component.log_weight


@pytest.mark.parametrize('changed', [
    {'weights': (.75, .25)}, {'goal_angles': (-.22, .35)},
])
def test_second_goal_mode_and_its_weight_affect_the_actual_full_objective_gradient(changed):
    outputs = []
    for options in ({}, changed):
        inference, result, selector = _problem(**options)
        selected, diagnostics = selector.step(result, .05)
        expected, _, _ = _oracle(inference, result, selector, selected)
        np.testing.assert_allclose(diagnostics['log_expected_density'], expected, atol=1e-10)
        baseline = np.array(diagnostics['gradient_at_command'])
        difference = diagnostics['gradient_difference_steps'][0]
        initial = _oracle(inference, result, selector, baseline)[0]
        displaced = _oracle(inference, result, selector, baseline+[difference])[0]
        np.testing.assert_allclose(diagnostics['objective_gradient'],
                                  [-(displaced-initial)/difference], rtol=1e-6, atol=1e-7)
        outputs.append(diagnostics)
    assert abs(outputs[0]['log_expected_density']-outputs[1]['log_expected_density']) > 1e-3
    assert np.linalg.norm(np.array(outputs[0]['objective_gradient'])
                          -outputs[1]['objective_gradient']) > 1e-3


def test_finite_log_mass_with_underflowed_weight_can_dominate_the_goal_overlap():
    inference, result, selector = _problem(underflow=True, goal_angles=(.08, .08))
    selected, diagnostics = selector.step(result, .05)
    expected, _, terms = _oracle(inference, result, selector, selected)
    assert result.stages[-1].components[1].weight == 0.
    assert result.stages[-1].components[1].log_weight == -1000.
    assert terms[1] > terms[0]+1000.
    np.testing.assert_allclose(diagnostics['log_expected_density'], expected, atol=1e-10)
    np.testing.assert_allclose(expected, terms[1], atol=1e-10)
    assert diagnostics['source_log_weights'] == [0., -1000.]
    assert len(diagnostics['selected_branches']) == 2
    assert diagnostics['selected_branches'][1]['log_weight'] == -1000.
    # Removing the positive-mass near-goal branch changes the actual criterion
    # and command gradient, despite its ordinary float weight underflowing.
    far_only = replace(result.stages[-1], components=(result.stages[-1].components[0],))
    reduced = SimpleNamespace(**dict(vars(result), stages=(far_only,)))
    _, without = MovingController(inference, URDF, GOAL_OFFSET, GOAL_ORIENTATION, MOTION).step(reduced, .05)
    assert diagnostics['log_expected_density'] > without['log_expected_density']+1000.
    assert np.linalg.norm(np.array(diagnostics['objective_gradient'])-without['objective_gradient']) > 1.


@pytest.mark.parametrize('underflow', [False, True])
def test_control_source_and_ros_wire_goal_mixture_share_ids_times_commands_and_log_weights(underflow):
    inference, result, selector = _problem(underflow=underflow)
    _, diagnostics = selector.step(result, .05)
    common, _ = load_inputs(CONFIG/'common.yaml', CONFIG/'no_sensors.yaml')
    settings = common['display']
    snapshot = build_snapshot(result, result.stages[-1], common)
    message = make_markers(snapshot, settings['seed'], settings['point_size'])
    payload = verify_wire_identity(snapshot, message, seed=settings['seed'], point_size=settings['point_size'])
    received = MarkerArray().deserialize(payload)
    assert serialized_marker_array(received) == payload
    # The public snapshot String preserves metadata that MarkerArray itself
    # does not encode, including the finite log mass of a transparent branch.
    metadata = String(json.dumps(snapshot_record(snapshot)))
    stream = io.BytesIO()
    metadata.serialize(stream)
    received_metadata = json.loads(String().deserialize(stream.getvalue()).data)
    states = {component.label: component for component in result.stages[-1].components}
    assert diagnostics['source_state_ids'] == [c.state.state_id for c in states.values()]
    assert diagnostics['source_weights'] == [c.weight for c in states.values()]
    assert diagnostics['source_log_weights'] == [c.log_weight for c in states.values()]
    rendered = [m for m in received.markers if m.type == Marker.SPHERE_LIST and m.ns.startswith('gaussian/')]
    assert len(rendered) == len(snapshot.points) == len(received_metadata['queries'])
    endpoints = {'origin': [0., 0., 0.], 'x': [settings['axis_length'], 0., 0.],
                 'y': [0., settings['axis_length'], 0.], 'z': [0., 0., settings['axis_length']]}
    paths = dict(result.link_paths, hand=result.hand_path, camera=result.camera_path, marker=result.marker_path)
    for point, record, marker in zip(snapshot.points, received_metadata['queries'], rendered):
        component = states[point.component_id]
        query = point_query(component.state, paths[point.frame_label], endpoints[point.axis],
                            frame=result.root_frame)
        assert record['component_id'] == record['source_component_id'] == component.label
        assert record['source_state_id'] == component.state.state_id
        assert record['weight'] == component.weight
        assert record['log_weight'] == component.log_weight
        assert record['source_command'] == diagnostics['source_command'] == list(component.state.command)
        assert record['source_stamp'] == diagnostics['source_stamp'] == component.state.stamp
        assert record['source_version'] == component.state.version
        assert marker.header.stamp.to_sec() == component.state.stamp
        assert marker.header.frame_id == query.frame == snapshot.frame_id
        np.testing.assert_array_equal(record['mean'], query.mean)
        np.testing.assert_array_equal(record['covariance'], query.covariance)
        np.testing.assert_allclose(marker.color.a, .85*component.weight, atol=1e-7)
    assert len({record['source_state_id'] for record in received_metadata['queries']}) == 2
    if underflow:
        assert any(record['weight'] == 0. and record['log_weight'] == -1000.
                   for record in received_metadata['queries'])

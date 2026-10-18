#!/usr/bin/python3
"""Compare held-gradient and local-Fisher execution on two frozen candidate laws.

Run from a sourced ROS workspace with the demo and ProbTF Python dependencies.
For example::

    /usr/bin/python3 validation/tools/compare_frozen_candidate_tracking.py \
        --output-dir /tmp/probtf_frozen_candidate_comparison

The fixture renders one initial image and IMU input using MovingSwitchScene,
then uses the simultaneous compatibility inference API. K Bayes-factor surprise
is explicitly disabled ONLY for this diagnostic fixture. Each execution model
receives one immutable packet. No further sensor, K, or gradient update occurs.
The scene does not execute the compared commands: this is not a physical robot
reaching experiment, a live throughput measurement, or a stability guarantee.

All generated scene files and full comparison JSON go under --output-dir.
The script does not add any generated file to Git or change configuration.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'src'))

ETA = 1e-6
GAMMA = 8.
WEIGHT = 2e6
ANGLE_DEG = 10.
RIDGE_ABSOLUTE = 1e5
RIDGE_RELATIVE = 1e-6
SOURCE_STAMP = 1792080000.
CSV_FIELDS = ('model', 'time', 'command', 'velocity', 'exact_frozen_J',
              'exact_frozen_V', 'exact_frozen_selected_objective',
              'estimated_relative_position_rms_m',
              'local_model_equilibrium_distance')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def modal_time_constants(eigenvalues):
    """Envelope/slow exponential time constants for fixed positive curvature."""
    result = []
    for eigenvalue in eigenvalues:
        discriminant = GAMMA * GAMMA - 4. * ETA * eigenvalue
        # The rational form avoids cancellation in the overdamped slow root.
        rate = (GAMMA / 2. if discriminant < 0. else
                2. * ETA * eigenvalue / (GAMMA + math.sqrt(discriminant)))
        result.append(1. / rate)
    return result


def run_comparison(output_dir, *, cases=('marker', 'imu_marker'), duration=30.,
                   dt=.005, diagnostic_period=.5):
    import yaml
    from probtf.spherical_law.first_order import pose_query
    from probtf_observation_fusion_demo.async_state import predict_result
    from probtf_observation_fusion_demo.async_stiffness import (
        PhysicalActionEngine, capture_stiffness_packet,
        capture_direct_stiffness_action, reanchor_retained_stiffness,
    )
    from probtf_observation_fusion_demo.moving_control import (
        MovingController, control_components,
    )
    from probtf_observation_fusion_demo.moving_inference import MovingInference
    from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene
    from probtf_observation_fusion_demo.running_gradient import (
        GradientPacket, RunningGradientTracker,
    )
    from probtf_observation_fusion_demo.selection_gradient import (
        selection_gradient_at_command,
    )
    from probtf_observation_fusion_demo.stiffness_identification import StiffnessIdentifier
    from probtf_observation_fusion_demo.visibility_factor import VisibilityFactor

    for name, value in (('duration', duration), ('dt', dt),
                        ('diagnostic_period', diagnostic_period)):
        if not math.isfinite(value) or value <= 0.:
            raise ValueError(name + ' must be positive and finite')
    if not cases or any(name not in ('marker', 'imu_marker') for name in cases):
        raise ValueError('Choose marker and/or imu_marker')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = MovingSwitchScene(output_dir / 'scene', imu_target_link='tool0')
    inputs = scene.sensor_inputs(scene.step(0., SOURCE_STAMP))
    settings = yaml.safe_load((PACKAGE / 'config/stiffness_identification.yaml').read_text())
    settings['bayes_factor_surprise'] = {'enabled': False}
    results = {}
    for name in cases:
        templates = yaml.safe_load((PACKAGE / ('config/' + name + '.yaml')).read_text())['observations']
        inference = MovingInference(scene.robot_description, templates)
        result = inference.update(inputs, 0)
        identifier = StiffnessIdentifier(scene.urdf, inference.body_ids, settings)
        result.records['stiffness'] = identifier.update(result)
        identifier.command_proposal(result)
        stiffness_packet = capture_stiffness_packet(result)
        engine = PhysicalActionEngine(scene.urdf, inference.body_ids, settings)
        predicted = predict_result(result, (), target_stamp=result.prior.stamp,
                                   frame_index=0, body_ids=inference.body_ids,
                                   actions=inference.actions)
        action = capture_direct_stiffness_action(engine, predicted, stiffness_packet)
        source, mapping = reanchor_retained_stiffness(engine, predicted, action)
        config = inference.settings['control']
        controller = MovingController(inference, scene.robot_description,
            config['goal_offset'], config['goal_orientation_wxyz'], config,
            integration_steps=config['integration_steps'])
        factor = VisibilityFactor(dict(enabled=True, weight=WEIGHT,
            penalty_mode='tolerance_cone', tolerance_angle_deg=ANGLE_DEG))
        controller.local_quadratic_model_settings = dict(enabled=True,
            regularization_absolute=RIDGE_ABSOLUTE,
            regularization_relative=RIDGE_RELATIVE)
        local = selection_gradient_at_command(controller, source, factor,
                                             command_map=mapping).record
        if local['status'] != 'ok':
            raise ValueError('Local model evaluation failed: ' + str(local.get('reason')))
        controller.local_quadratic_model_settings = {'enabled': False}
        constant = selection_gradient_at_command(controller, source, factor,
                                                command_map=mapping).record
        if constant['status'] != 'ok':
            raise ValueError('Constant gradient evaluation failed: ' + str(constant.get('reason')))
        identical = all(np.asarray(constant[key], dtype=np.float64).tobytes() ==
                        np.asarray(local[key], dtype=np.float64).tobytes()
                        for key in ('gradient', 'objective_at_source'))
        if not identical:
            raise AssertionError('The two execution models must have identical source J/g')
        equilibrium = local['local_model_diagnostics']['unconstrained_equilibrium_command']
        equilibrium = None if equilibrium is None else np.asarray(equilibrium)
        _, components = control_components(source)
        camera_relative_marker = tuple((edge, not inverse)
            for edge, inverse in reversed(source.camera_path)) + source.marker_path

        def diagnostics(command):
            command = np.asarray(command)
            task, branches = controller.objective_at_command(source, command,
                                                             command_map=mapping)
            weights = np.asarray([branch['weight'] for branch in branches])
            weights = weights / weights.sum()
            error2 = sum(weight * (
                np.linalg.norm(np.asarray(branch['relative_position_mean']) -
                               controller.goal_offset) ** 2 +
                np.trace(branch['relative_position_covariance']))
                for weight, branch in zip(weights, branches))
            states = [mapping.push(component.state, command,
                      inference.body_ids, inference.actions) for component in components]
            view = factor.evaluate(components, [pose_query(state, camera_relative_marker)
                                               for state in states])['penalty']
            return dict(exact_frozen_J=task, exact_frozen_V=view,
                exact_frozen_selected_objective=task + WEIGHT * view,
                estimated_relative_position_rms_m=float(math.sqrt(error2)),
                local_model_equilibrium_distance=(None if equilibrium is None else
                    float(np.linalg.norm(command - equilibrium))))

        trials = {}
        for label, record in (('constant', constant), ('local_fisher', local)):
            packet = GradientPacket(**{key: record[key] for key in
                GradientPacket.__dataclass_fields__ if key in record})
            tracker = RunningGradientTracker(source.prior.command, eta=ETA, gamma=GAMMA,
                joint_lower=controller.lower, joint_upper=controller.upper,
                initial_stamp=source.prior.stamp)
            adoption = tracker.accept(packet, source.prior.stamp)
            if not adoption['accepted']:
                raise ValueError('Packet rejected: ' + str(adoption))
            rows = [dict(time=0., command=list(source.prior.command),
                         velocity=[0.] * len(source.prior.command),
                         **diagnostics(source.prior.command))]
            halted, maximum_speed, safe_time = None, 0., 0.
            minimum = np.asarray(source.prior.command).copy()
            maximum = minimum.copy()
            next_diagnostic = diagnostic_period
            for index in range(int(math.ceil(duration / dt))):
                attempted_time = min(duration, (index + 1) * dt)
                interval = min(dt, duration - index * dt)
                if interval <= 0.:
                    break
                update = tracker.advance(interval)
                if update.record['halted']:
                    halted = dict(attempted_time_s=attempted_time,
                                  last_safe_time_s=safe_time, record=update.record)
                    break
                safe_time = attempted_time
                maximum_speed = max(maximum_speed, float(np.linalg.norm(update.velocity)))
                minimum = np.minimum(minimum, update.command)
                maximum = np.maximum(maximum, update.command)
                if safe_time + 1e-12 >= next_diagnostic:
                    rows.append(dict(time=safe_time, command=list(update.command),
                                     velocity=list(update.velocity),
                                     **diagnostics(update.command)))
                    while next_diagnostic <= safe_time + 1e-12:
                        next_diagnostic += diagnostic_period
            if rows[-1]['time'] != safe_time:
                rows.append(dict(time=safe_time, command=list(tracker.position),
                                 velocity=list(tracker.velocity),
                                 **diagnostics(tracker.position)))
            trials[label] = dict(physical_simulation_run=False,
                elapsed_model_time_s=safe_time, halted=halted,
                maximum_velocity_norm=maximum_speed,
                command_min=minimum.tolist(), command_max=maximum.tolist(), samples=rows)
            print(name, label, 'model seconds', safe_time, 'stop',
                  None if halted is None else halted['record'].get('reason'),
                  'selected objective', rows[-1]['exact_frozen_selected_objective'],
                  'estimated RMS', rows[-1]['estimated_relative_position_rms_m'], flush=True)
        results[name] = dict(
            scope='One initial full-body/goal/K law frozen throughout. Exact J/V/RMS queries are diagnostics only. No physical trajectory is run.',
            fixture='Rendered-image/IMU simultaneous compatibility fixture; K BFS explicitly disabled only here.',
            **{key: local[key] for key in ('source_stamp', 'source_command',
                'body_state_ids', 'goal_state_ids', 'parameter_density_id', 'selected_K',
                'gradient', 'curvature', 'local_model_diagnostics')},
            modal_envelope_time_constants_s=modal_time_constants(
                local['local_model_diagnostics']['eigenvalues']),
            eta=ETA, gamma=GAMMA, weight=WEIGHT, tolerance_angle_deg=ANGLE_DEG,
            requested_duration_s=duration, dt_s=dt,
            diagnostic_period_s=diagnostic_period,
            joint_lower=controller.lower.tolist(), joint_upper=controller.upper.tolist(),
            task_objective_and_gradient_bitwise_identical_between_execution_models=identical,
            final_sample_after_stop_velocity='Halted zero velocity; pre-stop motion velocity remains in halted.record.last_safe_velocity.',
            cases=trials)
        with (output_dir / (name + '_trajectory.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, CSV_FIELDS)
            writer.writeheader()
            for label, trial in trials.items():
                for row in trial['samples']:
                    writer.writerow(dict(model=label, **row))
    (output_dir / 'comparison.json').write_text(json.dumps(results, indent=2,
                                                         allow_nan=False) + '\n')
    source_files = [Path(__file__).resolve()] + [PACKAGE / 'src/probtf_observation_fusion_demo' / name
        for name in ('moving_control.py', 'visibility_factor.py', 'selection_gradient.py',
                     'local_curvature.py', 'running_gradient.py', 'local_quadratic_tracking.py')]
    configuration_files = [PACKAGE / 'config' / name for name in (
        'moving_estimation.yaml', 'stiffness_identification.yaml', 'marker.yaml', 'imu_marker.yaml')]
    manifest = dict(schema='probtf_frozen_candidate_comparison_manifest_v1',
        package_directory=str(PACKAGE), output_directory=str(output_dir.resolve()),
        source_stamp=SOURCE_STAMP, cases=list(cases), duration_s=duration,
        dt_s=dt, diagnostic_period_s=diagnostic_period,
        eta=ETA, gamma=GAMMA, visibility_weight=WEIGHT,
        penalty_mode='tolerance_cone', tolerance_angle_deg=ANGLE_DEG,
        regularization_absolute=RIDGE_ABSOLUTE, regularization_relative=RIDGE_RELATIVE,
        stiffness_BFS_enabled_in_this_fixture=False,
        physical_trajectory_run=False, monte_carlo_used=False,
        source_sha256={str(path.relative_to(PACKAGE)): sha256(path)
                       for path in source_files + configuration_files},
        limitations=['This frozen fixture uses the simultaneous compatibility inference API, not the asynchronous joint-batch acquisition pipeline.',
                     'No subsequent observation, K update, gradient update or packet switch occurs.',
                     'Estimated relative RMS is a candidate-distribution quantity, not actual physical tip error.',
                     'The nonlinear full criterion need not decrease under the fixed local quadratic model.',
                     'Joint-box failures are actual domain violations; no proposed command is clipped or executed.'])
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2,
                                                       allow_nan=False) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Destination for generated scene files, full JSON and CSVs')
    parser.add_argument('--cases', nargs='+', choices=('marker', 'imu_marker'),
                        default=('marker', 'imu_marker'))
    parser.add_argument('--duration', type=float, default=30., help='Command-model seconds')
    parser.add_argument('--dt', type=float, default=.005, help='Command integration interval')
    parser.add_argument('--diagnostic-period', type=float, default=.5)
    args = parser.parse_args()
    run_comparison(args.output_dir, cases=args.cases, duration=args.duration,
                   dt=args.dt, diagnostic_period=args.diagnostic_period)


if __name__ == '__main__':
    main()

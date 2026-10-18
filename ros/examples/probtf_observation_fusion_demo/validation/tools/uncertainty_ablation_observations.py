"""Withdrawn observation oracles retained as historical source.

Both construction and installation are disabled by the 2026-10-19 user
instruction; importing this module does not authorize a truth observation.

Nothing imports this module on the production path.  The diagnostic runner
records physical acquisition poses separately, then installs the patch in its
isolated inference process.  The oracle replaces the *derived central-board
observation*, never the individual image detector's native modes.  A baseline
visibility gate is retained by default, so accuracy and missingness are not
silently changed together.
"""

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from uncertainty_ablation_policy import reject_truth_injection

from prob_artag_detector.models import EstimationDiagnostics, PoseMixtureResult
from probtf.distributions.bingham_orientation import BinghamOrientation
from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
from probtf.distributions.stamped import TransformDistributionStamped
from probtf.distributions.transform_component import TransformComponent
from probtf.distributions.transform_distribution import TransformDistribution
from probtf.geometry import quat_to_rotmat, rotmat_to_quat
from probtf.provenance import ComponentProvenance, TransformProvenance
from probtf.spherical_law.first_order import bingham_from_rotation_coordinates
from probtf_observation_fusion_demo.scene import camera_model
from probtf_observation_fusion_demo.sensors import PreparedMarkerObservation


def _finite_pose(value, name):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(name + ' must be a finite physical acquisition 4x4 pose.')
    return pose


def prepare_oracle_marker_observation(
        common, image, stamp, *, camera_pose, marker_pose, observation_id,
        command, capture_stamp, translation_std_m=1e-4, rotation_std_rad=1e-3):
    """Construct one declared true camera-relative central-board mode.

    The two standard deviations are finite, diagnostic model parameters, not
    camera detector covariance estimates or a covariance floor.  No image is
    decoded here; the caller decides whether baseline visibility is required.
    Acquisition metadata is copied into the returned board provenance.
    """
    reject_truth_injection('oracle marker observation construction')
    stamp, capture_stamp = float(stamp), float(capture_stamp)
    if not np.isfinite(stamp) or stamp < 0. or capture_stamp != stamp:
        raise ValueError('Oracle truth and image must have the same original acquisition stamp.')
    observation_id = str(observation_id)
    if not observation_id or not observation_id.startswith('image:'):
        raise ValueError('The oracle requires the original image observation ID.')
    command = np.asarray(command, dtype=float)
    if command.ndim != 1 or not np.all(np.isfinite(command)):
        raise ValueError('The oracle needs the finite original sent command.')
    translation_std_m, rotation_std_rad = float(translation_std_m), float(rotation_std_rad)
    if (not np.isfinite(translation_std_m) or translation_std_m <= 0.
            or not np.isfinite(rotation_std_rad) or rotation_std_rad <= 0.):
        raise ValueError('Oracle uncertainty parameters must be positive and finite.')
    camera_pose = _finite_pose(camera_pose, 'camera_pose')
    marker_pose = _finite_pose(marker_pose, 'marker_pose')
    relative = np.linalg.solve(camera_pose, marker_pose)
    rotation = rotmat_to_quat(relative[:3, :3])
    rotation_covariance = rotation_std_rad ** 2 * np.eye(3)
    translation_covariance = translation_std_m ** 2 * np.eye(3)
    parameter = bingham_from_rotation_coordinates(rotation, rotation_covariance)
    detail = ('DIAGNOSTIC ORACLE: exact physical camera-relative board center at '
              'the original image acquisition; single true mode supplied instead '
              'of image ambiguity; finite Gaussian/Bingham surrogate uncertainty.')
    source_ids = (observation_id, 'diagnostic_physical_camera_and_board_capture')
    component_id = 'diagnostic_oracle_central_board:' + observation_id
    component = TransformComponent(component_id, 1.,
        BinghamOrientation.from_parameter_matrix(parameter,
            reference_quaternion_wxyz=rotation),
        ConditionalGaussianTranslation(relative[:3, 3], translation_covariance,
                                       np.zeros((3, 9))),
        provenance=ComponentProvenance(source_ids=source_ids,
            method='diagnostic_exact_capture_central_pose', detail=detail))
    stamped = TransformDistributionStamped('camera', 'marker', stamp,
        'camera__to__marker', 'diagnostic_uncertainty_ablation',
        TransformDistribution((component,)),
        provenance=TransformProvenance(source_ids=source_ids,
            method='diagnostic_exact_capture_central_pose', detail=detail))
    detected = PoseMixtureResult(stamped,
        EstimationDiagnostics(1, 1, 0, 0, 0, 0, ()),
        (quat_to_rotmat(rotation),), (component.translation.mean_at_reference,),
        (0.,), (1.,), (0,))
    provenance = dict(available=True, diagnostic_only=True,
        intervention='oracle_camera_relative_central_board',
        physical_truth_received=True, physical_truth_used_for_inference=True,
        original_image_observation_id=observation_id, source_stamp=stamp,
        capture_stamp=capture_stamp, source_command=command.tolist(),
        translation_std_m=translation_std_m, rotation_std_rad=rotation_std_rad,
        true_camera_pose=camera_pose.tolist(), true_marker_pose=marker_pose.tolist(),
        exact_camera_relative_pose=relative.tolist(),
        exact_translation_mean=relative[:3, 3].tolist(),
        exact_rotation_wxyz=rotation.tolist(),
        central_component_count=1, image_detector_native_modes_replaced=False,
        central_image_observation_bypassed=True, oracle_true_mode_supplied=True,
        board_fusion_bypassed=True, detector_ambiguity_removed=True,
        monte_carlo_used=False, detail=detail)
    descriptor = dict(component_id=component_id, parent_frame_id='camera',
        child_frame_id='marker', edge_id=stamped.edge_id, stamp=stamp,
        weight=1., log_weight=0., translation_mean=relative[:3, 3].tolist(),
        translation_covariance=translation_covariance.tolist(),
        rotation_wxyz=rotation.tolist(), bingham=parameter.tolist(),
        full_product_covariance=np.diag([translation_std_m**2]*3+
                                      [rotation_std_rad**2]*3).tolist(),
        lineage=[dict(observation_id=observation_id,
            method='authorized diagnostic exact physical central pose')],
        position_orientation_product_approximation=True,
        diagnostic_only=True, physical_truth_received=True)
    # Fresh-goal construction validates this same central-law identity even
    # for an intentional oracle. Keep the actual board record schema rather
    # than masquerading as a raw image observation or skipping that check.
    provenance.update(raw_observations=[],
        central_distribution=dict(parent_frame_id='camera', child_frame_id='marker',
            edge_id=stamped.edge_id, stamp=stamp, components=[descriptor],
            component_count=1, representation='diagnostic Gaussian x Bingham',
            **{key:value for key,value in descriptor.items()
               if key not in ('parent_frame_id','child_frame_id','edge_id','stamp')}))
    provenance['source_digest'] = hashlib.sha256(json.dumps(provenance,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return PreparedMarkerObservation(image, stamp, (), detected,
        camera_model(common['camera']), np.asarray(image).copy(), (), provenance)


def install_oracle_marker_patch(output, *, translation_std_m=1e-4,
                                rotation_std_rad=1e-3, require_detected=True):
    """Install an isolated worker patch and return a restoration callback.

    ``output/diagnostic_marker_truth/<image event id>.json`` must be written
    atomically by the diagnostic acquisition wrapper before supplying that
    event.  Each JSON object has ``event_id, stamp, command, camera_pose,
    marker_pose`` from the *same original acquisition request*.  Stamps are
    matched exactly; a missing/ambiguous source is an implementation error,
    never a reason to select nearby truth or invent an observation.

    With ``require_detected=True`` the original image pipeline is still run
    once.  Its missing observation stays missing, and its native raw results
    remain available for audit.  Only an available central observation is
    replaced.  ``False`` is the separately declared no-loss intervention.
    """
    reject_truth_injection('oracle marker inference patch')
    from probtf_observation_fusion_demo import sensors

    output = Path(output)
    directory = output / 'diagnostic_marker_truth'
    original = sensors.prepare_marker_observation
    loaded, truth_by_stamp = set(), {}

    def capture_at(stamp):
        for path in sorted(directory.glob('*.json')):
            if path.name in loaded:
                continue
            capture = json.loads(path.read_text())
            capture_stamp = float(capture['stamp'])
            if capture_stamp in truth_by_stamp:
                raise ValueError('More than one diagnostic image capture has the same stamp.')
            if path.stem != str(capture['event_id']):
                raise ValueError('Oracle sidecar filename differs from original event ID.')
            truth_by_stamp[capture_stamp] = capture
            loaded.add(path.name)
        if float(stamp) not in truth_by_stamp:
            raise ValueError('No exact original acquisition truth for this diagnostic image stamp.')
        return truth_by_stamp[float(stamp)]

    def prepare(common, image, stamp):
        baseline = original(common, image, stamp)
        capture = capture_at(stamp)
        if require_detected and baseline.detected is None:
            record = dict(baseline.board_fusion or {})
            record.update(diagnostic_only=True,
                intervention='oracle_camera_relative_central_board',
                intervention_applied=False, baseline_detection_required=True,
                baseline_marker_available=False,
                original_image_observation_id=capture['event_id'],
                source_stamp=float(stamp), source_command=list(capture['command']),
                reason='baseline image observation missing; oracle does not remove missingness')
            prepared = replace(baseline, board_fusion=record)
        else:
            prepared = prepare_oracle_marker_observation(common, image, stamp,
                camera_pose=capture['camera_pose'], marker_pose=capture['marker_pose'],
                observation_id=capture['event_id'], command=capture['command'],
                capture_stamp=capture['stamp'], translation_std_m=translation_std_m,
                rotation_std_rad=rotation_std_rad)
            record = dict(prepared.board_fusion)
            record.update(intervention_applied=True,
                baseline_detection_required=bool(require_detected),
                baseline_marker_available=baseline.detected is not None,
                baseline_central_component_count=(0 if baseline.detected is None
                    else len(baseline.detected.record.distribution.components)),
                raw_observations=(baseline.board_fusion or {}).get('raw_observations', []),
                baseline_board_fusion=baseline.board_fusion)
            prepared = replace(prepared, observations=baseline.observations,
                debug_image=baseline.debug_image,
                individual_results=baseline.individual_results, board_fusion=record)
        log = dict(prepared.board_fusion or {})
        # Individual native board records are already archived by the normal
        # diagnostic output. Keep this additional intervention ledger small.
        log.pop('baseline_board_fusion', None)
        with (output / 'diagnostic_marker_interventions.jsonl').open('a') as stream:
            stream.write(json.dumps(log, sort_keys=True, allow_nan=False) + '\n')
        return prepared

    modules = (sensors,) + tuple(importlib.import_module(
        'probtf_observation_fusion_demo.' + name) for name in
        ('moment_observation', 'batch_observation_factors', 'nonlinear_observation'))
    previous = [(module, module.prepare_marker_observation) for module in modules]
    previous_preparer = sensors.PREPARERS['marker']
    for module in modules:
        module.prepare_marker_observation = prepare
    sensors.PREPARERS['marker'] = prepare

    moment_module = modules[1]
    variational_module = importlib.import_module('probtf_observation_fusion_demo.variational_model')
    original_moment_prepare = moment_module.prepare_observation
    original_variational_prepare = variational_module.prepare_observation

    def prepare_native(common, observation, paths, image, stamp):
        prepared = original_moment_prepare(common, observation, paths, image, stamp)
        if prepared.prepared_marker is None:
            return prepared
        board = prepared.prepared_marker.board_fusion or {}
        if not board.get('intervention_applied', False):
            return prepared
        diagnostics = dict(prepared.diagnostics)
        diagnostics.update(diagnostic_only=True, physical_truth_received=True,
            physical_truth_used_for_inference=True,
            image_approximation=('diagnostic exact acquisition-relative central mode with '
                'declared finite Gaussian/Bingham uncertainty; not an image Hessian estimate'),
            detector_prior='diagnostic physical-truth central observation replacement',
            original_image_observation_id=board['original_image_observation_id'],
            source_command=board['source_command'], source_stamp=board['source_stamp'])
        return replace(prepared, diagnostics=diagnostics)

    moment_module.prepare_observation = prepare_native
    variational_module.prepare_observation = prepare_native

    def restore():
        for module, old in previous:
            module.prepare_marker_observation = old
        sensors.PREPARERS['marker'] = previous_preparer
        moment_module.prepare_observation = original_moment_prepare
        variational_module.prepare_observation = original_variational_prepare

    return restore

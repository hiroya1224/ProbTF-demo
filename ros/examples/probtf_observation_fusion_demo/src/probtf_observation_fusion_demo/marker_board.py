"""Known board geometry and image-only assumed-density central-pose fusion.

Every decoded marker keeps its native observation mixture. Only derived
central-board laws are reduced, when a unique modal chart and its required
inverse can actually be formed. Finite width is not a rejection criterion.
Otherwise all branches continue through the next marker product.
This is order-dependent assumed-density filtering of image-only local
Laplace laws, not an exact product of all native combinations on SE(3).
"""

from dataclasses import dataclass
import hashlib
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from prob_artag_detector.models import EstimationDiagnostics, PoseMixtureResult
from probtf.bingham import (
    bingham_second_moment as _python_bingham_second_moment,
    match_bingham_to_second_moment as _python_match_bingham_to_second_moment,
)
from .marker_board_native import (
    bingham_second_moment, match_bingham_to_second_moment, backend_record,
    diagonal_second_moment,
)
from probtf.distributions.bingham_orientation import BinghamOrientation
from probtf.distributions.conditional_translation import ConditionalGaussianTranslation
from probtf.distributions.stamped import TransformDistributionStamped
from probtf.distributions.transform_component import TransformComponent
from probtf.distributions.transform_distribution import TransformDistribution
from probtf.geometry import (
    quat_mul, quat_to_rotmat, quaternion_to_rotation_vector, rotmat_to_quat,
    rotation_vector_to_quaternion,
)
from probtf.spherical_law.first_order import (
    BinghamTransform, JointTransformState, _right_jacobian,
    bingham_from_rotation_coordinates, bingham_rotation_coordinates, pose_query,
)
from probtf.spherical_law.fusion import Gaussian, condition_gaussian
from probtf.spherical_law.support import covariance_support


@dataclass(frozen=True)
class BoardMarker:
    marker_id: int
    row: int
    column: int
    size_m: float
    center_to_marker_translation: tuple
    rotation_wxyz: tuple = (1., 0., 0., 0.)

    @property
    def marker_to_center_translation(self):
        return tuple(-value for value in self.center_to_marker_translation)


def board_layout(settings):
    """Return row-major 3x3 tags; positive board Y points toward the top row."""
    if int(settings.get('rows', 3)) != 3 or int(settings.get('columns', 3)) != 3:
        raise ValueError('This known marker board requires a 3x3 layout.')
    size = float(settings.get('marker_size', .072))
    pitch = float(settings.get('pitch', .09))
    ids = np.asarray(settings.get('ids', ((17, 18, 19), (20, 21, 22), (23, 24, 25))))
    if (not np.isfinite(size) or not np.isfinite(pitch) or size <= 0. or pitch <= size
            or ids.shape != (3, 3) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(set(ids.ravel().tolist())) != 9):
        raise ValueError('A marker board requires nine unique nonnegative IDs and positive nonoverlapping sizes.')
    return tuple(BoardMarker(int(ids[row, column]), row, column, size,
        ((column-1)*pitch, (1-row)*pitch, 0.)) for row in range(3) for column in range(3))


@dataclass(frozen=True)
class BoardFusionResult:
    detected: object
    record: dict


def _query_record(query):
    return dict(query_id=query.query_id, source_state_id=query.state_id,
        frame=query.frame, stamp=query.stamp, rotation_wxyz=query.rotation_wxyz.tolist(),
        **{name: getattr(query, name).tolist() for name in (
            'mode', 'mean', 'covariance', 'jacobian', 'cross_covariance', 'within', 'between')})


def _native_to_center(detected, marker, arithmetic_backend='native'):
    # Lazy import permits sensors.prepare_marker_observation to use this
    # helper, while sharing precisely its existing native-Hessian adapter.
    from .sensors import _marker_mode_chart
    components = tuple(detected.record.distribution.components)
    logs = np.asarray(detected.log_masses, dtype=float)
    weights = np.asarray(detected.weights, dtype=float)
    if (not components or logs.shape != (len(components),) or weights.shape != logs.shape
            or not np.all(np.isfinite(logs)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0.) or len({item.component_id for item in components}) != len(components)):
        raise ValueError('Board fusion requires every native marker mode and finite log mass.')
    normalized = logs-float(logsumexp(logs))
    if not np.allclose(weights, np.exp(normalized), rtol=1e-12, atol=1e-15):
        raise ValueError('Native board-marker float weights disagree with their log masses.')
    output, records = [], []
    for index, component in enumerate(components):
        rotation = component.orientation.mode_wxyz
        native, native_volume = _marker_mode_chart(component, quat_to_rotmat(rotation))
        covariance = np.zeros((12, 12))
        covariance[:6, :6] = native.covariance
        original = BinghamTransform('image_marker', native.mean[:3],
            component.orientation.parameter_matrix(), native.covariance[:3, :3])
        attachment = BinghamTransform('known_marker_to_center',
            marker.marker_to_center_translation, rotation_wxyz=marker.rotation_wxyz)
        state = JointTransformState((original, attachment), covariance,
            stamp=detected.record.stamp, component_id=component.component_id)
        query_function = pose_query
        if arithmetic_backend == 'native':
            from .body_native import pose_query as query_function
        query = query_function(state, (('image_marker', False), ('known_marker_to_center', False)),
            query_id='marker_board:{}:{}'.format(marker.marker_id, component.component_id),
            frame=detected.record.parent_frame_id)
        # Reexpress only this derived central branch at its corrected mean.
        # Position/rotation cross terms remain in the complete 6x6 law.
        chart = np.eye(6)
        chart[3:, 3:] = _right_jacobian(query.mean[3:])
        center_covariance = chart @ query.covariance @ chart.T
        center_covariance = .5*(center_covariance+center_covariance.T)
        quaternion = quat_mul(query.rotation_wxyz,
            rotation_vector_to_quaternion(query.mean[3:]))
        sign, volume = np.linalg.slogdet(chart @ query.jacobian[:, :6])
        if sign <= 0. or not np.isfinite(volume):
            raise ValueError('The known board attachment has an invalid local chart volume.')
        output.append(dict(label=component.component_id, translation=query.mean[:3],
            rotation=quaternion, covariance=center_covariance,
            log_weight=float(normalized[index]), push_chart_log_volume=float(volume)))
        records.append(dict(native_component_id=component.component_id,
            native_seed_index=None if not detected.seed_indices else int(detected.seed_indices[index]),
            native_log_mass=float(logs[index]), weight=float(weights[index]),
            log_weight=float(normalized[index]),
            native_translation_mean=component.translation.mean_at_reference.tolist(),
            native_translation_residual_covariance=component.translation.residual_covariance.tolist(),
            native_translation_rotation_coupling=component.translation.rotation_coupling.tolist(),
            native_rotation_wxyz=rotation.tolist(),
            native_bingham=component.orientation.parameter_matrix().tolist(),
            native_joint_mean=native.mean.tolist(), native_joint_covariance=native.covariance.tolist(),
            native_chart_log_volume=native_volume,
            known_marker_to_center_translation=list(marker.marker_to_center_translation),
            known_marker_to_center_rotation_wxyz=list(marker.rotation_wxyz),
            core_center_pose_query=_query_record(query),
            center_mean_translation=query.mean[:3].tolist(),
            center_mean_rotation_wxyz=quaternion.tolist(),
            center_full_covariance=center_covariance.tolist(),
            center_reexpression_jacobian=chart.tolist(),
            push_chart_log_volume=float(volume)))
    return tuple(output), records


def _in_chart(mode, reference):
    from .sensors import _rotation_log_right_jacobian_inverse
    coordinate = quaternion_to_rotation_vector(rotmat_to_quat(
        quat_to_rotmat(reference).T @ quat_to_rotmat(mode['rotation'])))
    chart = np.eye(6)
    chart[3:, 3:] = _rotation_log_right_jacobian_inverse(coordinate)
    covariance = chart @ mode['covariance'] @ chart.T
    sign, volume = np.linalg.slogdet(chart)
    if sign <= 0. or not np.isfinite(volume):
        raise ValueError('A board-image common chart has invalid volume.')
    return Gaussian(np.concatenate((mode['translation'], coordinate)), covariance), float(volume), chart


def _project_central(branches, reference, integration_steps, arithmetic_backend='native'):
    moment_function = (bingham_second_moment if arithmetic_backend == 'native'
                       else _python_bingham_second_moment)
    fit_function = (match_bingham_to_second_moment if arithmetic_backend == 'native'
                   else _python_match_bingham_to_second_moment)
    logs = np.asarray([item['log_mass'] for item in branches])
    log_evidence = float(logsumexp(logs))
    log_weights = logs-log_evidence
    weights = np.exp(log_weights)
    mean = sum(weight*item['law'].mean for weight, item in zip(weights, branches))
    within = sum(weight*item['law'].covariance for weight, item in zip(weights, branches))
    between = sum(weight*np.outer(item['law'].mean-mean, item['law'].mean-mean)
                  for weight, item in zip(weights, branches))
    covariance = within+between
    covariance_support(covariance, 'central board full within/between covariance')
    moments, parameters = [], []
    for item in branches:
        law = item['law']
        quaternion = quat_mul(reference, rotation_vector_to_quaternion(law.mean[3:]))
        chart = _right_jacobian(law.mean[3:])
        omega = chart @ law.covariance[3:, 3:] @ chart.T
        parameter = bingham_from_rotation_coordinates(quaternion, omega)
        parameters.append(parameter)
        moments.append(moment_function(parameter, integration_steps=integration_steps))
    target = sum(weight*moment for weight, moment in zip(weights, moments))
    fit_method = 'single branch Bingham preserved exactly'
    refinement_evaluations = 0
    refinement_skip_reason = None
    core_fit_max_error = None
    if len(branches) == 1:
        # This branch already is one Bingham. Preserve its parameters exactly;
        # a nonlinear numerical refit would add unnecessary fitting error.
        parameter = parameters[0]
    else:
        moment_eigenvalues = np.linalg.eigvalsh(target)
        if np.any(moment_eigenvalues <= 0.):
            raise ValueError('A finite central Bingham requires a positive definite quaternion second moment.')
        # The core fitter parametrizes strictly ordered gaps. Starting it
        # with a native isotropic/tied spectrum can trap an initial zero
        # gap. Initialize from all target moments instead; this separation
        # changes only optimization initialization, never the target law.
        initial = np.r_[-.5/moment_eigenvalues[:3]*np.array([1.02, 1., .98]), 0.]
        parameter = fit_function(target,
            integration_steps=integration_steps, initial_eigenvalues=initial)
        core_fit_max_error = float(np.max(np.abs(
            moment_function(parameter, integration_steps=integration_steps)-target)))
        fit_method = 'existing core quaternion second-moment fit'
        if core_fit_max_error > 5e-8:
            # Refine the same moment equation using relative residuals. The
            # core fit's absolute-loss stopping rule can stop early for very
            # concentrated laws; independent log concentrations also avoid
            # its tied-gap initialization issue. No core default is changed.
            eigenvalues, basis = np.linalg.eigh(target)
            spectrum = np.linalg.eigvalsh(parameter)
            concentrations = spectrum[-1]-spectrum[:3]
            if np.any(concentrations <= 0.):
                # A finite Bingham with tied maxima is a valid law. This
                # optional log-gap parameterization cannot represent zero;
                # retain the fit and let the actual modal-chart operation
                # below decide whether all Gaussian branches must continue.
                refinement_skip_reason = ('optional log-concentration refinement cannot '
                    'represent a zero modal gap; finite fitted Bingham retained unchanged')
                fit_method += '; optional log-concentration refinement skipped'
            else:
                def residual(log_concentrations):
                    diagonal = np.r_[-np.exp(log_concentrations), 0.]
                    if arithmetic_backend == 'native':
                        diagonal_moment = diagonal_second_moment(diagonal, integration_steps)
                    else:
                        diagonal_moment = np.diag(moment_function(
                            np.diag(diagonal), integration_steps=integration_steps))
                    return np.log(diagonal_moment[:3]/eigenvalues[:3])
                refined = least_squares(residual, np.log(concentrations),
                    ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100)
                diagonal = np.r_[-np.exp(refined.x), 0.]
                parameter = basis @ np.diag(diagonal) @ basis.T
                parameter = .5*(parameter+parameter.T)
                refinement_evaluations = int(refined.nfev)
                fit_method += '; deterministic relative-moment concentration refinement'
    fitted = moment_function(parameter, integration_steps=integration_steps)
    if not all(np.all(np.isfinite(item)) for item in (parameter, target, fitted)):
        raise ValueError('The central Bingham fit contains nonfinite parameters or moments.')
    fit_error = float(np.max(np.abs(fitted-target)))
    fit_tolerance_verified = bool(fit_error <= 1e-6)
    try:
        quaternion, omega = bingham_rotation_coordinates(parameter)
        domain_reason = None
    except (ValueError, np.linalg.LinAlgError) as error:
        # An actual nonunique mode or unavailable inverse cannot provide
        # this single chart. Preserve every full branch for the next tag;
        # finite covariance width alone never defers the projection.
        quaternion = omega = None
        domain_reason = str(error)
    product_covariance = None
    if omega is not None:
        product_covariance = np.zeros((6, 6))
        product_covariance[:3, :3] = covariance[:3, :3]
        product_covariance[3:, 3:] = omega
    record = dict(log_normalization=log_evidence,
        reference_rotation_wxyz=reference.tolist(), branch_count=len(branches),
        branch_weights=weights.tolist(), branch_log_weights=log_weights.tolist(),
        full_joint_mean=mean.tolist(), within_covariance=within.tolist(),
        between_covariance=between.tolist(), full_joint_covariance=covariance.tolist(),
        discarded_position_rotation_cross_covariance=(covariance[:3, 3:].tolist()
            if domain_reason is None else None),
        retained_position_rotation_cross_covariance=(None if domain_reason is None
            else covariance[:3, 3:].tolist()),
        quaternion_second_moment_target=target.tolist(),
        quaternion_second_moment_fitted=fitted.tolist(),
        quaternion_second_moment_fit_max_error=fit_error,
        quaternion_second_moment_fit_tolerance=1e-6,
        quaternion_second_moment_fit_tolerance_verified=fit_tolerance_verified,
        quaternion_second_moment_fit_residual_is_update_gate=False,
        quaternion_second_moment_fit_warning=(None if fit_tolerance_verified else
            'The finite central Bingham fit exceeds the diagnostic second-moment residual tolerance.'),
        bingham_fit_method=fit_method, core_bingham_fit_max_error=core_fit_max_error,
        bingham_fit_refinement_used=refinement_evaluations > 0,
        bingham_fit_refinement_skipped_reason=refinement_skip_reason,
        bingham_fit_refinement_function_evaluations=refinement_evaluations,
        bingham_parameter=parameter.tolist(), product_translation_mean=mean[:3].tolist(),
        product_translation_covariance=covariance[:3, :3].tolist(),
        product_rotation_wxyz=None if quaternion is None else quaternion.tolist(),
        product_covariance=None if product_covariance is None else product_covariance.tolist(),
        applied=domain_reason is None, deferred=domain_reason is not None,
        single_proxy_domain_reason=domain_reason,
        concentration_threshold_applied=False,
        maximum_local_rotation_variance_rad2=(None if omega is None else
            float(np.max(np.linalg.eigvalsh(omega)))),
        approximation=('central derived law only: translation moment Gaussian times a single Bingham fitted to quaternion second moments with its residual recorded; position/orientation cross discarded here'
            if domain_reason is None else 'single central approximation not applied; every full correlated Gaussian branch and normalized log mass retained'))
    if domain_reason is not None:
        return None, record
    return dict(translation=mean[:3], rotation=quaternion, covariance=product_covariance,
                bingham=parameter), record


def fuse_marker_board(detections, settings, *, stamp=None, integration_steps=120,
                      arithmetic_backend=None):
    """Fuse decoded tag mixtures into an explicitly approximated central law.

    Image corner likelihoods are conditionally independent given the board
    pose and configured calibration. No uncertain body/camera law or shared
    body prior is included here. The caller lifts this one central observation
    through its camera/body law once. ADF never changes the retained native
    observations, even if a float component weight underflows.
    """
    backend = arithmetic_backend
    if backend is None:
        backend = os.environ.get('PROBTF_MARKER_BOARD_BACKEND',
            settings.get('arithmetic_backend', 'native'))
    if backend == 'python_reference':
        backend = 'python'
    if backend not in ('native', 'python'):
        raise ValueError('Marker-board arithmetic backend must be native or python.')
    layout = {item.marker_id: item for item in board_layout(settings)}
    detected_by_id = dict(detections)
    if any(marker_id not in layout for marker_id in detected_by_id):
        raise ValueError('An observed ID is outside the configured marker board.')
    if not detected_by_id:
        return BoardFusionResult(None, dict(applied=False, reason='no configured board marker detected',
            raw_observations=[], monte_carlo_used=False, physical_truth_received=False))
    integration_steps = int(integration_steps)
    if integration_steps < 1:
        raise ValueError('Bingham moment integration requires a positive deterministic step count.')
    ordered = sorted(detected_by_id)
    first = detected_by_id[ordered[0]]
    capture_stamp = first.record.stamp if stamp is None else float(stamp)
    parent = first.record.parent_frame_id
    if any(item.record.stamp != capture_stamp or item.record.parent_frame_id != parent
           for item in detected_by_id.values()):
        raise ValueError('Every board marker must belong to the same camera image and capture stamp.')
    record = dict(applied=True, parent_frame_id=parent, child_frame_id='marker', stamp=capture_stamp,
        observed_marker_ids=ordered, configured_marker_ids=sorted(layout),
        fusion_order=ordered, raw_observations=[], steps=[],
        raw_native_mixtures_retained=True, central_component_count=1,
        method='marker-ID-ordered image-only assumed-density filtering; full current native mixture before each central-only product projection',
        exact_full_native_cartesian_product_claimed=False, order_independent_claimed=False,
        shared_body_prior_used=False, body_camera_uncertainty_used=False,
        body_camera_uncertainty_lift_count_in_this_helper=0,
        downstream_body_camera_lift_required_once=True,
        image_corner_likelihood_independence='per-marker Hessian corner likelihoods conditionally independent given central pose and configured calibrated camera; cross-marker image-error correlation not modeled',
        likelihood_chart_rule='each normalized native Gaussian is a scalar image likelihood; multiply by the common-chart |det J| before overlap weighting',
        central_position_orientation_product_approximation=True,
        monte_carlo_used=False, physical_truth_received=False, covariance_regularization_added=False,
        integration_steps=integration_steps, arithmetic_backend=backend,
        arithmetic_backend_source=backend_record(backend))
    proxy = None
    prior_branches = ()
    reference = None
    record['deferred_projection_marker_ids'] = []
    record['maximum_live_branch_count'] = 0
    for marker_id in ordered:
        native = detected_by_id[marker_id]
        modes, native_records = _native_to_center(native, layout[marker_id], backend)
        record['raw_observations'].append(dict(marker_id=marker_id,
            parent_frame_id=native.record.parent_frame_id, child_frame_id=native.record.child_frame_id,
            edge_id=native.record.edge_id, authority=native.record.authority, stamp=native.record.stamp,
            native_component_count=len(modes), components=native_records))
        if reference is None:
            reference = modes[0]['rotation']
        prior = prior_branches[0]['law'] if len(prior_branches) == 1 else None
        branches = []
        details = []
        for mode in modes:
            law, volume, chart = _in_chart(mode, reference)
            for previous in prior_branches or (None,):
                posterior, overlap = ((law, 0.) if previous is None else
                    condition_gaussian(previous['law'], law.mean, law.covariance))
                prior_log = 0. if previous is None else previous['log_mass']
                label = mode['label'] if previous is None else previous['label']+'*'+mode['label']
                log_mass = prior_log+mode['log_weight']+mode['push_chart_log_volume']+volume+float(overlap)
                branches.append(dict(label=label, law=posterior, log_mass=log_mass))
                details.append(dict(native_component_id=mode['label'], native_log_weight=mode['log_weight'],
                    prior_branch_label=None if previous is None else previous['label'],
                    prior_branch_log_weight=prior_log, product_branch_label=label,
                    prior_branch_mean=None if previous is None else previous['law'].mean.tolist(),
                    prior_branch_full_covariance=None if previous is None else previous['law'].covariance.tolist(),
                    push_chart_log_volume=mode['push_chart_log_volume'], common_chart_log_volume=volume,
                    common_chart_jacobian=chart.tolist(), common_chart_mean=law.mean.tolist(),
                    common_chart_full_covariance=law.covariance.tolist(),
                    prior_overlap_log_evidence=float(overlap), product_log_mass=log_mass,
                    product_mean=posterior.mean.tolist(), product_full_covariance=posterior.covariance.tolist()))
        proxy, projection = _project_central(branches, reference, integration_steps, backend)
        record['maximum_live_branch_count'] = max(record['maximum_live_branch_count'], len(branches))
        record['steps'].append(dict(marker_id=marker_id, central_prior_used=bool(prior_branches),
            prior_branch_count=len(prior_branches),
            central_prior_mean=None if prior is None else prior.mean.tolist(),
            central_prior_full_covariance=None if prior is None else prior.covariance.tolist(),
            branch_products=details, projection=projection))
        if proxy is None:
            record['deferred_projection_marker_ids'].append(marker_id)
            normalization = projection['log_normalization']
            prior_branches = tuple(dict(item, log_mass=item['log_mass']-normalization) for item in branches)
            # Its full covariance and common reference are unchanged.
        else:
            reference = proxy['rotation']
            prior_branches = (dict(label='central_after_'+str(marker_id),
                law=Gaussian(np.r_[proxy['translation'], np.zeros(3)], proxy['covariance']), log_mass=0.),)
    output_proxies = []
    final_log_weights = (0.,) if proxy is not None else tuple(item['log_mass'] for item in prior_branches)
    if proxy is not None:
        output_proxies.append(dict(proxy, lineage=tuple(item['label'] for item in branches)))
    else:
        # Reduction was unavailable. Keep every derived branch as its own
        # Gaussian x Bingham when the aggregate has no unique modal chart
        # or required inverse. No mixture mode is selected by its width.
        for branch in prior_branches:
            law = branch['law']
            quaternion = quat_mul(reference, rotation_vector_to_quaternion(law.mean[3:]))
            chart = _right_jacobian(law.mean[3:])
            parameter = bingham_from_rotation_coordinates(quaternion,
                chart @ law.covariance[3:, 3:] @ chart.T)
            quaternion, omega = bingham_rotation_coordinates(parameter)
            covariance = np.zeros((6, 6))
            covariance[:3, :3] = law.covariance[:3, :3]
            covariance[3:, 3:] = omega
            full_chart = np.eye(6)
            full_chart[3:, 3:] = chart
            before_product = full_chart @ law.covariance @ full_chart.T
            output_proxies.append(dict(translation=law.mean[:3], rotation=quaternion,
                covariance=covariance, bingham=parameter, lineage=(branch['label'],),
                pre_product_full_covariance=before_product,
                discarded_position_rotation_cross_covariance=before_product[:3, 3:]))
    final_weights = np.exp(final_log_weights)
    record.update(central_component_count=len(output_proxies),
        final_mixture_fallback_used=proxy is None,
        central_reduction_policy='single moment-matched proxy whenever its unique modal chart and required inverse are available; no concentration threshold; otherwise preserve every final branch as a mixture',
        concentration_threshold_applied=False,
        final_position_orientation_cross_discarded_per_output_component=True)
    signature = hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
    edge_id = parent+'__to__marker'
    components, descriptors = [], []
    from .sensors import _marker_mode_chart
    for index, (output, weight, log_weight) in enumerate(zip(output_proxies, final_weights, final_log_weights)):
        component = TransformComponent(edge_id+':board_adf:'+signature[:16]+':'+str(index), float(weight),
            BinghamOrientation.from_parameter_matrix(output['bingham']),
            ConditionalGaussianTranslation(output['translation'], output['covariance'][:3, :3], np.zeros((3, 9))))
        components.append(component)
        actual_rotation = component.orientation.mode_wxyz
        actual_law, _volume = _marker_mode_chart(component, quat_to_rotmat(actual_rotation))
        descriptors.append(dict(component_id=component.component_id,
            parent_frame_id=parent, child_frame_id='marker', edge_id=edge_id, stamp=capture_stamp,
            weight=float(weight), log_weight=float(log_weight), lineage=list(output['lineage']),
            translation_mean=output['translation'].tolist(),
            translation_covariance=output['covariance'][:3, :3].tolist(),
            rotation_wxyz=actual_rotation.tolist(), bingham=component.orientation.parameter_matrix().tolist(),
            full_product_covariance=actual_law.covariance.tolist(),
            pre_product_full_covariance=(None if 'pre_product_full_covariance' not in output else
                output['pre_product_full_covariance'].tolist()),
            discarded_position_rotation_cross_covariance=(None if 'discarded_position_rotation_cross_covariance' not in output else
                output['discarded_position_rotation_cross_covariance'].tolist()),
            position_orientation_product_approximation=True))
    stamped = TransformDistributionStamped(parent, 'marker', capture_stamp, edge_id,
        'probtf_marker_board_central_adf', TransformDistribution(tuple(components)))
    count = len(components)
    diagnostics = EstimationDiagnostics(count, count, 0, 0, 0, 0, ())
    detected = PoseMixtureResult(stamped, diagnostics,
        tuple(quat_to_rotmat(item.orientation.mode_wxyz) for item in components),
        tuple(item.translation.mean_at_reference for item in components),
        tuple(final_log_weights), tuple(float(value) for value in final_weights), tuple(range(count)))
    record['central_distribution'] = dict(parent_frame_id=parent, child_frame_id='marker',
        edge_id=edge_id, stamp=capture_stamp, components=descriptors, component_count=count,
        representation='single Gaussian x Bingham' if count == 1 else 'mixture of Gaussian x Bingham branches')
    if count == 1:
        record['central_distribution'].update(descriptors[0])
    record['source_digest'] = signature
    return BoardFusionResult(detected, record)

"""Independent marker-bearing factor for control selection, never inference.

Each camera-relative marker mean is evaluated before mixture averaging. This
factor does not certify detector success, marker corners, occlusion or front
surface visibility. Pose propagation and its complete shared candidate law
remain the caller's responsibility; this module only reads their queries.
"""

from copy import deepcopy
import math
from collections.abc import Mapping

import numpy as np


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _serial(value):
    if isinstance(value, Mapping):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_serial(item) for item in value]
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0. else '-Infinity'
    return deepcopy(value)


def direction_penalty(position, minimum_norm_m=1e-9):
    """Return phi(p) and its analytic position gradient for optical +z.

    ``(unit_z*unit-e_z)/norm`` equals z*p/norm**3-e_z/norm but
    avoids cubing the norm or multiplying large coordinates together.
    """
    position = np.asarray(position, dtype=float)
    minimum_norm_m = float(minimum_norm_m)
    if (not math.isfinite(minimum_norm_m) or minimum_norm_m <= 0.
            or position.shape != (3,) or not np.all(np.isfinite(position))):
        raise ValueError('Marker bearing needs three finite coordinates and a positive minimum norm.')
    radius = math.hypot(*position)
    if not math.isfinite(radius):
        raise ValueError('Marker position norm is non-finite.')
    if radius <= minimum_norm_m:
        raise ValueError('Marker position norm is degenerate at the configured numerical minimum.')
    unit = position/radius
    penalty = float(1.-unit[2])
    gradient = (unit[2]*unit-np.array([0., 0., 1.]))/radius
    if not math.isfinite(penalty) or not np.all(np.isfinite(gradient)):
        raise ValueError('Marker bearing or its derivative is non-finite.')
    return penalty, gradient


class VisibilityFactor:
    """Evaluate exp(-weight*V) in log form from complete per-mode queries.

    Config keys are ``enabled``, ``weight`` and ``minimum_norm_m`` (the alias
    ``min_norm_m`` is also accepted). With no explicit weight the factor is
    neutral. Optional ``intrinsics`` may contain fx/fy/cx/cy and width/height;
    a 3x3 matrix is also accepted. Pixel diagnostics never change the factor.
    """

    def __init__(self, config=None):
        config = {} if config is None else dict(config)
        self.enabled = bool(config.get('enabled', False))
        self.weight = float(config.get('weight', 0.))
        self.minimum_norm_m = float(config.get('minimum_norm_m', config.get('min_norm_m', 1e-9)))
        if (not math.isfinite(self.weight) or self.weight < 0.
                or not math.isfinite(self.minimum_norm_m) or self.minimum_norm_m <= 0.):
            raise ValueError('Visibility factor needs finite nonnegative weight and positive minimum norm.')
        self.intrinsics = self._intrinsics(config.get('intrinsics'))

    @staticmethod
    def _intrinsics(value):
        if value is None:
            return None
        if isinstance(value, Mapping):
            result = {name: float(value[name]) for name in ('fx', 'fy', 'cx', 'cy')}
            for name in ('width', 'height'):
                if name in value:
                    result[name] = float(value[name])
        else:
            matrix = np.asarray(value, dtype=float)
            if matrix.shape != (3, 3):
                raise ValueError('Visibility pixel diagnostics need a 3x3 camera matrix.')
            result = dict(fx=float(matrix[0, 0]), fy=float(matrix[1, 1]),
                          cx=float(matrix[0, 2]), cy=float(matrix[1, 2]))
        if (not all(math.isfinite(item) for item in result.values()) or result['fx'] <= 0.
                or result['fy'] <= 0.
                or any(result[name] <= 0. for name in ('width', 'height') if name in result)):
            raise ValueError('Visibility camera intrinsics must be finite with positive focal lengths/sizes.')
        return result

    def _projection(self, position):
        record = dict(predicted_pixel=None, predicted_pixel_inside_image=None,
                      pixel_projection_in_front=bool(position[2] > 0.),
                      pixel_projection_reason='intrinsics_not_configured')
        if self.intrinsics is None:
            return record
        if position[2] == 0.:
            record['pixel_projection_reason'] = 'zero_optical_depth'
            return record
        camera = self.intrinsics
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            pixel = np.array([camera['fx']*position[0]/position[2]+camera['cx'],
                              camera['fy']*position[1]/position[2]+camera['cy']])
        if not np.all(np.isfinite(pixel)):
            record['pixel_projection_reason'] = 'nonfinite_pixel_projection'
            return record
        record.update(predicted_pixel=pixel.tolist(),
                      pixel_projection_reason='front_projection' if position[2] > 0. else 'behind_camera_algebraic_projection')
        if 'width' in camera and 'height' in camera:
            record['predicted_pixel_inside_image'] = bool(position[2] > 0.
                and 0. <= pixel[0] < camera['width'] and 0. <= pixel[1] < camera['height'])
        return record

    @staticmethod
    def _weight(component):
        raw = _get(component, 'weight')
        if raw is not None and (not math.isfinite(float(raw)) or float(raw) < 0.):
            raise ValueError('Component weight must be finite and nonnegative.')
        log_weight = _get(component, 'log_weight')
        if log_weight is None:
            if raw is None:
                raise ValueError('A component needs a weight or log weight.')
            log_weight = math.log(float(raw)) if float(raw) > 0. else -math.inf
        log_weight = float(log_weight)
        if math.isnan(log_weight) or log_weight == math.inf:
            raise ValueError('Component log weight must be finite or negative infinity.')
        return None if raw is None else float(raw), log_weight

    def evaluate(self, components, camera_queries, *, source_ids=None):
        """Read matching component/query sequences, without pruning any mode.

        An invalid active mode makes the complete factor unavailable. Neither
        its mixture mass nor a behind-camera mode is silently discarded. A
        disabled or zero-weight factor returns an exact log-unit factor and
        does not inspect candidates, so the task-only path remains identical.
        """
        record = dict(enabled=self.enabled, weight=self.weight,
            minimum_norm_m=self.minimum_norm_m, optical_axis=[0., 0., 1.],
            definition='V=sum_k exp(log_weight_k-log_mass)*phi(mean_camera_relative_marker_k)',
            phi_definition='1-z/norm(position)', log_factor=None, penalty=None,
            weighted_penalty=None, per_mode_record=[], source_ids=_serial(source_ids),
            candidate_count=None, query_count=None,
            status='unavailable', reason=None, unit_factor_applied=False,
            query_evaluation_performed=False, original_candidate_measure_normalized=False,
            mixture_collapsed=False, source_laws_modified=False, monte_carlo_used=False,
            physical_truth_received=False, detector_probability=False,
            all_marker_corners_or_occlusion_certified=False,
            intrinsics=_serial(self.intrinsics))
        if not self.enabled or self.weight == 0.:
            record.update(status='unit_factor', reason='disabled' if not self.enabled else 'zero_weight',
                          log_factor=0., weighted_penalty=0., unit_factor_applied=True)
            return record
        components = () if components is None else tuple(components)
        camera_queries = () if camera_queries is None else tuple(camera_queries)
        record.update(candidate_count=len(components), query_count=len(camera_queries))
        if not components:
            record['reason'] = 'no_candidates'
            return record
        if len(components) != len(camera_queries):
            record['reason'] = 'component_query_count_mismatch'
            return record
        record['query_evaluation_performed'] = True
        log_weights, issues, metadata = [], [], []
        for index, (component, query) in enumerate(zip(components, camera_queries)):
            state = _get(component, 'state')
            mode = dict(index=index, label=_get(component, 'label'),
                source_component_state_id=_get(state, 'state_id'),
                query_id=_get(query, 'query_id'), source_state_id=_get(query, 'state_id'),
                source_component_id=_get(query, 'component_id'), source_frame=_get(query, 'frame'),
                source_stamp=_serial(_get(query, 'stamp')),
                source_command=_serial(_get(query, 'command')),
                source_version=_serial(_get(query, 'version')),
                input_weight=None, input_log_weight=None, normalized_weight=None,
                penalty=None, penalty_contribution=None, weighted_penalty_contribution=None,
                position_mean=None, position_covariance=None, mean=None, covariance=None,
                positive_optical_depth=None, front_or_back=None, optical_axis_angle_rad=None,
                analytic_position_gradient=None, directional_stationary=None,
                status='unavailable', reason=None)
            try:
                raw, log_weight = self._weight(component)
                mode.update(input_weight=raw, input_log_weight=_serial(log_weight))
                log_weights.append(log_weight)
            except (ValueError, TypeError, OverflowError) as error:
                log_weights.append(None)
                mode['reason'] = 'invalid_component_weight: '+str(error)
            try:
                mean = np.asarray(_get(query, 'mean'), dtype=float)
                covariance = np.asarray(_get(query, 'covariance'), dtype=float)
                if (mean.ndim != 1 or mean.size < 3 or not np.all(np.isfinite(mean))
                        or covariance.shape != (mean.size, mean.size)
                        or not np.all(np.isfinite(covariance))):
                    raise ValueError('Camera-relative marker query needs finite mean and matching full covariance.')
                mode.update(mean=mean.tolist(), covariance=covariance.tolist(),
                    position_mean=mean[:3].tolist(), position_covariance=covariance[:3, :3].tolist())
                position = mean[:3]
                penalty, derivative = direction_penalty(position, self.minimum_norm_m)
                radius = math.hypot(*position)
                cosine = float(position[2]/radius)
                mode.update(mean=mean.tolist(), covariance=covariance.tolist(),
                    position_mean=position.tolist(), position_covariance=covariance[:3, :3].tolist(),
                    position_norm_m=radius, penalty=penalty,
                    positive_optical_depth=bool(position[2] > 0.),
                    front_or_back='front' if position[2] > 0. else 'back' if position[2] < 0. else 'optical_plane',
                    optical_axis_angle_rad=math.acos(min(1., max(-1., cosine))),
                    analytic_position_gradient=derivative.tolist(),
                    directional_stationary=bool(np.all(derivative == 0.)), **self._projection(position))
                if mode['reason'] is None:
                    mode['status'] = 'ok'
            except (ValueError, TypeError, OverflowError) as error:
                mode['reason'] = 'invalid_or_degenerate_camera_position: '+str(error)
            if mode['status'] != 'ok':
                issues.append(dict(index=index, reason=mode['reason']))
            metadata.append((mode['source_stamp'], mode['source_command'], mode['source_version']))
            record['per_mode_record'].append(mode)
        finite = [item for item in log_weights if item is not None and math.isfinite(item)]
        if not finite:
            issues.append(dict(index=None, reason='zero_or_invalid_total_component_mass'))
        elif all(item is not None for item in log_weights):
            maximum = max(finite)
            scaled = [math.exp(item-maximum) for item in log_weights]
            total = math.fsum(scaled)
            record['component_log_mass'] = maximum+math.log(total)
            for mode, mass in zip(record['per_mode_record'], scaled):
                mode['normalized_weight'] = mass/total
        if metadata and any(item != metadata[0] for item in metadata[1:]):
            issues.append(dict(index=None, reason='inconsistent_query_stamp_command_or_version'))
        if issues:
            record.update(reason='invalid_complete_candidate_factor', unavailable_modes=issues)
            return record
        penalty = math.fsum(mode['normalized_weight']*mode['penalty'] for mode in record['per_mode_record'])
        weighted = self.weight*penalty
        if not math.isfinite(weighted):
            record['reason'] = 'nonfinite_weighted_penalty'
            return record
        for mode in record['per_mode_record']:
            mode['penalty_contribution'] = mode['normalized_weight']*mode['penalty']
            mode['weighted_penalty_contribution'] = self.weight*mode['penalty_contribution']
        record.update(status='ok', reason=None, penalty=penalty,
                      weighted_penalty=weighted, log_factor=-weighted)
        return record

#!/usr/bin/env python3
"""Stream a recorded motion run into an evidence-only evaluation summary.

Large raw sensor/image/estimation payloads are never loaded. Missing telemetry
produces null, rather than a reconstructed success or an assumed zero. Distance
residence is time weighted using linear interpolation between physical samples;
this is a sampled-path diagnostic, not continuous collision certification.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def vector(value):
    if not isinstance(value, (list, tuple)) or not value:
        return None
    result = [number(item) for item in value]
    return result if all(item is not None for item in result) else None


def difference(left, right):
    left, right = vector(left), vector(right)
    if left is None or right is None or len(left) != len(right):
        return None
    return max(abs(a-b) for a, b in zip(left, right))


def norm(value):
    value = vector(value)
    return None if value is None else math.sqrt(sum(item*item for item in value))


class Stats:
    def __init__(self):
        self.count = 0
        self.mean = self.m2 = 0.
        self.minimum = self.maximum = None

    def add(self, value):
        value = number(value)
        if value is None:
            return
        self.count += 1
        delta = value-self.mean
        self.mean += delta/self.count
        self.m2 += delta*(value-self.mean)
        self.minimum = value if self.minimum is None else min(value, self.minimum)
        self.maximum = value if self.maximum is None else max(value, self.maximum)

    def record(self):
        if not self.count:
            return None
        return dict(count=self.count, mean=self.mean, minimum=self.minimum,
                    maximum=self.maximum, population_std=math.sqrt(max(0., self.m2/self.count)))


class Lines:
    def __init__(self, path, warnings):
        self.path, self.warnings = Path(path), warnings
        self.rows = self.invalid = 0
        self.digest = None

    def __iter__(self):
        if not self.path.exists():
            return
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                    if not isinstance(row, dict):
                        raise ValueError('row is not an object')
                except (ValueError, UnicodeError) as error:
                    self.invalid += 1
                    if self.invalid <= 3:
                        self.warnings.append('%s:%d: %s' % (self.path.name, line_number, error))
                    continue
                self.rows += 1
                yield row
        self.digest = digest.hexdigest()

    def record(self):
        return dict(path=str(self.path.resolve()), exists=self.path.exists(), rows=self.rows,
                    invalid_rows=self.invalid, sha256=self.digest)


class Trace:
    """Bounded deterministic decimation; extrema and final point stay separate."""
    def __init__(self, limit):
        self.limit, self.stride, self.count, self.points = limit, 1, 0, []
        self.last = None

    def add(self, point):
        index = self.count
        self.count += 1
        self.last = point
        if index % self.stride == 0:
            self.points.append(point)
        if len(self.points) > self.limit:
            self.points = self.points[::2]
            self.stride *= 2

    def record(self):
        points = list(self.points)
        if self.last is not None and (not points or points[-1] != self.last):
            if len(points) >= self.limit:
                points[-1] = self.last
            else:
                points.append(self.last)
        return dict(raw_samples=self.count, retained_samples=len(points),
                    deterministic_index_stride=self.stride, final_point_added=True, samples=points)


def mapping(value):
    return value if isinstance(value, dict) else {}


def first(*values):
    return next((value for value in values if value is not None), None)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_manifest(path, warnings):
    if not path.exists():
        warnings.append('motion_manifest.json is missing; run mode/settings are not assumed.')
        return {}
    try:
        return mapping(json.loads(path.read_text()))
    except (OSError, ValueError) as error:
        warnings.append('motion_manifest.json could not be read: '+str(error))
        return {}


def configuration(manifest, options):
    gains = first(manifest.get('running_gradient_config'), manifest.get('running_gradient_options'),
                  manifest.get('running_gradient_gains'), {})
    gains = mapping(gains)
    eta = first(options.eta, number(gains.get('eta')), number(manifest.get('running_gradient_eta')))
    gamma = first(options.gamma, number(gains.get('gamma')), number(manifest.get('running_gradient_gamma')))
    threshold = first(options.near_threshold, number(manifest.get('reach_tolerance_m')),
                      number(manifest.get('reach_tolerance')))
    threshold_source = 'explicit --near-threshold' if options.near_threshold is not None else 'run manifest'
    if threshold is None:
        # This is explicitly labelled current config, never a frozen old-run setting.
        path = Path(__file__).resolve().parents[1]/'config/switch_press_scene.yaml'
        try:
            for line in path.read_text().splitlines():
                if line.startswith('reach_tolerance:'):
                    threshold = number(line.split(':', 1)[1].split('#', 1)[0].strip())
                    threshold_source = 'current repository scene config (not an archived run snapshot): '+str(path)
                    break
        except OSError:
            pass
    return dict(eta=eta, gamma=gamma,
                gain_source=('explicit CLI override' if options.eta is not None or options.gamma is not None
                             else 'run manifest' if eta is not None or gamma is not None else None),
                near_threshold_m=threshold,
                near_threshold_source=threshold_source if threshold is not None else None)


def physical_summary(lines, threshold, samples):
    trace, errors, intervals = Trace(samples), Stats(), Stats()
    previous = first_point = last_point = minimum_point = None
    residence = covered = 0.
    reached = Counter()
    modes = Counter()
    motion_steps = 0
    for row in lines:
        stamp = number(first(row.get('time'), row.get('simulated_elapsed_s'), row.get('stamp')))
        error = number(first(row.get('physical_error_m'), row.get('tip_goal_distance_m')))
        modes[str(row.get('action_mode', 'unrecorded'))] += 1
        reached[str(row.get('reached'))] += 1
        if stamp is None or error is None:
            previous = None
            continue
        point = dict(time_s=stamp, physical_distance_m=error)
        errors.add(error)
        trace.add(point)
        if first_point is None:
            first_point = point
        last_point = point
        if minimum_point is None or error < minimum_point['physical_distance_m']:
            minimum_point = point
        if previous is not None and stamp > previous['time_s']:
            dt = stamp-previous['time_s']
            intervals.add(dt)
            covered += dt
            if threshold is not None:
                before = previous['physical_distance_m']
                if before <= threshold and error <= threshold:
                    residence += dt
                elif (before <= threshold) != (error <= threshold):
                    inside_fraction = abs(threshold-(before if before <= threshold else error))/abs(error-before)
                    residence += dt*inside_fraction
            motion_steps += 1
        previous = point
    duration = None if first_point is None or last_point is None else last_point['time_s']-first_point['time_s']
    return dict(distance_m=errors.record(), initial=first_point, minimum=minimum_point, final=last_point,
                duration_s=duration, actual_simulation_sample_hz=(motion_steps/duration if duration else None),
                physical_interval_s=intervals.record(), distance_coverage_s=covered,
                near_threshold_m=threshold,
                near_residence_s=residence if threshold is not None and covered > 0. else None,
                near_residence_fraction=(residence/covered if threshold is not None and covered > 0. else None),
                near_residence_method='linear distance interpolation between consecutive valid physical samples',
                reached_samples=dict(reached), action_mode_samples=dict(modes), distance_trace=trace.record())


class VisibilityTimeline:
    """Acquisition-stamp diagnostics, never an inferred continuous detection."""
    def __init__(self, limit=1200):
        self.limit, self.previous, self.initial, self.final = limit, None, None, None
        self.visible = self.missing = self.invalid = self.nonincreasing = 0
        self.coverage = self.visible_time = 0.
        self.intervals, self.completed_losses, self.reacquisitions = Stats(), Stats(), Stats()
        self.loss = None
        self.episodes, self.episode_count = [], 0
        self.left_censored = self.right_censored = 0
        self.seen = False
        self.first_detection = None
        self.longest = None

    def _finish_loss(self, stamp, reacquired):
        episode = dict(self.loss)
        duration = stamp-episode['first_missing_acquisition_stamp']
        episode.update(right_censored=not reacquired,
            reacquired_acquisition_stamp=stamp if reacquired else None,
            reacquisition_time_s=duration if reacquired and not episode['left_censored'] else None,
            observed_loss_duration_s=duration if reacquired else None,
            censored_loss_duration_lower_bound_s=None if reacquired else duration,
            observation_end_acquisition_stamp=stamp)
        if reacquired:
            self.completed_losses.add(duration)
            if not episode['left_censored']:
                self.reacquisitions.add(duration)
        self.episode_count += 1
        self.left_censored += int(episode['left_censored'])
        self.right_censored += int(not reacquired)
        if len(self.episodes) < self.limit:
            self.episodes.append(episode)
        if self.longest is None or duration > self.longest['duration_or_lower_bound_s']:
            self.longest = dict(duration_or_lower_bound_s=duration, **episode)
        self.loss = None

    def add(self, stamp, visible):
        stamp = number(stamp)
        if stamp is None or not isinstance(visible, bool):
            self.invalid += 1
            if self.loss is not None:
                self._finish_loss(self.final, False)
            self.previous = None
            return False
        if self.final is not None and stamp <= self.final:
            self.nonincreasing += 1
            return False
        if self.initial is None:
            self.initial = stamp
        if self.previous is not None:
            dt = stamp-self.previous[0]
            self.intervals.add(dt)
            self.coverage += dt
            self.visible_time += dt if self.previous[1] else 0.
        self.visible += int(visible)
        self.missing += int(not visible)
        if not visible and self.loss is None:
            self.loss = dict(first_missing_acquisition_stamp=stamp,
                last_missing_acquisition_stamp=stamp, left_censored=not self.seen or self.previous is None,
                last_detected_acquisition_stamp=(None if self.previous is None or not self.previous[1]
                                                  else self.previous[0]))
        elif not visible:
            self.loss['last_missing_acquisition_stamp'] = stamp
        elif self.loss is not None:
            self._finish_loss(stamp, True)
        if visible:
            self.seen = True
            if self.first_detection is None:
                self.first_detection = stamp
        self.previous, self.final = (stamp, visible), stamp
        return True

    def record(self):
        if self.loss is not None:
            self._finish_loss(self.final, False)
        samples = self.visible+self.missing
        return dict(valid_samples=samples, detected_samples=self.visible, missing_samples=self.missing,
            detected_frame_fraction=self.visible/samples if samples else None,
            first_acquisition_stamp=self.initial, last_acquisition_stamp=self.final,
            sampled_span_s=None if self.initial is None else self.final-self.initial,
            preview_acquisition_hz=((samples-1)/(self.final-self.initial)
                if samples > 1 and self.final > self.initial else None),
            time_weighted_visible_fraction=self.visible_time/self.coverage if self.coverage > 0. else None,
            time_weighted_visible_s=self.visible_time if self.coverage > 0. else None,
            time_coverage_s=self.coverage if self.coverage > 0. else None,
            weighting='left hold only between recorded acquisition stamps; no extension beyond the last image',
            acquisition_intervals_s=self.intervals.record(), invalid_samples=self.invalid,
            nonincreasing_acquisition_samples=self.nonincreasing,
            first_detection_acquisition_stamp=self.first_detection,
            initial_acquisition_to_first_detection_s=(self.first_detection-self.initial
                if self.first_detection is not None and self.initial is not None else None),
            loss_episode_count=self.episode_count, completed_loss_duration_s=self.completed_losses.record(),
            left_censored_loss_episodes=self.left_censored, right_censored_loss_episodes=self.right_censored,
            reacquisition_time_s=self.reacquisitions.record(), longest_loss=self.longest,
            loss_episodes=self.episodes, loss_episode_records_truncated=self.episode_count > len(self.episodes),
            right_censoring='An unrecovered loss ends at the last acquired image as a duration lower bound; reacquisition time remains null.',
            loss_times_are_sampled_brackets=True)


def scene_geometry(run, manifest, explicit_path, warnings):
    """Load small declared geometry, checking a frozen run hash when available."""
    expected = mapping(manifest.get('physical_model_source')).get('scene_config_sha256')
    candidates = ([Path(explicit_path)] if explicit_path is not None else
        [run/'scene_config.yaml', run/'switch_press_scene.yaml', run/'config/switch_press_scene.yaml',
         Path(__file__).resolve().parents[1]/'config/switch_press_scene.yaml'])
    for path in candidates:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        archived = path.resolve().is_relative_to(run) if hasattr(Path, 'is_relative_to') else run in path.resolve().parents
        if expected is not None and expected != digest:
            continue
        if expected is None and explicit_path is None and not archived:
            continue
        try:
            import yaml
            settings = yaml.safe_load(raw)
            camera = mapping(settings.get('camera'))
            width, height, focal = (number(camera.get(key)) for key in ('width', 'height', 'focal_length'))
            if any(value is None or value <= 0. for value in (width, height, focal)):
                raise ValueError('camera dimensions/focal length are unavailable')
            board = mapping(settings.get('marker_board'))
            if board.get('enabled'):
                rows, columns = number(board.get('rows')), number(board.get('columns'))
                pitch, size = number(board.get('pitch')), number(board.get('marker_size'))
                if any(value is None or value <= 0. for value in (rows, columns, pitch, size)):
                    raise ValueError('marker-board dimensions are unavailable')
                black = [(columns-1)*pitch+size, (rows-1)*pitch+size]
                white = [(columns-1)*pitch+1.25*size, (rows-1)*pitch+1.25*size]
            else:
                size = number(mapping(settings.get('marker')).get('size'))
                if size is None or size <= 0.:
                    raise ValueError('marker size is unavailable')
                black, white = [size, size], [1.25*size, 1.25*size]
            return dict(status='available', source_path=str(path.resolve()), sha256=digest,
                expected_run_sha256=expected, matches_run_hash=None if expected is None else digest == expected,
                source_kind='explicit CLI geometry' if explicit_path is not None else
                    'archived run geometry' if archived else 'current repository file verified against run hash',
                camera=dict(width=width, height=height, fx=focal, fy=focal, cx=width/2., cy=height/2.,
                    distortion='zero, matching the demo renderer', near_plane_m=.02),
                board_sizes_m=dict(black_region=black, white_support=white),
                exposure_time_s=number(camera.get('exposure_time')),
                motion_blur_samples=number(camera.get('motion_blur_samples')))
        except (ImportError, ValueError, TypeError, AttributeError) as error:
            warnings.append('Posthoc visibility geometry unavailable: '+str(error))
    return dict(status='unavailable', reason='No declared scene configuration matched the run hash; pass --scene-config with the archived configuration.',
                expected_run_sha256=expected)


def matrix4(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    rows = [vector(row) for row in value]
    if not all(row is not None and len(row) == 4 for row in rows):
        return None
    if any(abs(a-b) > 1e-10 for a, b in zip(rows[3], [0., 0., 0., 1.])):
        return None
    if any(abs(sum(rows[k][i]*rows[k][j] for k in range(3))-(1. if i == j else 0.)) > 1e-8
           for i in range(3) for j in range(3)):
        return None
    determinant = (rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])
        -rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])
        +rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
    return rows if abs(determinant-1.) <= 1e-8 else None


def project_board(camera_pose, marker_pose, size, camera):
    """Truth-only posthoc pinhole projection; never exported to a controller."""
    camera_pose, marker_pose = matrix4(camera_pose), matrix4(marker_pose)
    if camera_pose is None or marker_pose is None:
        return None
    corners, depths = [], []
    for x, y in ((-size[0]/2., size[1]/2.), (size[0]/2., size[1]/2.),
                 (size[0]/2., -size[1]/2.), (-size[0]/2., -size[1]/2.)):
        world = [marker_pose[i][0]*x+marker_pose[i][1]*y+marker_pose[i][3] for i in range(3)]
        relative = [sum(camera_pose[j][i]*(world[j]-camera_pose[j][3]) for j in range(3)) for i in range(3)]
        depths.append(relative[2])
        corners.append(None if relative[2] <= 0. else [camera['fx']*relative[0]/relative[2]+camera['cx'],
                                                     camera['fy']*relative[1]/relative[2]+camera['cy']])
    inside = [pixel is not None and 0. <= pixel[0] < camera['width'] and 0. <= pixel[1] < camera['height']
              and depth > camera['near_plane_m'] for pixel, depth in zip(corners, depths)]
    return dict(all_corners_inside=all(inside), corners_inside=inside,
                pixels=corners, depths_m=depths, all_corners_in_front_of_renderer_near_plane=all(
                    depth > camera['near_plane_m'] for depth in depths))


def exposure_poses(row, geometry):
    history = row.get('camera_history')
    exposure, count = geometry.get('exposure_time_s'), geometry.get('motion_blur_samples')
    if exposure is None or count is None:
        return None, 'exposure configuration is unrecorded'
    if exposure <= 0. or count <= 1:
        return [row.get('camera_pose')], None
    if not isinstance(history, list):
        return None, 'physical exposure history is unrecorded'
    if len(history) < 2:
        return [row.get('camera_pose')], None
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
        times = np.array([item[0] for item in history], dtype=float)
        poses = np.array([item[1] for item in history], dtype=float)
        sampled = np.clip(np.linspace(times[-1]-exposure, times[-1], int(count)), times[0], times[-1])
        rotations = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))(sampled).as_matrix()
        result = []
        for stamp, rotation in zip(sampled, rotations):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = [np.interp(stamp, times, poses[:, axis, 3]) for axis in range(3)]
            result.append(pose.tolist())
        return result, None
    except (ImportError, ValueError, TypeError, IndexError) as error:
        return None, str(error)


def geometry_timeline_record(timeline):
    state = timeline.record()
    return dict(valid_samples=state['valid_samples'], all_corners_inside_samples=state['detected_samples'],
        some_corner_outside_samples=state['missing_samples'],
        all_corners_inside_frame_fraction=state['detected_frame_fraction'],
        time_weighted_all_corners_inside_fraction=state['time_weighted_visible_fraction'],
        time_weighted_all_corners_inside_s=state['time_weighted_visible_s'],
        time_coverage_s=state['time_coverage_s'], weighting=state['weighting'],
        first_acquisition_stamp=state['first_acquisition_stamp'],
        last_acquisition_stamp=state['last_acquisition_stamp'],
        invalid_samples=state['invalid_samples'],
        nonincreasing_acquisition_samples=state['nonincreasing_acquisition_samples'],
        geometry_is_not_an_image_detection=True)


def camera_summary(previews, acquisitions, geometry, samples):
    detector, marker_counts = VisibilityTimeline(samples), Stats()
    geometry_timelines = {name: VisibilityTimeline(samples) for name in ('black_region', 'white_support')}
    exposure_timelines = {name: VisibilityTimeline(samples) for name in geometry_timelines}
    projected_trace = Trace(samples)
    source_counts, paired, unavailable = Counter(), Counter(), Counter()
    first_image = last_image = None
    image_count = 0
    preview_iterator = iter(previews)
    pending = next(preview_iterator, None)

    def consume_preview(row):
        detector.add(row.get('stamp'), row.get('visible'))
        identifiers = row.get('visible_marker_ids')
        if isinstance(identifiers, list):
            marker_counts.add(len(identifiers))

    for row in acquisitions:
        kind = row.get('kind')
        source_counts[str(kind)] += 1
        if kind != 'image':
            continue
        stamp = number(row.get('stamp'))
        if stamp is None:
            unavailable['invalid_image_acquisition_stamp'] += 1
            continue
        image_count += 1
        first_image = stamp if first_image is None else min(first_image, stamp)
        last_image = stamp if last_image is None else max(last_image, stamp)
        detection = None
        while pending is not None:
            preview_stamp = number(pending.get('stamp'))
            if preview_stamp is not None and preview_stamp > stamp:
                break
            index = row.get('frame_index')
            if (preview_stamp == stamp and isinstance(index, int) and not isinstance(index, bool)
                    and index >= 0 and pending.get('index') == index
                    and isinstance(pending.get('visible'), bool)):
                detection = pending['visible']
            consume_preview(pending)
            pending = next(preview_iterator, None)
        if geometry.get('status') != 'available':
            continue
        point = dict(acquisition_stamp=stamp, frame_index=row.get('frame_index'), detected=detection, instantaneous={})
        poses, exposure_error = exposure_poses(row, geometry)
        if exposure_error is not None:
            unavailable['exposure_pose_interpolation: '+exposure_error] += 1
        for name, size in geometry['board_sizes_m'].items():
            projection = project_board(row.get('camera_pose'), row.get('marker_pose'), size, geometry['camera'])
            if projection is None:
                unavailable[name+'_physical_pose_unavailable'] += 1
                geometry_timelines[name].add(stamp, None)
                continue
            visible = projection['all_corners_inside']
            geometry_timelines[name].add(stamp, visible)
            point['instantaneous'][name] = projection
            if detection is not None:
                paired[name+':'+('all_corners_inside' if visible else 'some_corners_outside')+':'+
                       ('detected' if detection else 'not_detected')] += 1
            if poses is not None:
                projected = [project_board(pose, row.get('marker_pose'), size, geometry['camera']) for pose in poses]
                exposure_inside = None if any(value is None for value in projected) else all(
                    value['all_corners_inside'] for value in projected)
                exposure_timelines[name].add(stamp, exposure_inside)
                point.setdefault('exposure', {})[name] = dict(all_corners_inside_every_render_sample=exposure_inside,
                    temporal_pose_samples=len(poses))
        projected_trace.add(point)
    if pending is not None:
        consume_preview(pending)
    for pending in preview_iterator:
        consume_preview(pending)
    span = None if first_image is None else last_image-first_image
    return dict(raw_detector=detector.record(), visible_marker_count=marker_counts.record(),
        acquisition=dict(source_kind_counts=dict(source_counts), image_requests=image_count,
            first_image_acquisition_stamp=first_image, last_image_acquisition_stamp=last_image,
            image_acquisition_hz=(image_count-1)/span if image_count > 1 and span > 0. else None,
            previews_per_image_request=previews.rows/image_count if image_count else None,
            detection_rate_is_not_acquisition_rate=True),
        physical_geometry=dict(configuration=geometry, evaluation_only=True, controller_input=False,
            physical_truth_used_only_for_posthoc_projection=True,
            does_not_certify_detection_blur_occlusion_or_facing=True,
            pixel_bounds='0 <= u < width; 0 <= v < height; every corner depth > 0.02 m',
            instantaneous={name: geometry_timeline_record(timeline) for name, timeline in geometry_timelines.items()},
            exposure={name: geometry_timeline_record(timeline) for name, timeline in exposure_timelines.items()},
            detector_geometry_pairs=dict(paired), matching='exact acquisition stamp and physical frame index only',
            unavailable_reasons=dict(unavailable), projection_trace=projected_trace.record()))


def coefficients(gamma, dt):
    if gamma == 0.:
        return 1., dt, .5*dt*dt
    x = gamma*dt
    if abs(x) < 1e-3:
        return (math.exp(-x), dt*(1.-x/2.+x*x/6.-x**3/24.+x**4/120.),
                dt*dt*(.5-x/6.+x*x/24.-x**3/120.+x**4/720.))
    a = -math.expm1(-x)/gamma
    return math.exp(-x), a, (dt-a)/gamma


class KChanges:
    def __init__(self):
        self.count = self.changes = 0
        self.initial = self.final = self.minimum = self.maximum = None
        self.maximum_step = self.maximum_log_step = None
        self.density_ids = set()

    def add(self, value, density_id=None):
        value = vector(value)
        if value is None or any(item <= 0. for item in value):
            return
        if self.final is not None and len(value) != len(self.final):
            return
        self.count += 1
        if density_id is not None:
            self.density_ids.add(str(density_id))
        if self.initial is None:
            self.initial, self.minimum, self.maximum = list(value), list(value), list(value)
        else:
            delta = max(abs(a-b) for a, b in zip(value, self.final))
            log_delta = max(abs(math.log(a)-math.log(b)) for a, b in zip(value, self.final))
            self.changes += int(delta > 0.)
            self.maximum_step = delta if self.maximum_step is None else max(delta, self.maximum_step)
            self.maximum_log_step = log_delta if self.maximum_log_step is None else max(log_delta, self.maximum_log_step)
            self.minimum = [min(a, b) for a, b in zip(value, self.minimum)]
            self.maximum = [max(a, b) for a, b in zip(value, self.maximum)]
        self.final = list(value)

    def record(self):
        return None if not self.count else dict(samples=self.count, changed_samples=self.changes,
            distinct_density_ids=len(self.density_ids), initial=self.initial, final=self.final,
            per_joint_minimum=self.minimum, per_joint_maximum=self.maximum,
            maximum_absolute_step=self.maximum_step, maximum_absolute_log_step=self.maximum_log_step)


def command_summary(lines, configuration):
    join_position, join_velocity, equation_position, equation_velocity = Stats(), Stats(), Stats(), Stats()
    age, distance, lag, speed, acceleration = (Stats() for _ in range(5))
    statuses, policies, failures = Counter(), Counter(), Counter()
    previous_command = previous_velocity = None
    intervals, equation_intervals, held_intervals = 0, 0, 0
    held_duration = 0.
    no_goal_duration = image_missing_duration = None
    no_goal_samples = image_missing_samples = 0
    image_missing_hold_motion_duration = None
    image_missing_hold_motion_samples = 0
    no_goal_motion = no_goal_motion_duration = None
    no_goal_motion_samples = 0
    selected_k = KChanges()
    for row in lines:
        step = mapping(row.get('step'))
        execution = mapping(first(row.get('running_gradient_execution'), row.get('reaching_execution'),
                                  row.get('running_gradient')))
        behavior = mapping(row.get('behavior'))
        command = vector(first(step.get('end_command'), execution.get('command'), row.get('command')))
        before = vector(first(execution.get('position_before'), execution.get('command_before'),
                              step.get('start_command')))
        velocity = vector(first(execution.get('velocity'), row.get('velocity')))
        before_velocity = vector(first(execution.get('velocity_before'), execution.get('initial_velocity')))
        dt = number(first(execution.get('dt'),
                          (step['end_stamp']-step['start_stamp'] if isinstance(step.get('end_stamp'), (float, int))
                           and isinstance(step.get('start_stamp'), (float, int)) else None)))
        join_position.add(difference(before, previous_command))
        if before_velocity is not None:
            join_velocity.add(difference(before_velocity, previous_velocity))
        intervals += 1
        status = str(execution.get('status', 'unrecorded'))
        statuses[status] += 1
        policies[str(first(behavior.get('mode'), row.get('reaching_phase'), 'unrecorded'))] += 1
        if execution.get('stop_requested') or execution.get('halted'):
            failures[str(execution.get('reason', status))] += 1
        packet = mapping(execution.get('gradient_packet'))
        gradient = vector(first(execution.get('gradient'), packet.get('gradient')))
        age.add(execution.get('gradient_source_age_s'))
        distance.add(execution.get('gradient_source_command_distance_l2'))
        lag.add(execution.get('gradient_snapshot_update_lag'))
        speed.add(norm(velocity))
        acceleration.add(norm(execution.get('acceleration')))
        selected_k.add(first(step.get('selected_K'), execution.get('selected_K'), packet.get('selected_K')),
                       first(step.get('density_id'), execution.get('parameter_density_id'), packet.get('parameter_density_id')))
        eta, gamma = number(first(execution.get('eta'), configuration['eta'])), number(first(execution.get('gamma'), configuration['gamma']))
        if before_velocity is None:
            # This is recorded previous interval state, not an assumed initial zero.
            before_velocity = previous_velocity
        if (status == 'tracking' and gradient is not None and before is not None and before_velocity is not None
                and command is not None and velocity is not None and dt is not None and dt >= 0.
                and eta is not None and gamma is not None
                and len({len(value) for value in (before, before_velocity, command, velocity, gradient)}) == 1):
            try:
                decay, a_coeff, b_coeff = coefficients(gamma, dt)
                expected_command = [u+a_coeff*v-eta*b_coeff*g for u, v, g in zip(before, before_velocity, gradient)]
                expected_velocity = [decay*v-eta*a_coeff*g for v, g in zip(before_velocity, gradient)]
                equation_position.add(difference(command, expected_command))
                equation_velocity.add(difference(velocity, expected_velocity))
                equation_intervals += 1
            except (ValueError, OverflowError):
                pass
        if status == 'tracking' and dt is not None:
            held_intervals += 1
            held_duration += dt
        goal_available = first(execution.get('goal_available'), behavior.get('goal_available'))
        if goal_available is False and dt is not None:
            no_goal_samples += 1
            no_goal_duration = (no_goal_duration or 0.)+dt
            moving = difference(command, before)
            if moving is not None:
                no_goal_motion_samples += 1
                no_goal_motion_duration = (no_goal_motion_duration or 0.)+(dt if moving > 0. else 0.)
                no_goal_motion = bool(no_goal_motion) or moving > 0.
        marker_detected = first(behavior.get('marker_detected'), execution.get('marker_detected'), row.get('marker_detected'))
        if marker_detected is False and dt is not None:
            image_missing_samples += 1
            image_missing_duration = (image_missing_duration or 0.)+dt
            if status == 'tracking' and difference(command, before) is not None:
                image_missing_hold_motion_samples += 1
                image_missing_hold_motion_duration = (image_missing_hold_motion_duration or 0.)+(dt if difference(command, before) > 0. else 0.)
        previous_command, previous_velocity = command, velocity
    return dict(intervals=intervals, status_counts=dict(statuses), policy_counts=dict(policies),
        command_position_boundary_residual_inf_rad=join_position.record(),
        command_velocity_boundary_residual_inf_rad_s=join_velocity.record(),
        exact_hold_equation_position_residual_inf_rad=equation_position.record(),
        exact_hold_equation_velocity_residual_inf_rad_s=equation_velocity.record(),
        exact_hold_equation_verified_intervals=equation_intervals,
        command_speed_norm_rad_s=speed.record(), command_acceleration_norm_rad_s2=acceleration.record(),
        held_gradient_tracking_intervals=held_intervals, held_gradient_tracking_duration_s=held_duration,
        gradient_source_age_s=age.record(), gradient_source_command_distance_rad=distance.record(),
        gradient_snapshot_update_lag=lag.record(), selected_K_on_actual_intervals=selected_k.record(),
        fail_safe_tick_reasons=dict(failures),
        no_goal_telemetry_samples=no_goal_samples, no_goal_duration_s=no_goal_duration,
        no_goal_motion_detected=no_goal_motion, no_goal_motion_covered_samples=no_goal_motion_samples,
        no_goal_motion_duration_s=no_goal_motion_duration,
        raw_image_missing_telemetry_samples=image_missing_samples, raw_image_missing_duration_s=image_missing_duration,
        raw_image_missing_retained_gradient_motion_samples=image_missing_hold_motion_samples,
        raw_image_missing_retained_gradient_motion_duration_s=image_missing_hold_motion_duration,
        continuity_note='Equation residual, interval-state boundary continuity and packet-adoption invariance are separate checks; no unrecorded initial velocity is assumed.')


def adoption_summary(lines):
    counts, reasons = Counter(), Counter()
    age, displacement, lag, position_invariance, velocity_invariance = (Stats() for _ in range(5))
    selected_k = KChanges()
    for row in lines:
        counts['accepted' if row.get('accepted') is True else 'rejected'] += 1
        reasons[str(row.get('reason'))] += 1
        age.add(row.get('gradient_source_age_s'))
        displacement.add(row.get('gradient_source_command_distance_l2'))
        lag.add(row.get('gradient_snapshot_update_lag'))
        before = mapping(row.get('state_before'))
        after = mapping(row.get('state_after'))
        position_invariance.add(difference(first(row.get('position_before'), row.get('position'), before.get('position'), before.get('command')),
                                          first(row.get('position_after'), row.get('after_position'), after.get('position'), after.get('command'))))
        velocity_invariance.add(difference(first(row.get('velocity_before'), row.get('velocity'), before.get('velocity')),
                                          first(row.get('velocity_after'), row.get('after_velocity'), after.get('velocity'))))
        if row.get('accepted'):
            packet = mapping(first(row.get('gradient_packet'), row.get('packet')))
            selected_k.add(packet.get('selected_K'), packet.get('parameter_density_id'))
    return dict(counts=dict(counts), rejection_or_acceptance_reason_counts=dict(reasons),
                age_at_adoption_s=age.record(), command_displacement_at_adoption_rad=displacement.record(),
                snapshot_update_lag_at_adoption=lag.record(),
                packet_adoption_position_change_inf_rad=position_invariance.record(),
                packet_adoption_velocity_change_inf_rad_s=velocity_invariance.record(),
                selected_K_at_accepted_adoptions=selected_k.record())


def control_summary(lines, samples=1200):
    modes, statuses, reasons = Counter(), Counter(), Counter()
    computation, worker = Stats(), Stats()
    selected_k = KChanges()
    scalars = {name: Stats() for name in ('log_task_factor', 'log_view_factor', 'log_select_factor',
        'task_objective', 'view_penalty', 'view_weight', 'task_gradient_norm', 'view_gradient_norm',
        'weighted_view_gradient_norm', 'selected_gradient_norm', 'task_view_cosine',
        'task_weighted_view_cosine', 'factor_log_sum_residual', 'gradient_sum_residual_inf',
        'weighted_view_scale_residual_inf', 'view_log_factor_residual',
        'optical_axis_angle_rad', 'predicted_pixel_u', 'predicted_pixel_v',
        'visibility_computation_seconds', 'view_probe_computation_seconds',
        'task_and_FD_overhead_seconds_excluding_view_probes',
        'task_computation_seconds_including_view_probes', 'view_probe_count')}
    view_status, source_checks, mode_states = Counter(), Counter(), Counter()
    prediction_trace = Trace(samples)

    def cosine(left, right):
        left, right = vector(left), vector(right)
        if left is None or right is None or len(left) != len(right):
            return None
        denominator = norm(left)*norm(right)
        return None if denominator == 0. else sum(a*b for a, b in zip(left, right))/denominator

    def source_consistency(evaluation, visibility):
        source = mapping(visibility.get('source_ids'))
        aliases = dict(body_state_ids=('body_state_ids',), goal_state_ids=('goal_state_ids',),
            source_stamp=('source_stamp', 'stamp'), source_command=('source_command', 'command'),
            selected_K=('selected_K',), parameter_density_id=('parameter_density_id', 'density_id'),
            body_observation_source_stamp=('body_observation_source_stamp',),
            goal_observation_source_stamp=('goal_observation_source_stamp',))
        compared, mismatched = 0, []
        for name, names in aliases.items():
            found = next((key for key in names if key in source), None)
            if name not in evaluation or found is None:
                source_checks[name+':unrecorded'] += 1
                continue
            compared += 1
            same = evaluation[name] == source[found]
            source_checks[name+(':match' if same else ':mismatch')] += 1
            if not same:
                mismatched.append(name)
        source_checks['rows_checked'] += 1
        source_checks['rows_fully_checkable' if compared == len(aliases) else 'rows_partially_checkable'] += 1
        if mismatched:
            source_checks['rows_with_mismatch'] += 1
        return mismatched

    for row in lines:
        modes[str(first(row.get('control_mode'), mapping(row.get('controller')).get('control_mode'), 'unrecorded'))] += 1
        evaluation = mapping(row.get('gradient_evaluation'))
        if evaluation:
            statuses[str(evaluation.get('status'))] += 1
            reasons[str(evaluation.get('reason'))] += 1
            computation.add(evaluation.get('computation_seconds'))
            selected_k.add(evaluation.get('selected_K'), evaluation.get('parameter_density_id'))
            visibility = mapping(evaluation.get('visibility_factor'))
            task = vector(evaluation.get('task_gradient'))
            view = vector(evaluation.get('view_gradient'))
            weighted = vector(evaluation.get('weighted_view_gradient'))
            selected = vector(first(evaluation.get('selected_gradient'), evaluation.get('gradient')))
            for name in ('log_task_factor', 'log_view_factor', 'log_select_factor'):
                scalars[name].add(evaluation.get(name))
            log_task, log_view, log_select = (number(evaluation.get(name)) for name in
                ('log_task_factor', 'log_view_factor', 'log_select_factor'))
            scalars['task_objective'].add(-log_task if log_task is not None else None)
            scalars['view_penalty'].add(visibility.get('penalty'))
            scalars['view_weight'].add(visibility.get('weight'))
            scalars['visibility_computation_seconds'].add(visibility.get('computation_seconds'))
            for name in ('view_probe_computation_seconds',
                         'task_and_FD_overhead_seconds_excluding_view_probes',
                         'task_computation_seconds_including_view_probes', 'view_probe_count'):
                scalars[name].add(evaluation.get(name))
            for name, value in (('task', task), ('view', view), ('weighted_view', weighted), ('selected', selected)):
                scalars[name+'_gradient_norm'].add(norm(value))
            unweighted_cosine, weighted_cosine = cosine(task, view), cosine(task, weighted)
            scalars['task_view_cosine'].add(unweighted_cosine)
            scalars['task_weighted_view_cosine'].add(weighted_cosine)
            if weighted_cosine is not None:
                source_checks['gradient_direction_comparisons'] += 1
                source_checks['opposed_task_and_weighted_view_gradients'] += int(weighted_cosine < 0.)
            if all(value is not None for value in (log_task, log_view, log_select)):
                scalars['factor_log_sum_residual'].add(abs(log_select-log_task-log_view))
            weight, penalty = number(visibility.get('weight')), number(visibility.get('penalty'))
            if weight is not None and penalty is not None and log_view is not None:
                scalars['view_log_factor_residual'].add(abs(log_view+weight*penalty))
            if weight is not None and view is not None and weighted is not None:
                scalars['weighted_view_scale_residual_inf'].add(difference(weighted, [weight*value for value in view]))
            if (task is not None and weighted is not None and selected is not None
                    and len(task) == len(weighted) == len(selected)):
                scalars['gradient_sum_residual_inf'].add(difference(selected,
                    [a+b for a, b in zip(task, weighted)]))
            if visibility:
                view_status[str(visibility.get('status', 'unrecorded'))] += 1
                mismatched = source_consistency(evaluation, visibility)
                per_mode = visibility.get('per_mode_record')
                if isinstance(per_mode, list):
                    for mode in per_mode:
                        mode = mapping(mode)
                        scalars['optical_axis_angle_rad'].add(mode.get('optical_axis_angle_rad'))
                        pixel = vector(mode.get('predicted_pixel'))
                        if pixel is not None and len(pixel) == 2:
                            scalars['predicted_pixel_u'].add(pixel[0])
                            scalars['predicted_pixel_v'].add(pixel[1])
                        mode_states[str(mode.get('front_or_back', 'unrecorded'))] += 1
                        prediction_trace.add(dict(request_id=row.get('request_id'),
                            source_stamp=evaluation.get('source_stamp'),
                            source_command=evaluation.get('source_command'),
                            body_observation_source_stamp=evaluation.get('body_observation_source_stamp'),
                            goal_observation_source_stamp=evaluation.get('goal_observation_source_stamp'),
                            mode_source_stamp=mode.get('source_stamp'), mode_index=mode.get('index'),
                            source_component_state_id=mode.get('source_component_state_id'),
                            position_mean=mode.get('position_mean'), position_covariance=mode.get('position_covariance'),
                            optical_axis_angle_rad=mode.get('optical_axis_angle_rad'), predicted_pixel=pixel,
                            predicted_pixel_inside_image=mode.get('predicted_pixel_inside_image'),
                            front_or_back=mode.get('front_or_back'), status=mode.get('status'),
                            factor_source_mismatches=mismatched))
        worker.add(row.get('work_seconds'))
    return dict(mode_counts=dict(modes), gradient_status_counts=dict(statuses), gradient_reason_counts=dict(reasons),
                gradient_computation_seconds=computation.record(), total_worker_seconds=worker.record(),
                selected_K_at_gradient_evaluation=selected_k.record(),
                selection_factors=dict(statistics={name: value.record() for name, value in scalars.items()},
                    visibility_status_counts=dict(view_status), factor_source_identity_checks=dict(source_checks),
                    opposed_gradient_fraction=(source_checks['opposed_task_and_weighted_view_gradients']/
                        source_checks['gradient_direction_comparisons'] if source_checks['gradient_direction_comparisons'] else None),
                    predicted_front_back_counts=dict(mode_states), prediction_trace=prediction_trace.record(),
                    gradient_convention='task=grad(J); view=grad(V); weighted_view=lambda*grad(V); selected=task+weighted_view',
                    timing_convention='Top-level view_probe_computation_seconds sums all reused finite-difference view probes; task_computation_seconds_including_view_probes includes that work. They are not additive independent costs. visibility_computation_seconds is legacy nested telemetry only.',
                    component_aggregation='Each logged component is a separate diagnostic sample; no mean-point collapse or fake image synchronization.',
                    source_identity_checks_do_not_recompute_candidate_laws=True,
                    source_time_is_not_adoption_time=True))


def database_summary(path, warnings):
    if not path.exists():
        return None
    output = dict(path=str(path.resolve()), sensor_status_counts={}, sensors={}, errors=[],
                  result_metadata_only=True, raw_payloads_loaded=False)
    try:
        connection = sqlite3.connect('file:'+str(path.resolve())+'?mode=ro', uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')
        try:
            for row in connection.execute('SELECT kind,status,COUNT(*) n FROM events GROUP BY kind,status'):
                output['sensor_status_counts'].setdefault(row['kind'], {})[row['status']] = row['n']
            for row in connection.execute('SELECT kind,COUNT(*) n,MIN(stamp) first_stamp,MAX(stamp) last_stamp, '
                                           'SUM(attempts) attempts,MAX(attempts) maximum_attempts FROM events GROUP BY kind'):
                span = row['last_stamp']-row['first_stamp']
                output['sensors'][row['kind']] = dict(count=row['n'], first_acquisition_stamp=row['first_stamp'],
                    last_acquisition_stamp=row['last_stamp'], acquisition_span_s=span,
                    actual_simulation_acquisition_hz=(row['n']-1)/span if span > 0. else None,
                    attempts=row['attempts'], maximum_attempts=row['maximum_attempts'])
            processing, waiting = Stats(), Stats()
            fused = Counter()
            for row in connection.execute('SELECT kind,result_metadata_json,claimed_wall_stamp,completed_wall_stamp,ready_wall_stamp '
                                           'FROM events WHERE result_metadata_json IS NOT NULL'):
                metadata = mapping(json.loads(row['result_metadata_json']))
                fused[row['kind']+'_fused' if metadata.get('observation_fused') else row['kind']+'_not_fused'] += 1
                processing.add(metadata.get('inference_seconds'))
                if row['claimed_wall_stamp'] is not None and row['ready_wall_stamp'] is not None:
                    waiting.add(row['claimed_wall_stamp']-row['ready_wall_stamp'])
            output.update(total_events=sum(sum(value.values()) for value in output['sensor_status_counts'].values()),
                          observation_fusion_counts=dict(fused), inference_seconds=processing.record(),
                          ready_to_claim_wall_delay_s=waiting.record())
            for row in connection.execute('SELECT event_id,kind,status,error FROM events WHERE error IS NOT NULL LIMIT 20'):
                output['errors'].append(dict(row))
            output['error_events'] = connection.execute('SELECT COUNT(*) FROM events WHERE error IS NOT NULL').fetchone()[0]
            output['backlog_events'] = connection.execute("SELECT COUNT(*) FROM events WHERE status IN ('pending','processing','awaiting_payload')").fetchone()[0]
            output['failed_or_replay_events'] = connection.execute("SELECT COUNT(*) FROM events WHERE status IN ('failed','late_requires_replay')").fetchone()[0]
        finally:
            connection.close()
    except (sqlite3.Error, ValueError) as error:
        warnings.append('Observation DB summary failed: '+str(error))
        output['query_error'] = str(error)
    return output


def failure_summary(run, warnings):
    summaries = {}
    for name in ('execution_failure.jsonl', 'shutdown.jsonl', 'gradient_failures.jsonl',
                 'running_gradient_fail_safe.jsonl'):
        lines = Lines(run/name, warnings)
        examples = []
        reasons = Counter()
        for row in lines:
            reasons[str(first(row.get('reason'), row.get('status'), row.get('failed'), 'unrecorded'))] += 1
            if len(examples) < 3:
                examples.append(row)
        if lines.path.exists():
            summaries[name] = dict(file=lines.record(), reason_counts=dict(reasons), first_examples=examples)
    return summaries


def evaluate(options):
    run, warnings = Path(options.run_dir).resolve(), []
    manifest = load_manifest(run/'motion_manifest.json', warnings)
    config = configuration(manifest, options)
    logs = {name: Lines(run/name, warnings) for name in
            ('physical_trajectory.jsonl', 'commands.jsonl', 'gradient_adoptions.jsonl', 'control.jsonl',
             'camera_previews.jsonl', 'acquisition_requests.jsonl')}
    physical = physical_summary(logs['physical_trajectory.jsonl'], config['near_threshold_m'], options.distance_samples)
    commands = command_summary(logs['commands.jsonl'], config)
    adoptions = adoption_summary(logs['gradient_adoptions.jsonl'])
    control = control_summary(logs['control.jsonl'], options.distance_samples)
    geometry = scene_geometry(run, manifest, getattr(options, 'scene_config', None), warnings)
    camera = camera_summary(logs['camera_previews.jsonl'], logs['acquisition_requests.jsonl'],
                            geometry, options.distance_samples)
    mode = first(manifest.get('control_mode'), next(iter(control['mode_counts']), None))
    if mode in (None, 'unrecorded'):
        mode = 'unknown; not inferred from current repository default'
    return dict(schema='probtf_running_gradient_evaluation_v1', run_directory=str(run), control_mode=mode,
        evaluation_only=True, modifies_raw_inputs=False, old_run_performance_not_relabelled=True,
        configuration=config, physical=physical, commands=commands, gradient_adoptions=adoptions,
        control_worker=control, camera_visibility=camera,
        observations=database_summary(run/'observations.sqlite3', warnings),
        manifest={key: manifest.get(key) for key in (
            'execution_mode', 'execution_architecture', 'finished', 'estimation_drained', 'duration',
            'target_physics_hz', 'target_joint_hz', 'target_imu_hz', 'target_image_hz',
            'actual_motion_hz', 'actual_camera_hz', 'actual_joint_hz', 'actual_imu_hz',
            'actual_control_hz', 'actual_estimate_hz', 'reached', 'final_physical_error_m',
            'failure', 'collision_checking_enabled', 'reach_condition',
            'implementation_sha256', 'implementation_unchanged_during_execution')},
        files={name: lines.record() for name, lines in logs.items()},
        estimation_log=dict(path=str(run/'estimation.jsonl'), exists=(run/'estimation.jsonl').exists(),
            scanned=False, reason='Use SQLite event result metadata; avoid loading giant candidate records.'),
        failure_logs=failure_summary(run, warnings), warnings=warnings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--near-threshold', type=float, default=None, help='Physical distance tolerance, metres.')
    parser.add_argument('--distance-samples', type=int, default=1200, help='Bounded distance, projection and prediction trace sizes.')
    parser.add_argument('--eta', type=float, default=None, help='Explicit logged-run gain when absent from manifest.')
    parser.add_argument('--gamma', type=float, default=None, help='Explicit logged-run damping when absent from manifest.')
    parser.add_argument('--scene-config', type=Path, help='Archived geometry for posthoc truth-only board projection.')
    options = parser.parse_args(argv)
    if not options.run_dir.is_dir():
        parser.error('run_dir must be an existing directory')
    for name in ('near_threshold', 'eta', 'gamma'):
        value = getattr(options, name)
        if value is not None and (not math.isfinite(value) or value < 0. or name != 'gamma' and value == 0.):
            parser.error(name+' must be finite and positive (gamma may be zero)')
    if options.distance_samples < 2:
        parser.error('distance-samples must be at least 2')
    result = evaluate(options)
    encoded = json.dumps(finite_json(result), indent=2, sort_keys=True, allow_nan=False)+'\n'
    if options.output is None:
        sys.stdout.write(encoded)
    else:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
        print(str(options.output.resolve()))


if __name__ == '__main__':
    main()

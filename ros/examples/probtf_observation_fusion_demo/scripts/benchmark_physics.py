#!/usr/bin/env python3
"""Measure the native plant and physical frames, without inference or replay.

This is a prescribed-command physics benchmark, not a reaching-success test.
It also releases a small physical joint displacement under a held command to
measure actual nonlinear ringdown. Raw outputs stay in the requested directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from probtf_observation_fusion_demo.dynamic_physics import NativeDynamics
from probtf_observation_fusion_demo.moving_scene import MovingSwitchScene


def timing(values):
    samples = np.asarray(values)*1e3
    return dict(count=len(samples), mean_ms=float(samples.mean()),
        median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
        max_ms=float(samples.max()))


def ringdown(scene, initial, *, damping_ratio, require_oscillation=False):
    settings = dict(scene.config['plant']['dynamics'], damping_ratio=damping_ratio)
    native = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'], **settings)
    native.reset(initial.joints, command=initial.command)
    rest_energy = native.state()['diagnostics']['energy_j']
    displaced = np.array(initial.joints, copy=True); displaced[3] += .03
    native.reset(displaced, command=initial.command)
    before = native.state()
    trace = native.advance(5., initial.command, initial.command)
    after = native.state()
    displacement = trace['positions'][:, 3]-initial.joints[3]
    finite = bool(all(np.all(np.isfinite(trace[key])) for key in ('positions', 'velocities')))
    signs = np.sign(displacement[np.abs(displacement) > 1e-8])
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    if not finite or require_oscillation and crossings < 1:
        raise AssertionError('The realized nonlinear physical state failed the declared ringdown check.')
    if after['diagnostics']['energy_j']-rest_energy >= .1*(before['diagnostics']['energy_j']-rest_energy):
        raise AssertionError('The held-command ringdown did not dissipate its initial excess energy.')
    return dict(duration_s=5., held_command=True, finite=finite, damping_ratio=damping_ratio,
        displaced_joint_index=3, initial_displacement_rad=.03,
        displacement_min_rad=float(displacement.min()), displacement_max_rad=float(displacement.max()),
        equilibrium_crossings=crossings,
        initial_excess_energy_j=before['diagnostics']['energy_j']-rest_energy,
        final_excess_energy_j=after['diagnostics']['energy_j']-rest_energy,
        final_position_error_norm_rad=float(np.linalg.norm(after['position']-initial.joints)),
        peak_actual_speed_rad_s=float(np.max(np.abs(trace['velocities']))),
        diagnostics=after['diagnostics'])


def benchmark(output, *, duration=2., physics_hz=120.):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scene = MovingSwitchScene(output/'physical')
    if duration > scene.duration:
        raise ValueError('The benchmark duration cannot exceed the configured physical scene duration.')
    initial = scene.step(0., render_camera=False)
    settings = scene.config['plant']['dynamics']
    native = NativeDynamics(scene.urdf, scene.config['plant']['stiffness'], **settings)
    native.reset(initial.joints, command=initial.command)
    count = max(1, int(round(duration*physics_hz)))
    period = duration/count
    native_times, frame_times = [], []
    previous = np.asarray(initial.command)
    for index in range(count):
        elapsed = (index+1)*period
        command = np.array(initial.command, copy=True)
        command[3:5] += [.015*np.sin(2*np.pi*.7*elapsed), .01*np.sin(2*np.pi*.4*elapsed)]
        started = time.perf_counter()
        native.advance(period, previous, command)
        native_times.append(time.perf_counter()-started)
        started = time.perf_counter()
        frame = scene.step(elapsed, command=command, render_camera=False)
        frame_times.append(time.perf_counter()-started)
        np.testing.assert_allclose(frame.joints, native.state()['position'], rtol=0., atol=2e-9)
        previous = command
    # A physical initial displacement is released under a constant spring
    # command. No sinusoidal term or equilibrium replacement is in the plant.
    heavy = ringdown(scene, initial, damping_ratio=settings['damping_ratio'])
    low = ringdown(scene, initial, damping_ratio=.1, require_oscillation=True)
    package = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), package/'config/switch_press_scene.yaml']
    sources.extend((package/'native').glob('*'))
    sources.extend(package/'src/probtf_observation_fusion_demo'/name for name in
                   ('moving_scene.py', 'dynamic_physics.py'))
    summary = dict(test='prescribed-command physics throughput and nonlinear held-command ringdown',
        reaching_or_estimation_validation=False, wall_clock_paced=False,
        nominal_physics_hz=physics_hz, simulated_duration_s=duration,
        native_advance=timing(native_times),
        physical_frame_without_image_rendering=timing(frame_times),
        collision_checking_enabled=scene.collision_checking,
        image_rendering_or_ros_publication_included=False,
        default_heavy_damping_ringdown=heavy, low_damping_oscillation_check=low,
        physical_source=scene.physics_source,
        source_sha256={str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(sources)})
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--physics-hz', type=float, default=120.)
    args = parser.parse_args()
    if not all(np.isfinite(x) and x > 0. for x in (args.duration, args.physics_hz)):
        parser.error('duration and physics-hz must be finite and positive')
    print(json.dumps(benchmark(args.output, duration=args.duration, physics_hz=args.physics_hz), indent=2))


if __name__ == '__main__':
    main()

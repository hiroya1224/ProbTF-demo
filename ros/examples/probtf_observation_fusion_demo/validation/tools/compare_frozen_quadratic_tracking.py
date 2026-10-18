#!/usr/bin/python3
"""Compare one frozen quadratic model with holding only its source gradient.

This is a deterministic synthetic execution test, not a robot reaching trial
or a measurement of probabilistic inference throughput. Neither run receives
new gradient packets after its initial capture.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from probtf_observation_fusion_demo.local_quadratic_tracking import FrozenLocalQuadraticModel
from probtf_observation_fusion_demo.running_gradient import (
    GradientPacket, RunningGradientTracker, step_running_gradient,
)


def run_comparison(*, eta=1e-6, gamma=8., duration=30., dt=.01, trace_output=None):
    if not math.isfinite(duration) or not math.isfinite(dt) or duration <= 0. or dt <= 0.:
        raise ValueError('Positive finite duration and dt are required.')
    source = np.zeros(6)
    target = np.array([.04, -.03, .02, -.015, .01, -.005])
    angle = .4
    basis = np.eye(6)
    basis[:2, :2] = [[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]]
    curvature = (basis*np.array([1., 1.5, 2., 3., 5., 8.])*1e6) @ basis.T
    gradient = curvature @ (source-target)
    objective = lambda command: .5*float((command-target) @ curvature @ (command-target))
    trace = []
    results = {}
    for name, hessian in (('held_source_gradient', None), ('frozen_local_quadratic', curvature)):
        tracker = RunningGradientTracker(source, eta=eta, gamma=gamma,
            joint_lower=np.full(6, -1000.), joint_upper=np.full(6, 1000.))
        packet = GradientPacket(gradient=gradient, source_command=source,
            source_stamp=0., body_state_ids=('fixed synthetic body',),
            goal_state_ids=('fixed synthetic goal',), parameter_density_id=None,
            selected_K=None, objective_at_source=objective(source), computation_seconds=0.,
            finite_difference_diagnostics={}, curvature=hessian,
            local_model_method=None if hessian is None else 'exact synthetic fixed quadratic',
            packet_id=name)
        if not tracker.accept(packet, 0.)['accepted']:
            raise ValueError('Synthetic packet was not accepted.')
        elapsed, count, maximum_error = 0., 0, 0.
        cpu_started, wall_started = time.process_time(), time.perf_counter()
        trace.append(dict(mode=name, time_s=0., command=source.tolist(),
                          velocity=tracker.velocity.tolist(), objective=objective(source)))
        while elapsed < duration:
            interval = min(dt, duration-elapsed)
            result = tracker.advance(interval)
            if result.record['halted']:
                raise ValueError('Synthetic execution halted: '+str(result.record))
            elapsed = min(duration, elapsed+interval)
            count += 1
            error = float(np.linalg.norm(tracker.position-target))
            maximum_error = max(maximum_error, error)
            trace.append(dict(mode=name, time_s=elapsed, command=tracker.position.tolist(),
                velocity=tracker.velocity.tolist(), objective=objective(tracker.position)))
        cpu, wall = time.process_time()-cpu_started, time.perf_counter()-wall_started
        expected = (step_running_gradient(source, np.zeros(6), gradient, duration, eta, gamma)
                    if hessian is None else FrozenLocalQuadraticModel(source, gradient,
                        curvature).advance(source, np.zeros(6), duration, eta, gamma))
        results[name] = dict(final_command=tracker.position.tolist(),
            final_velocity=tracker.velocity.tolist(), final_objective=objective(tracker.position),
            final_distance_to_fixed_equilibrium=float(np.linalg.norm(tracker.position-target)),
            maximum_distance_to_fixed_equilibrium=maximum_error,
            whole_interval_command_discrepancy=float(np.max(np.abs(tracker.position-expected[0]))),
            whole_interval_velocity_discrepancy=float(np.max(np.abs(tracker.velocity-expected[1]))),
            source_packet_count=1, tick_count=count, wall_s=wall, cpu_s=cpu)
    if trace_output is not None:
        with Path(trace_output).open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['mode', 'time_s', 'objective']+
                ['u'+str(i) for i in range(6)]+['v'+str(i) for i in range(6)])
            for point in trace:
                writer.writerow([point['mode'], point['time_s'], point['objective']]+
                                point['command']+point['velocity'])
    summary = dict(schema='probtf_frozen_quadratic_tracking_comparison_v1',
        purpose='synthetic frozen model only; not robot reaching or inference speed',
        equation='u_dot=v; v_dot=-gamma*v-eta*(g_source+H*(u-u_source))',
        eta=eta, gamma=gamma, duration_s=duration, dt_s=dt,
        source_command=source.tolist(), source_gradient=gradient.tolist(),
        curvature=curvature.tolist(), fixed_equilibrium=target.tolist(),
        initial_objective=objective(source), results=results,
        trace_output=None if trace_output is None else str(trace_output),
        monte_carlo_used=False, core_propagations_per_tick=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return summary, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eta', type=float, default=1e-6)
    parser.add_argument('--gamma', type=float, default=8.)
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--dt', type=float, default=.01)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--trace-output', type=Path)
    parser.add_argument('--plot-output', type=Path)
    args = parser.parse_args()
    summary, trace = run_comparison(eta=args.eta, gamma=args.gamma,
        duration=args.duration, dt=args.dt, trace_output=args.trace_output)
    if args.plot_output is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 1, figsize=(8., 6.), sharex=True)
        for mode in summary['results']:
            points = [row for row in trace if row['mode'] == mode]
            stamp = [row['time_s'] for row in points]
            axes[0].semilogy(stamp, [max(row['objective'], np.finfo(float).tiny)
                                   for row in points], label=mode)
            axes[1].plot(stamp, [row['command'][0] for row in points], label=mode)
        axes[0].set_ylabel('Fixed true quadratic J')
        axes[1].set_ylabel('Command u0 [rad]')
        axes[1].set_xlabel('Time [s]')
        axes[1].axhline(summary['fixed_equilibrium'][0], color='black', linestyle=':')
        axes[0].legend()
        figure.tight_layout()
        figure.savefig(args.plot_output)
        plt.close(figure)
        summary['plot_output'] = str(args.plot_output)
    encoded = json.dumps(summary, indent=2, allow_nan=False)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded+'\n')


if __name__ == '__main__':
    main()

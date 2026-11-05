#!/usr/bin/env python3
"""PID maps for IMU-only navigation, using a structural full-state certificate.

No numerical pole is invented: translation symmetry proves rho >= 1 whenever
a hover Jacobian exists. Consequently the old strict all-state rho < 1 event
is empty for every PID and every physical/three-lag candidate.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
import sys
import time

os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import numpy as np

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE.parent/'first-order-lag/automation'
if str(AUTOMATION) not in sys.path:
    sys.path.insert(0,str(AUTOMATION))
from pid_voxel_domains import axes_for_box
from imu_pid_inputs import GROUPS, SOURCE, load_imu_pid_inputs
from imu_pid_observability import imu_only_stability_certificate
from imu_pid_plots import render_grid


def serializable(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,default=serializable,indent=2,
                                    ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def stamp(): return datetime.now(timezone.utc).isoformat()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gain_axes(size,gains,group,mode):
    reference = np.asarray(gains)[GROUPS.index(group)]
    domain = json.loads((AUTOMATION/'pid_voxel_domains.json').read_text())
    bounds = np.stack((float(domain['lower_factor'])*reference,
                       float(domain['upper_factor'])*reference))
    axes = axes_for_box(size,bounds,[reference],domain['minimum_relative_clearance'])
    return (axes[0],np.array([1.]),axes[2]) if mode=='plane' else axes


def run(args):
    destination = args.output/args.case/args.mode
    destination.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    bundle = load_imu_pid_inputs(args.case,args.source,args.nodes)
    certificate = imu_only_stability_certificate(bundle)
    files = [Path(__file__),HERE/'imu_pid_inputs.py',HERE/'imu_pid_plots.py',
             HERE/'imu_pid_observability.py',AUTOMATION/'pid_voxel_domains.json']
    provenance = dict(schema='grape/imu-only-full-state-pid-maps/v1',case=args.case,
        mode=args.mode,started_utc=stamp(),arguments=vars(args),
        input=bundle['metadata'],certificate=certificate,
        mean_parameters=bundle['estimate']['point_estimate'],
        mean_three_lags=bundle['nominal_taus'],controller_dt=bundle['controller_dt'],
        fixed_controller_gains=bundle['reference_gains'],
        source_sha256={str(p):sha(p) for p in files},
        measurement_policy=dict(mocap=False,external_position=False,external_velocity=False,
            position_and_velocity='estimated from IMU/history only',
            gyro_lag='tau_g*d(gyro_measured)/dt + gyro_measured = omega_true',
            accel_lag='tau_a*d(accel_measured)/dt + accel_measured = specific_force_true',
            thrust_lag='per-candidate estimated tau_thrust'),
        assessment_method='analytic_translation_observability',
        numerical_eigenvalues_computed=False,hover_equilibria_numerically_evaluated=False,
        criterion='strict asymptotic stability of ALL physical, sensor and estimator states',
        strict_stability_upper_bound=0.,conditional_spectral_radius_lower_bound=1.,
        samples_discarded=0,success_inputs_used=False,monte_carlo=False)
    identity = dict(provenance); identity.pop('started_utc')
    fingerprint = hashlib.sha256(json.dumps(identity,default=serializable,
        sort_keys=True).encode()).hexdigest()
    provenance['fingerprint'] = fingerprint
    if (destination/'manifest.json').exists():
        old = json.loads((destination/'manifest.json').read_text())
        if old['fingerprint'] != fingerprint:
            raise ValueError('A different analysis exists here; choose a separate output')
    write_json(destination/'manifest.json',provenance)
    np.savez_compressed(destination/'model_inputs.npz',
        joint_mean=bundle['mean'],joint_covariance=bundle['covariance'],
        joint_covariance_factor=bundle['factor'],joint_chart_basis=bundle['full_chart_basis'],
        deterministic_coordinates=bundle['coordinates'],three_lags=bundle['taus'],
        node_weights=bundle['weights'],source_model_decodable=bundle['valid_node_mask'],
        fixed_controller_gains=bundle['reference_gains'])
    status = dict(status='running',case=args.case,mode=args.mode,pid=os.getpid(),
                  started_utc=stamp(),completed_grids=[])
    stop = [False]
    def request_stop(_signum,_frame): stop[0]=True
    signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
    print(f'{args.case} / {args.mode}: full IMU-only navigation, all 3 estimated lags',flush=True)
    print('ANALYTIC CERTIFICATE: >=3 exact unit poles; full-state rho<1 is impossible.',flush=True)
    print('No exact numerical spectral radius or hover-valid count is being fabricated.',flush=True)
    try:
        for size in args.grid_sizes:
            for group in args.groups:
                if stop[0]:
                    status['status']='stopped'; return
                axes = gain_axes(size,bundle['reference_gains'],group,args.mode)
                shape = tuple(len(a) for a in axes)
                rate = np.zeros(shape)
                decided = np.ones(shape,bool)
                grid = destination/group/f'grid_{size}'
                grid.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(grid/'criterion_grid.npz',
                    p_axis=axes[0],i_axis=axes[1],d_axis=axes[2],
                    strict_stability_fraction=rate,criterion_decided=decided,
                    numerical_pole_evaluated=np.zeros(shape,bool),
                    strict_stability_upper_bound=np.array(0.),
                    conditional_spectral_radius_lower_bound=np.array(1.),
                    known_unit_pole_multiplicity_lower_bound=np.array(3),
                    node_count=np.array(args.nodes))
                meta = dict(schema=provenance['schema'],group=group,mode=args.mode,
                    requested_grid_size=size,actual_shape=shape,
                    full_fixed_controller_gains=bundle['reference_gains'],
                    assessment_method='analytic_translation_observability',
                    criterion_label='Strict full-state stability fraction (analytic certificate)',
                    numerical_eigenvalues_computed=False,hover_validity_evaluated=False,
                    strict_stability_upper_bound=0.,conditional_spectral_radius_lower_bound=1.,
                    known_unit_pole_multiplicity_lower_bound=3,
                    no_hover_case='no asymptotically stable hover exists either',
                    no_flight_failure_probability_claim=True,
                    input_node_count=args.nodes,all_input_nodes_retained=True,
                    source_model_decodable_count=int(bundle['valid_node_mask'].sum()),
                    exact_fixed_i=1. if args.mode=='plane' else None,
                    interpretation='This decides strict full-state stability; it does not compute the other poles or prove all motion control impossible.')
                write_json(grid/'assessment.json',meta)
                print(f'{group} grid {size}, actual {shape}: Q=0 certified; rendering',flush=True)
                render_grid(grid,args.case,group,*axes,rate,decided,
                            bundle['reference_gains'],args.mode,meta)
                status['completed_grids'].append(str(grid.relative_to(destination)))
                status.update(updated_utc=stamp(),elapsed_seconds=time.monotonic()-started,
                              latest_group=group,latest_grid=size,latest_directory=str(grid))
                write_json(destination/'status.json',status)
                print(f'{group} grid {size}: saved PNG/PDF and exact-I/3D criterion data',flush=True)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',error=type(error).__name__+': '+str(error)); raise
    finally:
        status.update(updated_utc=stamp(),elapsed_seconds=time.monotonic()-started)
        write_json(destination/'status.json',status)
    print('COMPLETED',destination,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('failure1','failure2'),required=True)
    parser.add_argument('--mode',choices=('3d','plane'),required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=Path('/tmp/grape-imu-only-pid-20261106'))
    parser.add_argument('--nodes',type=int,default=128)
    parser.add_argument('--groups',nargs='+',choices=GROUPS,default=['roll_pitch','yaw','xy','z'])
    parser.add_argument('--grid-sizes',nargs='+',type=int,default=[3,5,9,17,33,65])
    args = parser.parse_args()
    if args.nodes <= 0 or any(n<3 for n in args.grid_sizes):
        parser.error('positive node count and grid sizes >=3 required')
    run(args)


if __name__=='__main__': main()

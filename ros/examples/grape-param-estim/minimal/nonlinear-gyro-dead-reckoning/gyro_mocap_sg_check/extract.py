#!/usr/bin/env python3
"""Original-clock mocap/gyro input for a separate SG derivative diagnostic.

This diagnostic does not update a physical model or feed the sparse predictor.
No time alignment, bias correction, scale correction, smoothing or resampling
is applied by this extractor.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import numpy as np

HERE = Path(__file__).resolve().parent
MINIMAL = HERE.parents[1]
BAG = Path('/home/leus/catkin_ws/bags/grape-drone/20260612_grape_hovering/'
           '20260612_grape_hovering_8_2026-06-12-17-46-41.bag')
TOPICS = dict(gyro='/gimbalrotor/imu', mocap='/gimbalrotor/mocap/pose',
              converted='/gimbalrotor/sensor_plugin/imu1/ros_converted', static='/tf_static')
EVAL_INTERVAL = (50.000988619, 55.000988619)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _helpers():
    path = MINIMAL/'imu-first-order-lag/imu_data.py'
    spec = importlib.util.spec_from_file_location('sg_clock_existing_imu_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clock_statistics(header_ns, record_ns):
    header, record = np.asarray(header_ns, np.int64), np.asarray(record_ns, np.int64)
    if header.shape != record.shape or header.ndim != 1 or not len(header):
        raise ValueError('Paired original header/record clock vectors required')
    latency = (record-header).astype(float)*1e-9
    differences = np.diff(header).astype(float)*1e-9
    return dict(count=len(header), record_minus_header_seconds=dict(
        mean=float(np.mean(latency)), std=float(np.std(latency)), minimum=float(np.min(latency)),
        p05=float(np.percentile(latency,5)), median=float(np.median(latency)),
        p95=float(np.percentile(latency,95)), maximum=float(np.max(latency))),
        header_interval_seconds=dict(minimum=float(differences.min()),
            p05=float(np.percentile(differences,5)), median=float(np.median(differences)),
            p95=float(np.percentile(differences,95)), maximum=float(differences.max()),
            mean=float(differences.mean()), std=float(differences.std())),
        nonpositive_header_interval_count=int(np.count_nonzero(differences<=0)),
        nonpositive_record_interval_count=int(np.count_nonzero(np.diff(record)<=0)),
        mean_header_rate_hz=float((len(header)-1)/((header[-1]-header[0])*1e-9)),
        note='Record minus header is a timestamp/transport diagnostic; it does not identify sensor dynamics or physical latency on its own')


def extract(bag_path=BAG, support_interval=(39.5,60.5), eval_interval=EVAL_INTERVAL):
    import genpy
    import rosbag
    helper = _helpers()
    inverse = {topic:name for name,topic in TOPICS.items()}
    streams = {name:[] for name in TOPICS}
    with rosbag.Bag(str(bag_path),'r') as bag:
        origin = next(bag.read_messages(raw=True))[2].to_nsec()
        end_ns = origin + int(round((support_interval[1]+.2)*1e9))
        end = genpy.Time(end_ns//1_000_000_000,end_ns%1_000_000_000)
        for topic,message,record in bag.read_messages(topics=list(inverse),end_time=end):
            name=inverse[topic]
            stamp=message.stamp if name=='gyro' else message.header.stamp if name!='static' else record
            row=dict(stamp_ns=stamp.to_nsec(),record_ns=record.to_nsec(),source_index=len(streams[name]))
            if name=='gyro':
                row['gyro']=list(message.gyro)
            elif name=='mocap':
                q,p=message.pose.orientation,message.pose.position
                row.update(quaternion=[q.x,q.y,q.z,q.w],position=[p.x,p.y,p.z],frame=message.header.frame_id)
            elif name=='converted':
                omega=message.angular_velocity
                row.update(gyro=[omega.x,omega.y,omega.z],frame=message.header.frame_id)
            elif name=='static':
                row['transforms']=[]
                for transform in message.transforms:
                    p,q=transform.transform.translation,transform.transform.rotation
                    row['transforms'].append((transform.header.frame_id,transform.child_frame_id,
                        [p.x,p.y,p.z],[q.x,q.y,q.z,q.w]))
            streams[name].append(row)
    extrinsics=helper._verify_extrinsics(streams)
    data=dict(eval_interval=np.asarray(eval_interval,float),
              wide_interval=np.asarray([40.,60.]),source_support_interval=np.asarray(support_interval,float))
    metadata=dict(schema='grape/sg-mocap-gyro-original-clock-input/v1',
        bag_path=str(Path(bag_path).resolve()),bag_size_bytes=Path(bag_path).stat().st_size,
        bag_mtime_ns=Path(bag_path).stat().st_mtime_ns,full_bag_sha256_computed=False,
        bag_origin_record_ns=origin,source_support_interval_seconds=list(support_interval),
        eval_interval_seconds=list(eval_interval),wide_comparison_interval_seconds=[40.,60.],
        time_origin='first original bag record shared by all header and record arrays',
        time_alignment_applied=False,resampling_applied=False,bias_or_scale_adjustment_applied=False,
        raw_bag_copied=False,original_trajectory_or_filter_modified=False,
        use='read-only SG differentiation/gyro comparison and separate kinematic reconstruction diagnostic',
        clock_diagnostics={},topics=TOPICS,extrinsics=extrinsics,
        extractor_sha256=sha(Path(__file__)),extrinsic_helper_sha256=sha(MINIMAL/'imu-first-order-lag/imu_data.py'))
    for name in ('gyro','mocap','converted'):
        rows=streams[name]
        header=np.asarray([r['stamp_ns'] for r in rows],np.int64)
        local=(header-origin).astype(float)*1e-9
        selected=(local>=support_interval[0])&(local<=support_interval[1])
        chosen=[row for row,keep in zip(rows,selected) if keep]
        stamp=np.asarray([r['stamp_ns'] for r in chosen],np.int64)
        record=np.asarray([r['record_ns'] for r in chosen],np.int64)
        data[name+'_time']=(stamp-origin).astype(float)*1e-9
        data[name+'_record_time']=(record-origin).astype(float)*1e-9
        data[name+'_stamp_ns']=stamp
        data[name+'_record_ns']=record
        data[name+'_source_index']=np.asarray([r['source_index'] for r in chosen],np.int64)
        metadata['clock_diagnostics'][name]=clock_statistics(stamp,record)
        if name=='mocap':
            data['mocap_quaternion_xyzw']=np.asarray([r['quaternion'] for r in chosen],float)
            data['mocap_position']=np.asarray([r['position'] for r in chosen],float)
            metadata['mocap_frames']=sorted(set(row['frame'] for row in chosen))
            if any(frame.split('/')[-1]!='world' for frame in metadata['mocap_frames']):
                raise ValueError('Original mocap does not use the expected world frame')
        else:
            data[name+'_rad_s']=np.asarray([r['gyro'] for r in chosen],float)
            if name=='converted': metadata['converted_frames']=sorted(set(row['frame'] for row in chosen))
    # Exact matching by original acquisition stamp is diagnostic only: all
    # original arrays remain present, including unmatched measurements.
    _,left,right=np.intersect1d(data['gyro_stamp_ns'],data['converted_stamp_ns'],return_indices=True)
    difference=data['gyro_rad_s'][left]-data['converted_rad_s'][right]
    metadata['native_vs_converted_same_header_stamp']=dict(
        exact_timestamp_pair_count=len(left),native_count=len(data['gyro_time']),
        converted_count=len(data['converted_time']),
        maximum_absolute_gyro_difference_rad_s=float(np.max(np.abs(difference))) if len(left) else None,
        role='value/frame-copy check, never used to shift, replace or interpolate source measurements')
    return data,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag',type=Path,default=BAG)
    parser.add_argument('--output',type=Path,default=Path('/tmp/gyro-mocap-sg-20261113'))
    args=parser.parse_args()
    data,metadata=extract(args.bag)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'input.npz',**data)
    metadata['arrays_sha256']=sha(args.output/'input.npz')
    (args.output/'input.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),bytes=(args.output/'input.npz').stat().st_size,
        clock_diagnostics=metadata['clock_diagnostics'],
        native_vs_converted_same_header_stamp=metadata['native_vs_converted_same_header_stamp']),indent=2))


if __name__=='__main__':main()

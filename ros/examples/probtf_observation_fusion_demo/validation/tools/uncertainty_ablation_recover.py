#!/usr/bin/env python3
"""Finish only raw archival after the diagnostic's physical stop/shutdown.

No inference, K update, control or physical motion is replayed. Exact original
request times/commands/geometries remain fixed. The renderer's stateful noise
stream is advanced across the completed prefix, and its last committed image
is reproduced bitwise before any missing suffix is supplied. This is archival
recovery, never information that was available during the completed motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rospkg
import yaml

from uncertainty_ablation import acquisition_worker


class IteratorQueue:
    def __init__(self, values):
        self.values=iter(values)
    def get(self):
        return next(self.values,None)


class RecordReplies:
    def __init__(self,path):
        self.path=path
        self.count=0
        self.failure=None
    def put(self,value):
        record={key:value[key]for key in ('kind','sensor_kind','event_id',
            'event_sequence','frame_index','stamp','work_seconds','image_sha256',
            'image_path','failed','worker','traceback')if key in value}
        self.count+=1
        if value.get('failed'):
            self.failure=value
        with self.path.open('a')as stream:
            stream.write(json.dumps(record,separators=(',',':'))+'\n')


def recover(output):
    from probtf_observation_fusion_demo.sensor_event_store import SensorEventStore
    import probtf_observation_fusion_demo.async_workers as workers
    from probtf_observation_fusion_demo.board_camera import BoardCamera
    from probtf_switch_press_demo.vision import PlanarCamera
    output=Path(output)
    setup=json.loads((output/'diagnostic_setup.json').read_text())
    # Require a completed physical trial and an already closed worker set.
    if not (output/'shutdown.jsonl').is_file():
        raise ValueError('Raw recovery requires a closed, completed diagnostic run.')
    shutdown=json.loads((output/'shutdown.jsonl').read_text().splitlines()[-1])
    if not (output/'diagnostic_summary.json').is_file():
        raise ValueError('Raw recovery requires a closed trial with its physical result recorded.')
    requests=[json.loads(line)for line in (output/'acquisition_requests.jsonl').read_text().splitlines()]
    if any(request.get('kind')not in ('joint','imu','image')for request in requests):
        raise ValueError('The physical acquisition journal contains a non-sensor request.')
    store=SensorEventStore(output/'observations.sqlite3')
    before=store.stats()
    filled={row[0]:row[1]for row in store.connection.execute(
        'SELECT event_id,payload_hash FROM events WHERE payload_hash IS NOT NULL')}
    pending=[request for request in requests if request['event_id']not in filled]
    images=[request for request in requests if request['kind']=='image']
    prefix=[]
    missing_seen=False
    for request in images:
        if request['event_id']not in filled:
            missing_seen=True
        elif missing_seen:
            raise ValueError('The camera archive is not a completed FIFO prefix.')
        else:
            prefix.append(request)
    package=Path(rospkg.RosPack().get_path('probtf_observation_fusion_demo'))
    config=yaml.safe_load((package/'config/switch_press_scene.yaml').read_text())
    board=config.get('marker_board',{})
    ids=tuple(int(i)for row in board['ids']for i in row)if board.get('enabled')else(config['marker']['id'],)
    texture=Path(rospkg.RosPack().get_path('prob_artag_detector'))/'samples'/(
        'apriltag_36h11_id_{:03d}.png'.format(config['marker']['id']))
    specification=dict(board=board,camera=config['camera'],texture=str(texture),
        seed=config['seed'],marker_size=config['marker']['size'],marker_ids=ids)
    camera=(BoardCamera(texture,board,config['camera'],config['seed'])if board.get('enabled')
        else PlanarCamera(texture,config['marker']['size'],config['camera'],config['seed']))
    started=time.monotonic()
    # Draw the exact same normal arrays, rather than incorrectly assuming a
    # fixed number of PCG64 raw draws per normal variate. New diagnostics
    # retain the exact before/after RNG state of their last completed image.
    shape=(camera.camera.height,camera.camera.width)
    checkpoint_path=output/'diagnostic_camera_rng_checkpoint.json'
    checkpoint=json.loads(checkpoint_path.read_text())if checkpoint_path.is_file()else None
    checkpoint_index=(next((index for index,request in enumerate(prefix)
        if request['event_id']==checkpoint['event_id']),None)if checkpoint is not None else None)
    verification_index=len(prefix)-1 if checkpoint_index is None else checkpoint_index
    if checkpoint_index is not None:
        camera.rng.bit_generator.state=checkpoint['state_before_image']
    for _ in ([]if checkpoint_index is not None else prefix[:verification_index]):
        camera.rng.normal(0.,config['camera']['noise_std'],shape)
    bitwise_check=None
    if prefix:
        request=prefix[verification_index]
        image=camera.render(np.asarray(request['camera_pose']),np.asarray(request['marker_pose']),
            tuple((stamp,np.asarray(pose))for stamp,pose in request['camera_history']))
        expected=store.get(request['event_id']).payload['image_sha256']
        actual=hashlib.sha256(image.tobytes()).hexdigest()
        bitwise_check=dict(event_id=request['event_id'],expected_sha256=expected,
                           reproduced_sha256=actual,equal=actual==expected)
        if actual!=expected:
            raise ValueError('The original stateful camera noise prefix did not reproduce bitwise.')
        for _ in prefix[verification_index+1:]:
            camera.rng.normal(0.,config['camera']['noise_std'],shape)
    store.close()

    def remaining():
        for request in pending:
            if request['kind']=='image':
                request=dict(request,camera_pose=np.asarray(request['camera_pose']),
                    marker_pose=np.asarray(request['marker_pose']),
                    camera_history=tuple((stamp,np.asarray(pose))for stamp,pose in request['camera_history']))
            yield request
        yield None

    original_board,original_planar=workers.BoardCamera,workers.PlanarCamera
    workers.BoardCamera=lambda *args,**kwargs:camera
    workers.PlanarCamera=lambda *args,**kwargs:camera
    replies=RecordReplies(output/'diagnostic_raw_recovery_events.jsonl')
    try:
        # Recovery only renders/archives the original raw measurements. Never
        # reinstall a historical oracle or recreate its inference sidecars.
        acquisition_worker('baseline',IteratorQueue(remaining()),replies,
                           specification,str(output/'observations.sqlite3'),str(output))
    finally:
        workers.BoardCamera,workers.PlanarCamera=original_board,original_planar
    if replies.failure is not None:
        raise RuntimeError(replies.failure)
    store=SensorEventStore(output/'observations.sqlite3')
    after=store.stats()
    missing=[request['event_id']for request in requests if
             store.get(request['event_id'],include_payload=False).payload_hash is None]
    record=dict(physical_motion_or_estimation_replayed=False,
        results_used_in_completed_motion=False,raw_request_count=len(requests),
        completed_requested_motion=bool(shutdown['motion_finished']),
        before=before,after=after,recovered_event_count=len(pending),
        remaining_missing_event_ids=missing,all_raw_archived=not missing,
        completed_camera_prefix=len(prefix),noise_prefix_bitwise_check=bitwise_check,
        wall_seconds=time.monotonic()-started)
    store.close()
    (output/'diagnostic_raw_recovery.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    print(json.dumps(recover(parser.parse_args().output),indent=2))

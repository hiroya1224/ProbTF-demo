"""Boundary checks for the isolated diagnostic, not controller mirror tests."""
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

from uncertainty_ablation import (QueueProxy, control_truth,
                                 install_coordinator_proxies)


def test_baseline_queue_wrappers_preserve_payload_objects_and_numbers(tmp_path):
    outgoing, incoming = Queue(), Queue()
    session = SimpleNamespace(control_requests=outgoing,control_replies=incoming,
                              parameter_packets={},output=tmp_path)
    install_coordinator_proxies(session,'baseline')
    matrix = np.arange(36,dtype=float).reshape(6,6)
    request = dict(packet=None,result=object(),covariance=matrix)
    session.control_requests.put(request)
    assert outgoing.get_nowait() is request
    incoming.put(request)
    assert session.control_replies.get_nowait() is request
    assert request['covariance'] is matrix
    assert sorted(request) == ['covariance','packet','result']


def _session():
    frame = SimpleNamespace(stamp=100.25,time=.25,command=np.arange(6)*.01,
        joints=np.arange(6)*.02,link_poses={'base_link':np.eye(4)})
    scene = SimpleNamespace(marker=np.eye(4),goal=np.array([1.,2.,3.]),
        config={'plant':{'stiffness':[350.,125.,95.,60.,40.,30.]}})
    return SimpleNamespace(frame=frame,frames=[object(),frame],scene=scene)


@pytest.mark.parametrize('stamp,index',[(100.24,1),(100.25,0)])
def test_truth_capture_refuses_neighboring_time_or_frame(stamp,index):
    with pytest.raises(ValueError,match='exact control source frame'):
        control_truth(_session(),dict(stamp=stamp,index=index))


def test_truth_capture_uses_physical_pose_at_matching_source_only():
    session = _session()
    truth = control_truth(session,dict(stamp=100.25,index=1))
    np.testing.assert_array_equal(truth['physical_joint_position'],session.frame.joints)
    np.testing.assert_array_equal(truth['command'],session.frame.command)
    assert truth['stamp'] == session.frame.stamp
    assert truth['time'] == session.frame.time
    assert truth['true_K'] == [350.,125.,95.,60.,40.,30.]
    session.frame.joints[0] = 99.
    assert truth['physical_joint_position'][0] == 0.


def test_oracle_packet_is_registered_before_normal_receive_can_restore_action(tmp_path):
    session = _session()
    session.output = tmp_path
    session.control_requests,session.control_replies = Queue(),Queue()
    session.parameter_packets = {}
    incoming = session.control_replies
    install_coordinator_proxies(session,'true_K')
    packet = SimpleNamespace(packet_id='explicit-diagnostic-known-K')
    message = dict(_diagnostic_estimated_packet=packet,
                   action_packet_ref={'estimated_packet_id':packet.packet_id})
    incoming.put(message)
    received = session.control_replies.get_nowait()
    assert received is message
    assert session.parameter_packets[received['action_packet_ref']['estimated_packet_id']] is packet


def test_oracle_control_truth_is_separate_from_original_request(tmp_path):
    session = _session()
    outgoing = Queue()
    session.control_requests,session.control_replies = outgoing,Queue()
    session.parameter_packets,session.output = {},tmp_path
    install_coordinator_proxies(session,'oracle_goal')
    request = dict(stamp=session.frame.stamp,index=1,request_id=7)
    session.control_requests.put(request)
    transmitted = outgoing.get_nowait()
    assert transmitted is not request
    assert '_diagnostic_truth' not in request
    assert transmitted['_diagnostic_truth']['stamp'] == session.frame.stamp
    assert (tmp_path/'diagnostic_control_truth.jsonl').is_file()

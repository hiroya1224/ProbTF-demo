"""Timeline auditing must not reinterpret arrival order or assume permanent loss."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('occlusion_summary',
    Path(__file__).with_name('summarize_marker_occlusion.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_visibility_reacquisition_retains_each_original_period():
    samples = [dict(stamp=float(index), visible=value) for index, value in enumerate(
        (True, True, False, False, True, False, False))]
    result = module.visibility_periods(samples)
    assert result['initial_seen']
    assert result['reacquisitions'] == 1
    assert [item['visible'] for item in result['periods']] == [True, False, True, False]
    assert result['terminal_lost_interval']['first_stamp'] == 5.
    assert result['terminal_lost_interval']['samples'] == 2


def test_initially_missing_is_not_claimed_as_observed_then_reacquired():
    result = module.visibility_periods([dict(stamp=0., visible=False),
                                       dict(stamp=1., visible=True)])
    assert not result['initial_seen']
    assert result['reacquisitions'] == 0
    assert result['terminal_lost_interval'] is None


def test_post_loss_fusion_counts_original_acquisition_time_not_list_order():
    # The older image may arrive after the newer IMU. Its acquisition stays old.
    updates = [dict(kind='imu', acquisition_stamp=11., observation_ids=['imu:2'],
                    simple_inference=dict(goal_observed=True)),
               dict(kind='image', acquisition_stamp=8., observation_ids=['image:1'],
                    simple_inference=dict(goal_observed=True)),
               dict(kind='image', acquisition_stamp=12., observation_ids=[],
                    simple_inference=dict(goal_observed=True))]
    plans = [dict(adoption_stamp=13., start_command=[0., 0.], target_command=[.01, 0.],
                  diagnostics=dict(goal_observed=True))]
    result = module.after_loss_counts(updates, plans, 10.)
    assert result['counts']['fused_imu'] == 1
    assert result['counts'].get('fused_image', 0) == 0
    assert result['counts']['processed_image'] == 1
    assert result['counts']['plans_sending_nonzero_command_increment'] == 1


def test_no_terminal_loss_does_not_invent_imu_only_tail():
    result = module.after_loss_counts([], [], None)
    assert not result['available']


def test_physical_end_without_estimation_drain_is_not_a_final_trial(tmp_path):
    (tmp_path/'motion_manifest.json').write_text(json.dumps(dict(
        finished=True, estimation_drained=False, simulated_elapsed_s=180., duration=180.)))
    with pytest.raises(RuntimeError, match='complete estimation drain'):
        module.summarize(tmp_path)

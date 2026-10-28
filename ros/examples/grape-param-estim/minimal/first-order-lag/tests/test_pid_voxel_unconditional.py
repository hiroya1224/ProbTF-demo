"""Regression against success-conditioned populations and controller leakage."""
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'automation'))
from pid_voxel_survival import sample_stability


def test_the_previously_excluded_plant_remains_in_the_denominator():
    radii = np.full((1, 128), .9)
    radii[0, 26] = 1.0000195565456116
    result = sample_stability(radii, np.array([True]))
    assert result.sample_count == 128
    assert result.stable_count[0] == 127
    assert result.stable_fraction[0] == 127 / 128
    assert result.stable_fraction[0] != 1
    assert result.valid_count[0] == 128
    # 95% and every-sample stability are distinct requirements.
    assert result.alpha_classification[0] == 3
    assert sample_stability(radii, np.array([True]), 1).alpha_classification[0] == 2


def test_unresolved_stays_in_N_and_pending_is_not_a_zero_stability_result():
    radii = np.array([[.8, 1.2, np.nan, np.inf], [.7, .6, .8, .9]])
    result = sample_stability(radii, np.array([True, False]))
    assert result.stable_fraction[0] == .25
    assert result.stable_count[0] == 1
    assert result.valid_count[0] == 2
    assert result.unresolved_count[0] == 2
    assert np.isnan(result.stable_fraction[1])
    assert result.stable_count[1] == result.valid_count[1] == result.unresolved_count[1] == -1
    assert result.alpha_classification[1] == 0


@pytest.mark.parametrize('module_name,evaluator_name', [
    ('pid_gain_voxels', 'BatchedForwardMarginEvaluator'),
    ('pid_gain_voxels_interior', 'ExpandedGainEvaluator'),
    ('pid_gain_voxels_fast', 'FastGainEvaluator'),
])
def test_success_is_loaded_only_after_the_failure_context_map_is_finished(
        module_name, evaluator_name, tmp_path, monkeypatch):
    module = importlib.import_module(module_name)
    failure_gains = np.array([[3, .1, 1], [5, 1, 2.5], [20, 1, 8], [4, 1, 2.]])
    success_gains = failure_gains * 3.7  # Outside the failure-defined map.
    failure_path, success_path = tmp_path/'failure.json', tmp_path/'success.json'
    failure_path.write_text('{}')
    success_path.write_text('{}')
    model_path = tmp_path/'model.json'
    model_path.write_text('{}')
    events = []

    def load(path):
        is_success = Path(path) == success_path
        events.append(('load_success' if is_success else 'load_failure', None))
        return {'test_gains': success_gains if is_success else failure_gains,
                'input': {'vehicle_model': str(model_path)},
                'controller_timing': {'median_seconds': .01}, 'case_name': 'failure'}

    class Evaluator:
        def __init__(self, **kwargs):
            self._cache = {}
            assert 'reference_gains' not in kwargs or np.array_equal(kwargs['reference_gains'], failure_gains)

        def evaluate_many(self, values, **kwargs):
            values = np.asarray(values)
            events.append(('evaluate', values.copy()))
            return [SimpleNamespace(
                gains=value.reshape(4, 3)*module.GAIN_CAPS,
                q=value, delta=.1, worst_spectral_radius=.9, worst_sample_index=0,
                spectral_radius=np.full(4, .9), pole_valid_mask=np.ones(4, bool),
                near_kink_mask=np.zeros(4, bool), trim_vectors=np.zeros((4, 10)),
                safe_on_sample_set=True) for value in values]

        def close(self):
            pass

    monkeypatch.setattr(module, 'load_estimate_json', load)
    monkeypatch.setattr(module, '_recorded_gain_matrix', lambda estimate: estimate['test_gains'])
    monkeypatch.setattr(module, 'load_vehicle_model', lambda path: SimpleNamespace(parameters={}))
    monkeypatch.setattr(module, 'actuator_parameters_from_estimate', lambda estimate: {})
    monkeypatch.setattr(module, '_sobol_quotient_coordinates', lambda *args: (np.zeros((4, 13)), {}))
    monkeypatch.setattr(module, 'quotient_to_scale_free_plants', lambda *args: [None]*4)
    monkeypatch.setattr(module, 'input_fingerprint', lambda *args: ('test-map', {}))
    monkeypatch.setattr(module, 'source_commit', lambda: 'test')
    monkeypatch.setattr(module, evaluator_name, Evaluator)
    monkeypatch.setattr(module, 'plot_volume', lambda *args: None)
    monkeypatch.setattr(module, 'write_index', lambda *args: None)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    args = module.parser().parse_args([
        '--estimate-json', str(failure_path), '--success-json', str(success_path),
        '--output', str(tmp_path/'output'), '--samples', '4', '--workers', '1',
        '--grid-sizes', '3', '--groups', 'roll_pitch'])
    module.analyze(args)
    load_success = next(i for i, event in enumerate(events) if event[0] == 'load_success')
    before = [values for kind, values in events[:load_success] if kind == 'evaluate']
    assert len(before) > 1
    for values in before:
        gains = values.reshape(-1, 4, 3)*module.GAIN_CAPS
        np.testing.assert_allclose(gains[:, [0, 1, 3]], np.broadcast_to(failure_gains[[0, 1, 3]], gains[:, [0, 1, 3]].shape))
    after = [values for kind, values in events[load_success:] if kind == 'evaluate']
    assert len(after) == 1
    np.testing.assert_allclose(after[0].reshape(4, 3)*module.GAIN_CAPS, success_gains)
    refs = json.loads((tmp_path/'output/references.json').read_text())
    assert refs['success_comparison_input']['role'] == 'post-map full-12-gain comparison only'


@pytest.mark.parametrize('module_name', ['pid_gain_voxels', 'pid_gain_voxels_interior', 'pid_gain_voxels_fast'])
def test_default_needs_no_success_file_and_success_path_cannot_change_map_identity(module_name):
    module = importlib.import_module(module_name)
    args = module.parser().parse_args([])
    assert args.success_json is None
    assert args.output.endswith('_failure_context')
    estimate = Path(args.estimate_json)
    model = Path(module.load_estimate_json(estimate)['input']['vehicle_model'])
    first, first_inputs = module.input_fingerprint(args, estimate, None, model)
    args.success_json = '/nonexistent/comparison-only.json'
    second, second_inputs = module.input_fingerprint(args, estimate, Path(args.success_json), model)
    assert first == second
    assert first_inputs == second_inputs
    assert args.success_json not in first_inputs

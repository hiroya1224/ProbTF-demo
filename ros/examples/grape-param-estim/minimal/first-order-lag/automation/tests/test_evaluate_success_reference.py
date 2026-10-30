from pathlib import Path
import json
import sys

import numpy as np
import pytest

AUTOMATION = Path(__file__).resolve().parents[1]
if str(AUTOMATION) not in sys.path:
    sys.path.insert(0, str(AUTOMATION))
import evaluate_success_reference as comparison


def test_rate_keeps_nonfinite_and_boundary_samples_in_original_denominator():
    radii = np.array([.9, 1., 1.1, np.nan, np.inf])
    before = radii.copy()
    result = comparison.stability_summary(radii)
    assert result["sample_count"] == 5
    assert result["stable_count"] == 1
    assert result["valid_count"] == 3
    assert result["unresolved_count"] == 2
    assert result["stable_fraction"] == .2
    assert result["spectral_radius"] == [.9, 1., 1.1, None, None]
    np.testing.assert_array_equal(radii, before)


def completed_data(root):
    for case in comparison.CASES:
        folder = root / case
        folder.mkdir()
        (folder / "manifest.json").write_text(json.dumps({"arguments": {"success_json": None}}))
        (folder / "status.json").write_text(json.dumps({"status": "complete"}))
        for group in comparison.GROUPS:
            for size in (33, 65):
                volume = folder / group / f"grid_{size}"
                volume.mkdir(parents=True)
                (volume / "status.json").write_text(json.dumps({"complete": True, "pending": 0}))


def test_post_map_comparison_requires_both_failures_and_all_resolutions_complete(tmp_path):
    completed_data(tmp_path)
    assert tuple(comparison.completed_inputs(tmp_path)) == comparison.CASES
    (tmp_path / "failure2/yaw/grid_65/status.json").write_text(json.dumps({"complete": False, "pending": 1}))
    with pytest.raises(ValueError, match="not complete"):
        comparison.completed_inputs(tmp_path)


def test_success_dependent_map_cannot_be_used_as_a_failure_only_comparison(tmp_path):
    completed_data(tmp_path)
    (tmp_path / "failure1/manifest.json").write_text(json.dumps({"arguments": {"success_json": "old_success.json"}}))
    with pytest.raises(ValueError, match="must not depend"):
        comparison.completed_inputs(tmp_path)

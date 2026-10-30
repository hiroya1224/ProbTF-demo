from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from publish_results import compact_arrays


def test_compaction_preserves_full_state_gram_and_cross_terms():
    factor = np.arange(12 * 17, dtype=float).reshape(1, 12, 17) / 100
    covariance = factor @ factor.transpose(0, 2, 1)
    original = {"model_state_covariance_factor": factor,
                "model_covariance_state": covariance,
                "model_position_covariance_factor_fc": factor[:, :3],
                "model_rotation_covariance_factor": factor[:, 3:6],
                "orientation_A_recursive": np.eye(4)[None], "native_time": np.arange(3)}
    compact = compact_arrays(original)
    assert "model_covariance_state" not in compact
    kept = compact["model_state_covariance_factor"]
    np.testing.assert_array_equal(kept @ kept.transpose(0, 2, 1), covariance)
    assert np.any(covariance[0, :3, 3:6] != 0)
    np.testing.assert_array_equal(compact["orientation_A_recursive"], original["orientation_A_recursive"])

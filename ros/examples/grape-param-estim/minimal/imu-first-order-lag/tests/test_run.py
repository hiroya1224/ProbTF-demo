from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run
import imu_model
from test_imu_model import synthetic_data


def test_outputs_cannot_overwrite_mocap_branch():
    with pytest.raises(ValueError, match="outside"):
        run.require_separate_output(HERE.parent / "first-order-lag/outputs/new")
    assert run.require_separate_output(HERE / "outputs") == HERE / "outputs"


def test_cache_rejects_changed_sensor_values_on_identical_clock():
    data = synthetic_data()
    vehicle = HERE.parent / "grape_vehicle_model.json"
    fit = {"estimate": {"provenance": imu_model.input_provenance(data, vehicle)}}
    run.validate_fit_inputs(fit, data, vehicle)
    data["gyro"][10, 1] += .001
    with pytest.raises(ValueError, match="exact native IMU inputs"):
        run.validate_fit_inputs(fit, data, vehicle)

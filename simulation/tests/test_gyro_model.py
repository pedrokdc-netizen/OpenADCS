import math

import numpy as np
import pytest

from openadcs_sim.gyro_model import GyroConfig, GyroModel


def make_config(**overrides):
    z = np.zeros(3)
    values = dict(sensor_from_body=np.eye(3), scale_error=z, misalignment=np.eye(3), bias_rad_s=z,
                  bias_temperature_rad_s_per_c=z, bias_temperature_quadratic_rad_s_per_c2=z,
                  reference_temperature_c=20.0, temperature_range_c=(-40.0, 85.0),
                  angle_random_walk_rad_s_rthz=z, full_scale_rad_s=10.0)
    values.update(overrides)
    return GyroConfig(**values)


def test_ideal_gyro_passes_rate_through_with_frame_rotation():
    rotation = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    model = GyroModel(make_config(sensor_from_body=rotation))
    reading = model.sample(0.0, [0.1, 0.2, 0.3], 20.0)
    assert reading.valid and reading.status == "OK"
    assert reading.rate_sensor_rad_s == pytest.approx(rotation @ [0.1, 0.2, 0.3])


def test_scale_bias_and_temperature_terms():
    model = GyroModel(make_config(
        scale_error=np.array([0.01, 0, 0]), bias_rad_s=np.array([0.001, 0, 0]),
        bias_temperature_rad_s_per_c=np.array([1e-4, 0, 0]),
        bias_temperature_quadratic_rad_s_per_c2=np.array([1e-5, 0, 0])))
    reading = model.sample(0.0, [1.0, 0, 0], 30.0)
    assert reading.rate_sensor_rad_s[0] == pytest.approx(1.01 + 0.001 + 1e-3 + 1e-3)


def test_arw_statistics_and_seeding():
    sigma = 1e-3

    def run(seed):
        model = GyroModel(make_config(angle_random_walk_rad_s_rthz=np.full(3, sigma), seed=seed))
        return np.array([model.sample(i * 0.1, [0, 0, 0], 20.0).rate_sensor_rad_s for i in range(1, 4001)])
    a = run(1)
    assert np.array_equal(a, run(1))
    assert a.std() == pytest.approx(sigma / math.sqrt(0.1), rel=0.1)


def test_rate_random_walk_grows():
    model = GyroModel(make_config(rate_random_walk_rad_s2_rthz=np.full(3, 1e-4), seed=2))
    values = [model.sample(float(i), [0, 0, 0], 20.0).rate_sensor_rad_s for i in range(2000)]
    assert np.abs(values[-1]).max() > 1e-4


def test_saturation_quantization_and_temperature_domain():
    model = GyroModel(make_config(full_scale_rad_s=1.0, resolution_rad_s=0.1))
    reading = model.sample(0.0, [5.0, 0.234, 0], 20.0)
    assert reading.saturated and reading.status == "SATURATED"
    assert reading.rate_sensor_rad_s == pytest.approx([1.0, 0.2, 0.0])
    out = model.sample(1.0, [0, 0, 0], 100.0)
    assert not out.valid and out.status == "TEMPERATURE_OUT_OF_RANGE"
    assert np.all(np.isnan(out.rate_sensor_rad_s))


def test_validation_and_time_order():
    with pytest.raises(ValueError):
        make_config(sensor_from_body=np.diag([1.0, 1, -1]))
    with pytest.raises(ValueError):
        make_config(bias_instability_rad_s=np.full(3, 1e-4))
    with pytest.raises(ValueError):
        make_config(scale_error=np.array([-1.0, 0, 0]))
    model = GyroModel(make_config())
    model.sample(1.0, [0, 0, 0], 20.0)
    with pytest.raises(ValueError):
        model.sample(1.0, [0, 0, 0], 20.0)

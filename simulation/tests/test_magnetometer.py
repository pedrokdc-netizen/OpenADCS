import numpy as np
import pytest

from openadcs_sim.magnetometer import MagnetometerConfig, MagnetometerModel


def config(**overrides):
    values = dict(
        sensor_from_body=np.eye(3), bias_t=np.zeros(3), scale_error=np.zeros(3),
        misalignment=np.eye(3), white_noise_sigma_t=np.zeros(3),
        residual_field_sensor_t=np.zeros(3), bias_temperature_t_per_c=np.zeros(3),
        scale_temperature_per_c=np.zeros(3), reference_temperature_c=20.0,
        temperature_range_c=(-40.0, 85.0), full_scale_t=1e-3,
    )
    values.update(overrides)
    return MagnetometerConfig(**values)


def test_ideal_configuration_returns_mounted_field():
    mount = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
    reading = MagnetometerModel(config(sensor_from_body=mount)).sample(0, [1e-5, 2e-5, 3e-5], 20)
    np.testing.assert_allclose(reading.field_sensor_t, [2e-5, -1e-5, 3e-5])
    assert reading.valid and not reading.saturated


def test_error_terms_are_applied_with_documented_order():
    model = MagnetometerModel(config(
        bias_t=[1e-7, 0, 0], scale_error=[.01, 0, 0], residual_field_sensor_t=[1e-6, 0, 0],
        bias_temperature_t_per_c=[1e-9, 0, 0], scale_temperature_per_c=[1e-3, 0, 0],
    ))
    reading = model.sample(0, [1e-5, 0, 0], 30)
    expected = (1 + .01 + 1e-3 * 10) * (1e-5 + 1e-6) + 1e-7 + 1e-9 * 10
    assert reading.field_sensor_t[0] == pytest.approx(expected)


def test_noise_is_seeded_and_statistically_configured():
    first = MagnetometerModel(config(white_noise_sigma_t=[1e-7] * 3, seed=4))
    second = MagnetometerModel(config(white_noise_sigma_t=[1e-7] * 3, seed=4))
    a = np.array([first.sample(i, [0, 0, 0], 20).field_sensor_t for i in range(4000)])
    b = np.array([second.sample(i, [0, 0, 0], 20).field_sensor_t for i in range(4000)])
    np.testing.assert_array_equal(a, b)
    assert a.std(axis=0) == pytest.approx([1e-7] * 3, rel=.08)


def test_gauss_markov_noise_is_correlated_with_requested_sigma():
    model = MagnetometerModel(config(markov_sigma_t=[1e-7] * 3, markov_time_constant_s=50., seed=2))
    values = np.array([model.sample(i, [0, 0, 0], 20).field_sensor_t[0] for i in range(6000)])
    assert values.std() == pytest.approx(1e-7, rel=.25)
    assert np.corrcoef(values[:-1], values[1:])[0, 1] > .9


def test_saturation_quantization_and_temperature_domain():
    model = MagnetometerModel(config(resolution_t=1e-8))
    saturated = model.sample(0, [2e-3, 0, 0], 20)
    assert saturated.saturated and saturated.status == "SATURATED"
    assert saturated.field_sensor_t[0] == pytest.approx(1e-3)
    assert model.sample(1, [1.23456e-5, 0, 0], 20).field_sensor_t[0] == pytest.approx(1.235e-5)
    invalid = model.sample(2, [1e-5, 0, 0], 100)
    assert not invalid.valid and invalid.status == "TEMPERATURE_OUT_OF_RANGE"
    assert np.all(np.isnan(invalid.field_sensor_t))


def test_invalid_inputs_and_configuration_are_rejected():
    with pytest.raises(ValueError, match="rotation"):
        config(sensor_from_body=np.diag([1., 1, -1]))
    with pytest.raises(ValueError, match="gain positive"):
        config(scale_error=[-1, 0, 0])
    with pytest.raises(ValueError, match="markov_time_constant_s"):
        config(markov_sigma_t=[1e-8] * 3)
    with pytest.raises(ValueError, match="inside temperature_range_c"):
        config(reference_temperature_c=200.)
    model = MagnetometerModel(config())
    model.sample(1, [0, 0, 0], 20)
    with pytest.raises(ValueError, match="strictly increasing"):
        model.sample(1, [0, 0, 0], 20)
    with pytest.raises(ValueError, match="finite 3-vector"):
        model.sample(2, [np.nan, 0, 0], 20)

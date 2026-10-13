from dataclasses import replace
import math

import numpy as np
import pytest

from openadcs_sim.sensor_processing import (
    AdcsInputs, GyroConfig, Illumination, Measurement, NativeProcessing,
    SampleHeader, SensorKind, Status, SunConfig,
)


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
ROTATION = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
FOV = math.radians(36.7)


@pytest.fixture
def sun_config():
    # Synthetic geometry/calibration, not a flight noise or instrument profile.
    return SunConfig(
        (0.01, 0.01), (0.002, 0.002), (0.0,) * 4, (1.0,) * 4,
        (0.1, 0.1), FOV, IDENTITY,
    )


@pytest.fixture
def gyro_config():
    return GyroConfig(
        20.0, 0.0, 40.0, (0.0,) * 9, (1.0,) * 3, 35.0, IDENTITY,
    )


def channels_for(x_angle, y_angle=0.0, brightness=1.0):
    values = []
    for angle in (x_angle, y_angle):
        ratio = 2 * 0.002 / 0.01 * math.tan(angle)
        values.extend((brightness * (1 - ratio), brightness * (1 + ratio)))
    return values


def test_sun_psd_relation_mounting_and_intensity_invariance(native, sun_config):
    expected = np.array([math.tan(math.radians(20)), math.tan(math.radians(-10)), 1])
    expected /= np.linalg.norm(expected)
    config = replace(sun_config, sensor_to_body=ROTATION)
    for brightness in (0.1, 1.0, 100.0):
        measurement = native.sun(
            config, SampleHeader(100, 1, 0),
            channels_for(math.radians(20), math.radians(-10), brightness),
            Illumination.DIRECT_SUN,
        )
        assert measurement.header.status == Status.OK
        np.testing.assert_allclose(measurement.body, np.array(ROTATION).reshape(3, 3) @ expected)
        assert np.linalg.norm(measurement.body) == pytest.approx(1)


@pytest.mark.parametrize("angle,status", [
    (FOV, Status.OK), (-FOV, Status.OK),
    (FOV + 1e-7, Status.OUT_OF_FOV), (-FOV - 1e-7, Status.OUT_OF_FOV),
])
def test_exact_fov_threshold(native, sun_config, angle, status):
    result = native.sun(
        sun_config, SampleHeader(100, 1, 0), channels_for(angle),
        Illumination.DIRECT_SUN,
    )
    assert result.header.status == status
    assert (result.body is None) == (status != Status.OK)


def test_conservative_cone_rejects_diagonal_outside_fov(native, sun_config):
    result = native.sun(
        sun_config, SampleHeader(100, 1, 0),
        channels_for(math.radians(30), math.radians(30)), Illumination.DIRECT_SUN,
    )
    assert result.header.status == Status.OUT_OF_FOV


@pytest.mark.parametrize("channels,illumination,status", [
    ((1, 1, 1, 1), Illumination.DARK, Status.NO_ILLUMINATION),
    ((1, 1, 1, 1), Illumination.UNVERIFIED, Status.ILLUMINATION_UNKNOWN),
    ((0, 0, 1, 1), Illumination.DIRECT_SUN, Status.NO_ILLUMINATION),
    ((0.01,) * 4, Illumination.DIRECT_SUN, Status.NO_ILLUMINATION),
    ((-1, 1, 1, 1), Illumination.DIRECT_SUN, Status.INVALID_INPUT),
    ((math.nan, 1, 1, 1), Illumination.DIRECT_SUN, Status.INVALID_INPUT),
    ((math.inf, 1, 1, 1), Illumination.DIRECT_SUN, Status.INVALID_INPUT),
])
def test_sun_invalid_readings_never_become_truth(native, sun_config, channels, illumination, status):
    result = native.sun(sun_config, SampleHeader(100, 1, 0), channels, illumination)
    assert result.header.status == status
    assert result.body is None


def test_sun_channel_calibration(native, sun_config):
    config = replace(
        sun_config, channel_offset=(1, 2, 3, 4), channel_gain=(2, 3, 4, 5),
    )
    signal = channels_for(math.radians(15))
    channels = [
        value * gain + offset for value, gain, offset
        in zip(signal, config.channel_gain, config.channel_offset)
    ]
    result = native.sun(config, SampleHeader(0, 1, 0), channels, Illumination.DIRECT_SUN)
    np.testing.assert_allclose(result.body, [math.sin(math.radians(15)), 0, math.cos(math.radians(15))])


@pytest.mark.parametrize("changes", [
    {"active_length_m": (0, 0.01)},
    {"mask_height_m": (0.002, -0.002)},
    {"channel_gain": (1, 1, 0, 1)},
    {"minimum_signal": (0, 0.1)},
    {"half_fov_rad": FOV + 1e-7},
    {"half_fov_rad": math.nan},
    {"sensor_to_body": (1, 0, 0, 0, 1, 0, 0, 0, -1)},
    {"sensor_to_body": (1,) * 9},
])
def test_bad_or_unqualified_sun_config_is_explicit(native, sun_config, changes):
    with pytest.raises(ValueError, match="configuration"):
        native.sun(
            replace(sun_config, **changes), SampleHeader(0, 1, 0),
            (1,) * 4, Illumination.DIRECT_SUN,
        )


def test_gyro_temperature_polynomial_gain_and_mounting(native, gyro_config):
    config = replace(
        gyro_config, bias_polynomial_rad_s=(0.1, 0.01, 0.001, -0.1, 0, 0, 0.2, 0, 0),
        gain=(2, 1, 0.5), sensor_to_body=ROTATION,
    )
    result = native.gyro(config, SampleHeader(100, 2, 0), (1.3, 0.9, 0.7), 30)
    assert result.header.status == Status.OK
    np.testing.assert_allclose(result.body, [-1, 0.5, 1], atol=1e-12)


@pytest.mark.parametrize("rate,temperature,status", [
    ((35, 0, 0), 20, Status.SATURATED),
    ((-35, 0, 0), 20, Status.SATURATED),
    ((1, 2, 3), 41, Status.UNQUALIFIED),
    ((1, 2, 3), -1, Status.UNQUALIFIED),
    ((math.inf, 0, 0), 20, Status.INVALID_INPUT),
    ((1, 2, 3), math.nan, Status.INVALID_INPUT),
])
def test_gyro_fault_boundaries(native, gyro_config, rate, temperature, status):
    result = native.gyro(gyro_config, SampleHeader(100, 2, 0), rate, temperature)
    assert result.header.status == status
    assert result.body is None


@pytest.mark.parametrize("changes", [
    {"gain": (1, 0, 1)}, {"maximum_raw_rate_rad_s": 0},
    {"minimum_temperature_c": 21}, {"bias_polynomial_rad_s": (math.nan,) * 9},
    {"minimum_temperature_c": -274},
    {"sensor_to_body": (-1, 0, 0, 0, 1, 0, 0, 0, 1)},
])
def test_invalid_gyro_calibration(native, gyro_config, changes):
    with pytest.raises(ValueError, match="configuration"):
        native.gyro(replace(gyro_config, **changes), SampleHeader(0, 2, 0), (1, 2, 3), 20)


def test_driver_fault_preserved_and_payload_hidden(native, sun_config, gyro_config):
    header = SampleHeader(100, 1, 0, Status.DRIVER_FAULT | Status.SATURATED)
    for result in (
        native.sun(sun_config, header, (1,) * 4, Illumination.DIRECT_SUN),
        native.gyro(gyro_config, header, (1, 2, 3), 20),
    ):
        assert result.header == header
        assert result.body is None


def reading(kind, sensor_id=1, timestamp=100, status=Status.OK):
    return Measurement(
        SampleHeader(timestamp, sensor_id, 0, status), kind,
        (0.0, 0.0, 1.0) if status == Status.OK else None,
    )


def test_adcs_receives_segregated_processors_without_estimating_attitude(native, sun_config, gyro_config):
    inputs = AdcsInputs(native)
    sun = native.sun(sun_config, SampleHeader(100, 1, 0), (1,) * 4, Illumination.DIRECT_SUN)
    gyro = native.gyro(gyro_config, SampleHeader(100, 2, 0), (1, 2, 3), 20)
    assert inputs.submit(sun) == inputs.submit(gyro) == Status.OK
    assert inputs.submit(reading(SensorKind.SUN, 3)) == Status.OK
    health = inputs.health(100, 50, 50)
    assert health.valid_sun_count == 2
    assert health.valid_gyro_count == 1
    assert health.instantaneous_reference_rank == 2
    assert not health.attitude_available
    assert health.status == Status.OK


def test_empty_core_does_not_invent_attitude(native):
    health = AdcsInputs(native).health(100, 1, 1)
    assert health.valid_sun_count == health.valid_gyro_count == 0
    assert health.instantaneous_reference_rank == 0
    assert not health.attitude_available


def test_freshness_threshold_and_time_order(native):
    inputs = AdcsInputs(native)
    assert inputs.submit(reading(SensorKind.SUN)) == Status.OK
    assert inputs.health(150, 50, 50).valid_sun_count == 1
    health = inputs.health(151, 50, 50)
    assert health.status == Status.STALE
    assert health.stale_count == health.invalid_count == 1
    assert health.valid_sun_count == 0
    health = inputs.health(99, 50, 50)
    assert health.status == Status.TIME_ORDER
    assert health.future_count == 1
    assert inputs.submit(reading(SensorKind.SUN, timestamp=99)) == Status.TIME_ORDER
    assert inputs.submit(reading(SensorKind.SUN)) == Status.TIME_ORDER
    assert inputs.health(100, 50, 50).valid_sun_count == 1


def test_new_fault_invalidates_old_measurement_but_not_redundant_sensor(native):
    inputs = AdcsInputs(native)
    inputs.submit(reading(SensorKind.SUN, 1))
    inputs.submit(reading(SensorKind.SUN, 2))
    assert inputs.submit(reading(SensorKind.SUN, 1, 101, Status.DRIVER_FAULT)) == Status.DRIVER_FAULT
    health = inputs.health(101, 50, 50)
    assert health.valid_sun_count == 1
    assert health.invalid_count == 1
    assert health.status == Status.DRIVER_FAULT
    assert inputs.submit(reading(SensorKind.SUN, 1, 102)) == Status.OK
    assert inputs.health(102, 50, 50).valid_sun_count == 2


@pytest.mark.parametrize("kind", [
    SensorKind.COARSE_SUN, SensorKind.IMU, SensorKind.HORIZON,
])
def test_unqualified_models_cannot_be_enabled(native, kind):
    inputs = AdcsInputs(native)
    assert inputs.submit(reading(kind)) == Status.UNQUALIFIED
    assert inputs.health(100, 50, 50).valid_sun_count == 0


def test_sensor_ids_capacity_and_invalid_external_sun(native):
    inputs = AdcsInputs(native)
    for sensor_id in range(16):
        assert inputs.submit(reading(SensorKind.SUN, sensor_id)) == Status.OK
    assert inputs.submit(reading(SensorKind.GYRO, 1, 101)) == Status.INVALID_INPUT
    assert inputs.submit(reading(SensorKind.SUN, 16)) == Status.CAPACITY
    bad = Measurement(SampleHeader(101, 1, 1), SensorKind.SUN, (0, 0, 0))
    assert inputs.submit(bad) == Status.INVALID_INPUT
    assert inputs.health(101, 50, 50).valid_sun_count == 15
    assert inputs.submit(reading(SensorKind.GPS, 2, 102)) == Status.INVALID_INPUT


def test_config_fault_is_not_mistaken_for_bad_health_age(native):
    inputs = AdcsInputs(native)
    assert inputs.submit(reading(SensorKind.SUN, status=Status.INVALID_CONFIG)) == Status.INVALID_CONFIG
    assert inputs.health(100, 50, 50).invalid_count == 1
    with pytest.raises(ValueError, match="greater than zero"):
        inputs.health(100, 0, 50)


@pytest.mark.parametrize("changes", [
    {"timestamp_ns": -1}, {"timestamp_ns": 2**64}, {"sensor_id": 2**32},
    {"sequence": -1}, {"sensor_id": True},
    {"status": 1.5},
])
def test_metadata_not_silently_truncated(changes):
    with pytest.raises(ValueError, match="unsigned"):
        SampleHeader(**dict({"timestamp_ns": 0, "sensor_id": 1, "sequence": 0}, **changes))


def test_invalid_python_payload_and_library_path(native):
    with pytest.raises(ValueError, match="usable vector"):
        Measurement(SampleHeader(0, 1, 0, Status.DRIVER_FAULT), SensorKind.SUN, (1, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        Measurement(SampleHeader(0, 1, 0), SensorKind.GYRO, (math.nan, 0, 0))
    with pytest.raises(FileNotFoundError):
        NativeProcessing(native.library_path.parent / "missing.dll")


def test_c_null_pointer_contracts(native):
    library = native._library
    assert library.openadcs_sun_process(None, None, None) == Status.INVALID_INPUT
    assert library.openadcs_gyro_process(None, None, None) == Status.INVALID_INPUT
    assert library.openadcs_accept(None, SensorKind.SUN, None) == Status.INVALID_INPUT
    assert library.openadcs_health(None, 0, 1, 1, None) == Status.INVALID_INPUT
    library.openadcs_init(None)


def test_scalar_array_shape_validation(native, sun_config):
    with pytest.raises(ValueError, match="exactly 4"):
        native.sun(sun_config, SampleHeader(0, 1, 0), (1, 2), Illumination.DIRECT_SUN)


def test_enum_values_cannot_wrap_into_qualified_states(native, sun_config):
    with pytest.raises(ValueError, match="Illumination"):
        native.sun(sun_config, SampleHeader(0, 1, 0), (1,) * 4, 2**32 + 1)
    with pytest.raises(ValueError, match="SensorKind"):
        Measurement(SampleHeader(0, 1, 0), 2**32 + 1, (0, 0, 1))

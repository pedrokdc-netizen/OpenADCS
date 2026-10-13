from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
import sys

import numpy as np
import pytest
from astropy.time import Time

from openadcs_sim.common import vector3
from openadcs_sim.orbit import OrbitState
from openadcs_sim.orion_b16 import (
    OrionB16, OrionConfig, ReceiverConditions, navigation_frame,
)
from openadcs_sim.sensor_processing import AdcsInputs, SampleHeader, Status


EPOCH = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


@pytest.fixture
def config():
    return OrionConfig(3.0, 0.1, 10.0, 2.0, 42)


@pytest.fixture
def conditions():
    return ReceiverConditions(8, -130, 7800, 9, 25.0, (2.0, 1.7, 1, 1.4, 1))


def truth(index, rate=1, epoch=EPOCH):
    return OrbitState(
        epoch + timedelta(seconds=index / rate),
        vector3([6778137, 0, 0]), vector3([0, 7000, -20]),
    )


def acquired(config, conditions):
    return OrionB16(replace(config, cold_start_s=0), EPOCH).sample(truth(0), conditions)


@pytest.mark.parametrize("mode,seconds", [("hot", 18), ("warm", 46), ("cold", 50)])
def test_spec_startup_threshold(config, conditions, mode, seconds):
    model = OrionB16(replace(config, start_mode=mode), EPOCH)
    for index in range(seconds):
        result = model.sample(truth(index), conditions)
        assert result.phase == "ACQUIRING"
        assert result.measurement.header.status == Status.NO_FIX
        assert result.measurement.position_ecef_m is None
        assert result.measurement.gps_week == result.measurement.tow_s == 0
    result = model.sample(truth(seconds), conditions)
    assert result.phase == "TRACKING"
    assert result.measurement.header.status == Status.OK


def test_reacquisition_and_tracking_thresholds(config, conditions):
    model = OrionB16(replace(config, cold_start_s=0), EPOCH)
    assert model.sample(truth(0), conditions).phase == "TRACKING"
    assert model.sample(truth(1), replace(conditions, signal_dbm=-165)).phase == "TRACKING"
    assert model.sample(truth(2), replace(conditions, signal_dbm=-165.01)).phase == "REACQUIRING"
    assert model.sample(truth(3), replace(conditions, signal_dbm=-164)).phase == "REACQUIRING"
    for index in range(4, 20):
        assert model.sample(truth(index), replace(conditions, signal_dbm=-160)).phase == "REACQUIRING"
    assert model.sample(truth(20), replace(conditions, signal_dbm=-160)).phase == "TRACKING"


def test_acquisition_dwell_resets_on_signal_loss(config, conditions):
    model = OrionB16(replace(config, cold_start_s=2), EPOCH)
    model.sample(truth(0), conditions)
    model.sample(truth(1), replace(conditions, satellites_available=3))
    assert model.sample(truth(2), conditions).phase == "ACQUIRING"
    assert model.sample(truth(3), conditions).phase == "ACQUIRING"
    assert model.sample(truth(4), conditions).phase == "TRACKING"


def test_power_cycle_restarts_selected_mode(config, conditions):
    model = OrionB16(replace(config, cold_start_s=1), EPOCH)
    model.sample(truth(0), conditions)
    model.sample(truth(1), conditions)
    assert model.sample(truth(2), replace(conditions, power_on=False)).phase == "POWERED_OFF"
    assert model.sample(truth(3), conditions).phase == "ACQUIRING"
    assert model.sample(truth(4), conditions).phase == "TRACKING"


def test_power_off_and_out_of_domain_do_not_emit_wire_packets(config, conditions):
    off = acquired(config, replace(conditions, power_on=False))
    outside = acquired(config, replace(conditions, supply_voltage_v=0))
    assert off.navigation_output() is None
    assert outside.navigation_output() is None


@pytest.mark.parametrize("changes", [
    {"supply_voltage_v": 2.99}, {"supply_voltage_v": 3.61},
    {"temperature_c": -40.01}, {"temperature_c": 85.01},
    {"inertial_speed_m_s": 10000.01}, {"acceleration_m_s2": 39.20001},
])
def test_out_of_datasheet_domain_no_fake_solution(config, conditions, changes):
    result = acquired(config, replace(conditions, **changes))
    assert result.phase == "OUT_OF_DOMAIN"
    assert result.measurement.header.status == Status.NO_FIX | Status.UNQUALIFIED
    assert result.measurement.position_ecef_m is None


def test_domain_edges_and_cold_signal_edge(config, conditions):
    result = acquired(config, replace(
        conditions, supply_voltage_v=3.0, temperature_c=-40,
        inertial_speed_m_s=10000, acceleration_m_s2=39.2, signal_dbm=-148,
    ))
    assert result.measurement.header.status == Status.OK
    assert acquired(config, replace(conditions, signal_dbm=-148.01)).phase == "ACQUIRING"


@pytest.mark.parametrize("rate", [1, 2, 4, 5, 8, 10, 20])
def test_tick_cadence(config, conditions, rate):
    model = OrionB16(replace(config, update_rate_hz=rate), EPOCH)
    for index in range(3):
        result = model.sample(truth(index, rate), conditions)
        assert result.measurement.header.timestamp_ns == index * (1_000_000_000 // rate)
    with pytest.raises(ValueError, match="tick"):
        model.sample(truth(5, rate), conditions)


def test_gaussian_cep_and_user_sigmas(config):
    model = OrionB16(replace(config, position_correlation_s=0, velocity_correlation_s=0), EPOCH)
    errors = [model._errors() for _ in range(20000)]
    position = np.array([p for p, _ in errors])
    velocity = np.array([v for _, v in errors])
    assert np.median(np.linalg.norm(position[:, :2], axis=1)) == pytest.approx(2.0, rel=0.025)
    assert np.std(position[:, 2]) == pytest.approx(3.0, rel=0.025)
    np.testing.assert_allclose(np.std(velocity, axis=0), 0.1, rtol=0.025)


def test_correlation_and_reproducibility(config, conditions):
    model = OrionB16(config, EPOCH)
    values = np.array([model._errors()[0] for _ in range(12000)])
    observed = np.corrcoef(values[:-1, 0], values[1:, 0])[0, 1]
    assert observed == pytest.approx(math.exp(-0.1), abs=0.02)
    first = acquired(config, conditions)
    second = acquired(config, conditions)
    assert first == second
    assert config.provenance()["flight_validated"] is False


def test_binary_round_trip_and_native_adcs_health(native, config, conditions):
    original = acquired(config, conditions).measurement
    frame = navigation_frame(original)
    assert len(frame) == 66
    decoded = native.gps_decode(frame, original.header)
    assert decoded.header.status == Status.OK
    np.testing.assert_allclose(decoded.position_ecef_m, original.position_ecef_m, atol=0.0051, rtol=0)
    np.testing.assert_allclose(decoded.velocity_ecef_m_s, original.velocity_ecef_m_s, atol=0.0051, rtol=0)
    assert decoded.ellipsoid_height_m - decoded.msl_height_m == pytest.approx(25, abs=0.011)
    inputs = AdcsInputs(native, maximum_gps_age_ns=1_000_000_000)
    assert inputs.submit(decoded) == Status.OK
    health = inputs.health(0, 1, 1)
    assert health.navigation_available and health.valid_gps_count == 1
    assert not health.attitude_available and health.instantaneous_reference_rank == 0
    assert inputs.health(1_000_000_001, 1, 1).status == Status.STALE
    assert inputs.submit(decoded) == Status.TIME_ORDER


def test_model_errors_are_rotated_from_enu_not_added_in_ecef(config, conditions, monkeypatch):
    model = OrionB16(replace(config, cold_start_s=0), EPOCH)
    monkeypatch.setattr(model, "_errors", lambda: (np.array([1, 2, 3]), np.array([4, 5, 6])))
    measured = model.sample(truth(0), conditions).measurement
    np.testing.assert_allclose(measured.position_ecef_m, [6778140, 1, 2])
    np.testing.assert_allclose(measured.velocity_ecef_m_s, [6, 7004, -15])


def test_native_gps_capacity_identity_and_redundancy(native, config, conditions):
    pvt = acquired(config, conditions).measurement
    inputs = AdcsInputs(native, maximum_gps_age_ns=1_000_000_000)
    for sensor_id in range(100, 104):
        assert inputs.submit(replace(pvt, header=SampleHeader(0, sensor_id, 0))) == Status.OK
    assert inputs.submit(replace(pvt, header=SampleHeader(0, 104, 0))) == Status.CAPACITY
    fault = replace(
        pvt, header=SampleHeader(1, 100, 1, Status.NO_FIX),
        position_ecef_m=None, velocity_ecef_m_s=None, latitude_deg=None,
        longitude_deg=None, ellipsoid_height_m=None, msl_height_m=None, dop=None,
    )
    assert inputs.submit(fault) == Status.NO_FIX
    health = inputs.health(1, 1, 1)
    assert health.valid_gps_count == 3 and health.invalid_count == 1
    assert health.navigation_available and not health.attitude_available
    from openadcs_sim.sensor_processing import Measurement, SensorKind
    assert inputs.submit(Measurement(SampleHeader(1, 101, 1), SensorKind.SUN, (0, 0, 1))) == Status.INVALID_INPUT
    other = AdcsInputs(native, maximum_gps_age_ns=1)
    other.submit(Measurement(SampleHeader(0, 100, 0), SensorKind.SUN, (0, 0, 1)))
    assert other.submit(pvt) == Status.INVALID_INPUT


def test_independent_documented_vendor_frame(native):
    # AN0037 navigation-message example; independent of our encoder.
    frame = bytes.fromhex(
        "A0 A1 00 3B A8 02 08 06 04 02 32 18 18 0E C5 E1 99 48 20 78 ED "
        "00 00 2E 3B 00 00 26 93 00 93 00 93 00 93 00 93 00 93 "
        "EE 35 4D 30 1D 99 AA 37 0F D7 0B 74 "
        "00 00 00 00 00 00 00 00 00 00 00 00 F5 0D 0A"
    )
    decoded = native.gps_decode(frame, SampleHeader(123, 100, 7))
    assert decoded.header.status == Status.OK
    assert decoded.gps_week == 1540
    assert decoded.position_ecef_m[0] == pytest.approx(-2984967.20)
    assert decoded.ellipsoid_height_m == pytest.approx(118.35)
    assert decoded.satellites_used == 8


@pytest.mark.parametrize("index", [0, 2, 4, 20, 63, 65])
def test_corrupt_frames_rejected(native, config, conditions, index):
    original = acquired(config, conditions).measurement
    frame = bytearray(navigation_frame(original))
    frame[index] ^= 1
    decoded = native.gps_decode(bytes(frame), original.header)
    assert decoded.header.status == Status.PROTOCOL_ERROR
    assert decoded.position_ecef_m is None


@pytest.mark.parametrize("field", ["fix", "satellites", "tow", "latitude", "origin"])
def test_checksum_valid_but_semantically_invalid_frame(native, config, conditions, field):
    original = acquired(config, conditions).measurement
    frame = bytearray(navigation_frame(original))
    if field == "fix":
        frame[5] = 4
    elif field == "satellites":
        frame[6] = 3
    elif field == "tow":
        frame[9:13] = (60_480_000).to_bytes(4, "big")
    elif field == "latitude":
        frame[13:17] = (910_000_000).to_bytes(4, "big")
    else:
        frame[39:51] = bytes(12)
    checksum = 0
    for byte in frame[4:63]:
        checksum ^= byte
    frame[63] = checksum
    result = native.gps_decode(bytes(frame), original.header)
    assert result.header.status == Status.PROTOCOL_ERROR
    assert result.position_ecef_m is None


def test_2d_fix_is_not_a_valid_3d_navigation_solution(native, config, conditions):
    original = acquired(config, conditions).measurement
    frame = bytearray(navigation_frame(original))
    frame[5] = 1
    frame[63] ^= 2 ^ 1
    result = native.gps_decode(bytes(frame), original.header)
    assert result.fix_mode == 1 and result.header.status == Status.NO_FIX
    assert result.velocity_ecef_m_s is None


def test_no_fix_protocol_invalidates_previous_solution(native, config, conditions):
    model = OrionB16(replace(config, cold_start_s=0), EPOCH)
    valid = model.sample(truth(0), conditions).measurement
    no_fix = model.sample(truth(1), replace(conditions, antenna_available=False)).measurement
    inputs = AdcsInputs(native, maximum_gps_age_ns=2_000_000_000)
    inputs.submit(valid)
    decoded = native.gps_decode(navigation_frame(no_fix), SampleHeader(1_000_000_000, 100, 1))
    assert decoded.header.status == Status.NO_FIX
    assert inputs.submit(decoded) == Status.NO_FIX
    assert not inputs.health(1_000_000_000, 1, 1).navigation_available
    with pytest.raises(ValueError, match="maximum_gps_age"):
        AdcsInputs(native).submit(valid)
    assert native.gps_decode(b"", valid.header).header.status == Status.PROTOCOL_ERROR


def test_gps_week_rollover_and_clock_bias(native, config, conditions):
    epoch = Time(2440 * 604800 - 0.001, format="gps").utc.to_datetime(timezone=timezone.utc)
    model = OrionB16(replace(config, cold_start_s=0, clock_bias_s=0.002), epoch)
    result = model.sample(truth(0, epoch=epoch), conditions)
    assert result.measurement.gps_week == 2440
    assert result.measurement.tow_s == pytest.approx(0.001, abs=1e-6)
    decoded = native.gps_decode(navigation_frame(result.measurement), result.measurement.header)
    assert decoded.gps_week == 2440 and decoded.tow_s == 0


@pytest.mark.parametrize("changes", [
    {"vertical_sigma_m": -1}, {"velocity_sigma_m_s": math.nan},
    {"update_rate_hz": 3}, {"update_rate_hz": True}, {"seed": -1},
    {"start_mode": "unknown"}, {"position_correlation_s": math.inf},
])
def test_invalid_receiver_configuration(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_cli_exports_orbit_driven_model(tmp_path, monkeypatch):
    from openadcs_sim import gps_cli
    from openadcs_sim.scenario import training_tle
    tle = tmp_path / "training.tle"
    tle.write_text(training_tle(), encoding="ascii")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "receiver": {
            "vertical_sigma_m": 3, "velocity_sigma_m_s": 0.1,
            "position_correlation_s": 10, "velocity_correlation_s": 2, "seed": 42,
            "update_rate_hz": 2, "cold_start_s": 0,
        },
        "conditions": {
            "satellites_available": 8, "signal_dbm": -130,
            "inertial_speed_m_s": 0, "acceleration_m_s2": 9,
            "geoid_undulation_m": 0, "dop": [2, 1.7, 1, 1.4, 1],
        },
    }), encoding="utf-8")
    output, binary = tmp_path / "gps.jsonl", tmp_path / "gps.bin"
    monkeypatch.setattr(sys, "argv", [
        "gps_cli", "--tle", str(tle), "--profile", str(profile),
        "--start", EPOCH.isoformat(), "--samples", "3",
        "--output", str(output), "--binary-output", str(binary),
    ])
    gps_cli.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 4
    assert records[0]["provenance"]["flight_validated"] is False
    assert records[1]["measurement"]["header"]["status"] == 0
    assert 7000 < records[1]["conditions"]["inertial_speed_m_s"] < 10000
    assert len(binary.read_bytes()) == 132
    assert records[1]["truth_position_ecef_m"] != records[1]["measurement"]["position_ecef_m"]

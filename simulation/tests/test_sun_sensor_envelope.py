import math

import numpy as np
import pytest

from openadcs_sim import sun_sensor_array as ssa
from openadcs_sim.black_sun_sensor import rotation_from_vector
from openadcs_sim.sun_sensor_envelope import EnvelopeSpec, run_envelope


def test_rotation_from_vector():
    r = rotation_from_vector([0, 0, math.radians(90)])
    assert r @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)
    assert np.allclose(rotation_from_vector([0, 0, 0]), np.eye(3))


def test_mount_error_produces_matching_fused_error():
    errors = [[0, 0, 0]] * 6
    errors = [[0.0, 0.5, 0.0]] * 6
    config = ssa.SunSensorArrayConfig(1e-6, albedo_coefficient=0, mount_errors_deg=errors)
    sun = np.array([0.0, 0.0, 1.0])
    error = ssa.measure_once(config, sun).error_deg
    assert 0.4 < error < 0.6


def test_mount_sigma_is_seeded_and_zero_by_default():
    base = ssa.SunSensorArray(ssa.SunSensorArrayConfig(0.1)).mount_errors_rad()
    assert np.all(base == 0)
    a = ssa.SunSensorArray(ssa.SunSensorArrayConfig(0.1, mount_sigma_deg=1.0, seed=3)).mount_errors_rad()
    b = ssa.SunSensorArray(ssa.SunSensorArrayConfig(0.1, mount_sigma_deg=1.0, seed=3)).mount_errors_rad()
    assert np.array_equal(a, b) and a.std() > 0


def test_centroid_bias_shifts_fused_vector():
    config = ssa.SunSensorArrayConfig(1e-6, albedo_coefficient=0, centroid_bias_px=(2.0, 0.0))
    assert ssa.measure_once(config, [0, 0, 1]).error_deg > 0.05


def test_envelope_small_run():
    report = run_envelope(EnvelopeSpec(sun_directions=40, monte_carlo_runs=3))
    kinds = [c["kind"] for c in report["cases"]]
    assert kinds.count("baseline") == 1 and kinds.count("corner") == 256 and kinds.count("monte_carlo") == 3
    summary = report["summary"]
    assert summary["baseline"]["availability"] == 1.0
    assert summary["corner"]["worst_rms_deg"] > summary["baseline"]["rms_deg"]
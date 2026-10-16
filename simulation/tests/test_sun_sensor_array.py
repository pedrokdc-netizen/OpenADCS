import json
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from openadcs_sim import EnvironmentSimulator, XYZGeomagModel
from openadcs_sim import sun_sensor_array as ssa
from openadcs_sim.attitude import spacecraft_view
from openadcs_sim.common import vector3
from openadcs_sim.orbit import OrbitState
from pathlib import Path

from openadcs_sim.orbit import SGP4Propagator
from openadcs_sim.scenario import KERNEL_NAMES, Scenario, baseline_tle, decode_sample, export_results
from openadcs_sim.sun import SunState

NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


def random_directions(n, seed=3):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_120_deg_cube_covers_the_whole_sphere_with_overlap():
    array = ssa.SunSensorArray(ssa.SunSensorArrayConfig(1e-6, 120.0))
    counts = np.array([array.measure(float(k), d).valid_heads for k, d in enumerate(random_directions(3000))])
    assert counts.min() >= 1
    assert counts.max() <= 3
    assert 0.10 < np.mean(counts >= 2) < 0.30   # about 19 % seen by two or more heads


def test_default_cube_is_120_deg():
    assert ssa.coverage_fraction(ssa.SunSensorArrayConfig(1e-6), n=2000)["at_least_one"] == 1.0


def test_standard_68_deg_lens_leaves_blind_regions():
    result = ssa.coverage_fraction(ssa.SunSensorArrayConfig(1e-6, 68.7), n=3000)
    assert 0.25 < result["at_least_one"] < 0.5


def test_coverage_grid_matches_the_per_head_model():
    cfg = ssa.SunSensorArrayConfig(1e-6, 120.0)
    az, el, count = ssa.coverage_grid(cfg, 36, 18)
    array = ssa.SunSensorArray(cfg)
    for n, (i, j) in enumerate(((3, 4), (10, 2), (15, 20), (17, 30), (0, 17))):
        a, e = math.radians(az[j]), math.radians(el[i])
        d = [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
        assert array.measure(float(n), d).valid_heads == count[i, j]
    assert count.min() >= 1


def test_fused_error_scales_with_centroid_sigma_and_stays_small():
    directions = random_directions(400, seed=5)

    def rms(sigma):
        array = ssa.SunSensorArray(ssa.SunSensorArrayConfig(sigma, 120.0, seed=7))
        errors = [array.measure(float(k), d).error_deg for k, d in enumerate(directions)]
        return math.sqrt(np.mean(np.square(errors)))

    perfect, fine, coarse = rms(1e-6), rms(0.1), rms(1.0)
    assert perfect < 1e-5
    assert 0.001 < fine < 0.05
    assert 5 < coarse / fine < 20


def test_eclipse_gives_no_valid_heads_and_no_error():
    m = ssa.measure_once(ssa.SunSensorArrayConfig(), [0, 0, 1], sunlit=False)
    assert m.valid_heads == 0 and m.error_deg is None
    assert not m.fused.valid
    assert {r.status for r in m.heads} == {"ECLIPSE"}


def test_measurement_json_round_trip_with_missing_values():
    for sunlit in (True, False):
        m = ssa.measure_once(ssa.SunSensorArrayConfig(0.3), [0.3, -0.5, 0.8], sunlit)
        data = json.loads(json.dumps(m.to_dict(), allow_nan=False))
        back = ssa.SunSensorMeasurement.from_dict(data)
        assert back.valid_heads == m.valid_heads
        assert back.sunlit == sunlit
        assert [r.status for r in back.heads] == [r.status for r in m.heads]
        assert back.error_deg == pytest.approx(m.error_deg) if m.error_deg is not None else back.error_deg is None


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        ssa.SunSensorArrayConfig(centroid_sigma_px=-1)
    with pytest.raises(ValueError):
        ssa.SunSensorArrayConfig(diagonal_fov_deg=180)


class Orbit:
    def propagate(self, timestamp):
        return OrbitState(timestamp, vector3([6878137, 0, 0]), vector3([0, 7600, 0]))


class Sun:
    def __init__(self, direction):
        self.direction = vector3(direction)

    def sample(self, timestamp, spacecraft):
        return SunState(self.direction, 1.5e11)


def test_environment_runs_the_six_heads_from_the_body_frame_sun_vector():
    plain = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([0, 0, 1])).sample(NOW)
    assert plain.sun_sensor is None and "sun_sensor" not in plain.to_dict()

    sensor = ssa.SunSensorArray(ssa.SunSensorArrayConfig(0.1))
    sample = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([0, 0, 1]), sensor).sample(NOW)
    expected = spacecraft_view(sample, None).sun_body_direction
    assert sample.sun_sensor.sun_true_body == pytest.approx(expected)
    assert sample.sun_sensor.sunlit
    assert sample.sun_sensor.valid_heads >= 1
    assert sample.sun_sensor.error_deg < 0.1
    assert sample.sun_direction_itrs == pytest.approx(plain.sun_direction_itrs)
    assert json.loads(json.dumps(sample.to_dict(), allow_nan=False))["sun_sensor"]["valid_heads"] >= 1


def test_environment_marks_eclipse_when_the_earth_is_between():
    sensor = ssa.SunSensorArray()
    sample = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([-1, 0, 0]), sensor).sample(NOW)
    assert not sample.sun_sensor.sunlit and sample.sun_sensor.valid_heads == 0


def test_csv_export_adds_sensor_columns_only_when_present(tmp_path):
    with_sensor = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([0, 0, 1]), ssa.SunSensorArray()).sample(NOW)
    without = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([0, 0, 1])).sample(NOW)
    on, off = tmp_path / "on.csv", tmp_path / "off.csv"
    kernels = Path(__file__).resolve().parents[1] / "data" / "kernels"
    if not (kernels / KERNEL_NAMES[0]).is_file():
        pytest.skip("Download kernels first")
    tle = baseline_tle().splitlines()[-2:]
    scenario = Scenario((tle[0], tle[1]), kernels, SGP4Propagator(*tle).epoch_utc, 0.0, 1.0)
    export_results(on, scenario, [with_sensor], "complete")
    export_results(off, scenario, [without], "complete")
    assert "sun_sensor_error_deg" in on.read_text(encoding="utf-8").splitlines()[0]
    assert "sun_sensor" not in off.read_text(encoding="utf-8")
    again = decode_sample(json.dumps(with_sensor.to_dict()))
    assert again.sun_sensor.valid_heads == with_sensor.sun_sensor.valid_heads


def test_run_history_tracks_eclipse_and_decimates():
    from datetime import timedelta
    from dataclasses import replace

    base = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([0, 0, 1])).sample(NOW)
    samples = [replace(base, time_utc=NOW + timedelta(seconds=10 * k)) for k in range(30)]
    samples += [replace(base, time_utc=NOW + timedelta(seconds=300 + 10 * k),
                        sun_direction_itrs=vector3([-1, 0, 0])) for k in range(10)]
    history = ssa.run_history(samples, lambda s: spacecraft_view(s, None), ssa.SunSensorArrayConfig(), step_s=60)
    assert len(history.t_min) == 7
    assert history.sunlit.sum() == 5 and (~history.sunlit).sum() == 2
    assert np.all(np.isnan(history.error_deg[~history.sunlit]))
    summary = history.summary()
    assert summary["availability_sunlit"] == 1.0 and summary["rms_deg"] < 0.1


def test_run_history_can_be_cancelled():
    base = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun([0, 0, 1])).sample(NOW)
    history = ssa.run_history([base], lambda s: spacecraft_view(s, None), ssa.SunSensorArrayConfig(),
                              cancelled=lambda: True)
    assert len(history.t_min) == 0

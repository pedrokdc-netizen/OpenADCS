import dataclasses
import math

import numpy as np
import pytest

from openadcs_sim.horizon_camera import AtmosphereConfig, CameraConfig, CameraMount
from openadcs_sim.horizon_montecarlo import (extract_error_law, no_scene_variability, run_trials,
                                             summarise)
from openadcs_sim.horizon_processing import _tangent_basis
from openadcs_sim.horizon_sensor import (DEFAULT_ERROR_LAWS, Esp32Config, HorizonArchitecture,
                                         HorizonErrorLaw, HorizonSensorConfig, HorizonSensorModel,
                                         fuse_nadir_estimates, nadir_from_roll_pitch,
                                         roll_pitch_covariance, roll_pitch_from_nadir)

CLEAN = AtmosphereConfig(enabled=False)
N0 = np.array([0.0, 0.0, 1.0])


def config(arch, **kw):
    return HorizonSensorConfig(architecture=arch, atmosphere=CLEAN, **kw)


def test_roll_pitch_round_trip():
    for roll, pitch in ((0.0, 0.0), (3.0, -2.0), (-10.0, 20.0)):
        n = nadir_from_roll_pitch(math.radians(roll), math.radians(pitch))
        assert np.degrees(roll_pitch_from_nadir(n)) == pytest.approx([roll, pitch], abs=1e-9)


def test_timing_budget_defaults():
    for arch in HorizonArchitecture:
        t = HorizonSensorConfig(architecture=arch).timing()
        assert t.output_rate_hz == pytest.approx(10.0)
        assert t.processing_s < 0.02 and t.cpu_load < 0.5          # ESP32 budget (ASSUMPTION costs)
        assert t.readout_s == pytest.approx(9920 * 8 / 10e6 * 1.3, rel=1e-6)
        assert t.latency_s < 0.2
    single = HorizonSensorConfig(architecture=HorizonArchitecture.SINGLE).timing()
    fused = HorizonSensorConfig(architecture=HorizonArchitecture.FUSED).timing()
    assert fused.latency_s > single.latency_s + 0.04               # 9.9 kB over a 2 Mbaud UART
    assert fused.transfer_s == pytest.approx(9920 * 10 / 2e6)


def test_averaging_lowers_output_rate_and_rate_limit():
    fast = HorizonSensorConfig(camera=CameraConfig(fps=10.0, frames_averaged=1)).timing()
    slow = HorizonSensorConfig(camera=CameraConfig(fps=10.0, frames_averaged=4)).timing()
    assert slow.output_rate_hz == pytest.approx(2.5)
    assert slow.max_rate_dps < fast.max_rate_dps
    assert slow.latency_s > fast.latency_s


def test_slow_link_can_limit_the_fused_rate():
    slow_link = Esp32Config(uart_baud=115200.0)
    t = HorizonSensorConfig(architecture=HorizonArchitecture.FUSED, esp32=slow_link).timing()
    assert t.output_rate_hz < 10.0


def test_architecture_and_mode_validation():
    with pytest.raises(ValueError):
        HorizonSensorConfig(mode="bogus")
    with pytest.raises(ValueError):
        HorizonSensorConfig(architecture=HorizonArchitecture.INDEPENDENT, mounts=(CameraMount(),))


@pytest.mark.parametrize("arch", list(HorizonArchitecture))
def test_image_mode_measures_nadir_without_scene_variability(arch):
    model = HorizonSensorModel(config(arch), seed=3)
    n = nadir_from_roll_pitch(math.radians(1.5), math.radians(-2.0))
    reading = model.measure_now(n, 6871.0, 10.0, 30.0)
    assert reading.valid and reading.status.startswith("OK")
    assert np.degrees(np.linalg.norm(roll_pitch_from_nadir(reading.n_body) - roll_pitch_from_nadir(n))) < 0.6
    assert reading.cov_roll_pitch_rad2.shape == (2, 2) and np.all(np.linalg.eigvalsh(reading.cov_roll_pitch_rad2) > 0)
    assert len(reading.cameras) == (1 if arch is HorizonArchitecture.SINGLE else 2)


def test_two_cameras_observe_both_axes_better_than_one():
    n = nadir_from_roll_pitch(math.radians(2.0), math.radians(-1.0))
    single = HorizonSensorModel(config(HorizonArchitecture.SINGLE), seed=5).measure_now(n, 6871.0, 0.0)
    both = HorizonSensorModel(config(HorizonArchitecture.INDEPENDENT), seed=5).measure_now(n, 6871.0, 0.0)
    assert np.sqrt(np.trace(both.cov_roll_pitch_rad2)) < 0.6 * np.sqrt(np.trace(single.cov_roll_pitch_rad2))


def test_fused_architecture_estimates_limb_height_gamma():
    model = HorizonSensorModel(config(HorizonArchitecture.FUSED), seed=2)
    reading = model.measure_now(nadir_from_roll_pitch(0.01, 0.02), 6871.0, 0.0)
    assert reading.valid and math.isfinite(reading.gamma_deg) and 65.0 < reading.gamma_deg < 73.0


def test_cold_polar_scene_degrades_availability():
    atm = AtmosphereConfig(warm_temperature_k=240.0, polar_drop_k=0.0, enabled=False)
    model = HorizonSensorModel(HorizonSensorConfig(architecture=HorizonArchitecture.SINGLE, atmosphere=atm), seed=1)
    reading = model.measure_now(N0, 6871.0, 80.0)
    assert not reading.valid and reading.status in ("TOO_FEW_POINTS", "NO_LIMB", "ARC_TOO_SHORT")


def test_rate_gate_rejects_fast_tumble():
    model = HorizonSensorModel(config(HorizonArchitecture.SINGLE), seed=1)
    reading = model.measure_now(N0, 6871.0, 0.0, rate_dps=model.timing.max_rate_dps * 2)
    assert not reading.valid and reading.status == "RATE_TOO_HIGH"


def test_step_applies_rate_limit_and_latency():
    model = HorizonSensorModel(config(HorizonArchitecture.SINGLE, mode="law"), seed=1)
    delivered = []
    for k in range(60):
        t = 0.01 * k
        reading = model.step(t, N0, 6871.0, 0.0)
        if reading is not None:
            delivered.append((t, reading))
    assert 5 <= len(delivered) <= 7                        # 10 Hz over 0.6 s
    for t, reading in delivered:
        assert reading.available_time_s <= t + 1e-9
        assert reading.available_time_s - reading.sample_time_s == pytest.approx(model.timing.latency_s)


def test_law_mode_is_seeded_and_matches_law_statistics():
    law = HorizonErrorLaw(bias_deg=(0.2, -0.1), sigma_white_deg=(0.3, 0.1), sigma_corr_deg=(0.0, 0.0),
                          tau_s=10.0, availability=1.0)
    cfg = HorizonSensorConfig(architecture=HorizonArchitecture.SINGLE, mode="law", error_law=law)
    def run(seed):
        m = HorizonSensorModel(cfg, seed=seed)
        return np.array([m.measure_now(N0, 6871.0, 0.0, t_s=float(k)).roll_pitch_deg for k in range(3000)])
    a, b = run(1), run(1)
    assert np.array_equal(a, b)
    assert a.mean(0) == pytest.approx([0.2, -0.1], abs=0.03)
    assert a.std(0) == pytest.approx([0.3, 0.1], rel=0.1)


def test_law_bands_select_by_latitude():
    law = HorizonErrorLaw(bias_deg=(0, 0), sigma_corr_deg=(1.0, 1.0),
                          bands=((30.0, 0, 0, 0.1, 0.1), (90.1, 0, 0, 2.0, 2.0)))
    assert law.at_latitude(10.0)[1] == pytest.approx([0.1, 0.1])
    assert law.at_latitude(-75.0)[1] == pytest.approx([2.0, 2.0])


def test_default_laws_are_populated_and_ordered_sensibly():
    for arch, law in DEFAULT_ERROR_LAWS.items():
        assert 0.0 < law.availability <= 1.0 and law.bands
        low, mid, high = (band[3] for band in law.bands)
        assert low < mid < high                              # worse near the poles (floor / clouds)
    assert DEFAULT_ERROR_LAWS[HorizonArchitecture.INDEPENDENT].availability > \
        DEFAULT_ERROR_LAWS[HorizonArchitecture.SINGLE].availability


def test_information_fusion_reduces_covariance_and_weights_by_confidence():
    n = N0
    basis = _tangent_basis(n)
    n1 = n + 0.01 * basis[:, 0]; n1 /= np.linalg.norm(n1)
    est1 = (n1, _tangent_basis(n1), np.diag([1e-4, 1e-2]))
    n2 = n - 0.01 * basis[:, 1]; n2 /= np.linalg.norm(n2)
    est2 = (n2, _tangent_basis(n2), np.diag([1e-2, 1e-4]))
    fused, fbasis, cov = fuse_nadir_estimates([est1, est2])
    assert np.trace(cov) < min(np.trace(est1[2]), np.trace(est2[2]))
    d1, d2 = basis.T @ fused
    assert abs(d1 - 0.01) < 0.002 and abs(d2 + 0.01) < 0.002


def test_roll_pitch_covariance_propagation():
    basis = _tangent_basis(N0)
    cov = roll_pitch_covariance(N0, basis, np.diag([4e-6, 1e-6]))
    assert np.sqrt(np.diag(cov)).max() == pytest.approx(2e-3, rel=0.05)


def test_monte_carlo_harness_and_law_extraction():
    result = run_trials(6, 500.0, 11, atmosphere=no_scene_variability(AtmosphereConfig()))
    for arch in HorizonArchitecture:
        assert result.valid_fraction[arch.value] > 0.5
    summary = summarise(result.errors_deg["independent"])
    law = extract_error_law(summary, summary, 0.9)
    assert isinstance(law, HorizonErrorLaw) and law.availability == 0.9
    assert dataclasses.is_dataclass(law)

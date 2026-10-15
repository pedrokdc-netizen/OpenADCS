import numpy as np
import pytest

from openadcs_sim.adcs_control import (AdcsMode, AttitudeEKF, quaternion_from_vector, rotation_from_vector, triad)
from openadcs_sim.closed_loop import AnalyticEnvironment, ClosedLoopSimulation, angle_deg
from openadcs_sim.spacecraft import pilot1_baseline


def make(**kwargs):
    defaults = dict(initial_rate_rad_s=(0.05, -0.04, 0.03), initial_attitude_offset_rad=(1.0, -1.5, 2.0))
    defaults.update(kwargs)
    return ClosedLoopSimulation(AnalyticEnvironment(), pilot1_baseline(), **defaults)


def test_triad_recovers_rotation():
    rotation = rotation_from_vector(np.array([0.3, -0.2, 0.5]))
    r1, r2 = np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0.2])
    r2 /= np.linalg.norm(r2)
    estimate = triad(rotation.T @ r1, r1, rotation.T @ r2, r2)
    assert angle_deg(estimate, rotation) < 1e-6


def test_ekf_converges_and_estimates_gyro_bias():
    rng = np.random.default_rng(1)
    truth = rotation_from_vector(np.array([0.4, 0.1, -0.3]))
    bias = np.array([0.002, -0.001, 0.0015])
    ekf = AttitudeEKF()
    ekf.initialise(rotation_from_vector(np.array([0.5, 0.0, -0.2])))
    refs = [np.array([1.0, 0, 0]), np.array([0, 0.6, 0.8]), np.array([0.3, -0.9, 0.1])]
    for _ in range(600):
        ekf.predict(bias + 1e-5 * rng.standard_normal(3), 1.0)
        for r in refs:
            r = r / np.linalg.norm(r)
            ekf.update_vector(truth.T @ r + 1e-3 * rng.standard_normal(3), r, 2e-3)
    from openadcs_sim.closed_loop import quaternion_to_matrix
    assert angle_deg(quaternion_to_matrix(ekf.q), truth) < 0.5
    assert np.linalg.norm(ekf.bias - bias) < 5e-4


@pytest.fixture(scope="module")
def run():
    sim = make()
    return sim, sim.run(3500)


def test_detumble_then_nominal(run):
    _, result = run
    assert result.mode[0] is AdcsMode.DETUMBLE
    assert result.mode[-1] is AdcsMode.NOMINAL
    assert result.rate_rad_s[-1] < 0.005


def test_estimate_converges_in_sun_and_eclipse(run):
    _, result = run
    window = slice(600, None)
    assert np.percentile(result.estimation_error_deg[window], 95) < 2.5
    eclipse = ~result.sunlit[window]
    assert eclipse.any()
    assert any("horizon" in u for u in np.array(result.used, dtype=object)[window][eclipse])


def test_nadir_pointing_accuracy(run):
    _, result = run
    window = slice(600, None)
    assert np.median(result.pointing_error_deg[window]) < 1.0
    assert result.pointing_error_deg[window].max() < 3.0


def test_wheels_stay_unsaturated_and_biased(run):
    sim, result = run
    assert result.wheel_momentum_nms[600:].max() < 0.4 * sim.wheels.spec.max_momentum_nms


def test_wheel_unloading_reduces_momentum():
    sim = make(initial_wheel_speeds_rad_s=(0.0, 700.0, 0.0, 0.0), initial_rate_rad_s=(0, 0, 0),
               initial_attitude_offset_rad=(0, 0, 0))
    result = sim.run(2500)
    assert result.wheel_momentum_nms[:5].max() > result.wheel_momentum_nms[-200:].mean()
    assert np.abs(result.dipole_am2).max() > 0

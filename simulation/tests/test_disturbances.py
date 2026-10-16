import numpy as np
import pytest

from openadcs_sim.closed_loop import AnalyticEnvironment, ClosedLoopSimulation
from openadcs_sim.conditions import ThermalNode, illumination
from openadcs_sim.disturbances import (DisturbanceConfig, DisturbanceModel, atmospheric_density, j2_acceleration,
                                       EARTH_RADIUS_M)
from openadcs_sim.spacecraft import pilot1_baseline


def test_atmosphere_decreases_and_matches_table_point():
    assert atmospheric_density(500e3) == pytest.approx(6.967e-13)
    assert atmospheric_density(400e3) > atmospheric_density(500e3) > atmospheric_density(600e3)
    assert atmospheric_density(500e3, scale=2.0) == pytest.approx(2 * 6.967e-13)


def test_j2_matches_known_equatorial_and_polar_values():
    r = np.array([EARTH_RADIUS_M + 500e3, 0.0, 0.0])
    a = j2_acceleration(r)
    assert a[0] == pytest.approx(-1.5 * 1.08262668e-3 * 3.986004418e14 * EARTH_RADIUS_M ** 2 / np.linalg.norm(r) ** 5 * r[0])
    p = j2_acceleration(np.array([0.0, 0.0, EARTH_RADIUS_M + 500e3]))
    assert p[2] > 0 and a[0] < 0 and abs(p[2]) == pytest.approx(2 * abs(a[0]), rel=1e-6)


def test_drag_force_opposes_motion_and_torque_follows_lever_arm():
    model = DisturbanceModel(DisturbanceConfig(com_offset_m=(0, 0, 0)))
    force, torque = model.aerodynamic(np.array([0.0, 0.0, 7600.0]), 7e-13)
    assert force[2] < 0 and abs(force[0]) < 1e-18 and np.allclose(torque, 0, atol=1e-20)   # symmetric box, no offset
    shifted = DisturbanceModel(DisturbanceConfig(com_offset_m=(0.01, 0, 0)))
    force2, torque2 = shifted.aerodynamic(np.array([0.0, 0.0, 7600.0]), 7e-13)
    assert force2[2] == pytest.approx(force[2]) and abs(torque2[1]) > 0


def test_solar_pressure_pushes_away_from_sun_and_vanishes_in_eclipse():
    model = DisturbanceModel(DisturbanceConfig(com_offset_m=(0, 0, 0)))
    force, _ = model.solar(np.array([0.0, 0.0, 1.0]))
    assert force[2] < 0
    area = 0.2 * 0.1   # the +Z/-Z faces
    assert abs(force[2]) == pytest.approx(4.56e-6 * area * (1 + 2 * 0.2 / 3 + 0.1), rel=1e-6)
    env = AnalyticEnvironment()
    r, v = np.array([-7e6, 0, 0.0]), np.array([0, 7500.0, 0])
    lit = model.torques(np.eye(3), r, v, np.zeros(3), np.array([1.0, 0, 0]), True).solar_nm
    dark = model.torques(np.eye(3), r, v, np.zeros(3), np.array([1.0, 0, 0]), False).solar_nm
    assert not np.any(dark) and lit is not None and env is not None


def test_disturbance_torques_are_small_but_nonzero_and_loop_stays_stable():
    env = AnalyticEnvironment()
    sample = env(0.0)
    d = DisturbanceModel().torques(np.eye(3), sample.r_gcrs_m, sample.v_gcrs_m_s, sample.b_gcrs_t, sample.sun_gcrs,
                                   sample.sunlit)
    assert 1e-10 < np.linalg.norm(d.total_nm) < 1e-5
    sim = ClosedLoopSimulation(env, pilot1_baseline(), initial_rate_rad_s=(0.05, -0.04, 0.03),
                               initial_attitude_offset_rad=(1.0, -1.5, 2.0))
    result = sim.run(3600)
    pointing = np.asarray(result.pointing_error_deg)[800:]
    assert np.median(pointing) < 1.0 and pointing.max() < 3.0


def test_illumination_and_thermal_node_cycle():
    env = AnalyticEnvironment()
    lit = [t for t in range(0, 5600, 100) if env(t).sunlit]
    dark = [t for t in range(0, 5600, 100) if not env(t).sunlit]
    assert lit and dark
    assert illumination(env(lit[0]).r_gcrs_m, env.sun, True).solar_w_m2 == pytest.approx(1361.0)
    assert illumination(env(dark[0]).r_gcrs_m, env.sun, False).solar_w_m2 == 0.0
    node = ThermalNode()
    temperatures = []
    for t in range(0, 20000):
        sample = env(t)
        temperatures.append(node.step(1.0, illumination(sample.r_gcrs_m, sample.sun_gcrs, sample.sunlit)))
    last_orbit = temperatures[-5600:]
    assert max(last_orbit) - min(last_orbit) > 10 and -100 < min(last_orbit) and max(last_orbit) < 80

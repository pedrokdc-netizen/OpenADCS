import math
from datetime import datetime, timedelta, timezone

import numpy as np

from openadcs_sim.attitude import NadirPointingAttitude
from openadcs_sim.dynamics import (MU_EARTH_M3_S2, DynamicsAttitudeModel, matrix_to_quaternion,
                                   quaternion_to_matrix)
from openadcs_sim.spacecraft import SpacecraftConfig, pilot1_baseline

GG_STABLE = SpacecraftConfig("gg-stable", 0.5, np.diag([2.0e-6, 3.0e-6, 1.5e-6]))   # Iy > Ix > Iz, y principal

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
A = 6878e3
N = math.sqrt(MU_EARTH_M3_S2 / A ** 3)


def orbit(t_s):
    c, s = math.cos(N * t_s), math.sin(N * t_s)
    return np.array([A * c, A * s, 0.0]), np.array([-A * N * s, A * N * c, 0.0])


def run(model, times):
    states = []
    for t in times:
        r, v = orbit(t)
        states.append(model.propagate(EPOCH + timedelta(seconds=t), r, v))
    return states


def test_quaternion_matrix_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        m = quaternion_to_matrix(q)
        assert np.allclose(m.T @ m, np.eye(3)) and math.isclose(np.linalg.det(m), 1.0)
        assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(m)), m)


def test_torque_free_tumble_conserves_momentum_and_energy():
    model = DynamicsAttitudeModel(pilot1_baseline(), max_step_s=0.5, gravity_gradient=False,
                                  initial_rate_body_rad_s=(0.05, 0.2, 0.1))
    states = run(model, range(0, 3601, 60))
    h0, e0 = states[0].angular_momentum_gcrs, states[0].kinetic_energy_j
    for s in states:
        assert np.linalg.norm(s.angular_momentum_gcrs - h0) < 1e-5 * np.linalg.norm(h0)
        assert math.isclose(s.kinetic_energy_j, e0, rel_tol=1e-5)
    assert not np.allclose(states[-1].omega_body_rad_s, states[0].omega_body_rad_s)   # asymmetric body: rate moves


def test_free_orbit_rate_follows_nadir_without_disturbances():
    model = DynamicsAttitudeModel(GG_STABLE, gravity_gradient=False)
    states = run(model, (0, 1500, 3000, 5400))
    for s in states:
        r, v = orbit(s.time_s)
        reference = NadirPointingAttitude().orientation(EPOCH, r, v).body_to_gcrs
        assert np.allclose(quaternion_to_matrix(s.quaternion), reference, atol=1e-5)
        assert np.allclose(s.omega_body_rad_s, [0, -N, 0], atol=1e-9)


def test_gravity_gradient_keeps_a_nadir_start_close_to_nadir():
    model = DynamicsAttitudeModel(GG_STABLE)
    states = run(model, range(0, 5401, 300))
    worst = 0.0
    for s in states:
        r, v = orbit(s.time_s)
        reference = NadirPointingAttitude().orientation(EPOCH, r, v).body_to_gcrs
        cosine = (np.trace(quaternion_to_matrix(s.quaternion).T @ reference) - 1) / 2
        worst = max(worst, math.degrees(math.acos(min(1.0, cosine))))
    assert worst < 1.0
    assert np.linalg.norm(states[3].torque_body_nm) > 0


def test_external_torque_gives_angular_acceleration():
    inertia = pilot1_baseline().inertia_kg_m2
    torque = np.array([1e-6, 0.0, 0.0])
    model = DynamicsAttitudeModel(pilot1_baseline(), gravity_gradient=False, external_torque=lambda t, q, w: torque)
    start, end = run(model, (0, 10))
    expected = np.linalg.solve(inertia, torque) * 10
    assert np.allclose(end.omega_body_rad_s - start.omega_body_rad_s, expected, rtol=0.02, atol=1e-4)


def test_scrubbing_backwards_reproduces_earlier_state():
    model = DynamicsAttitudeModel(pilot1_baseline(), initial_rate_body_rad_s=(0.01, 0.0, 0.02))
    forward = run(model, (0, 600, 1200))
    again = run(model, (600,))[0]
    assert np.allclose(again.quaternion, forward[1].quaternion)
    mid = run(model, (900,))[0]
    last = run(model, (1200,))[0]
    assert np.allclose(last.quaternion, forward[2].quaternion, atol=1e-6)
    assert mid.time_s == 900


def test_implements_attitude_model_protocol():
    model = DynamicsAttitudeModel(pilot1_baseline())
    r, v = orbit(0)
    state = model.orientation(EPOCH, r, v)
    assert state.body_to_gcrs.shape == (3, 3)

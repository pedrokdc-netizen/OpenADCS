import math
from datetime import datetime, timedelta, timezone

import numpy as np

from openadcs_sim.actuators import (RW0003_INERTIA_KG_M2, ActuatorCommand, MagnetorquerSet, MagnetorquerSpec,
                                    ReactionWheelArray, ReactionWheelSpec, bdot_dipole, pyramid_axes, scaled_rw0003)
from openadcs_sim.dynamics import MU_EARTH_M3_S2, DynamicsAttitudeModel
from openadcs_sim.spacecraft import pilot1_baseline

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
A = 6878e3
N = math.sqrt(MU_EARTH_M3_S2 / A ** 3)


def orbit(t_s):
    c, s = math.cos(N * t_s), math.sin(N * t_s)
    return np.array([A * c, A * s, 0.0]), np.array([-A * N * s, A * N * c, 0.0])


def run(model, times):
    return [model.propagate(EPOCH + timedelta(seconds=t), *orbit(t)) for t in times]


def frictionless(spec=None):
    spec = spec or scaled_rw0003()
    return ReactionWheelSpec(spec.name, spec.mass_kg, spec.inertia_kg_m2, spec.max_torque_nm, spec.max_speed_rad_s,
                             0.0, 0.0)


def test_scaled_wheel_is_ten_gram_and_smaller_than_rw0003():
    spec = scaled_rw0003()
    ratio = (0.010 / 0.050) ** (5 / 3)
    assert spec.mass_kg == 0.010
    assert math.isclose(spec.inertia_kg_m2, RW0003_INERTIA_KG_M2 * ratio)
    assert math.isclose(spec.max_torque_nm, 1.0e-3 * ratio)
    assert math.isclose(spec.max_momentum_nms, 3.0e-3 * ratio)


def test_pyramid_axes_are_unit_and_equal_momentum_on_all_axes():
    axes = pyramid_axes()
    assert axes.shape == (3, 4) and np.allclose(np.linalg.norm(axes, axis=0), 1.0)
    assert np.allclose(axes @ axes.T, 4 / 3 * np.eye(3))   # isotropic: equal momentum capacity about each axis


def test_distribution_reproduces_commanded_body_torque():
    array = ReactionWheelArray(frictionless())
    wanted = np.array([2e-5, -1e-5, 3e-5])
    wheel = array.wheel_torques_for_body_torque(wanted)
    assert np.allclose(array.reaction_torque_body(wheel, np.zeros(4)), wanted)


def test_wheel_torque_and_speed_saturate():
    array = ReactionWheelArray(frictionless())
    spec = array.spec
    assert np.allclose(array.motor_torque(np.full(4, 1.0), np.zeros(4)), spec.max_torque_nm)
    at_limit = np.full(4, spec.max_speed_rad_s)
    assert np.allclose(array.motor_torque(np.full(4, 1.0), at_limit), 0.0)               # cannot speed up further
    assert np.allclose(array.motor_torque(np.full(4, -1.0), at_limit), -spec.max_torque_nm)   # can slow down


def test_friction_slows_a_spinning_wheel():
    array = ReactionWheelArray()
    speeds = np.full(4, 500.0)
    assert np.all(array.speed_derivative(np.zeros(4), speeds) < 0)
    assert np.allclose(array.friction(np.zeros(4)), 0.0)


def test_total_angular_momentum_is_conserved_with_wheels():
    array = ReactionWheelArray(frictionless())

    def controller(t, q, omega, speeds, b):
        return ActuatorCommand(body_torque_nm=np.array([5e-5, -3e-5, 4e-5]) * math.sin(t / 30.0))

    model = DynamicsAttitudeModel(pilot1_baseline(), max_step_s=0.1, gravity_gradient=False, wheels=array,
                                  controller=controller, initial_rate_body_rad_s=(0.01, 0.02, 0.0))
    states = run(model, range(0, 201, 20))
    h0 = states[0].angular_momentum_gcrs
    for s in states:
        assert np.linalg.norm(s.angular_momentum_gcrs - h0) < 1e-9
    assert np.linalg.norm(states[5].wheel_momentum_body_nms) > 0       # wheels actually took momentum
    assert not np.allclose(states[5].omega_body_rad_s, states[0].omega_body_rad_s)


def test_saturated_wheels_limit_the_torque_on_the_body():
    array = ReactionWheelArray(frictionless())
    model = DynamicsAttitudeModel(pilot1_baseline(), max_step_s=0.5, gravity_gradient=False, wheels=array,
                                  controller=lambda *args: ActuatorCommand(body_torque_nm=np.array([1.0, 0, 0])))
    state = run(model, [0, 10])[1]
    assert np.linalg.norm(state.torque_body_nm) <= 4 * array.spec.max_torque_nm * (1 + 1e-9)
    assert np.all(np.abs(state.wheel_speeds_rad_s) <= array.spec.max_speed_rad_s * 1.001)


def test_magnetorquer_torque_is_perpendicular_to_field_and_saturates():
    torquers = MagnetorquerSet(MagnetorquerSpec(max_dipole_am2=0.1))
    b = np.array([0.0, 0.0, 3e-5])
    torque = torquers.torque(np.array([1.0, 0.0, 0.5]), b)
    assert np.allclose(torque, [0.0, -0.1 * 3e-5, 0.0])          # m = (0.1, 0, 0.1) after saturation
    assert abs(torque @ b) < 1e-20
    assert np.allclose(torquers.dipole([5.0, -5.0, 0.0]), [0.1, -0.1, 0.0])


def test_bdot_detumbles_the_spacecraft():
    field = np.array([1.0, 2.0, 3.0]) * 1e-5     # fixed in GCRS for the test

    def controller(t, q, omega, speeds, b_body):
        return ActuatorCommand(dipole_am2=bdot_dipole(-np.cross(omega, b_body), 1e5))

    model = DynamicsAttitudeModel(pilot1_baseline(), max_step_s=1.0, gravity_gradient=False,
                                  magnetorquers=MagnetorquerSet(), controller=controller,
                                  magnetic_field=lambda t, r: field, initial_rate_body_rad_s=(0.03, 0.04, 0.02))
    states = run(model, range(0, 3601, 300))
    assert states[-1].kinetic_energy_j < 0.5 * states[0].kinetic_energy_j

"""Rigid-body attitude dynamics: torque-driven truth attitude and body rate.

State: unit quaternion q = [w, x, y, z] (body -> GCRS) and body rate omega_B [rad/s].
    q_dot     = 0.5 * q (x) [0, omega_B]
    I omega_dot = tau - omega x (I omega)
Torques: gravity gradient (optional) plus an external body-frame torque hook, which is where the actuator
and disturbance models plug in.  The class implements the ``AttitudeModel`` protocol, so it replaces the ideal
nadir reference in ``spacecraft_view``/``EnvironmentSimulator``.  Evaluating an earlier time restarts from the
nearest stored checkpoint, so the GUI can scrub backwards.
"""

import bisect
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from .actuators import ActuatorCommand, MagnetorquerSet, ReactionWheelArray
from .attitude import AttitudeState, NadirPointingAttitude
from .common import Vector, vector3
from .spacecraft import SpacecraftConfig

MU_EARTH_M3_S2 = 3.986004418e14
ExternalTorque = Callable[[float, np.ndarray, np.ndarray], np.ndarray]   # (t_s, q, omega_B) -> torque_B [N m]
# (t_s, q, omega_B, wheel_speeds, B_body) -> command; B_body is None without a magnetic field model
Controller = Callable[[float, np.ndarray, np.ndarray, np.ndarray, "np.ndarray | None"], ActuatorCommand]
MagneticField = Callable[[float, np.ndarray], np.ndarray]                # (t_s, r_gcrs_m) -> B_gcrs [T]


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    trace = np.trace(m)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = np.array([s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4)
        q[1 + i] = s / 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    return q / np.linalg.norm(q)


@dataclass(frozen=True)
class DynamicsState:
    time_s: float                 # seconds since the first evaluated epoch
    quaternion: np.ndarray        # body -> GCRS, [w, x, y, z]
    omega_body_rad_s: np.ndarray
    torque_body_nm: np.ndarray    # total torque at this instant
    angular_momentum_gcrs: np.ndarray
    kinetic_energy_j: float       # body only
    wheel_speeds_rad_s: np.ndarray | None = None
    wheel_momentum_body_nms: np.ndarray | None = None
    dipole_am2: np.ndarray | None = None


class DynamicsAttitudeModel:
    """Torque-driven rigid-body attitude; starts nadir-pointing at the orbit rate plus ``initial_rate_body``."""

    def __init__(self, spacecraft: SpacecraftConfig, max_step_s: float = 2.0, gravity_gradient: bool = True,
                 external_torque: ExternalTorque | None = None, initial_rate_body_rad_s=(0.0, 0.0, 0.0),
                 mu_m3_s2: float = MU_EARTH_M3_S2, wheels: ReactionWheelArray | None = None,
                 magnetorquers: MagnetorquerSet | None = None, controller: Controller | None = None,
                 magnetic_field: MagneticField | None = None, initial_wheel_speeds_rad_s=None,
                 initial_attitude_offset_rad=(0.0, 0.0, 0.0)):
        if not max_step_s > 0:
            raise ValueError("max_step_s must be positive")
        self.spacecraft = spacecraft
        self.inertia = np.array(spacecraft.inertia_kg_m2, dtype=float)
        self.inertia_inverse = np.linalg.inv(self.inertia)
        self.max_step_s = float(max_step_s)
        self.gravity_gradient = gravity_gradient
        self.external_torque = external_torque
        self.initial_rate_body = np.array(vector3(initial_rate_body_rad_s))
        self.mu = float(mu_m3_s2)
        self.initial_attitude_offset = np.array(vector3(initial_attitude_offset_rad))   # rotation vector from nadir
        self.wheels = wheels
        self.magnetorquers = magnetorquers
        self.controller = controller
        self.magnetic_field = magnetic_field
        count = wheels.count if wheels else 0
        self.initial_wheel_speeds = (np.zeros(count) if initial_wheel_speeds_rad_s is None
                                     else np.array(initial_wheel_speeds_rad_s, dtype=float))
        if self.initial_wheel_speeds.shape != (count,):
            raise ValueError("initial_wheel_speeds_rad_s must have one entry per wheel")
        self._lock = threading.RLock()
        self._epoch: datetime | None = None
        self._times: list[float] = []
        self._checkpoints: list[tuple] = []   # (q, omega, r_gcrs, v_gcrs), parallel to _times
        self.last_state: DynamicsState | None = None

    def reset(self) -> None:
        with self._lock:
            self._epoch = None
            self._times.clear()
            self._checkpoints.clear()
            self.last_state = None

    def _actuation(self, t: float, q: np.ndarray, omega: np.ndarray, r_gcrs: np.ndarray, wheel_speeds: np.ndarray):
        """Return (torque from magnetorquers, wheel reaction torque, wheel speed derivative, dipole)."""
        zero = np.zeros(3)
        if self.controller is None or (self.wheels is None and self.magnetorquers is None):
            return zero, zero, np.zeros_like(wheel_speeds), zero
        b_body = None
        if self.magnetic_field is not None:
            b_body = quaternion_to_matrix(q).T @ np.asarray(self.magnetic_field(t, r_gcrs), dtype=float)
        command = self.controller(t, q, omega, wheel_speeds, b_body)
        reaction, speed_dot, magnetic, dipole = zero, np.zeros_like(wheel_speeds), zero, zero
        if self.wheels is not None:
            wheel_command = self.wheels.wheel_torques_for_body_torque(command.body_torque_nm)
            if command.wheel_torque_offset_nm is not None:
                wheel_command = wheel_command + command.wheel_torque_offset_nm
            reaction = self.wheels.reaction_torque_body(wheel_command, wheel_speeds)
            speed_dot = self.wheels.speed_derivative(wheel_command, wheel_speeds)
        if self.magnetorquers is not None and b_body is not None:
            dipole = self.magnetorquers.dipole(command.dipole_am2)
            magnetic = np.cross(dipole, b_body)
        return magnetic, reaction, speed_dot, dipole

    def _torque(self, t: float, q: np.ndarray, omega: np.ndarray, r_gcrs: np.ndarray,
                wheel_speeds: np.ndarray) -> np.ndarray:
        torque = np.zeros(3)
        if self.gravity_gradient:
            r_body = quaternion_to_matrix(q).T @ r_gcrs
            radius = float(np.linalg.norm(r_body))
            torque += 3.0 * self.mu / radius ** 5 * np.cross(r_body, self.inertia @ r_body)
        if self.external_torque is not None:
            torque += np.asarray(self.external_torque(t, q, omega), dtype=float)
        magnetic, reaction, _, _ = self._actuation(t, q, omega, r_gcrs, wheel_speeds)
        return torque + magnetic + reaction

    def _wheel_momentum(self, wheel_speeds: np.ndarray) -> np.ndarray:
        return self.wheels.momentum_body(wheel_speeds) if self.wheels is not None else np.zeros(3)

    def _derivative(self, t, y):
        q, omega, r, v, w = y[0:4], y[4:7], y[7:10], y[10:13], y[13:]
        torque = self._torque(t, q, omega, r, w)
        _, _, speed_dot, _ = self._actuation(t, q, omega, r, w)
        q_dot = 0.5 * quaternion_multiply(q, np.concatenate(([0.0], omega)))
        h_total = self.inertia @ omega + self._wheel_momentum(w)
        omega_dot = self.inertia_inverse @ (torque - np.cross(omega, h_total))
        acceleration = -self.mu * r / float(np.linalg.norm(r)) ** 3   # two-body orbit between supplied samples
        return np.concatenate((q_dot, omega_dot, v, acceleration, speed_dot))

    def _step(self, t, dt, y):
        k1 = self._derivative(t, y)
        k2 = self._derivative(t + dt / 2, y + dt / 2 * k1)
        k3 = self._derivative(t + dt / 2, y + dt / 2 * k2)
        k4 = self._derivative(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y[0:4] /= np.linalg.norm(y[0:4])
        return y
    def _initial(self, position: np.ndarray, velocity: np.ndarray):
        reference = NadirPointingAttitude().orientation(self._epoch, position, velocity).body_to_gcrs
        angle = float(np.linalg.norm(self.initial_attitude_offset))
        if angle > 0:
            k = self.initial_attitude_offset / angle
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            reference = reference @ (np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx)
        omega_gcrs = np.cross(position, velocity) / float(position @ position)
        omega = reference.T @ omega_gcrs + self.initial_rate_body
        return matrix_to_quaternion(reference), omega

    def propagate(self, time_utc: datetime, position_gcrs_m: Vector, velocity_gcrs_m_s: Vector) -> DynamicsState:
        position = np.array(vector3(position_gcrs_m))
        velocity = np.array(vector3(velocity_gcrs_m_s))
        with self._lock:
            if self._epoch is None:
                self._epoch = time_utc
                q, omega = self._initial(position, velocity)
                self._times = [0.0]
                self._checkpoints = [(q, omega, position, velocity, self.initial_wheel_speeds.copy())]
            target = (time_utc - self._epoch).total_seconds()
            index = bisect.bisect_right(self._times, target) - 1
            if index < 0:
                raise ValueError("Requested time precedes the dynamics epoch")
            t0 = self._times[index]
            q, omega, r_start, v_start, wheel_speeds = self._checkpoints[index]
            if target > t0:
                steps = max(1, int(np.ceil((target - t0) / self.max_step_s)))
                dt = (target - t0) / steps
                y = np.concatenate((q, omega, r_start, v_start, wheel_speeds))
                for n in range(steps):
                    y = self._step(t0 + n * dt, dt, y)
                q, omega, wheel_speeds = y[0:4], y[4:7], y[13:]
                # store the supplied (SGP4) state so two-body drift never accumulates between checkpoints
                insert = bisect.bisect_left(self._times, target)
                entry = (q, omega, position, velocity, wheel_speeds)
                if insert < len(self._times) and self._times[insert] == target:
                    self._checkpoints[insert] = entry
                else:
                    self._times.insert(insert, target)
                    self._checkpoints.insert(insert, entry)
            torque = self._torque(target, q, omega, position, wheel_speeds)
            _, _, _, dipole = self._actuation(target, q, omega, position, wheel_speeds)
            matrix = quaternion_to_matrix(q)
            wheel_momentum = self._wheel_momentum(wheel_speeds)
            state = DynamicsState(target, q.copy(), omega.copy(), torque,
                                  matrix @ (self.inertia @ omega + wheel_momentum),
                                  0.5 * float(omega @ self.inertia @ omega),
                                  wheel_speeds.copy(), wheel_momentum, dipole)
            self.last_state = state
            return state
    def orientation(self, time_utc: datetime, position_gcrs_m: Vector, velocity_gcrs_m_s: Vector) -> AttitudeState:
        state = self.propagate(time_utc, position_gcrs_m, velocity_gcrs_m_s)
        return AttitudeState(quaternion_to_matrix(state.quaternion), "Rigid-body dynamics (gravity gradient + external torque)")

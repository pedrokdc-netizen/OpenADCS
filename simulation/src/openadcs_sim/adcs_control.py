"""ADCS algorithms: sensor fusion (multiplicative EKF) and the mode-based control law.

Estimator: MEKF, state = attitude quaternion q (body -> GCRS) + gyro bias b; error state (dtheta, db) in the body
frame.  The gyro drives the prediction, and unit-vector measurements correct it:
    Sun (sunlit only), magnetic field (reference from the field model at the GNSS position),
    nadir from the horizon sensor (eclipse / no valid Sun only).
    R_true = R_est (I + [dtheta]x),   b_meas = b_ref + [b_ref]x dtheta   ->   H = [[b_ref]x  0]
A UKF/IMM was NOT needed: the measurement models are nearly linear and measurement availability (eclipse, Sun
loss) is handled by gating, see ``tests/test_adcs_control.py`` and the README for the accuracy actually reached.

Control modes:
    DETUMBLE     B-dot with the magnetorquers, m = -k dB/dt = k w x B  (wheels idle)
    ACQUISITION  wheels damp the rate while the filter is initialised (TRIAD from Sun+magnetic, or magnetic+nadir)
                 and converges
    NOMINAL      nadir pointing by wheels, PD on the estimated attitude error with gyroscopic feed-forward;
                 magnetorquers unload the wheels (m = B x tau / |B|^2, tau = -h_w / T) whenever they are enabled
"""

from dataclasses import dataclass, field
import dataclasses
from enum import Enum

import numpy as np

from .actuators import ActuatorCommand, MagnetorquerSpec, ReactionWheelArray
from .attitude import NadirPointingAttitude
from .dynamics import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix


def skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def rotation_from_vector(v) -> np.ndarray:
    """exp([v]x) (Rodrigues)."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3) + skew(v)
    k = skew(np.asarray(v) / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def quaternion_from_vector(v) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        q = np.concatenate(([1.0], 0.5 * np.asarray(v)))
        return q / np.linalg.norm(q)
    return np.concatenate(([np.cos(angle / 2)], np.sin(angle / 2) * np.asarray(v) / angle))


def triad(b1, r1, b2, r2) -> np.ndarray:
    """Body -> reference rotation from two vector pairs; the first pair is trusted exactly."""
    def frame(a, c):
        t1 = a / np.linalg.norm(a)
        t2 = np.cross(a, c)
        t2 /= np.linalg.norm(t2)
        return np.column_stack((t1, t2, np.cross(t1, t2)))
    return frame(r1, r2) @ frame(b1, b2).T


class OrbitCoaster:
    """Onboard two-body propagator: holds the last GNSS fix and coasts through outages (no J2/drag)."""

    MU = 3.986004418e14

    def __init__(self, max_step_s: float = 10.0):
        self.max_step, self.t, self.r, self.v = max_step_s, None, None, None

    def _acc(self, r):
        return -self.MU * r / np.linalg.norm(r) ** 3

    def update(self, t: float, r, v):
        if r is not None and v is not None:
            self.t, self.r, self.v = t, np.asarray(r, float), np.asarray(v, float)
        elif self.t is not None:
            remaining = t - self.t
            while remaining > 1e-9:
                h = min(self.max_step, remaining)
                k1r, k1v = self.v, self._acc(self.r)
                k2r, k2v = self.v + 0.5 * h * k1v, self._acc(self.r + 0.5 * h * k1r)
                k3r, k3v = self.v + 0.5 * h * k2v, self._acc(self.r + 0.5 * h * k2r)
                k4r, k4v = self.v + h * k3v, self._acc(self.r + h * k3r)
                self.r = self.r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
                self.v = self.v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
                remaining -= h
            self.t = t
        return self.r, self.v


class AdcsMode(str, Enum):
    DETUMBLE = "DETUMBLE"
    ACQUISITION = "ACQUISITION"
    NOMINAL = "NOMINAL"


@dataclass(frozen=True)
class AdcsInputs:
    t_s: float
    gyro_rad_s: np.ndarray | None = None
    mag_body_t: np.ndarray | None = None
    sun_body: np.ndarray | None = None
    sun_sigma_rad: float = 0.01
    sunlit: bool = True
    nadir_body: np.ndarray | None = None
    nadir_sigma_rad: float = 0.02
    wheel_speeds_rad_s: np.ndarray | None = None
    r_gcrs_m: np.ndarray | None = None            # GNSS
    v_gcrs_m_s: np.ndarray | None = None          # GNSS
    b_ref_gcrs_t: np.ndarray | None = None        # magnetic field model evaluated at the GNSS position
    sun_ref_gcrs: np.ndarray | None = None        # onboard ephemeris


@dataclass(frozen=True)
class AdcsStatus:
    mode: AdcsMode
    initialised: bool
    quaternion: np.ndarray
    bias_rad_s: np.ndarray
    sigma_deg: np.ndarray            # 1-sigma attitude, per body axis
    wheel_momentum_nms: float
    desaturating: bool
    used: tuple = ()                 # measurements fused this cycle


class AttitudeEKF:
    def __init__(self, arw_rad_s_rthz: float = 2e-4, bias_rw_rad_s2_rthz: float = 2e-6,
                 initial_attitude_sigma_rad: float = np.radians(5.0), initial_bias_sigma_rad_s: float = 5e-3):
        self.sigma_v, self.sigma_u = arw_rad_s_rthz, bias_rw_rad_s2_rthz
        self.initial_attitude_sigma, self.initial_bias_sigma = initial_attitude_sigma_rad, initial_bias_sigma_rad_s
        self.initialised = False
        self.q = np.array([1.0, 0, 0, 0])
        self.bias = np.zeros(3)
        self.P = np.eye(6)

    def initialise(self, rotation_body_to_gcrs: np.ndarray) -> None:
        self.q = matrix_to_quaternion(rotation_body_to_gcrs)
        self.bias = np.zeros(3)
        self.P = np.diag([self.initial_attitude_sigma ** 2] * 3 + [self.initial_bias_sigma ** 2] * 3)
        self.initialised = True

    @property
    def rotation(self) -> np.ndarray:
        return quaternion_to_matrix(self.q)

    def rate(self, gyro_rad_s) -> np.ndarray:
        return np.asarray(gyro_rad_s, dtype=float) - self.bias

    def predict(self, gyro_rad_s, dt: float) -> None:
        omega = self.rate(gyro_rad_s)
        theta = omega * dt
        self.q = quaternion_multiply(self.q, quaternion_from_vector(theta))
        self.q /= np.linalg.norm(self.q)
        phi = np.eye(6)
        phi[:3, :3] = rotation_from_vector(-theta)
        phi[:3, 3:] = -np.eye(3) * dt
        su2, sv2 = self.sigma_u ** 2, self.sigma_v ** 2
        q = np.zeros((6, 6))
        q[:3, :3] = np.eye(3) * (sv2 * dt + su2 * dt ** 3 / 3)
        q[:3, 3:] = q[3:, :3] = -np.eye(3) * su2 * dt ** 2 / 2
        q[3:, 3:] = np.eye(3) * su2 * dt
        self.P = phi @ self.P @ phi.T + q

    def update_vector(self, body_unit, reference_unit, sigma_rad: float) -> None:
        predicted = self.rotation.T @ reference_unit
        h = np.zeros((3, 6))
        h[:, :3] = skew(predicted)
        r = np.eye(3) * sigma_rad ** 2
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)
        dx = k @ (body_unit - predicted)
        self.q = quaternion_multiply(self.q, quaternion_from_vector(dx[:3]))
        self.q /= np.linalg.norm(self.q)
        self.bias = self.bias + dx[3:]
        i_kh = np.eye(6) - k @ h
        self.P = i_kh @ self.P @ i_kh.T + k @ r @ k.T

    def sigma_deg(self) -> np.ndarray:
        return np.degrees(np.sqrt(np.diag(self.P)[:3]))


@dataclass(frozen=True)
class AdcsConfig:
    inertia_kg_m2: np.ndarray
    wheels: ReactionWheelArray
    magnetorquer: MagnetorquerSpec = field(default_factory=MagnetorquerSpec)
    bandwidth_rad_s: float = 0.08
    damping: float = 0.8
    detumble_enter_rad_s: float = 0.10
    detumble_exit_rad_s: float = 0.03
    integral_ratio: float = 0.2              # integral zero at this fraction of the bandwidth
    integral_limit_nm: float = 1.0e-5        # anti-windup clamp on the integral torque
    bdot_gain: float = 5.0e4                 # A m^2 s / T per (rad/s . T)
    nominal_enter_deg: float = 2.0           # 1-sigma attitude needed for NOMINAL
    nominal_exit_deg: float = 6.0
    mag_sigma_rad: float = np.radians(2.0)   # field model + sensor + residual dipole
    unload_enter_fraction: float = 0.25      # of one wheel's momentum capacity (norm of body momentum)
    unload_exit_fraction: float = 0.10
    unload_time_constant_s: float = 600.0
    friction_compensation: float = 0.8       # fraction of the modelled wheel friction fed forward
    wheel_bias_rad_s: float = 500.0          # null-space speed bias, ~250 rad/s per wheel (0 disables)
    wheel_bias_time_constant_s: float = 20.0
    mtq_off_every: int = 5                   # one cycle in N with torquers off so the magnetometer reads clean


class AdcsController:
    def __init__(self, config: AdcsConfig, ekf: AttitudeEKF | None = None):
        self.config = config
        self.ekf = ekf or AttitudeEKF()
        self.inertia = np.array(config.inertia_kg_m2, dtype=float)
        self.mode = AdcsMode.DETUMBLE
        self._last_t: float | None = None
        self._unloading = False
        self._cycle = 0
        self._last_dipole = np.zeros(3)
        self._integral = np.zeros(3)
        self._dt = 0.0
        self.coaster = OrbitCoaster()

    def _null_bias_torque(self, speeds: np.ndarray) -> np.ndarray:
        """Torque along the pyramid null space that holds a zero-net-momentum speed bias.

        Keeps every wheel away from zero speed, where Coulomb friction flips sign and the
        wheel stalls; the null-space direction produces no torque on the body."""
        cfg = self.config
        null = np.linalg.svd(cfg.wheels.axes)[2][-1]
        null = null if null.sum() >= 0 else -null
        if cfg.wheel_bias_rad_s <= 0:
            return np.zeros(len(speeds))
        error = cfg.wheel_bias_rad_s * np.linalg.norm(null) - null @ speeds
        limit = 0.2 * cfg.wheels.spec.max_torque_nm
        torque = np.clip(cfg.wheels.spec.inertia_kg_m2 * error / cfg.wheel_bias_time_constant_s, -limit, limit)
        return torque * null

    # ------------------------------------------------------------ estimation
    def _estimate(self, u: AdcsInputs) -> tuple:
        ekf, cfg, used = self.ekf, self.config, []
        dt = None if self._last_t is None else u.t_s - self._last_t
        self._last_t = u.t_s
        self._dt = dt or 0.0
        if ekf.initialised and u.gyro_rad_s is not None and dt:
            ekf.predict(u.gyro_rad_s, dt)
        pairs = []   # (body, reference, sigma, name), most accurate first
        if u.sunlit and u.sun_body is not None and u.sun_ref_gcrs is not None:
            pairs.append((u.sun_body, u.sun_ref_gcrs / np.linalg.norm(u.sun_ref_gcrs), u.sun_sigma_rad, "sun"))
        if u.nadir_body is not None and u.r_gcrs_m is not None and not (u.sunlit and u.sun_body is not None):
            pairs.append((u.nadir_body, -u.r_gcrs_m / np.linalg.norm(u.r_gcrs_m), u.nadir_sigma_rad, "horizon"))
        if (u.mag_body_t is not None and u.b_ref_gcrs_t is not None and not np.any(self._last_dipole)
                and np.linalg.norm(u.mag_body_t) > 0):
            pairs.append((u.mag_body_t / np.linalg.norm(u.mag_body_t),
                          u.b_ref_gcrs_t / np.linalg.norm(u.b_ref_gcrs_t), cfg.mag_sigma_rad, "mag"))
        if not ekf.initialised:
            for i in range(len(pairs) - 1):
                b1, r1, _, n1 = pairs[i]
                b2, r2, _, n2 = pairs[i + 1]
                if np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1))) > 15 and \
                        np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1))) < 165:
                    ekf.initialise(triad(b1, r1, b2, r2))
                    used.append("triad:" + n1 + "+" + n2)
                    break
        elif pairs:
            for b, r, sigma, name in pairs:
                ekf.update_vector(b, r, sigma)
                used.append(name)
        return tuple(used)

    # --------------------------------------------------------------- control
    def _wheel_momentum(self, u: AdcsInputs) -> np.ndarray:
        if u.wheel_speeds_rad_s is None:
            return np.zeros(3)
        return self.config.wheels.momentum_body(u.wheel_speeds_rad_s)

    def _set_mode(self, rate: float) -> None:
        cfg = self.config
        if rate > cfg.detumble_enter_rad_s:
            self.mode = AdcsMode.DETUMBLE
            self._integral = np.zeros(3)
            return
        if self.mode is AdcsMode.DETUMBLE and rate < cfg.detumble_exit_rad_s:
            self.mode = AdcsMode.ACQUISITION
        if self.mode is AdcsMode.DETUMBLE:
            return
        sigma = float(np.max(self.ekf.sigma_deg())) if self.ekf.initialised else np.inf
        if self.mode is AdcsMode.ACQUISITION and sigma < cfg.nominal_enter_deg:
            self.mode = AdcsMode.NOMINAL
        elif self.mode is AdcsMode.NOMINAL and sigma > cfg.nominal_exit_deg:
            self.mode = AdcsMode.ACQUISITION

    def _nominal_torque(self, u: AdcsInputs, omega: np.ndarray, h_w: np.ndarray) -> np.ndarray:
        cfg = self.config
        kp = cfg.bandwidth_rad_s ** 2 * self.inertia
        kd = 2 * cfg.damping * cfg.bandwidth_rad_s * self.inertia
        h_total = self.inertia @ omega + h_w
        torque = np.cross(omega, h_total)
        if u.r_gcrs_m is None or u.v_gcrs_m_s is None:
            return torque - kd @ omega
        reference = NadirPointingAttitude().orientation(None, u.r_gcrs_m, u.v_gcrs_m_s).body_to_gcrs
        rate_ref = np.linalg.norm(np.cross(u.r_gcrs_m, u.v_gcrs_m_s)) / float(u.r_gcrs_m @ u.r_gcrs_m)
        re = reference.T @ self.ekf.rotation
        error = 0.5 * np.array([re[2, 1] - re[1, 2], re[0, 2] - re[2, 0], re[1, 0] - re[0, 1]])
        omega_ref_body = re.T @ np.array([0.0, -rate_ref, 0.0])
        ki = cfg.integral_ratio * cfg.bandwidth_rad_s * kp
        if np.linalg.norm(error) < 0.2:
            self._integral = np.clip(self._integral + ki @ error * self._dt,
                                     -cfg.integral_limit_nm, cfg.integral_limit_nm)
        return torque - kp @ error - kd @ (omega - omega_ref_body) - self._integral

    def _unload_dipole(self, u: AdcsInputs, h_w: np.ndarray) -> np.ndarray:
        cfg = self.config
        if u.b_ref_gcrs_t is None or not self.ekf.initialised:
            return np.zeros(3)
        b_body = self.ekf.rotation.T @ u.b_ref_gcrs_t
        norm2 = float(b_body @ b_body)
        if norm2 <= 0:
            return np.zeros(3)
        tau = -h_w / cfg.unload_time_constant_s
        return np.cross(b_body, tau) / norm2

    def update(self, u: AdcsInputs) -> tuple[ActuatorCommand, AdcsStatus]:
        cfg = self.config
        r_nav, v_nav = self.coaster.update(u.t_s, u.r_gcrs_m, u.v_gcrs_m_s)
        u = dataclasses.replace(u, r_gcrs_m=r_nav, v_gcrs_m_s=v_nav)
        used = self._estimate(u)
        gyro = np.zeros(3) if u.gyro_rad_s is None else np.asarray(u.gyro_rad_s, dtype=float)
        omega = self.ekf.rate(gyro) if self.ekf.initialised else gyro
        h_w = self._wheel_momentum(u)
        self._set_mode(float(np.linalg.norm(omega)))
        self._cycle += 1
        capacity = cfg.wheels.spec.max_momentum_nms
        h_norm = float(np.linalg.norm(h_w))
        if h_norm > cfg.unload_enter_fraction * capacity:
            self._unloading = True
        elif h_norm < cfg.unload_exit_fraction * capacity:
            self._unloading = False
        torque = np.zeros(3)
        dipole = np.zeros(3)
        if self.mode is AdcsMode.DETUMBLE:
            if u.mag_body_t is not None:
                dipole = cfg.bdot_gain * np.cross(omega, u.mag_body_t)
        else:
            if self.mode is AdcsMode.NOMINAL:
                torque = self._nominal_torque(u, omega, h_w)
            else:
                kd = 2 * cfg.damping * cfg.bandwidth_rad_s * self.inertia
                torque = np.cross(omega, self.inertia @ omega + h_w) - kd @ omega
            if self._unloading:
                dipole = self._unload_dipole(u, h_w)
        limit = cfg.magnetorquer.max_dipole_am2
        dipole = np.clip(dipole, -limit, limit)
        if self._cycle % cfg.mtq_off_every == 0:
            dipole = np.zeros(3)
        self._last_dipole = dipole
        status = AdcsStatus(self.mode, self.ekf.initialised, self.ekf.q.copy(), self.ekf.bias.copy(),
                            self.ekf.sigma_deg(), h_norm, self._unloading and self.mode is not AdcsMode.DETUMBLE,
                            used)
        offset = None
        if self.mode is not AdcsMode.DETUMBLE and u.wheel_speeds_rad_s is not None:
            speeds = np.asarray(u.wheel_speeds_rad_s)
            offset = cfg.friction_compensation * cfg.wheels.friction(speeds)
            offset = offset + self._null_bias_torque(speeds)
        return ActuatorCommand(body_torque_nm=torque, dipole_am2=dipole, wheel_torque_offset_nm=offset), status

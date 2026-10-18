"""Closed-loop ADCS simulation: truth dynamics + sensor models + ADCS (MEKF and control) + actuators.

Each ADCS cycle (``dt_s``): truth state -> sensor models (gyro, magnetometer, six-head Sun sensor, horizon sensor
equivalent error law) -> ``AdcsController.update`` -> actuator command held (zero-order hold) over the next cycle.
The environment (orbit, magnetic field, Sun, eclipse) comes from an ``Environment`` callable in GCRS: either the
analytic one used by the tests or ``RealEnvironment`` (SGP4 + XYZgeomag + Huira/SPICE).
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

from .actuators import ActuatorCommand, MagnetorquerSet, ReactionWheelArray
from .adcs_control import AdcsConfig, AdcsController, AdcsInputs, AdcsMode
from .attitude import NadirPointingAttitude
from .conditions import ThermalConfig, ThermalNode, illumination
from .disturbances import DisturbanceConfig, DisturbanceModel
from .dynamics import MU_EARTH_M3_S2, DynamicsAttitudeModel, quaternion_to_matrix
from .gyro_model import GyroConfig, GyroModel
from .horizon_sensor import HorizonSensorConfig, HorizonSensorModel
from .magnetometer import MagnetometerConfig, MagnetometerModel
from .spacecraft import SpacecraftConfig
from .sun_sensor_array import SunSensorArray

EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class EnvSample:
    r_gcrs_m: np.ndarray
    v_gcrs_m_s: np.ndarray
    b_gcrs_t: np.ndarray
    sun_gcrs: np.ndarray
    sunlit: bool


Environment = Callable[[float], EnvSample]   # t_s since the start -> sample


def cylinder_sunlit(r: np.ndarray, sun: np.ndarray) -> bool:
    along = float(r @ sun)
    return not (along < 0 and float(np.linalg.norm(r - along * sun)) < EARTH_RADIUS_M)


def dipole_field(r: np.ndarray, moment_axis: np.ndarray, b0_t: float = 3.12e-5) -> np.ndarray:
    radius = float(np.linalg.norm(r))
    rhat = r / radius
    return b0_t * (EARTH_RADIUS_M / radius) ** 3 * (3 * float(moment_axis @ rhat) * rhat - moment_axis)


class AnalyticEnvironment:
    """Circular orbit, tilted static dipole, fixed Sun direction and a cylindrical Earth shadow."""

    def __init__(self, altitude_m: float = 500e3, inclination_deg: float = 51.6, raan_deg: float = 0.0,
                 sun_gcrs=(0.9, 0.35, 0.2), dipole_tilt_deg: float = 11.0):
        self.a = EARTH_RADIUS_M + altitude_m
        self.n = math.sqrt(MU_EARTH_M3_S2 / self.a ** 3)
        i, o = math.radians(inclination_deg), math.radians(raan_deg)
        self.p = np.array([math.cos(o), math.sin(o), 0.0])
        self.q = np.array([-math.sin(o) * math.cos(i), math.cos(o) * math.cos(i), math.sin(i)])
        self.sun = np.array(sun_gcrs, dtype=float) / np.linalg.norm(sun_gcrs)
        tilt = math.radians(dipole_tilt_deg)
        self.axis = np.array([math.sin(tilt), 0.0, -math.cos(tilt)])

    def __call__(self, t_s: float) -> EnvSample:
        c, s = math.cos(self.n * t_s), math.sin(self.n * t_s)
        r = self.a * (c * self.p + s * self.q)
        v = self.a * self.n * (-s * self.p + c * self.q)
        return EnvSample(r, v, dipole_field(r, self.axis), self.sun, cylinder_sunlit(r, self.sun))


class RealEnvironment:
    """SGP4 + WMM2025 (XYZgeomag) + Huira/SPICE Sun, expressed in GCRS; sampled every ``step_s`` and interpolated."""

    def __init__(self, simulator, start_utc: datetime, step_s: float = 10.0):
        self.simulator, self.start, self.step = simulator, start_utc, step_s
        self._cache: dict[int, EnvSample] = {}

    def _node(self, k: int) -> EnvSample:
        if k not in self._cache:
            import astropy.units as u
            from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
            from astropy.time import Time
            sample = self.simulator.sample(self.start + timedelta(seconds=k * self.step))
            instant = Time(sample.time_utc)
            fixed = ITRS(CartesianRepresentation(sample.position_itrs_m * u.m, differentials=CartesianDifferential(
                sample.velocity_itrs_m_s * u.m / u.s)), obstime=instant)
            inertial = fixed.transform_to(GCRS(obstime=instant))
            r = np.array(inertial.cartesian.xyz.to_value(u.m))
            v = np.array(inertial.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s))
            basis = ITRS(CartesianRepresentation(np.eye(3) * u.m), obstime=instant)
            m = basis.transform_to(GCRS(obstime=instant)).cartesian.xyz.to_value(u.m)
            sun = m @ np.asarray(sample.sun_direction_itrs)
            self._cache[k] = EnvSample(r, v, m @ np.asarray(sample.magnetic_field_itrs_t), sun / np.linalg.norm(sun),
                                       cylinder_sunlit(r, sun / np.linalg.norm(sun)))
        return self._cache[k]

    def __call__(self, t_s: float) -> EnvSample:
        k = int(math.floor(t_s / self.step))
        a, b = self._node(k), self._node(k + 1)
        x = t_s / self.step - k
        h = self.step
        h00, h10, h01, h11 = 2 * x ** 3 - 3 * x ** 2 + 1, x ** 3 - 2 * x ** 2 + x, -2 * x ** 3 + 3 * x ** 2, x ** 3 - x ** 2
        r = h00 * a.r_gcrs_m + h10 * h * a.v_gcrs_m_s + h01 * b.r_gcrs_m + h11 * h * b.v_gcrs_m_s
        v = (1 - x) * a.v_gcrs_m_s + x * b.v_gcrs_m_s
        sun = (1 - x) * a.sun_gcrs + x * b.sun_gcrs
        sun /= np.linalg.norm(sun)
        return EnvSample(r, v, (1 - x) * a.b_gcrs_t + x * b.b_gcrs_t, sun, cylinder_sunlit(r, sun))


def default_gyro(seed: int = 1) -> GyroModel:
    z, deg = np.zeros(3), math.pi / 180
    return GyroModel(GyroConfig(
        sensor_from_body=np.eye(3), scale_error=z + 5e-4, misalignment=np.eye(3), bias_rad_s=np.array([1, -1, 0.5]) * 0.05 * deg,
        bias_temperature_rad_s_per_c=z, bias_temperature_quadratic_rad_s_per_c2=z, reference_temperature_c=20.0,
        temperature_range_c=(-40.0, 85.0), angle_random_walk_rad_s_rthz=z + 0.005 * deg,
        full_scale_rad_s=10.0, seed=seed))   # ARW ~0.005 deg/s/sqrt(Hz): a MEMS-grade ASSUMPTION


def default_magnetometer(seed: int = 2) -> MagnetometerModel:
    z = np.zeros(3)
    return MagnetometerModel(MagnetometerConfig(
        sensor_from_body=np.eye(3), bias_t=z + 100e-9, scale_error=z + 0.005, misalignment=np.eye(3),
        white_noise_sigma_t=z + 100e-9, residual_field_sensor_t=z, bias_temperature_t_per_c=z,
        scale_temperature_per_c=z, reference_temperature_c=20.0, temperature_range_c=(-40.0, 85.0),
        full_scale_t=1e-3, seed=seed))


@dataclass(frozen=True)
class GnssConfig:
    """Orion B-16 style PVT errors (values of examples/orion_b16_assumptions.json; ASSUMPTIONS)."""
    horizontal_cep_m: float = 2.0
    vertical_sigma_m: float = 3.0
    velocity_sigma_m_s: float = 0.1
    position_correlation_s: float = 10.0
    velocity_correlation_s: float = 2.0
    outage_windows_s: tuple = ()          # ((start, end), ...) signal loss; fix returns after reacquisition_s
    reacquisition_s: float = 16.0         # Orion B-16 published reacquisition dwell


class GnssNoiseModel:
    """First-order Gauss-Markov position/velocity errors in the local radial/along/cross frame, GCRS output."""

    def __init__(self, config: GnssConfig | None = None, seed: int = 0):
        self.config, self.rng = config or GnssConfig(), np.random.default_rng(seed)
        self._pos = self._vel = self._r_meas = None
        self._last_t: float | None = None

    @staticmethod
    def _step(prev, sigma, tau, dt, rng):
        innovation = sigma * rng.normal(size=3)
        if prev is None or tau <= 0:
            return innovation
        phi = math.exp(-dt / tau)
        return phi * prev + math.sqrt(1 - phi * phi) * innovation

    def available(self, t: float) -> bool:
        return not any(a <= t < b + self.config.reacquisition_s for a, b in self.config.outage_windows_s)

    def measure(self, t: float, r: np.ndarray, v: np.ndarray):
        """Returns (r, v), or (None, None) while there is no fix."""
        c = self.config
        if not self.available(t):
            self._pos = self._vel = self._r_meas = self._last_t = None
            return None, None
        dt = 1.0 if self._last_t is None else max(t - self._last_t, 1e-9)
        if self._last_t is not None and dt < 1.0 - 1e-9:      # 1 Hz receiver: hold the last fix
            return self._r_meas, self._v_meas
        self._last_t = t
        horizontal = c.horizontal_cep_m / math.sqrt(2 * math.log(2))
        self._pos = self._step(self._pos, np.array([c.vertical_sigma_m, horizontal, horizontal]),
                               c.position_correlation_s, dt, self.rng)
        self._vel = self._step(self._vel, np.full(3, c.velocity_sigma_m_s), c.velocity_correlation_s, dt, self.rng)
        up = r / np.linalg.norm(r)
        cross = np.cross(r, v)
        cross /= np.linalg.norm(cross)
        along = np.cross(cross, up)
        frame = np.column_stack((up, along, cross))
        self._r_meas, self._v_meas = r + frame @ self._pos, v + frame @ self._vel
        return self._r_meas, self._v_meas


@dataclass
class ClosedLoopResult:
    t_s: np.ndarray
    estimation_error_deg: np.ndarray
    pointing_error_deg: np.ndarray       # truth vs ideal nadir reference
    rate_rad_s: np.ndarray               # truth body rate norm
    wheel_momentum_nms: np.ndarray
    dipole_am2: np.ndarray
    mode: list
    sunlit: np.ndarray
    used: list = field(default_factory=list)

    def window(self, start_s: float, end_s: float | None = None) -> slice:
        end = self.t_s[-1] + 1 if end_s is None else end_s
        mask = np.nonzero((self.t_s >= start_s) & (self.t_s <= end))[0]
        return slice(mask[0], mask[-1] + 1)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(a.T @ b) - 1) / 2))))


class ClosedLoopSimulation:
    def __init__(self, env: Environment, spacecraft: SpacecraftConfig, epoch: datetime | None = None,
                 dt_s: float = 1.0, wheels: ReactionWheelArray | None = None, adcs_config: AdcsConfig | None = None,
                 initial_rate_rad_s=(0.0, 0.0, 0.0), initial_attitude_offset_rad=(0.0, 0.0, 0.0),
                 initial_wheel_speeds_rad_s=None, seed: int = 0, use_horizon: bool = True,
                 gnss: GnssConfig | None = GnssConfig(), disturbances: DisturbanceConfig | None = DisturbanceConfig(),
                 thermal: ThermalConfig | None = None, gyro_config: GyroConfig | None = None,
                 magnetometer_config: MagnetometerConfig | None = None, sun_config=None, horizon_config=None):
        self.env, self.dt = env, dt_s
        self.epoch = epoch or datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.wheels = wheels or ReactionWheelArray()
        self.config = adcs_config or AdcsConfig(np.array(spacecraft.inertia_kg_m2), self.wheels)
        self.adcs = AdcsController(self.config)
        self.command = ActuatorCommand()
        self.mtq = MagnetorquerSet(self.config.magnetorquer)
        self.disturbance_model = DisturbanceModel(disturbances) if disturbances is not None else None   # None = none
        self.dynamics = DynamicsAttitudeModel(
            spacecraft, max_step_s=dt_s, wheels=self.wheels, magnetorquers=self.mtq,
            controller=lambda *args: self.command, magnetic_field=lambda t, r: self.env(t).b_gcrs_t,
            initial_rate_body_rad_s=initial_rate_rad_s, initial_attitude_offset_rad=initial_attitude_offset_rad,
            initial_wheel_speeds_rad_s=initial_wheel_speeds_rad_s, disturbance=self._disturbance_torque)
        self.gyro = GyroModel(gyro_config) if gyro_config else default_gyro(seed + 1)
        self.mag = MagnetometerModel(magnetometer_config) if magnetometer_config else default_magnetometer(seed + 2)
        self.gnss = GnssNoiseModel(gnss, seed + 4) if gnss is not None else None   # None = ideal GNSS
        self.thermal = ThermalNode(thermal) if thermal is not None else None   # None = constant 20 C
        self._temperature_c = 20.0
        self.sun_sensor = SunSensorArray(sun_config) if sun_config else SunSensorArray()
        self.horizon = HorizonSensorModel(horizon_config or HorizonSensorConfig(mode="law"), seed=seed + 3) if use_horizon else None

    def _disturbance_torque(self, t, q, omega, r, v):
        if self.disturbance_model is None:
            return np.zeros(3)
        env = self.env(t)
        return self.disturbance_model.torques(quaternion_to_matrix(q), r, v, env.b_gcrs_t, env.sun_gcrs,
                                              env.sunlit).total_nm

    def _inputs(self, t: float, env: EnvSample, truth) -> AdcsInputs:
        rotation = quaternion_to_matrix(truth.quaternion)
        gyro = self.gyro.sample(t, truth.omega_body_rad_s, self._temperature_c)
        mag = self.mag.sample(t, rotation.T @ env.b_gcrs_t, self._temperature_c)
        sun = self.sun_sensor.measure(t, rotation.T @ env.sun_gcrs, env.sunlit)
        nadir, nadir_sigma = None, 0.02
        if self.horizon is not None:
            r_hat = env.r_gcrs_m / np.linalg.norm(env.r_gcrs_m)
            reading = self.horizon.step(t, rotation.T @ -r_hat, np.linalg.norm(env.r_gcrs_m) / 1e3,
                                        math.degrees(math.asin(r_hat[2])))
            if reading is not None and reading.valid:
                nadir = reading.n_body
                nadir_sigma = math.sqrt(max(np.trace(reading.cov_roll_pitch_rad2) / 2, 1e-8))
        r_nav, v_nav = (env.r_gcrs_m, env.v_gcrs_m_s) if self.gnss is None else self.gnss.measure(
            t, env.r_gcrs_m, env.v_gcrs_m_s)
        return AdcsInputs(
            t_s=t, gyro_rad_s=gyro.rate_sensor_rad_s if gyro.valid else None,
            mag_body_t=mag.field_sensor_t if mag.valid else None,
            sun_body=sun.fused.sun_body if sun.fused.valid else None,
            sun_sigma_rad=max(float(sun.fused.sigma_rad), math.radians(0.3)) if sun.fused.valid else 0.01,
            sunlit=env.sunlit, nadir_body=nadir, nadir_sigma_rad=nadir_sigma,
            wheel_speeds_rad_s=truth.wheel_speeds_rad_s,             r_gcrs_m=r_nav, v_gcrs_m_s=v_nav,
            b_ref_gcrs_t=env.b_gcrs_t, sun_ref_gcrs=env.sun_gcrs)

    def run(self, duration_s: float, progress: Callable[[float], None] | None = None) -> ClosedLoopResult:
        steps = int(duration_s / self.dt)
        out = {k: [] for k in ("t", "est", "ptg", "rate", "h", "dip", "mode", "sun", "used")}
        reference = NadirPointingAttitude()
        for k in range(steps):
            t = k * self.dt
            env = self.env(t)
            if self.thermal is not None:
                self._temperature_c = self.thermal.step(self.dt, illumination(env.r_gcrs_m, env.sun_gcrs, env.sunlit))
            truth = self.dynamics.propagate(self.epoch + timedelta(seconds=t), env.r_gcrs_m, env.v_gcrs_m_s)
            command, status = self.adcs.update(self._inputs(t, env, truth))
            self.command = command
            rotation = quaternion_to_matrix(truth.quaternion)
            ref = reference.orientation(None, env.r_gcrs_m, env.v_gcrs_m_s).body_to_gcrs
            out["t"].append(t)
            out["est"].append(angle_deg(quaternion_to_matrix(status.quaternion), rotation) if status.initialised else 180.0)
            out["ptg"].append(angle_deg(rotation, ref))
            out["rate"].append(float(np.linalg.norm(truth.omega_body_rad_s)))
            out["h"].append(float(np.linalg.norm(truth.wheel_momentum_body_nms)))
            out["dip"].append(self.mtq.dipole(command.dipole_am2))
            out["mode"].append(status.mode)
            out["sun"].append(env.sunlit)
            out["used"].append(status.used)
            if progress and k % 500 == 0:
                progress(t / duration_s)
        return ClosedLoopResult(np.array(out["t"]), np.array(out["est"]), np.array(out["ptg"]), np.array(out["rate"]),
                                np.array(out["h"]), np.array(out["dip"]), out["mode"], np.array(out["sun"]), out["used"])

"""Actuator models: a 4-wheel pyramid of custom 10 g reaction wheels and a 3-axis magnetorquer set.

Simple model (imbalance, back-EMF, torque ripple and the current loop are not modelled yet):
    J_w w_dot_i = u_i - tau_friction_i(w_i)             u_i = motor torque, limited to +-tau_max and, at the
                                                        speed limit, to the direction that reduces |w|
    tau_friction = c_coulomb tanh(w/w_eps) + c_viscous w
    reaction torque on the body = -A (u - tau_friction),   wheel momentum h_w = A J_w w
Magnetorquer: tau = m x B_body with each dipole component limited to +-m_max.

The 10 g wheel is a scaled-down Sinclair RW-0.003 (RW-0.003: 50 g, 1 mNm, 3 mNms nominal, 8500 rpm).  The
scaling below is an ASSUMPTION, not a datasheet: same speed limit, geometry scaled by (mass ratio)^(1/3), so
rotor inertia and torque scale as (mass ratio)^(5/3); friction terms are assumed fractions of the torque limit.
"""

from dataclasses import dataclass, field

import numpy as np

RPM_TO_RAD_S = 2.0 * np.pi / 60.0

# Sinclair RW-0.003 (Rocket Lab data sheet, via search summaries): 50 g, 1 mNm, 3 mNms nominal at 8500 rpm
RW0003_MASS_KG = 0.050
RW0003_MAX_TORQUE_NM = 1.0e-3
RW0003_MAX_SPEED_RAD_S = 8500.0 * RPM_TO_RAD_S
RW0003_INERTIA_KG_M2 = 3.0e-3 / RW0003_MAX_SPEED_RAD_S   # inferred from nominal momentum at the speed limit


@dataclass(frozen=True)
class ReactionWheelSpec:
    name: str
    mass_kg: float
    inertia_kg_m2: float
    max_torque_nm: float
    max_speed_rad_s: float
    coulomb_friction_nm: float
    viscous_friction_nm_s: float
    friction_speed_eps_rad_s: float = 1.0

    @property
    def max_momentum_nms(self) -> float:
        return self.inertia_kg_m2 * self.max_speed_rad_s


def scaled_rw0003(mass_kg: float = 0.010) -> ReactionWheelSpec:
    scale = (mass_kg / RW0003_MASS_KG) ** (5.0 / 3.0)
    max_torque = RW0003_MAX_TORQUE_NM * scale
    return ReactionWheelSpec(
        name=f"Custom {mass_kg * 1000:.0f} g wheel (scaled RW-0.003)", mass_kg=mass_kg,
        inertia_kg_m2=RW0003_INERTIA_KG_M2 * scale, max_torque_nm=max_torque,
        max_speed_rad_s=RW0003_MAX_SPEED_RAD_S,
        coulomb_friction_nm=0.02 * max_torque,                                  # assumed: 2 % of max torque
        viscous_friction_nm_s=0.03 * max_torque / RW0003_MAX_SPEED_RAD_S)       # assumed: 3 % at max speed


def pyramid_axes(skew_deg: float = 54.7356) -> np.ndarray:
    """Four unit spin axes (3x4, columns) on a pyramid about +Z; the default skew gives equal momentum on all axes."""
    beta = np.radians(skew_deg)
    azimuths = np.radians([0.0, 90.0, 180.0, 270.0])
    return np.vstack((np.sin(beta) * np.cos(azimuths), np.sin(beta) * np.sin(azimuths), np.full(4, np.cos(beta))))


class ReactionWheelArray:
    def __init__(self, spec: ReactionWheelSpec | None = None, axes: np.ndarray | None = None):
        self.spec = spec or scaled_rw0003()
        self.axes = pyramid_axes() if axes is None else np.array(axes, dtype=float)
        if self.axes.shape[0] != 3:
            raise ValueError("axes must be 3xN")
        self.axes = self.axes / np.linalg.norm(self.axes, axis=0)
        self.count = self.axes.shape[1]
        self.distribution = np.linalg.pinv(self.axes)    # N x 3

    def wheel_torques_for_body_torque(self, body_torque_nm: np.ndarray) -> np.ndarray:
        """Motor torques (N m, per wheel) whose reaction gives the requested torque on the body."""
        return -self.distribution @ np.asarray(body_torque_nm, dtype=float)

    def motor_torque(self, command: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        s = self.spec
        u = np.clip(command, -s.max_torque_nm, s.max_torque_nm)
        # taper the torque over the last 10 % of speed so the limit is smooth for the integrator
        headroom = np.clip((s.max_speed_rad_s - np.abs(speeds)) / (0.10 * s.max_speed_rad_s), 0.0, 1.0)
        return np.where(u * speeds > 0, u * headroom, u)

    def friction(self, speeds: np.ndarray) -> np.ndarray:
        s = self.spec
        return s.coulomb_friction_nm * np.tanh(speeds / s.friction_speed_eps_rad_s) + s.viscous_friction_nm_s * speeds

    def speed_derivative(self, command: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        return (self.motor_torque(command, speeds) - self.friction(speeds)) / self.spec.inertia_kg_m2

    def momentum_body(self, speeds: np.ndarray) -> np.ndarray:
        return self.axes @ (self.spec.inertia_kg_m2 * np.asarray(speeds, dtype=float))

    def reaction_torque_body(self, command: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        return -self.axes @ (self.motor_torque(command, speeds) - self.friction(speeds))


@dataclass(frozen=True)
class MagnetorquerSpec:
    name: str = "Generic 3-axis CubeSat torquer (assumed)"
    max_dipole_am2: float = 0.2     # assumed per-axis limit
    residual_dipole_am2: tuple = field(default=(0.0, 0.0, 0.0))


class MagnetorquerSet:
    """Three orthogonal torquers along the body axes."""

    def __init__(self, spec: MagnetorquerSpec | None = None):
        self.spec = spec or MagnetorquerSpec()

    def dipole(self, command_am2: np.ndarray) -> np.ndarray:
        limit = self.spec.max_dipole_am2
        return np.clip(np.asarray(command_am2, dtype=float), -limit, limit) + np.array(self.spec.residual_dipole_am2)

    def torque(self, command_am2: np.ndarray, magnetic_field_body_t: np.ndarray) -> np.ndarray:
        return np.cross(self.dipole(command_am2), magnetic_field_body_t)


@dataclass(frozen=True)
class ActuatorCommand:
    body_torque_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))   # wanted on the body, via the wheels
    dipole_am2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wheel_torque_offset_nm: np.ndarray | None = None   # per-wheel motor torque added to the distributed command (friction compensation)


def bdot_dipole(b_dot_body_t_s: np.ndarray, gain: float) -> np.ndarray:
    """B-dot detumble law m = -k dB/dt."""
    return -gain * np.asarray(b_dot_body_t_s, dtype=float)

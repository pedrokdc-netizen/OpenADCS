"""Environmental disturbances on a small satellite: aerodynamic drag, solar radiation pressure, residual dipole, J2.

All vectors are SI.  The spacecraft is a set of flat faces (normal, area, lever arm about the centre of mass) in the
body frame.  Gravity gradient lives in ``dynamics.py``; this module adds the other torques and the J2 acceleration.

References: Vallado, *Fundamentals of Astrodynamics and Applications* (exponential atmosphere, Table 8-4; J2);
Wertz, *Spacecraft Attitude Determination and Control* (flat-plate SRP and free-molecular drag).

Every numeric default below (dimensions, coefficients, residual dipole, density scale) is an ASSUMPTION for a
~2U body, not a measured property of Pilot1; replace them from CAD/materials data when available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EARTH_RADIUS_M = 6378137.0
EARTH_ROTATION_RAD_S = 7.2921159e-5
J2 = 1.08262668e-3
MU_EARTH_M3_S2 = 3.986004418e14
SOLAR_PRESSURE_PA = 4.56e-6          # 1361 W/m^2 / c, at 1 AU

# Vallado exponential atmosphere: (base altitude km, density kg/m^3, scale height km)
_ATMOSPHERE = (
    (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554), (50, 1.057e-3, 8.382),
    (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799), (90, 3.396e-6, 5.382),
    (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473), (130, 8.484e-9, 12.636),
    (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523), (180, 5.464e-10, 29.740), (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298), (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835), (700, 3.614e-14, 88.667),
    (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00))


def atmospheric_density(altitude_m: float, scale: float = 1.0) -> float:
    """Exponential density (kg/m^3); ``scale`` multiplies it to represent solar activity (about 0.3 quiet to 5 high)."""
    h = max(altitude_m / 1e3, 0.0)
    base = _ATMOSPHERE[0]
    for row in _ATMOSPHERE:
        if row[0] <= h:
            base = row
    return scale * base[1] * math.exp(-(h - base[0]) / base[2])


def j2_acceleration(r: np.ndarray) -> np.ndarray:
    """J2 perturbing acceleration in an Earth-centred frame whose Z axis is the pole (GCRS to ~0.3 deg)."""
    radius = float(np.linalg.norm(r))
    k = 1.5 * J2 * MU_EARTH_M3_S2 * EARTH_RADIUS_M ** 2 / radius ** 5
    z2 = (r[2] / radius) ** 2
    return k * np.array([r[0] * (5 * z2 - 1), r[1] * (5 * z2 - 1), r[2] * (5 * z2 - 3)])


@dataclass(frozen=True)
class DisturbanceConfig:
    dimensions_m: tuple = (0.20, 0.10, 0.10)          # body X, Y, Z extent (assumed 2U box)
    com_offset_m: tuple = (0.005, 0.005, 0.010)       # centre of mass minus geometric centre (assumed)
    drag_coefficient: float = 2.2
    specular_reflectivity: float = 0.10
    diffuse_reflectivity: float = 0.20
    residual_dipole_am2: tuple = (0.003, -0.002, 0.004)   # body-fixed residual magnetic moment (assumed)
    density_scale: float = 1.0
    drag: bool = True
    srp: bool = True
    residual_magnetic: bool = True


@dataclass(frozen=True)
class DisturbanceTorques:
    aerodynamic_nm: np.ndarray
    solar_nm: np.ndarray
    magnetic_nm: np.ndarray

    @property
    def total_nm(self) -> np.ndarray:
        return self.aerodynamic_nm + self.solar_nm + self.magnetic_nm


class DisturbanceModel:
    def __init__(self, config: DisturbanceConfig | None = None):
        self.config = config or DisturbanceConfig()
        dims = np.asarray(self.config.dimensions_m, float)
        offset = np.asarray(self.config.com_offset_m, float)
        faces = []
        for axis in range(3):
            area = float(np.prod(np.delete(dims, axis)))
            for sign in (1.0, -1.0):
                normal = np.zeros(3)
                normal[axis] = sign
                faces.append((normal, area, normal * dims[axis] / 2 - offset))
        self.faces = faces
        self.dipole = np.asarray(self.config.residual_dipole_am2, float)

    def aerodynamic(self, v_rel_body: np.ndarray, density: float) -> tuple:
        """Free-molecular flat-plate drag; returns (force, torque) about the centre of mass."""
        speed = float(np.linalg.norm(v_rel_body))
        if speed == 0.0:
            return np.zeros(3), np.zeros(3)
        flow = v_rel_body / speed
        force, torque = np.zeros(3), np.zeros(3)
        for normal, area, arm in self.faces:
            cos = float(normal @ flow)
            if cos <= 0.0:       # only leading faces (outward normal towards the oncoming flow) are hit
                continue
            f = -0.5 * density * self.config.drag_coefficient * speed ** 2 * area * cos * flow
            force += f
            torque += np.cross(arm, f)
        return force, torque

    def solar(self, sun_body: np.ndarray) -> tuple:
        """Flat-plate SRP with specular and diffuse reflection; ``sun_body`` is the unit vector towards the Sun."""
        ps, pd = self.config.specular_reflectivity, self.config.diffuse_reflectivity
        force, torque = np.zeros(3), np.zeros(3)
        for normal, area, arm in self.faces:
            cos = float(normal @ sun_body)
            if cos <= 0.0:
                continue
            f = -SOLAR_PRESSURE_PA * area * cos * ((1 - ps) * sun_body + (2 * ps * cos + 2 * pd / 3) * normal)
            force += f
            torque += np.cross(arm, f)
        return force, torque

    def torques(self, rotation_body_to_gcrs: np.ndarray, r_gcrs: np.ndarray, v_gcrs: np.ndarray,
                b_gcrs_t: np.ndarray, sun_gcrs: np.ndarray, sunlit: bool) -> DisturbanceTorques:
        to_body = rotation_body_to_gcrs.T
        aero = solar = np.zeros(3)
        if self.config.drag:
            wind = v_gcrs - np.cross([0.0, 0.0, EARTH_ROTATION_RAD_S], r_gcrs)    # co-rotating atmosphere
            density = atmospheric_density(float(np.linalg.norm(r_gcrs)) - EARTH_RADIUS_M, self.config.density_scale)
            aero = self.aerodynamic(to_body @ wind, density)[1]
        if self.config.srp and sunlit:
            solar = self.solar(to_body @ sun_gcrs)[1]
        magnetic = np.cross(self.dipole, to_body @ b_gcrs_t) if self.config.residual_magnetic else np.zeros(3)
        return DisturbanceTorques(np.asarray(aero), np.asarray(solar), magnetic)

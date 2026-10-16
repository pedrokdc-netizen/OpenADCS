"""Earth-albedo geometry for the Sun sensor heads.

Earth is a Lambertian sphere of Bond albedo ``a`` lit by the Sun.  A head whose face is tilted by
``beta`` from nadir receives ``a * S * cos(zeta) * F(beta, rho)`` of reflected light, where ``F`` is the
exact sphere-to-plane view factor, ``rho`` the Earth angular radius and ``zeta`` the Sun zenith angle at the
sub-satellite point (the lit-fraction of the disk is approximated by the sub-satellite illumination).
ASSUMPTIONS: spherical Earth, uniform albedo, no spectral weighting, no penumbra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6378137.0
EARTH_BOND_ALBEDO = 0.30          # mean Earth Bond albedo (orbit-averaged literature value)


@dataclass(frozen=True)
class AlbedoContext:
    nadir_body: np.ndarray        # unit vector, spacecraft -> Earth centre, body frame
    earth_angle_rad: float        # Earth angular radius seen from the spacecraft
    reflectance: float            # albedo * cos(Sun zenith at the sub-satellite point), 0 on the night side

    def view_factor(self, cos_beta: float) -> float:
        return sphere_view_factor(math.acos(max(-1.0, min(1.0, cos_beta))), self.earth_angle_rad)


def sphere_view_factor(beta: float, rho: float) -> float:
    """View factor from a plane (normal tilted ``beta`` from nadir) to the Earth cap of angular radius ``rho``."""
    if rho <= 0 or beta >= math.pi / 2 + rho:
        return 0.0
    h = 1.0 / math.sin(rho)
    if beta <= math.pi / 2 - rho:
        return math.cos(beta) / h ** 2
    x = math.sqrt(h * h - 1.0)
    y = min(1.0, max(-1.0, x / math.tan(beta))) if beta != math.pi / 2 else 0.0
    root = math.sqrt(1.0 - y * y)
    value = (math.cos(beta) * math.acos(-y) - x * math.sin(beta) * root) / (math.pi * h * h) \
        + math.atan(math.sin(beta) * root / x) / math.pi
    return max(0.0, value)


def albedo_context(position_itrs_m, sun_direction_itrs, earth_center_body_m, coefficient: float = EARTH_BOND_ALBEDO,
                   ) -> AlbedoContext | None:
    """Context for one sample; ``None`` when albedo is disabled (coefficient 0)."""
    if coefficient <= 0:
        return None
    position = np.asarray(position_itrs_m, float)
    distance = float(np.linalg.norm(position))
    sun = np.asarray(sun_direction_itrs, float)
    sun = sun / np.linalg.norm(sun)
    nadir = np.asarray(earth_center_body_m, float)
    nadir = nadir / np.linalg.norm(nadir)
    rho = math.asin(min(1.0, EARTH_RADIUS_M / distance))
    return AlbedoContext(nadir, rho, coefficient * max(0.0, float(sun @ position) / distance))


def context_from_view(sample, view, coefficient: float = EARTH_BOND_ALBEDO) -> AlbedoContext | None:
    return albedo_context(sample.position_itrs_m, sample.sun_direction_itrs, view.earth_center_body_m, coefficient)

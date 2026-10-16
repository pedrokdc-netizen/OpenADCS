"""Orbital illumination and a lumped thermal node: the environment inputs that drive sensor temperature.

Fluxes follow the standard LEO thermal-environment decomposition (direct solar, Earth albedo, Earth IR) with
a sphere-to-plate view factor (R/r)^2 for a nadir-facing surface.  The thermal node is one lumped body, so the
temperature it returns is an orbit-scale trend, not a component-level thermal analysis.  All thermal parameters
are ASSUMPTIONS for a 0.5 kg unit; the fluxes are textbook means (Gilmore, *Spacecraft Thermal Control Handbook*).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SOLAR_CONSTANT_W_M2 = 1361.0
EARTH_IR_W_M2 = 237.0
BOND_ALBEDO = 0.30
STEFAN_BOLTZMANN = 5.670374419e-8
EARTH_RADIUS_M = 6378137.0


@dataclass(frozen=True)
class IlluminationSample:
    solar_w_m2: float          # direct flux on a surface facing the Sun (0 in eclipse)
    albedo_w_m2: float         # reflected flux on a nadir-facing surface
    earth_ir_w_m2: float       # Earth IR on a nadir-facing surface


def illumination(r_gcrs_m: np.ndarray, sun_gcrs: np.ndarray, sunlit: bool) -> IlluminationSample:
    radius = float(np.linalg.norm(r_gcrs_m))
    view = (EARTH_RADIUS_M / radius) ** 2
    cos_zenith = max(float(r_gcrs_m @ sun_gcrs) / radius, 0.0)      # Sun elevation at the sub-satellite point
    return IlluminationSample(SOLAR_CONSTANT_W_M2 if sunlit else 0.0,
                              BOND_ALBEDO * SOLAR_CONSTANT_W_M2 * cos_zenith * view, EARTH_IR_W_M2 * view)


@dataclass(frozen=True)
class ThermalConfig:
    heat_capacity_j_k: float = 450.0            # 0.5 kg x 900 J/kg/K
    absorptivity: float = 0.6
    emissivity: float = 0.8
    sun_area_m2: float = 0.03                   # mean projected area to the Sun, tumbling-averaged
    earth_area_m2: float = 0.02                 # area facing Earth for albedo and IR
    radiating_area_m2: float = 0.10
    internal_power_w: float = 1.0
    initial_temperature_c: float = 20.0


class ThermalNode:
    """Single-node heat balance, integrated with sub-steps (explicit Euler is stable for these time constants)."""

    def __init__(self, config: ThermalConfig | None = None):
        self.config = config or ThermalConfig()
        self.temperature_c = self.config.initial_temperature_c

    def step(self, dt_s: float, flux: IlluminationSample) -> float:
        c = self.config
        kelvin = self.temperature_c + 273.15
        remaining = dt_s
        while remaining > 1e-9:
            h = min(remaining, 5.0)
            absorbed = (c.absorptivity * (c.sun_area_m2 * flux.solar_w_m2 + c.earth_area_m2 * flux.albedo_w_m2)
                        + c.emissivity * c.earth_area_m2 * flux.earth_ir_w_m2 + c.internal_power_w)
            emitted = c.emissivity * STEFAN_BOLTZMANN * c.radiating_area_m2 * kelvin ** 4
            kelvin += h * (absorbed - emitted) / c.heat_capacity_j_k
            remaining -= h
        self.temperature_c = kelvin - 273.15
        return self.temperature_c

    def reset(self) -> None:
        self.temperature_c = self.config.initial_temperature_c

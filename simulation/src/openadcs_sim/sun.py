from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import huira
from huira.units import Meter
from huira.visible8 import Scene
import numpy as np
import spiceypy as spice

from .common import Vector, utc_time, vector3


@dataclass(frozen=True)
class SunState:
    direction_itrs: Vector
    distance_m: float


class HuiraSunModel:
    """Huira solar scene with geometric SPICE Sun direction (no aberration)."""

    def __init__(self, kernels: Iterable[Path]):
        paths = tuple(Path(path).resolve(strict=True) for path in kernels)
        if not paths:
            raise ValueError("Supply leap-second, planetary SPK, and Earth orientation kernels")
        for path in paths:
            # Huira and SpiceyPy have separate CSPICE kernel pools.
            huira.spice.furnsh(path)
            spice.furnsh(str(path))
        self.scene = Scene()
        self.earth_frame = self.scene.root.new_spice_subframe("EARTH", "ITRF93")
        self.sun = self.scene.root.new_instance(self.scene.new_sun_light())
        self.sun.set_spice_origin("SUN")
        self.spacecraft_frame = self.earth_frame.new_subframe()

    def sample(self, timestamp: datetime, position_itrs_m: Vector) -> SunState:
        timestamp = utc_time(timestamp)
        position = vector3(position_itrs_m)
        time = huira.Time(timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f"))
        # ITRF93 is SPICE's realization of ITRS; outputs are geocentric, not barycentric.
        sun_km, _ = spice.spkpos("SUN", time.et(), "ITRF93", "NONE", "EARTH")
        offset = vector3(sun_km) * 1000.0 - position
        distance = float(np.linalg.norm(offset))
        if not np.isfinite(distance) or distance <= 0:
            raise RuntimeError("SPICE returned an invalid spacecraft-to-Sun distance")
        self.spacecraft_frame.set_position(*(Meter(float(component)) for component in position))
        return SunState(vector3(offset / distance), distance)

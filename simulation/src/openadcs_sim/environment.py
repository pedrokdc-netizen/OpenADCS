from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from .common import Vector, utc_time
from .orbit import OrbitState
from .sun import SunState


class OrbitModel(Protocol):
    def propagate(self, timestamp: datetime) -> OrbitState: ...


class MagneticModel(Protocol):
    def field(self, timestamp: datetime, position_itrs_m: Vector) -> Vector: ...


class SunModel(Protocol):
    def sample(self, timestamp: datetime, position_itrs_m: Vector) -> SunState: ...


@dataclass(frozen=True)
class EnvironmentSample:
    time_utc: datetime
    position_itrs_m: Vector
    velocity_itrs_m_s: Vector
    magnetic_field_itrs_t: Vector
    sun_direction_itrs: Vector
    sun_distance_m: float

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "time_utc": self.time_utc.isoformat(),
            "frame": "ITRS",
            "position_m": self.position_itrs_m.tolist(),
            "velocity_m_s": self.velocity_itrs_m_s.tolist(),
            "magnetic_field_t": self.magnetic_field_itrs_t.tolist(),
            "sun_direction": self.sun_direction_itrs.tolist(),
            "sun_distance_m": self.sun_distance_m,
        }


class EnvironmentSimulator:
    def __init__(self, orbit: OrbitModel, magnetic: MagneticModel, sun: SunModel):
        self.orbit = orbit
        self.magnetic = magnetic
        self.sun = sun

    def sample(self, timestamp: datetime) -> EnvironmentSample:
        timestamp = utc_time(timestamp)
        state = self.orbit.propagate(timestamp)
        field = self.magnetic.field(timestamp, state.position_itrs_m)
        sun = self.sun.sample(timestamp, state.position_itrs_m)
        return EnvironmentSample(
            timestamp, state.position_itrs_m, state.velocity_itrs_m_s,
            field, sun.direction_itrs, sun.distance_m,
        )

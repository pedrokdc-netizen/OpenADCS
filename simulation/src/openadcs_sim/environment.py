from dataclasses import dataclass, replace
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
    sun_sensor: object | None = None   # SunSensorMeasurement when a sun-sensor array is attached

    def to_dict(self) -> dict[str, object]:
        record = {
            "schema_version": 1,
            "time_utc": self.time_utc.isoformat(),
            "frame": "ITRS",
            "position_m": self.position_itrs_m.tolist(),
            "velocity_m_s": self.velocity_itrs_m_s.tolist(),
            "magnetic_field_t": self.magnetic_field_itrs_t.tolist(),
            "sun_direction": self.sun_direction_itrs.tolist(),
            "sun_distance_m": self.sun_distance_m,
        }
        if self.sun_sensor is not None:
            record["sun_sensor"] = self.sun_sensor.to_dict()
        return record


class EnvironmentSimulator:
    def __init__(self, orbit: OrbitModel, magnetic: MagneticModel, sun: SunModel,
                 sun_sensor=None, attitude_model=None):
        self.orbit = orbit
        self.magnetic = magnetic
        self.sun = sun
        self.sun_sensor = sun_sensor           # optional SunSensorArray (six OV5640 cube heads)
        self.attitude_model = attitude_model   # body attitude used to express the Sun in the body frame

    def sample(self, timestamp: datetime) -> EnvironmentSample:
        timestamp = utc_time(timestamp)
        state = self.orbit.propagate(timestamp)
        field = self.magnetic.field(timestamp, state.position_itrs_m)
        sun = self.sun.sample(timestamp, state.position_itrs_m)
        sample = EnvironmentSample(
            timestamp, state.position_itrs_m, state.velocity_itrs_m_s,
            field, sun.direction_itrs, sun.distance_m,
        )
        if self.sun_sensor is None:
            return sample
        from .albedo import context_from_view
        from .attitude import spacecraft_view   # lazy: attitude imports this module
        from .horizon_scene import in_eclipse

        view = spacecraft_view(sample, self.attitude_model)
        sunlit = not in_eclipse(sample.position_itrs_m, sample.sun_direction_itrs)
        measurement = self.sun_sensor.measure(
            timestamp.timestamp(), view.sun_body_direction, sunlit,
            context_from_view(sample, view, self.sun_sensor.config.albedo_coefficient))
        return replace(sample, sun_sensor=measurement)


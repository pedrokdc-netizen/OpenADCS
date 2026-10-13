from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import astropy.units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec, WGS72
from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv, verify_checksum

from .common import Vector, utc_time, vector3


@dataclass(frozen=True)
class OrbitState:
    time_utc: datetime
    position_itrs_m: Vector
    velocity_itrs_m_s: Vector


class SGP4Propagator:
    """Vallado SGP4 with TLE-compatible WGS72 gravity and Astropy TEME -> ITRS."""

    def __init__(self, line1: str, line2: str, max_tle_age: timedelta = timedelta(days=7)):
        if max_tle_age <= timedelta(0):
            raise ValueError("Maximum TLE age must be positive")
        lines = (line1.rstrip("\r\n"), line2.rstrip("\r\n"))
        for number, line in enumerate(lines, start=1):
            if (
                len(line) != 69 or not line.isascii()
                or not line.startswith(f"{number} ") or not line[-1].isdigit()
            ):
                raise ValueError("Expected two standard 69-character TLE lines")
            verify_checksum(line)
        if lines[0][2:7] != lines[1][2:7]:
            raise ValueError("TLE lines describe different satellites")
        # The accelerated parser is permissive; first validate with the strict parser.
        twoline2rv(*lines, wgs72)
        self._satellite = Satrec.twoline2rv(*lines, WGS72)
        self.epoch_utc = Time(
            self._satellite.jdsatepoch, self._satellite.jdsatepochF,
            format="jd", scale="utc",
        ).to_datetime(timezone=timezone.utc)
        self.max_tle_age = max_tle_age

    def propagate(self, timestamp: datetime) -> OrbitState:
        timestamp = utc_time(timestamp)
        if abs(timestamp - self.epoch_utc) > self.max_tle_age:
            raise ValueError("Propagation exceeds configured TLE age; supply a recent TLE")
        time = Time(timestamp, scale="utc")
        error, position, velocity = self._satellite.sgp4(time.jd1, time.jd2)
        if error:
            raise RuntimeError(f"SGP4 failed ({error}): {SGP4_ERRORS[error]}")
        representation = CartesianRepresentation(
            vector3(position) * u.km,
            differentials=CartesianDifferential(vector3(velocity) * u.km / u.s),
        )
        fixed = TEME(representation, obstime=time).transform_to(ITRS(obstime=time))
        return OrbitState(
            timestamp,
            vector3(fixed.cartesian.xyz.to_value(u.m)),
            vector3(fixed.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s)),
        )

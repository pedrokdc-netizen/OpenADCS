from datetime import datetime

import astropy.units as u
from astropy.coordinates import EarthLocation
import numpy as np

from . import _xyzgeomag
from .common import Vector, utc_time, vector3


class XYZGeomagModel:
    """WMM2025 main field only: ITRS position in meters, ITRS field in tesla."""

    def field(self, timestamp: datetime, position_itrs_m: Vector) -> Vector:
        timestamp = utc_time(timestamp)
        position = vector3(position_itrs_m)
        if not 2025 <= timestamp.year < 2030:
            raise ValueError("WMM2025 is valid from 2025-01-01 through 2029-12-31")
        location = EarthLocation.from_geocentric(*position, unit=u.m)
        altitude = location.height.to_value(u.m)
        if not -1000 <= altitude <= 850000:
            raise ValueError("WMM2025 requires geodetic altitude between -1 and 850 km")
        start = datetime(timestamp.year, 1, 1, tzinfo=timestamp.tzinfo)
        end = datetime(timestamp.year + 1, 1, 1, tzinfo=timestamp.tzinfo)
        year = timestamp.year + (timestamp - start) / (end - start)
        # XYZgeomag uses float internally; avoid rounding late 2029 into epoch 2030.
        year = min(year, float(np.nextafter(np.float32(2030), np.float32(2025))))
        return vector3(_xyzgeomag.field(year, *position))

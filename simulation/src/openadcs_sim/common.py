from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float64]


def utc_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Simulation timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def vector3(value: Sequence[float] | Vector) -> Vector:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("Expected a finite three-component vector")
    result.setflags(write=False)
    return result

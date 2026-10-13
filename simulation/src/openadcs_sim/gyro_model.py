"""Three-axis MEMS rate-gyro behavioral model (gyro-only IMU).

ASSUMPTION-BASED: the error terms are the standard gyro model used by Basilisk's
`imuSensor` and AcubeSAT's gyro-noise studies (scale factor, misalignment, bias,
angle random walk, bias instability, rate random walk, quantization, saturation),
with a quadratic temperature bias that matches the polynomial corrected by the
flight-derived C99 `gyro.c` (ESTCube-1 style temperature calibration). No
parameter has a default and none is flight validated. The output is the RAW
sensor-frame rate, the input expected by the C99 correction.
"""

from dataclasses import dataclass, field
import math

import numpy as np

SCHEMA_VERSION = 1


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


@dataclass(frozen=True)
class GyroConfig:
    sensor_from_body: np.ndarray
    scale_error: np.ndarray
    misalignment: np.ndarray
    bias_rad_s: np.ndarray
    bias_temperature_rad_s_per_c: np.ndarray
    bias_temperature_quadratic_rad_s_per_c2: np.ndarray
    reference_temperature_c: float
    temperature_range_c: tuple[float, float]
    angle_random_walk_rad_s_rthz: np.ndarray
    full_scale_rad_s: float
    rate_random_walk_rad_s2_rthz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_instability_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bias_instability_time_constant_s: float = 0.0
    resolution_rad_s: float = 0.0
    seed: int = 0

    def __post_init__(self):
        for name in ("sensor_from_body", "misalignment"):
            matrix = np.asarray(getattr(self, name), dtype=float)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite 3x3 matrix")
            object.__setattr__(self, name, matrix)
        rotation = self.sensor_from_body
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9) or np.linalg.det(rotation) <= 0:
            raise ValueError("sensor_from_body must be a proper right-handed rotation")
        for name in ("scale_error", "bias_rad_s", "bias_temperature_rad_s_per_c",
                     "bias_temperature_quadratic_rad_s_per_c2", "angle_random_walk_rad_s_rthz",
                     "rate_random_walk_rad_s2_rthz", "bias_instability_rad_s"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        for name in ("angle_random_walk_rad_s_rthz", "rate_random_walk_rad_s2_rthz",
                     "bias_instability_rad_s"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} must be non-negative")
        if np.any(self.scale_error <= -1):
            raise ValueError("scale_error must keep each axis gain positive")
        low, high = self.temperature_range_c
        if not (math.isfinite(low) and math.isfinite(high) and -273.15 < low < high):
            raise ValueError("temperature_range_c must be a finite increasing range above absolute zero")
        if not (low <= self.reference_temperature_c <= high):
            raise ValueError("reference_temperature_c must lie inside temperature_range_c")
        if not (math.isfinite(self.full_scale_rad_s) and self.full_scale_rad_s > 0):
            raise ValueError("full_scale_rad_s must be positive and finite")
        if not (math.isfinite(self.resolution_rad_s) and self.resolution_rad_s >= 0):
            raise ValueError("resolution_rad_s must be non-negative and finite")
        if not (math.isfinite(self.bias_instability_time_constant_s)
                and self.bias_instability_time_constant_s >= 0):
            raise ValueError("bias_instability_time_constant_s must be non-negative and finite")
        if np.any(self.bias_instability_rad_s > 0) and self.bias_instability_time_constant_s == 0:
            raise ValueError("bias_instability_rad_s requires a positive time constant")


@dataclass(frozen=True)
class GyroReading:
    rate_sensor_rad_s: np.ndarray
    valid: bool
    saturated: bool
    temperature_c: float
    status: str


class GyroModel:
    """Stateful, seeded model; samples must have strictly increasing time."""

    def __init__(self, config: GyroConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._walk = np.zeros(3)
        self._instability = np.zeros(3)
        self._last_time_s: float | None = None

    def sample(self, time_s: float, rate_body_rad_s, temperature_c: float) -> GyroReading:
        config = self.config
        rate = _vector(rate_body_rad_s, "rate_body_rad_s")
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("Gyro samples must have strictly increasing time")
        dt = None if self._last_time_s is None else time_s - self._last_time_s
        self._last_time_s = time_s
        low, high = config.temperature_range_c
        if not (math.isfinite(temperature_c) and low <= temperature_c <= high):
            return GyroReading(np.full(3, np.nan), False, False, temperature_c,
                               "TEMPERATURE_OUT_OF_RANGE")
        delta_t = temperature_c - config.reference_temperature_c
        if dt is not None:
            self._walk += config.rate_random_walk_rad_s2_rthz * math.sqrt(dt) * self._rng.normal(0.0, 1.0, 3)
            self._step_instability(dt)
            white = config.angle_random_walk_rad_s_rthz / math.sqrt(dt) * self._rng.normal(0.0, 1.0, 3)
        else:
            self._instability = self._rng.normal(0.0, 1.0, 3) * config.bias_instability_rad_s
            white = np.zeros(3)
        bias = config.bias_rad_s + config.bias_temperature_rad_s_per_c * delta_t \
            + config.bias_temperature_quadratic_rad_s_per_c2 * delta_t ** 2
        raw = (1 + config.scale_error) * (config.misalignment @ (config.sensor_from_body @ rate)) \
            + bias + self._walk + self._instability + white
        saturated = bool(np.any(np.abs(raw) >= config.full_scale_rad_s))
        raw = np.clip(raw, -config.full_scale_rad_s, config.full_scale_rad_s)
        if config.resolution_rad_s > 0:
            raw = np.round(raw / config.resolution_rad_s) * config.resolution_rad_s
        return GyroReading(raw, True, saturated, temperature_c, "SATURATED" if saturated else "OK")

    def _step_instability(self, dt: float) -> None:
        config = self.config
        sigma = config.bias_instability_rad_s
        if not np.any(sigma > 0):
            return
        phi = math.exp(-dt / config.bias_instability_time_constant_s)
        self._instability = phi * self._instability \
            + math.sqrt(1 - phi * phi) * sigma * self._rng.normal(0.0, 1.0, 3)

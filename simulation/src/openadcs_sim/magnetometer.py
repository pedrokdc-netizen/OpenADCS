"""Three-axis magnetometer behavioral model.

ASSUMPTION-BASED: the structure follows the generic linear instrument error
model used by Basilisk's magnetometer (bias, scale, Gauss-Markov noise,
saturation) extended with misalignment, temperature coefficients, quantization
and a residual-spacecraft-field term. The extensions mirror the ESTCube-1
in-orbit findings (residual magnetic moment biasing the magnetometers and a
temperature dependence calibrated in orbit). No parameter default exists: every
value is a user characterization, and none is flight validated.
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
class MagnetometerConfig:
    sensor_from_body: np.ndarray
    bias_t: np.ndarray
    scale_error: np.ndarray
    misalignment: np.ndarray
    white_noise_sigma_t: np.ndarray
    residual_field_sensor_t: np.ndarray
    bias_temperature_t_per_c: np.ndarray
    scale_temperature_per_c: np.ndarray
    reference_temperature_c: float
    temperature_range_c: tuple[float, float]
    full_scale_t: float
    resolution_t: float = 0.0
    markov_sigma_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    markov_time_constant_s: float = 0.0
    seed: int = 0

    def __post_init__(self):
        rotation = np.asarray(self.sensor_from_body, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("sensor_from_body must be a finite 3x3 rotation")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9) or np.linalg.det(rotation) <= 0:
            raise ValueError("sensor_from_body must be a proper right-handed rotation")
        matrix = np.asarray(self.misalignment, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("misalignment must be a finite 3x3 matrix")
        object.__setattr__(self, "sensor_from_body", rotation)
        object.__setattr__(self, "misalignment", matrix)
        for name in ("bias_t", "scale_error", "white_noise_sigma_t", "residual_field_sensor_t",
                     "bias_temperature_t_per_c", "scale_temperature_per_c", "markov_sigma_t"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if np.any(self.white_noise_sigma_t < 0) or np.any(self.markov_sigma_t < 0):
            raise ValueError("noise sigmas must be non-negative")
        if np.any(self.scale_error <= -1):
            raise ValueError("scale_error must keep each axis gain positive")
        low, high = self.temperature_range_c
        if not (math.isfinite(low) and math.isfinite(high) and -273.15 < low < high):
            raise ValueError("temperature_range_c must be a finite increasing range above absolute zero")
        if not (low <= self.reference_temperature_c <= high):
            raise ValueError("reference_temperature_c must lie inside temperature_range_c")
        if not (math.isfinite(self.full_scale_t) and self.full_scale_t > 0):
            raise ValueError("full_scale_t must be positive and finite")
        if not (math.isfinite(self.resolution_t) and self.resolution_t >= 0):
            raise ValueError("resolution_t must be non-negative and finite")
        if not (math.isfinite(self.markov_time_constant_s) and self.markov_time_constant_s >= 0):
            raise ValueError("markov_time_constant_s must be non-negative and finite")
        if np.any(self.markov_sigma_t > 0) and self.markov_time_constant_s == 0:
            raise ValueError("markov_sigma_t requires a positive markov_time_constant_s")


@dataclass(frozen=True)
class MagnetometerReading:
    field_sensor_t: np.ndarray
    valid: bool
    saturated: bool
    temperature_c: float
    status: str


class MagnetometerModel:
    """Stateful, seeded model; samples must have strictly increasing time."""

    def __init__(self, config: MagnetometerConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._markov = np.zeros(3)
        self._last_time_s: float | None = None

    def sample(self, time_s: float, field_body_t, temperature_c: float) -> MagnetometerReading:
        config = self.config
        field = _vector(field_body_t, "field_body_t")
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("Magnetometer samples must have strictly increasing time")
        dt = None if self._last_time_s is None else time_s - self._last_time_s
        self._last_time_s = time_s
        low, high = config.temperature_range_c
        if not (math.isfinite(temperature_c) and low <= temperature_c <= high):
            return MagnetometerReading(np.full(3, np.nan), False, False, temperature_c,
                                       "TEMPERATURE_OUT_OF_RANGE")
        delta_t = temperature_c - config.reference_temperature_c
        scale = 1 + config.scale_error + config.scale_temperature_per_c * delta_t
        noise_state = self._markov_step(dt)
        ideal = config.sensor_from_body @ field + config.residual_field_sensor_t
        raw = scale * (config.misalignment @ ideal) + config.bias_t \
            + config.bias_temperature_t_per_c * delta_t + noise_state \
            + self._rng.normal(0.0, 1.0, 3) * config.white_noise_sigma_t
        saturated = bool(np.any(np.abs(raw) >= config.full_scale_t))
        raw = np.clip(raw, -config.full_scale_t, config.full_scale_t)
        if config.resolution_t > 0:
            raw = np.round(raw / config.resolution_t) * config.resolution_t
        return MagnetometerReading(raw, True, saturated, temperature_c,
                                   "SATURATED" if saturated else "OK")

    def _markov_step(self, dt: float | None) -> np.ndarray:
        config = self.config
        sigma = config.markov_sigma_t
        if not np.any(sigma > 0):
            return self._markov
        if dt is None:
            self._markov = self._rng.normal(0.0, 1.0, 3) * sigma
        else:
            phi = math.exp(-dt / config.markov_time_constant_s)
            self._markov = phi * self._markov + math.sqrt(1 - phi * phi) * sigma * self._rng.normal(0.0, 1.0, 3)
        return self._markov

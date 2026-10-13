"""Image-based micro Sun sensor ("black sun" CMOS) behavioral model.

UNVALIDATED EXCEPTION: the structure follows Lee and Shah, Sensors 2019, 19(4),
739 (PMC6386896): a pinhole camera whose Sun centroid is found in pixel space,
converted to azimuth/elevation with calibrated intrinsics, and several heads
fused into one body-frame vector. The paper is ground-tested only; no flight
heritage was found. Space effects (radiation, thermal drift, albedo, Earth
glare) are not in the paper, so they are not modeled here. Every parameter is a
user characterization; `PAPER_REFERENCE_NOISE_DEG` is only a reported ground
figure for orientation, never a default.

Fusion is inverse-covariance weighting of independent head measurements. It is
NOT the Covariance Projection Method of the paper, which treats unknown
cross-correlation.
"""

from dataclasses import dataclass
import math

import numpy as np

SCHEMA_VERSION = 1
PAPER_REFERENCE_NOISE_DEG = {"single_raw": 0.125, "fused_three_heads": 0.071}


def _rotation(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 rotation")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9) or np.linalg.det(matrix) <= 0:
        raise ValueError(f"{name} must be a proper right-handed rotation")
    return matrix


def _pair(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2-vector")
    return array


@dataclass(frozen=True)
class BlackSunHeadConfig:
    sensor_from_body: np.ndarray
    image_size_px: tuple[int, int]
    focal_length_px: float
    principal_point_px: tuple[float, float]
    half_fov_deg: float
    centroid_sigma_px: float
    centroid_bias_px: tuple[float, float] = (0.0, 0.0)
    dropout_probability: float = 0.0
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "sensor_from_body", _rotation(self.sensor_from_body, "sensor_from_body"))
        width, height = self.image_size_px
        if not (int(width) == width and int(height) == height and width > 0 and height > 0):
            raise ValueError("image_size_px must be positive integers")
        if not (math.isfinite(self.focal_length_px) and self.focal_length_px > 0):
            raise ValueError("focal_length_px must be positive and finite")
        principal = _pair(self.principal_point_px, "principal_point_px")
        if not (0 <= principal[0] <= width and 0 <= principal[1] <= height):
            raise ValueError("principal_point_px must lie inside the image")
        object.__setattr__(self, "principal_point_px", tuple(principal))
        object.__setattr__(self, "centroid_bias_px", tuple(_pair(self.centroid_bias_px, "centroid_bias_px")))
        if not (math.isfinite(self.half_fov_deg) and 0 < self.half_fov_deg < 90):
            raise ValueError("half_fov_deg must be in (0, 90)")
        if not (math.isfinite(self.centroid_sigma_px) and self.centroid_sigma_px >= 0):
            raise ValueError("centroid_sigma_px must be non-negative and finite")
        if not (0 <= self.dropout_probability <= 1):
            raise ValueError("dropout_probability must be within [0, 1]")


@dataclass(frozen=True)
class BlackSunReading:
    sun_body: np.ndarray
    valid: bool
    centroid_px: tuple[float, float] | None
    sigma_rad: float
    status: str


class BlackSunHeadModel:
    """One camera head. Samples must have strictly increasing time."""

    def __init__(self, config: BlackSunHeadConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._last_time_s: float | None = None

    def sample(self, time_s: float, sun_body, sunlit: bool = True) -> BlackSunReading:
        config = self.config
        sun = np.asarray(sun_body, dtype=float)
        if sun.shape != (3,) or not np.all(np.isfinite(sun)):
            raise ValueError("sun_body must be a finite 3-vector")
        norm = float(np.linalg.norm(sun))
        if norm == 0:
            raise ValueError("sun_body must be non-zero")
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("Samples must have strictly increasing time")
        self._last_time_s = time_s
        nan = np.full(3, np.nan)
        sigma_rad = config.centroid_sigma_px / config.focal_length_px
        if not sunlit:
            return BlackSunReading(nan, False, None, sigma_rad, "ECLIPSE")
        sensor = config.sensor_from_body @ (sun / norm)
        if sensor[2] <= math.cos(math.radians(config.half_fov_deg)):
            return BlackSunReading(nan, False, None, sigma_rad, "OUT_OF_FOV")
        cx, cy = config.principal_point_px
        u = cx + config.focal_length_px * sensor[0] / sensor[2]
        v = cy + config.focal_length_px * sensor[1] / sensor[2]
        width, height = config.image_size_px
        if not (0 <= u <= width and 0 <= v <= height):
            return BlackSunReading(nan, False, None, sigma_rad, "OUT_OF_IMAGE")
        if self._rng.random() < config.dropout_probability:
            return BlackSunReading(nan, False, None, sigma_rad, "CENTROID_LOST")
        u += config.centroid_bias_px[0] + self._rng.normal(0.0, config.centroid_sigma_px)
        v += config.centroid_bias_px[1] + self._rng.normal(0.0, config.centroid_sigma_px)
        if not (0 <= u <= width and 0 <= v <= height):
            return BlackSunReading(nan, False, None, sigma_rad, "OUT_OF_IMAGE")
        measured = np.array([(u - cx) / config.focal_length_px, (v - cy) / config.focal_length_px, 1.0])
        measured /= np.linalg.norm(measured)
        return BlackSunReading(config.sensor_from_body.T @ measured, True, (u, v), sigma_rad, "OK")


def fuse_heads(readings: list[BlackSunReading]) -> BlackSunReading:
    """Inverse-covariance fusion of independent valid head vectors (body frame)."""
    valid = [r for r in readings if r.valid]
    if not valid:
        return BlackSunReading(np.full(3, np.nan), False, None, math.nan, "NO_VALID_HEAD")
    if any(not r.sigma_rad > 0 for r in valid):
        raise ValueError("fusion needs a positive per-head sigma_rad")
    mean = sum(r.sun_body / r.sigma_rad ** 2 for r in valid)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        return BlackSunReading(np.full(3, np.nan), False, None, math.nan, "FUSION_DEGENERATE")
    fused = mean / norm
    sigma = 1.0 / math.sqrt(sum(1.0 / r.sigma_rad ** 2 for r in valid))
    return BlackSunReading(fused, True, None, sigma, "OK")

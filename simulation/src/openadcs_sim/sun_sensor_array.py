"""Six-head OV5640 cube Sun sensor: per-sample measurement, coverage map and run history.

Heads sit on the cube faces (+/-X, +/-Y, +/-Z of the body frame), boresights along the face normals.
Each head is the ``black_sun_sensor`` pinhole centroid model with the OV5640 geometry
(2592x1944 px, 1.4 um, 120 deg diagonal FOV baseline; the vendor f = 1.78 mm is only 103.8 deg as a pure pinhole, so 120 deg stands in for the fisheye).  The centroid error is a user
characterisation (ASSUMPTION; the module datasheet gives none).  Lens distortion, the rolling
shutter, glare and radiation are NOT modelled.  Earth albedo is a simple centroid pull (see `albedo.py`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .albedo import EARTH_BOND_ALBEDO, AlbedoContext, context_from_view
from .black_sun_sensor import (
    CUBE_FACES, OV5640_DIAGONAL_FOV_DEG, BlackSunReading, fuse_heads, ov5640_cube_heads, ov5640_head_config)

HEAD_NAMES = ("+Z", "-Z", "+X", "-X", "+Y", "-Y")


@dataclass(frozen=True)
class SunSensorArrayConfig:
    centroid_sigma_px: float = 0.1
    diagonal_fov_deg: float = OV5640_DIAGONAL_FOV_DEG
    dropout_probability: float = 0.0
    seed: int = 0
    albedo_coefficient: float = EARTH_BOND_ALBEDO     # 0 disables the albedo model
    albedo_leak_fraction: float = 1.0                 # 0 disables the albedo pull (see black_sun_sensor)
    veiling_glare: float = 0.005                      # typical lens veiling-glare ratio (ASSUMPTION)
    exposure_peak_fraction: float = 0.8               # typical auto-exposure Sun peak, fraction of full scale
    threshold_fraction: float = 0.1                   # typical centroid threshold, fraction of full scale
    centroid_bias_sigma_px: float = 0.0               # per-head, per-axis Gaussian residual bias, 1-sigma
    centroid_bias_px: tuple[float, float] = (0.0, 0.0)  # common residual calibration bias on every head
    mount_sigma_deg: float = 0.0                      # per-axis Gaussian head misalignment, 1-sigma
    mount_errors_deg: tuple | None = None             # explicit 6x3 override [deg] (envelope corners)

    def __post_init__(self):
        if not (math.isfinite(self.centroid_sigma_px) and self.centroid_sigma_px > 0):
            raise ValueError("centroid_sigma_px must be positive and finite")
        if not (math.isfinite(self.diagonal_fov_deg) and 0 < self.diagonal_fov_deg < 180):
            raise ValueError("diagonal_fov_deg must be in (0, 180)")
        if not (math.isfinite(self.albedo_coefficient) and 0 <= self.albedo_coefficient <= 1):
            raise ValueError("albedo_coefficient must be within [0, 1]")
        if not (math.isfinite(self.albedo_leak_fraction) and 0 <= self.albedo_leak_fraction <= 1):
            raise ValueError("albedo_leak_fraction must be within [0, 1]")
        if not (math.isfinite(self.centroid_bias_sigma_px) and self.centroid_bias_sigma_px >= 0):
            raise ValueError("centroid_bias_sigma_px must be non-negative and finite")
        if not (math.isfinite(self.mount_sigma_deg) and self.mount_sigma_deg >= 0):
            raise ValueError("mount_sigma_deg must be non-negative and finite")
        if self.mount_errors_deg is not None and np.shape(self.mount_errors_deg) != (6, 3):
            raise ValueError("mount_errors_deg must be 6x3")
        bias = np.asarray(self.centroid_bias_px, dtype=float)
        if bias.shape != (2,) or not np.all(np.isfinite(bias)):
            raise ValueError("centroid_bias_px must be a finite 2-vector")


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _reading_to_dict(reading: BlackSunReading) -> dict:
    return {"valid": reading.valid, "status": reading.status,
            "centroid_px": None if reading.centroid_px is None else [float(v) for v in reading.centroid_px],
            "sigma_rad": _finite(reading.sigma_rad), "albedo_ratio": float(reading.albedo_ratio),
            "albedo_bias_px": [float(v) for v in reading.albedo_bias_px],
            "sun_body": [float(v) for v in reading.sun_body] if reading.valid else None}


def _reading_from_dict(record: dict) -> BlackSunReading:
    vector = np.array(record["sun_body"], dtype=float) if record.get("sun_body") else np.full(3, np.nan)
    centroid = record.get("centroid_px")
    sigma = record.get("sigma_rad")
    return BlackSunReading(vector, bool(record["valid"]), None if centroid is None else tuple(centroid),
                           math.nan if sigma is None else float(sigma), str(record["status"]),
                           float(record.get("albedo_ratio", 0.0)), tuple(record.get("albedo_bias_px", (0.0, 0.0))))


@dataclass(frozen=True)
class SunSensorMeasurement:
    heads: tuple[BlackSunReading, ...]
    fused: BlackSunReading
    sun_true_body: np.ndarray
    sunlit: bool

    @property
    def valid_heads(self) -> int:
        return sum(r.valid for r in self.heads)

    @property
    def error_deg(self) -> float | None:
        """Angle between the fused vector and the true body-frame Sun direction."""
        if not self.fused.valid:
            return None
        cosine = float(np.clip(self.fused.sun_body @ self.sun_true_body, -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def to_dict(self) -> dict:
        error = self.error_deg
        return {"sunlit": self.sunlit, "valid_heads": self.valid_heads,
                "sun_true_body": [float(v) for v in self.sun_true_body],
                "fused": _reading_to_dict(self.fused), "fused_error_deg": error,
                "heads": [_reading_to_dict(r) for r in self.heads]}

    @classmethod
    def from_dict(cls, record: dict) -> "SunSensorMeasurement":
        return cls(tuple(_reading_from_dict(r) for r in record["heads"]), _reading_from_dict(record["fused"]),
                   np.array(record["sun_true_body"], dtype=float), bool(record["sunlit"]))


class SunSensorArray:
    """Stateful six-head array; ``time_s`` must increase strictly between calls."""

    def __init__(self, config: SunSensorArrayConfig | None = None):
        self.config = config or SunSensorArrayConfig()
        self.heads = ov5640_cube_heads(self.config.centroid_sigma_px, self.config.seed,
                                       self.config.dropout_probability, self.config.diagonal_fov_deg,
                                       self.config.albedo_leak_fraction, self.mount_errors_rad(),
                                       self.config.centroid_bias_px, self.centroid_biases_px(),
                                       veiling_glare=self.config.veiling_glare,
                                       exposure_peak_fraction=self.config.exposure_peak_fraction,
                                       threshold_fraction=self.config.threshold_fraction)

    def mount_errors_rad(self) -> np.ndarray:
        """True head misalignments [rad], one rotation vector per head (seeded, reproducible)."""
        if self.config.mount_errors_deg is not None:
            return np.radians(np.asarray(self.config.mount_errors_deg, dtype=float))
        rng = np.random.default_rng(self.config.seed + 7919)
        return np.radians(rng.normal(0.0, self.config.mount_sigma_deg, (6, 3)))

    def centroid_biases_px(self) -> np.ndarray:
        """Per-head residual centroid biases [px] (seeded Gaussian, zero by default)."""
        rng = np.random.default_rng(self.config.seed + 104729)
        return rng.normal(0.0, self.config.centroid_bias_sigma_px, (6, 2))

    def measure(self, time_s: float, sun_body, sunlit: bool = True,
                albedo: AlbedoContext | None = None) -> SunSensorMeasurement:
        truth = np.asarray(sun_body, dtype=float)
        truth = truth / np.linalg.norm(truth)
        readings = [head.sample(time_s, truth, sunlit, albedo) for head in self.heads]
        return SunSensorMeasurement(tuple(readings), fuse_heads(readings), truth, sunlit)


def measure_once(config: SunSensorArrayConfig, sun_body, sunlit: bool = True,
                 albedo: AlbedoContext | None = None) -> SunSensorMeasurement:
    """Single deterministic measurement (fresh heads, seeded by ``config.seed``)."""
    return SunSensorArray(config).measure(0.0, sun_body, sunlit, albedo)


def coverage_grid(config: SunSensorArrayConfig | None = None, az_points: int = 144, el_points: int = 72):
    """Number of heads that see each body-frame direction: (azimuth_deg, elevation_deg, count)."""
    config = config or SunSensorArrayConfig()
    az = np.linspace(-180.0, 180.0, az_points)
    el = np.linspace(-90.0, 90.0, el_points)
    azg, elg = np.meshgrid(np.radians(az), np.radians(el))
    direction = np.stack([np.cos(elg) * np.cos(azg), np.cos(elg) * np.sin(azg), np.sin(elg)], axis=-1)
    count = np.zeros(azg.shape, dtype=int)
    for boresight, x_axis in CUBE_FACES:
        z = np.array(boresight, float)
        x = np.array(x_axis, float)
        head = ov5640_head_config(np.vstack([x, np.cross(z, x), z]), diagonal_fov_deg=config.diagonal_fov_deg)
        sensor = direction @ head.sensor_from_body.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u = head.principal_point_px[0] + head.focal_length_px * sensor[..., 0] / sensor[..., 2]
            v = head.principal_point_px[1] + head.focal_length_px * sensor[..., 1] / sensor[..., 2]
        width, height = head.image_size_px
        count += ((sensor[..., 2] > math.cos(math.radians(head.half_fov_deg)))
                  & (u >= 0) & (u <= width) & (v >= 0) & (v <= height))
    return az, el, count


def coverage_fraction(config: SunSensorArrayConfig | None = None, n: int = 40000, seed: int = 0) -> dict:
    """Monte Carlo share of the sphere seen by >=1 and >=2 heads."""
    rng = np.random.default_rng(seed)
    array = SunSensorArray(SunSensorArrayConfig(
        1e-6, (config or SunSensorArrayConfig()).diagonal_fov_deg))
    counts = np.empty(n, dtype=int)
    for k in range(n):
        s = rng.normal(size=3)
        counts[k] = array.measure(float(k), s / np.linalg.norm(s)).valid_heads
    return {"at_least_one": float(np.mean(counts >= 1)), "at_least_two": float(np.mean(counts >= 2)),
            "max_heads": int(counts.max())}


@dataclass(frozen=True)
class SunSensorHistory:
    t_min: np.ndarray
    valid_heads: np.ndarray
    error_deg: np.ndarray      # NaN where no fused estimate
    sunlit: np.ndarray

    def summary(self) -> dict:
        lit = self.sunlit
        result = {"sunlit_samples": int(lit.sum()), "eclipse_samples": int((~lit).sum())}
        if lit.any():
            err = self.error_deg[lit]
            ok = np.isfinite(err)
            result["availability_sunlit"] = float(ok.mean())
            result["rms_deg"] = float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else math.nan
            result["p95_deg"] = float(np.percentile(err[ok], 95)) if ok.any() else math.nan
            result["multi_head_fraction"] = float(np.mean(self.valid_heads[lit] >= 2))
        return result


def run_history(samples: Sequence, view_of: Callable, config: SunSensorArrayConfig, step_s: float = 60.0,
                progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> SunSensorHistory:
    """Run the array over the stored samples (decimated to ``step_s``)."""
    from .horizon_scene import in_eclipse

    times = np.array([(s.time_utc - samples[0].time_utc).total_seconds() for s in samples])
    chosen, last = [], -1e18
    for i, t in enumerate(times):
        if t - last >= step_s - 1e-6:
            chosen.append(i)
            last = t
    array = SunSensorArray(config)
    valid, error, lit = [], [], []
    for n, i in enumerate(chosen):
        if cancelled is not None and cancelled():
            break
        sample = samples[i]
        sunlit = not in_eclipse(sample.position_itrs_m, sample.sun_direction_itrs)
        view = view_of(sample)
        measurement = array.measure(float(times[i]), view.sun_body_direction, sunlit,
                                    context_from_view(sample, view, config.albedo_coefficient))
        error_deg = measurement.error_deg
        valid.append(measurement.valid_heads)
        error.append(math.nan if error_deg is None else error_deg)
        lit.append(sunlit)
        if progress is not None:
            progress(n + 1, len(chosen))
    count = len(valid)
    return SunSensorHistory(times[chosen][:count] / 60.0, np.array(valid), np.array(error), np.array(lit, bool))

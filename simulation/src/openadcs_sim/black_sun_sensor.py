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


def rotation_from_vector(vector) -> np.ndarray:
    """Rodrigues rotation matrix for a rotation vector [rad]."""
    vector = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    axis = vector / angle
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


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
    albedo_leak_fraction: float = 0.0   # 0 disables the albedo pull; 1 = full weight of above-threshold background
    veiling_glare: float = 0.005        # scattered / direct light ratio of the lens (typical small wide-angle lens)
    exposure_peak_fraction: float = 0.8  # Sun peak as a fraction of full scale (auto-exposure keeps headroom)
    threshold_fraction: float = 0.1     # centroid threshold as a fraction of full scale
    mount_error_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)   # true head rotation error (rotation vector, body frame)

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
        if not (math.isfinite(self.albedo_leak_fraction) and 0 <= self.albedo_leak_fraction <= 1):
            raise ValueError("albedo_leak_fraction must be within [0, 1]")
        if not (0 < self.exposure_peak_fraction <= 1 and 0 <= self.threshold_fraction < self.exposure_peak_fraction):
            raise ValueError("need 0 <= threshold_fraction < exposure_peak_fraction <= 1")
        if not (math.isfinite(self.veiling_glare) and 0 <= self.veiling_glare <= 1):
            raise ValueError("veiling_glare must be within [0, 1]")
        error = np.asarray(self.mount_error_rad, dtype=float)
        if error.shape != (3,) or not np.all(np.isfinite(error)):
            raise ValueError("mount_error_rad must be a finite 3-vector")
        object.__setattr__(self, "mount_error_rad", tuple(float(v) for v in error))


@dataclass(frozen=True)
class BlackSunReading:
    sun_body: np.ndarray
    valid: bool
    centroid_px: tuple[float, float] | None
    sigma_rad: float
    status: str
    albedo_ratio: float = 0.0          # Earth-reflected / direct irradiance on the head face
    albedo_bias_px: tuple[float, float] = (0.0, 0.0)


class BlackSunHeadModel:
    """One camera head. Samples must have strictly increasing time."""

    def __init__(self, config: BlackSunHeadConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        # The head physically sits at R_err @ R_nominal; the estimator only knows R_nominal.
        self._true_from_body = rotation_from_vector(config.mount_error_rad) @ config.sensor_from_body
        self._last_time_s: float | None = None

    def sample(self, time_s: float, sun_body, sunlit: bool = True, albedo=None) -> BlackSunReading:
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
        sensor = self._true_from_body @ (sun / norm)
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
        ratio, bias = self._albedo_bias(albedo, sensor, u, v)
        u += config.centroid_bias_px[0] + bias[0] + self._rng.normal(0.0, config.centroid_sigma_px)
        v += config.centroid_bias_px[1] + bias[1] + self._rng.normal(0.0, config.centroid_sigma_px)
        if not (0 <= u <= width and 0 <= v <= height):
            return BlackSunReading(nan, False, None, sigma_rad, "OUT_OF_IMAGE")
        measured = np.array([(u - cx) / config.focal_length_px, (v - cy) / config.focal_length_px, 1.0])
        measured /= np.linalg.norm(measured)
        return BlackSunReading(config.sensor_from_body.T @ measured, True, (u, v), sigma_rad, "OK", ratio, bias)

    def _albedo_bias(self, albedo, sensor, u: float, v: float):
        """Centroid pull toward the Earth-lit region.

        Earth light reaches the image as veiling glare: background b = glare * (E_albedo / E_sun) * peak.  Only the part
        above the centroid threshold T enters the centroid: gain = leak * clip((b - T) / (peak - T), 0, 1).
        """
        config = self.config
        if albedo is None or albedo.reflectance <= 0 or config.albedo_leak_fraction <= 0:
            return 0.0, (0.0, 0.0)
        nadir = self._true_from_body @ albedo.nadir_body
        ratio = albedo.reflectance * albedo.view_factor(float(nadir[2])) / float(sensor[2])
        if ratio <= 0:
            return 0.0, (0.0, 0.0)
        cx, cy = config.principal_point_px
        reach = math.hypot(*config.image_size_px) / 2.0
        if nadir[2] > 0.05:
            pu, pv = config.focal_length_px * nadir[0] / nadir[2], config.focal_length_px * nadir[1] / nadir[2]
        else:
            pu, pv = nadir[0] * reach, nadir[1] * reach
        radius = math.hypot(pu, pv)
        if radius > reach:
            pu, pv = pu * reach / radius, pv * reach / radius
        peak, threshold = config.exposure_peak_fraction, config.threshold_fraction
        background = config.veiling_glare * ratio * peak
        gain = config.albedo_leak_fraction * min(1.0, max(0.0, (background - threshold) / (peak - threshold)))
        return ratio, (gain * (cx + pu - u), gain * (cy + pv - v))


OV5640_IMAGE_PX = (2592, 1944)
OV5640_STANDARD_DIAGONAL_FOV_DEG = 68.7   # cameramodule.com SINCEREFIRST M5101 lens, TV distortion < 1 %
OV5640_PIXEL_UM = 1.4
OV5640_FOCAL_MM = 1.78   # vendor focal length (aperture 2 mm); the module is sold as a 120 deg fisheye


def diagonal_fov_from_focal(focal_mm: float) -> float:
    """Pinhole diagonal FOV [deg] of the 2592x1944 / 1.4 um sensor for a focal length [mm]."""
    half_diagonal_mm = math.hypot(*OV5640_IMAGE_PX) / 2.0 * OV5640_PIXEL_UM * 1e-3
    return math.degrees(2.0 * math.atan(half_diagonal_mm / focal_mm))


# f = 1.78 mm is 103.7 deg as a pure pinhole; the 120 deg baseline stands in for the fisheye projection.
OV5640_DIAGONAL_FOV_DEG = 120.0   # user-confirmed target; hardware assumed to deliver it (fisheye mapping later)
CUBE_FACES = (  # (boresight, image x axis) in the body frame
    ((0, 0, 1), (1, 0, 0)), ((0, 0, -1), (1, 0, 0)), ((1, 0, 0), (0, 1, 0)),
    ((-1, 0, 0), (0, 1, 0)), ((0, 1, 0), (1, 0, 0)), ((0, -1, 0), (1, 0, 0)))


def ov5640_head_config(sensor_from_body, centroid_sigma_px: float = 0.1, seed: int = 0,
                       dropout_probability: float = 0.0,
                       diagonal_fov_deg: float = OV5640_DIAGONAL_FOV_DEG,
                       albedo_leak_fraction: float = 0.0, mount_error_rad=(0.0, 0.0, 0.0),
                       centroid_bias_px=(0.0, 0.0), **head_options) -> BlackSunHeadConfig:
    """OV5640 head, 2592x1944 px, ideal pinhole with the given diagonal FOV (lens distortion ignored).

    ``centroid_sigma_px`` is an ASSUMPTION; the module datasheet gives none.
    """
    width, height = OV5640_IMAGE_PX
    half_diagonal_px = math.hypot(width, height) / 2.0
    focal = half_diagonal_px / math.tan(math.radians(diagonal_fov_deg) / 2.0)
    return BlackSunHeadConfig(sensor_from_body, OV5640_IMAGE_PX, focal, (width / 2.0, height / 2.0),
                              diagonal_fov_deg / 2.0, centroid_sigma_px,
                              dropout_probability=dropout_probability, seed=seed,
                              centroid_bias_px=centroid_bias_px, albedo_leak_fraction=albedo_leak_fraction,
                              mount_error_rad=mount_error_rad, **head_options)


def ov5640_cube_heads(centroid_sigma_px: float = 0.1, seed: int = 0,
                      dropout_probability: float = 0.0,
                      diagonal_fov_deg: float = OV5640_DIAGONAL_FOV_DEG,
                      albedo_leak_fraction: float = 0.0, mount_errors_rad=None,
                      centroid_bias_px=(0.0, 0.0), centroid_biases_px=None, **head_options) -> list[BlackSunHeadModel]:
    """Six OV5640 heads on the cube faces (+/-X, +/-Y, +/-Z), boresights along the face normals."""
    heads = []
    for index, (boresight, x_axis) in enumerate(CUBE_FACES):
        z = np.array(boresight, dtype=float)
        x = np.array(x_axis, dtype=float)
        rotation = np.vstack([x, np.cross(z, x), z])
        heads.append(BlackSunHeadModel(ov5640_head_config(
            rotation, centroid_sigma_px, seed * 10 + index, dropout_probability, diagonal_fov_deg,
            albedo_leak_fraction, (0.0, 0.0, 0.0) if mount_errors_rad is None else mount_errors_rad[index],
            tuple(np.asarray(centroid_bias_px, float) + (0.0 if centroid_biases_px is None else np.asarray(centroid_biases_px[index], float)),
            ), **head_options)))
    return heads


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

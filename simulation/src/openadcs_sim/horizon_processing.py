"""Horizon-sensor image processing reference (limb detection + nadir estimation).

This is the PYTHON REFERENCE for the future ESP32-S3 firmware: every stage is plain
array arithmetic with fixed iteration counts so it can be ported to C99/float32 and checked
against golden vectors (``dtype=np.float32`` emulates the ESP32-S3 single-precision FPU).

Measurement model.  For a spherical limb the image of the limb is the cone of rays at the
angle ``gamma`` from the nadir ``n``; every limb ray ``d`` satisfies ``d.n = cos(gamma)``.
Two estimators are provided:

* ``fit_cone_nadir``  (flight candidate): range-aided -- ``gamma`` follows from the known orbit
  radius and an assumed limb height, so only the two nadir angles are unknown.  Optionally
  ``gamma`` is solved together with a Gaussian prior (needs two well separated cameras).
* ``conic_nadir``     (reference): free ellipse fit in normalised coordinates followed by the
  eigen-decomposition of the conic, i.e. the sphere special case of the Modenini et al.
  closed-form attitude-from-ellipsoid method.  It needs the limb to span a large arc and is
  ill-conditioned for the 45 x 34 deg field of a single Thermal-45 camera at LEO.  It is
  kept to quantify that statement, not as a flight candidate.

Yaw about the nadir is not observable from a (near) spherical Earth and is not produced.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .horizon_camera import (CameraConfig, EARTH_A_KM, EARTH_F, band_radiance, band_temperature,
                             convolve_separable, limb_radius_km, _gaussian_kernel)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProcessingConfig:
    smooth_sigma_px: float = 0.8
    edge_threshold_sigmas: float = 6.0     # gradient threshold in units of the gradient noise
    dark_side_offset_px: float = 2.0       # limb must have "space" (floor) on its dark side
    dark_side_margin_k: float = 2.5
    gate_deg: float = 5.0                  # accept edge points within this angle of the predicted limb
    min_points: int = 24
    assumed_limb_height_km: float = 38.0
    point_sigma_px: float = 0.35           # a-priori sub-pixel edge accuracy used for weights/covariance
    iterations: int = 8
    huber_k: float = 1.5
    gamma_prior_sigma_deg: float = 0.25    # ~10 km of limb-height uncertainty at 500 km altitude
    min_arc_deg: float = 20.0              # accepted arc length (clock angle) of the detected limb
    plateau_offset_px: float = 5.0         # where the Earth plateau temperature is sampled
    compensate_floor_shift: bool = True    # apply the limb-height table for the sensor floor clip


@dataclass
class LimbPoints:
    pixels: np.ndarray      # (N, 2) sub-pixel (col, row)
    strength: np.ndarray    # (N,) gradient magnitude [K/px]
    candidates: int         # edge candidates before gating (work estimate)
    plateau_k: np.ndarray | None = None   # Earth-side plateau temperature sampled behind each point


@dataclass
class NadirFit:
    valid: bool
    status: str
    n_cam: np.ndarray       # unit vector camera -> Earth centre, camera frame
    cov_tangent: np.ndarray # covariance [rad^2] in the tangent basis ``basis`` (2x2, or 3x3 with gamma)
    basis: np.ndarray       # (3, 2) tangent basis e1, e2 in the camera frame
    gamma_rad: float
    n_points: int
    rms_rad: float
    arc_deg: float
    iterations: int


def gamma_from_geometry(r_km: float, latitude_deg: float, heading_deg: float, clock_rad: float,
                        limb_height_km: float) -> float:
    radius = float(limb_radius_km(latitude_deg, heading_deg, clock_rad, r_km)) + limb_height_km
    return math.asin(min(1.0, radius / r_km))


def _sobel(image: np.ndarray):
    p = np.pad(image, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]) / 8.0
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]) / 8.0
    return gx, gy


def _gradient_noise_gain(kernel: np.ndarray) -> float:
    """RMS gain of (Gaussian smooth, Sobel-x) on white noise of unit variance."""
    smooth = np.outer(kernel, kernel)
    sobel = np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    full = np.zeros((smooth.shape[0] + 2, smooth.shape[1] + 2))
    for a in range(3):
        for b in range(3):
            full[a:a + smooth.shape[0], b:b + smooth.shape[1]] += sobel[a, b] * smooth
    return float(math.sqrt(np.sum(full ** 2)))


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x = np.clip(x, 0, w - 1.0001)
    y = np.clip(y, 0, h - 1.0001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


def repair_dead_pixels(image: np.ndarray, dead_mask: np.ndarray | None) -> np.ndarray:
    """Replace factory-mapped non-functional pixels by the mean of their valid 4-neighbours."""
    if dead_mask is None or not dead_mask.any():
        return image
    valid = (~dead_mask).astype(image.dtype)
    value = np.where(dead_mask, 0, image)
    pv, pw = np.pad(value, 1, mode="edge"), np.pad(valid, 1, mode="edge")
    total = pv[:-2, 1:-1] + pv[2:, 1:-1] + pv[1:-1, :-2] + pv[1:-1, 2:]
    count = pw[:-2, 1:-1] + pw[2:, 1:-1] + pw[1:-1, :-2] + pw[1:-1, 2:]
    fill = np.where(count > 0, total / np.maximum(count, 1), image.mean())
    return np.where(dead_mask, fill, image)


def detect_limb(image_k: np.ndarray, camera: CameraConfig, config: ProcessingConfig,
                dtype=np.float64, dead_mask: np.ndarray | None = None) -> LimbPoints:
    """Sub-pixel gradient-ridge limb detector (non-maximum suppression + parabolic refinement)."""
    image = repair_dead_pixels(np.asarray(image_k, dtype=dtype), dead_mask)
    kernel = _gaussian_kernel(config.smooth_sigma_px).astype(dtype)
    smooth = convolve_separable(image, kernel).astype(dtype)
    gx, gy = _sobel(smooth)
    mag = np.hypot(gx, gy)

    sigma = camera.noise_sigma_k()
    threshold = config.edge_threshold_sigmas * max(sigma * _gradient_noise_gain(kernel), 0.05)

    inner = (slice(2, -2), slice(2, -2))
    jj, ii = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    m0 = mag[inner]
    gxi, gyi = gx[inner], gy[inner]
    norm = np.maximum(m0, 1e-12)
    ux, uy = gxi / norm, gyi / norm
    xi, yi = ii[inner].astype(dtype), jj[inner].astype(dtype)
    mp = _bilinear(mag, xi + ux, yi + uy)
    mm = _bilinear(mag, xi - ux, yi - uy)
    peak = (m0 > threshold) & (m0 >= mp) & (m0 > mm)

    denom = mm - 2 * m0 + mp
    offset = np.where(np.abs(denom) > 1e-12, 0.5 * (mm - mp) / np.where(denom == 0, 1, denom), 0.0)
    peak &= np.abs(offset) <= 0.75
    candidates = int(peak.sum())

    # the limb is the outermost edge: its dark (space) side must read at the sensor floor
    dark = _bilinear(smooth, xi - config.dark_side_offset_px * ux, yi - config.dark_side_offset_px * uy)
    peak &= dark <= camera.floor_k + config.dark_side_margin_k + 2.0 * sigma

    px = (xi + offset * ux)[peak]
    py = (yi + offset * uy)[peak]
    plateau = _bilinear(smooth, xi + config.plateau_offset_px * ux, yi + config.plateau_offset_px * uy)[peak]
    return LimbPoints(np.column_stack([px, py]), m0[peak], candidates, plateau)


#: Limb height of the calibration scene (Modenini 2020 apparent limb).  The table only absorbs the
#: floor / temperature-mapping shift; a mismatch between this and ``assumed_limb_height_km`` stays an error.
REFERENCE_LIMB_HEIGHT_KM = 38.0


@lru_cache(maxsize=64)
def limb_offset_table(camera: CameraConfig, config: ProcessingConfig, r_km: float,
                      azimuth_deg: float = 0.0, off_nadir_deg: float = 66.0):
    """Floor-shift calibration table: plateau temperature [K] -> gamma offset [rad].

    The thermal floor clips the cold side of the limb ramp, so the apparent limb moves with the
    Earth plateau temperature.  The table is built by running the *actual* noiseless chain
    (render -> read -> detect -> fit) at a known nadir and finding, per plateau temperature, the
    scalar gamma offset that zeroes the strong-axis error (the fit is linear in that offset, so
    two runs and a secant give the root).  On the ESP32 it is a ~30 entry constant table; it must
    be re-derived from hardware data during validation.
    """
    import dataclasses
    from .horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraMount, CameraUnit,
                                 SceneTruth, render_radiance)
    mount = CameraMount(azimuth_deg, off_nadir_deg)
    rot = mount.rotation_body_from_cam()
    nadir_body = np.array([0.0, 0.0, 1.0])
    n_cam = rot.T @ nadir_body
    strong = np.array([math.cos(math.radians(azimuth_deg)), math.sin(math.radians(azimuth_deg)), 0.0])
    clean = dataclasses.replace(camera, fpn_sigma_k=0.0, dead_pixel_fraction=0.0, quantization_k=0.0,
                                frames_averaged=max(camera.frames_averaged, 64))
    cfg = dataclasses.replace(config, compensate_floor_shift=False)
    gamma = gamma_from_geometry(r_km, 0.0, 0.0, math.radians(azimuth_deg), REFERENCE_LIMB_HEIGHT_KM)
    temps = np.arange(200.0, 361.0, 8.0)
    gains = np.zeros_like(temps)
    for index, plateau in enumerate(temps):
        atmosphere = AtmosphereConfig(enabled=False, warm_temperature_k=float(plateau), polar_drop_k=0.0)
        rng = np.random.default_rng(1)
        truth = SceneTruth(nadir_body, r_km, 0.0, 0.0)
        radiance = render_radiance(clean, mount, truth, AtmosphereRealization(atmosphere, rng), atmosphere, rng)
        frame = CameraUnit(clean, 1).read_frame(radiance, rng)
        errors = []
        for delta in (0.0, math.radians(1.0)):
            fit, points = estimate_nadir(frame, clean, cfg, n_cam, gamma + delta)
            errors.append(float((rot @ fit.n_cam - nadir_body) @ strong) if fit.valid else np.nan)
        slope = (errors[1] - errors[0]) / math.radians(1.0)
        gains[index] = -errors[0] / slope if np.isfinite(slope) and abs(slope) > 1e-6 else np.nan
    good = np.isfinite(gains)
    return temps[good], gains[good]


def pixels_to_rays(pixels: np.ndarray, camera: CameraConfig) -> np.ndarray:
    cx, cy = camera.principal_point
    d = np.column_stack([(pixels[:, 0] - cx) / camera.focal_px, (pixels[:, 1] - cy) / camera.focal_px,
                         np.ones(len(pixels))])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _tangent_basis(n: np.ndarray) -> np.ndarray:
    seed = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(seed, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return np.column_stack([e1, e2])


def _arc_deg(rays: np.ndarray, n: np.ndarray) -> float:
    basis = _tangent_basis(n)
    clock = np.arctan2(rays @ basis[:, 1], rays @ basis[:, 0])
    if len(clock) < 2:
        return 0.0
    order = np.sort(clock)
    gaps = np.diff(np.concatenate([order, [order[0] + 2 * math.pi]]))
    return math.degrees(2 * math.pi - gaps.max())


def fit_cone_nadir(rays: np.ndarray, n0: np.ndarray, gamma_rad: float, sigma_rad: float,
                   config: ProcessingConfig, solve_gamma: bool = False,
                   gamma_prior_sigma_rad: float | None = None, dtype=np.float64,
                   gamma_offsets: np.ndarray | None = None) -> NadirFit:
    """Robust (Huber) Gauss-Newton fit of ``d.n = cos(gamma)`` for the nadir ``n``."""
    rays = np.asarray(rays, dtype=dtype)
    n = np.asarray(n0, dtype=dtype) / np.linalg.norm(n0)
    gamma = float(gamma_rad)
    gamma0 = gamma
    offsets = np.zeros(len(rays), dtype=dtype) if gamma_offsets is None else np.asarray(gamma_offsets, dtype=dtype)
    prior = gamma_prior_sigma_rad if gamma_prior_sigma_rad else math.radians(config.gamma_prior_sigma_deg)
    n_params = 3 if solve_gamma else 2
    basis = _tangent_basis(n)
    JtWJ = np.eye(n_params)
    scale = sigma_rad
    iterations = 0
    for iterations in range(1, config.iterations + 1):
        basis = _tangent_basis(n)
        f = np.clip(rays @ n, -0.999999, 0.999999)
        angle = np.arccos(f)
        res = angle - (gamma + offsets)
        s = np.sqrt(1.0 - f * f)
        cols = [-(rays @ basis[:, 0]) / s, -(rays @ basis[:, 1]) / s]
        if solve_gamma:
            cols.append(-np.ones_like(s))
        J = np.column_stack(cols)
        scale = max(1.4826 * float(np.median(np.abs(res - np.median(res)))), sigma_rad)
        w = np.minimum(1.0, config.huber_k * scale / np.maximum(np.abs(res), 1e-12))
        JtWJ = (J * w[:, None]).T @ J
        rhs = -(J * w[:, None]).T @ res
        if solve_gamma:
            JtWJ = JtWJ + np.diag([0.0, 0.0, scale ** 2 / prior ** 2])
            rhs = rhs - np.array([0.0, 0.0, scale ** 2 / prior ** 2 * (gamma - gamma0)])
        step = np.linalg.solve(JtWJ + 1e-12 * np.eye(n_params), rhs)
        n = n + step[0] * basis[:, 0] + step[1] * basis[:, 1]
        n = n / np.linalg.norm(n)
        if solve_gamma:
            gamma = gamma + step[2]
        if np.max(np.abs(step[:2])) < 1e-7:
            break
    f = np.clip(rays @ n, -0.999999, 0.999999)
    res = np.arccos(f) - (gamma + offsets)
    cov = scale ** 2 * np.linalg.inv(JtWJ + 1e-12 * np.eye(n_params))
    rms = float(np.sqrt(np.mean(res ** 2)))
    return NadirFit(True, "OK", n.astype(float), np.asarray(cov, dtype=float), basis.astype(float),
                    float(gamma), len(rays), rms, _arc_deg(rays, n), iterations)


def floor_shift_gamma_offsets(plateau_k: np.ndarray, camera: CameraConfig, config: ProcessingConfig,
                              r_km: float, gamma_rad: float, azimuth_deg: float = 0.0,
                              off_nadir_deg: float = 66.0) -> np.ndarray:
    temps, offsets = limb_offset_table(camera, config, round(float(r_km) / 25.0) * 25.0,
                                       azimuth_deg, off_nadir_deg)
    return np.interp(plateau_k, temps, offsets)


def conic_nadir(rays: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Free conic fit (Halir-Flusser) in normalised coordinates -> nadir (camera frame), gamma."""
    if len(rays) < 8:
        return None
    x = rays[:, 0] / rays[:, 2]
    y = rays[:, 1] / rays[:, 2]
    mx, my, s = x.mean(), y.mean(), max(x.std(), y.std(), 1e-9)
    xn, yn = (x - mx) / s, (y - my) / s
    D1 = np.column_stack([xn ** 2, xn * yn, yn ** 2])
    D2 = np.column_stack([xn, yn, np.ones_like(xn)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        M = np.array([M[2] / 2, -M[1], M[0] / 2])
        values, vectors = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    cond = 4 * vectors[0] * vectors[2] - vectors[1] ** 2
    ok = np.where(cond > 0)[0]
    if len(ok) == 0:
        return None
    a1 = np.real(vectors[:, ok[0]])
    a = np.concatenate([a1, T @ a1])
    A, B, C, Dd, E, F = a
    Cn = np.array([[A, B / 2, Dd / 2], [B / 2, C, E / 2], [Dd / 2, E / 2, F]])
    H = np.array([[1 / s, 0, 0], [0, 1 / s, 0], [-mx / s, -my / s, 1.0]])
    Cm = H.T @ Cn @ H
    vals, vecs = np.linalg.eigh(Cm)
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] *= -1
    gaps = [abs(vals[i] - np.mean(np.delete(vals, i))) for i in range(3)]
    k = int(np.argmax(gaps))
    n = vecs[:, k]
    if n[2] < 0:
        n = -n
    ratio = np.mean(np.delete(vals, k)) / (vals[k] if vals[k] != 0 else 1e-30)
    cos2 = -ratio if ratio < 0 else None
    gamma = math.acos(math.sqrt(min(1.0, cos2 / (1 + cos2)))) if cos2 and cos2 > 0 else float("nan")
    return n / np.linalg.norm(n), gamma


def gather_limb_rays(image_k: np.ndarray, camera: CameraConfig, config: ProcessingConfig,
                     n_prior_cam: np.ndarray, gamma_rad: float, dtype=np.float64,
                     dead_mask: np.ndarray | None = None, r_km: float | None = None,
                     mount: tuple[float, float] = (0.0, 66.0)):
    """Detect the limb, gate it around the predicted cone and attach floor-shift offsets.

    Returns ``(rays, gamma_offsets_or_None, points, status)``; ``status`` is "OK" or a failure code.
    """
    points = detect_limb(image_k, camera, config, dtype, dead_mask)
    if len(points.pixels) == 0:
        return np.zeros((0, 3)), None, points, "NO_LIMB"
    rays = pixels_to_rays(points.pixels, camera)
    gate = np.abs(np.arccos(np.clip(rays @ n_prior_cam, -1, 1)) - gamma_rad) < math.radians(config.gate_deg)
    rays = rays[gate]
    points = LimbPoints(points.pixels[gate], points.strength[gate], points.candidates, points.plateau_k[gate])
    if len(rays) < config.min_points:
        return rays, None, points, "TOO_FEW_POINTS"
    offsets = None
    if config.compensate_floor_shift and r_km is not None:
        offsets = floor_shift_gamma_offsets(points.plateau_k, camera, config, r_km, gamma_rad,
                                             mount[0], mount[1])
    return rays, offsets, points, "OK"


def estimate_nadir(image_k: np.ndarray, camera: CameraConfig, config: ProcessingConfig,
                   n_prior_cam: np.ndarray, gamma_rad: float, dtype=np.float64,
                   dead_mask: np.ndarray | None = None, r_km: float | None = None,
                   mount: tuple[float, float] = (0.0, 66.0)) -> tuple[NadirFit, LimbPoints]:
    """Full single-camera chain: detect -> gate around the predicted limb -> robust cone fit."""
    rays, offsets, points, status = gather_limb_rays(image_k, camera, config, n_prior_cam, gamma_rad,
                                                     dtype, dead_mask, r_km, mount)
    if status != "OK":
        return _invalid(status, n_prior_cam, len(rays)), points
    fit = fit_cone_nadir(rays, n_prior_cam, gamma_rad, config.point_sigma_px * camera.ifov_rad, config,
                         dtype=dtype, gamma_offsets=offsets)
    if fit.arc_deg < config.min_arc_deg:
        fit.valid, fit.status = False, "ARC_TOO_SHORT"
    return fit, points


def _invalid(status: str, n_prior: np.ndarray, n_points: int = 0) -> NadirFit:
    return NadirFit(False, status, np.asarray(n_prior, dtype=float), np.full((2, 2), np.inf),
                    _tangent_basis(np.asarray(n_prior, dtype=float)), float("nan"), n_points,
                    float("nan"), 0.0, 0)

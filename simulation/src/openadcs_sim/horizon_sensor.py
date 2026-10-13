"""Infrared limb ("horizon") sensor digital model: 1 or 2 Waveshare Thermal-45 cameras.

Status: NOT FLIGHT PROVEN.  This is an original concept (Modenini 2020 / Nguyen et al. 2020 style
Earth-limb imaging with a low-cost LWIR core), so no flight-heritage noise model exists.  The
model is therefore built bottom-up and every parameter is flagged ASSUMPTION or sourced:

* image acquisition  -> :mod:`horizon_camera`   (MI0802 datasheet noise / floor / pixel pitch)
* image processing   -> :mod:`horizon_processing` (reference chain to be ported to the ESP32-S3)
* this module        -> architectures, ESP32 timing, ADCS-level fusion and the equivalent error law

Measurement: the nadir unit vector in the body frame (roll/pitch follow from it).  Yaw about nadir
is unobservable from the horizon alone.  Body frame: +Z nadir, +X velocity.
    roll  = atan2(-n_y, n_z)        pitch = asin(n_x)

Three architectures (see ``simulation/docs/HORIZON_SENSOR.md``):
    SINGLE       one camera
    INDEPENDENT  two cameras, each fits its own nadir; the ADCS fuses two results
    FUSED        two cameras; camera B sends its (averaged) frame to camera A's ESP32, which
                 fits both limbs jointly (and may refine the limb height gamma)
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraConfig, CameraMount,
                             CameraUnit, SceneTruth, render_radiance)
from .horizon_processing import (ProcessingConfig, _arc_deg, _tangent_basis,
                                 estimate_nadir, fit_cone_nadir, gamma_from_geometry,
                                 gather_limb_rays)


class HorizonArchitecture(str, Enum):
    SINGLE = "single"
    INDEPENDENT = "independent"
    FUSED = "fused"


# ----------------------------------------------------------------------------- ESP32 timing
@dataclass(frozen=True)
class Esp32Config:
    """ESP32-S3 resource model.  ALL cycle costs are ASSUMPTIONS until measured on target
    with ``esp_timer_get_time()`` (the embedded tests in ``simulation/tests`` enforce budgets)."""
    cpu_mhz: float = 240.0                 # ESP32-S3 max CPU clock (Espressif datasheet)
    spi_hz: float = 10e6                   # MI0802 SPI clock: 10 / 14 / 20 MHz min/typ/max
    spi_overhead: float = 1.3              # command/handshake overhead factor, ASSUMPTION
    link: str = "uart"                     # inter-camera link: "uart" (1 port exposed) or "spi"
    uart_baud: float = 2_000_000.0         # ASSUMPTION: 2 Mbaud, 10 bits per byte on the wire
    link_spi_hz: float = 10e6
    cycles_per_pixel_filter: float = 100.0  # 5-tap separable smooth + Sobel + |grad| (sqrt)
    cycles_per_candidate: float = 400.0     # NMS, parabolic refine, bilinear plateau sample
    cycles_per_point_iteration: float = 350.0  # acos, sqrt, 2x2 normal-equation update per point
    cycles_fixed: float = 30_000.0          # dead-pixel repair, setup, UART output
    memory_factor: float = 1.5              # PSRAM / cache-miss penalty on the above
    frame_bytes_per_pixel: int = 2          # uint16 centi-kelvin frame
    sync_skew_s: float = 0.5e-3             # ASSUMPTION: GPIO frame-sync accuracy (fused arch.)

    def frame_bytes(self, camera: CameraConfig) -> int:
        return camera.width * camera.height * self.frame_bytes_per_pixel

    def readout_s(self, camera: CameraConfig) -> float:
        return self.frame_bytes(camera) * 8.0 / self.spi_hz * self.spi_overhead

    def transfer_s(self, n_bytes: int) -> float:
        if self.link == "uart":
            return n_bytes * 10.0 / self.uart_baud
        return n_bytes * 8.0 / self.link_spi_hz * self.spi_overhead

    def processing_s(self, n_pixels: int, candidates: int, points: int, iterations: int) -> float:
        cycles = (self.cycles_per_pixel_filter * n_pixels + self.cycles_per_candidate * candidates
                  + self.cycles_per_point_iteration * points * iterations + self.cycles_fixed)
        return cycles * self.memory_factor / (self.cpu_mhz * 1e6)


@dataclass(frozen=True)
class HorizonTiming:
    """Derived rate / latency budget of one architecture (nominal load)."""
    output_period_s: float
    output_rate_hz: float
    latency_s: float               # exposure-centre -> data available at the ADCS MCU
    readout_s: float
    processing_s: float
    transfer_s: float
    max_rate_dps: float            # above this the limb smears by more than ``smear_budget_px``
    cpu_load: float                # fraction of the output period spent computing on the ESP32


# ----------------------------------------------------------------------------- error law
@dataclass(frozen=True)
class HorizonErrorLaw:
    """Equivalent measurement-error law (per body axis: roll, pitch) extracted by Monte Carlo.

        e_k = bias + sigma_white * w_k + sigma_corr * gm_k,    gm: unit first-order Gauss-Markov

    ``sigma_white`` is the image-sensor + processing noise (noise-only runs, robust sigma);
    ``sigma_corr`` is the scene term (limb height, clouds, cold regions near the sensor floor) with a
    conservative inflation ``max(std, p95/1.96)``.  Because scene quality depends strongly on
    latitude, ``bands`` holds ``(abs_latitude_max_deg, bias_roll, bias_pitch, corr_roll, corr_pitch)``
    rows that override ``bias_deg``/``sigma_corr_deg`` (the all-latitude headline values).
    The correlation time ``tau_s`` is an ASSUMPTION (the limb scene changes with orbital motion).
    """
    bias_deg: tuple[float, float] = (0.0, 0.0)
    sigma_white_deg: tuple[float, float] = (0.1, 0.1)
    sigma_corr_deg: tuple[float, float] = (0.5, 0.5)
    tau_s: float = 120.0
    availability: float = 0.95
    bands: tuple = ()

    def total_sigma_deg(self) -> tuple[float, float]:
        return tuple(math.hypot(w, c) for w, c in zip(self.sigma_white_deg, self.sigma_corr_deg))

    def at_latitude(self, latitude_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """(bias_deg, sigma_corr_deg) for the given latitude."""
        for lat_max, br, bp, cr, cp in self.bands:
            if abs(latitude_deg) < lat_max:
                return np.array([br, bp]), np.array([cr, cp])
        return np.array(self.bias_deg), np.array(self.sigma_corr_deg)


#: 500 km, 300-trial Monte Carlo (seed 7, ``simulation/docs/horizon_montecarlo.json``).  Used by the
#: "law" mode; regenerate with ``python -m openadcs_sim.horizon_montecarlo`` after any model change.
DEFAULT_ERROR_LAWS: dict[HorizonArchitecture, HorizonErrorLaw] = {
    HorizonArchitecture.SINGLE: HorizonErrorLaw(
        bias_deg=(-0.23, -0.429), sigma_white_deg=(0.16, 0.041),
        sigma_corr_deg=(1.884, 0.772), tau_s=120.0, availability=0.82,
        bands=((30.0, 0.026, -0.27, 1.113, 0.643), (60.0, -0.33, -0.523, 1.845, 0.698), (90.1, -0.678, -0.649, 3.329, 1.263))),
    HorizonArchitecture.INDEPENDENT: HorizonErrorLaw(
        bias_deg=(0.218, -0.32), sigma_white_deg=(0.05, 0.048),
        sigma_corr_deg=(1.341, 0.9), tau_s=120.0, availability=0.91,
        bands=((30.0, 0.197, -0.183, 0.363, 0.299), (60.0, 0.3, -0.413, 0.722, 0.65), (90.1, 0.14, -0.422, 2.584, 1.649))),
    HorizonArchitecture.FUSED: HorizonErrorLaw(
        bias_deg=(0.171, -0.261), sigma_white_deg=(0.123, 0.131),
        sigma_corr_deg=(1.348, 0.968), tau_s=120.0, availability=0.91,
        bands=((30.0, 0.111, -0.145, 0.492, 0.555), (60.0, 0.171, -0.313, 0.833, 0.8), (90.1, 0.274, -0.386, 2.508, 1.626))),
}


# ----------------------------------------------------------------------------- config / reading
@dataclass(frozen=True)
class HorizonSensorConfig:
    architecture: HorizonArchitecture = HorizonArchitecture.INDEPENDENT
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    esp32: Esp32Config = field(default_factory=Esp32Config)
    mounts: tuple[CameraMount, ...] = (CameraMount(0.0, 66.0), CameraMount(90.0, 66.0))
    mode: str = "image"                    # "image" (full chain) or "law" (equivalent error law)
    error_law: HorizonErrorLaw | None = None
    nominal_points: int = 100              # nominal processing load for the timing budget
    nominal_candidates: int = 250
    smear_budget_px: float = 1.0
    phase_s: float = 0.0                   # offset of the free-running sample clock

    def __post_init__(self):
        if self.mode not in ("image", "law"):
            raise ValueError("mode must be 'image' or 'law'")
        if len(self.mounts) < self.n_cameras:
            raise ValueError("not enough camera mounts for the architecture")

    @property
    def n_cameras(self) -> int:
        return 1 if self.architecture is HorizonArchitecture.SINGLE else 2

    def timing(self) -> HorizonTiming:
        cam, esp = self.camera, self.esp32
        integration = cam.frames_averaged / cam.fps
        n_px = cam.width * cam.height
        readout = esp.readout_s(cam)
        transfer = 0.0
        if self.architecture is HorizonArchitecture.FUSED:
            transfer = esp.transfer_s(esp.frame_bytes(cam))
            proc = esp.processing_s(2 * n_px, 2 * self.nominal_candidates, 2 * self.nominal_points, 8)
            period = max(integration, transfer + proc + readout)
        else:
            proc = esp.processing_s(n_px, self.nominal_candidates, self.nominal_points, 8)
            period = max(integration, readout + proc)
        if self.architecture is HorizonArchitecture.INDEPENDENT:
            # each camera ships a ~40 byte result to the ADCS MCU, negligible but counted
            transfer = esp.transfer_s(40)
        latency = integration / 2.0 + readout + transfer + proc
        # smear: thermal lag + motion during the averaged exposure
        blur_time = cam.thermal_time_constant_s + (cam.frames_averaged - 1) / (2.0 * cam.fps)
        max_rate = math.degrees(self.smear_budget_px * cam.ifov_rad / blur_time)
        return HorizonTiming(period, 1.0 / period, latency, readout, proc, transfer, max_rate,
                             min(1.0, (proc + readout) / period))


@dataclass
class HorizonCameraResult:
    index: int
    status: str
    n_points: int
    arc_deg: float
    rms_rad: float


@dataclass
class HorizonReading:
    sample_time_s: float           # exposure centre (truth time of the measurement)
    available_time_s: float        # when the ADCS MCU receives it (sample + latency)
    valid: bool
    status: str
    n_body: np.ndarray             # estimated nadir unit vector, body frame
    roll_pitch_deg: np.ndarray     # (2,)
    cov_roll_pitch_rad2: np.ndarray  # (2, 2)
    architecture: HorizonArchitecture
    cameras: list[HorizonCameraResult] = field(default_factory=list)
    gamma_deg: float = float("nan")


# ----------------------------------------------------------------------------- math helpers
def nadir_from_roll_pitch(roll_rad: float, pitch_rad: float) -> np.ndarray:
    return np.array([math.sin(pitch_rad), -math.sin(roll_rad) * math.cos(pitch_rad),
                     math.cos(roll_rad) * math.cos(pitch_rad)])


def roll_pitch_from_nadir(n: np.ndarray) -> np.ndarray:
    return np.array([math.atan2(-n[1], n[2]), math.asin(max(-1.0, min(1.0, n[0])))])


def roll_pitch_covariance(n: np.ndarray, basis: np.ndarray, cov_tangent: np.ndarray) -> np.ndarray:
    """Propagate a tangent-plane covariance on the nadir vector to (roll, pitch)."""
    base = roll_pitch_from_nadir(n)
    eps = 1e-6
    jac = np.zeros((2, 2))
    for k in range(2):
        moved = n + eps * basis[:, k]
        delta = roll_pitch_from_nadir(moved / np.linalg.norm(moved)) - base
        jac[:, k] = (delta + math.pi) % (2 * math.pi) / eps - math.pi / eps if False else delta / eps
    return jac @ cov_tangent[:2, :2] @ jac.T


def fuse_nadir_estimates(estimates: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                         iterations: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Information-form fusion of nadir estimates ``(n_body, basis_body (3x2), cov (2x2))``.

    This is the ADCS-level fusion of the INDEPENDENT architecture (a direct, tiny computation
    that is also suitable for the ADCS MCU).  Returns ``(n, basis, cov)``.
    """
    if not estimates:
        raise ValueError("no estimates to fuse")
    n = np.asarray(estimates[0][0], dtype=float)
    for _ in range(iterations):
        basis = _tangent_basis(n)
        info = np.zeros((2, 2))
        rhs = np.zeros(2)
        for n_i, b_i, c_i in estimates:
            w = np.linalg.inv(np.asarray(c_i, dtype=float))
            h = b_i.T @ basis
            info += h.T @ w @ h
            rhs += h.T @ w @ (-(b_i.T @ n))
        delta = np.linalg.solve(info, rhs)
        n = n + basis @ delta
        n /= np.linalg.norm(n)
    basis = _tangent_basis(n)
    info = np.zeros((2, 2))
    for n_i, b_i, c_i in estimates:
        h = b_i.T @ basis
        info += h.T @ np.linalg.inv(np.asarray(c_i, dtype=float)) @ h
    return n, basis, np.linalg.inv(info)


# ----------------------------------------------------------------------------- the model
class HorizonSensorModel:
    """Stateful, seeded model of the horizon sensor (rate limited, latency and dropouts modelled).

    Call :meth:`step` every simulation tick with the true nadir vector and orbit geometry.  It
    returns the newest reading that has been delivered to the ADCS MCU by ``t_s`` (or ``None``).
    The ADCS side never sees truth; ``n_prior_body`` is the ADCS's current nadir estimate (used
    to gate the limb search) and defaults to the nominal +Z.
    """

    def __init__(self, config: HorizonSensorConfig | None = None, seed: int = 0):
        self.config = config or HorizonSensorConfig()
        self.rng = np.random.default_rng(seed)
        self.units = [CameraUnit(self.config.camera, seed * 7 + 1 + i) for i in range(self.config.n_cameras)]
        self.timing = self.config.timing()
        self._next_capture_s = self.config.phase_s
        self._pending: deque[HorizonReading] = deque()
        self._gm = np.zeros(2)
        self._last_t: float | None = None
        self._law = self.config.error_law or DEFAULT_ERROR_LAWS[self.config.architecture]
        self.last_processing_s = self.timing.processing_s

    # ------------------------------------------------------------------ public
    def step(self, t_s: float, n_body_true: np.ndarray, r_km: float, latitude_deg: float,
             heading_deg: float = 0.0, rate_dps: float = 0.0,
             n_prior_body: np.ndarray | None = None) -> HorizonReading | None:
        if t_s + 1e-12 >= self._next_capture_s:
            reading = self._capture(t_s, np.asarray(n_body_true, dtype=float), r_km, latitude_deg,
                                    heading_deg, rate_dps, n_prior_body)
            self._pending.append(reading)
            period = self.timing.output_period_s
            self._next_capture_s = max(self._next_capture_s + period, t_s + 1e-9)
        delivered = None
        while self._pending and self._pending[0].available_time_s <= t_s + 1e-12:
            delivered = self._pending.popleft()
        return delivered

    def measure_now(self, n_body_true, r_km, latitude_deg, heading_deg=0.0, rate_dps=0.0,
                    n_prior_body=None, t_s: float = 0.0) -> HorizonReading:
        """Process one capture immediately (no rate limiting) - used by the Monte Carlo runs."""
        return self._capture(t_s, np.asarray(n_body_true, dtype=float), r_km, latitude_deg,
                             heading_deg, rate_dps, n_prior_body)

    # ------------------------------------------------------------------ internals
    def _capture(self, t_s, n_true, r_km, lat, heading, rate_dps, n_prior_body) -> HorizonReading:
        timing = self.timing
        sample_t = t_s
        available = t_s + timing.latency_s
        if abs(rate_dps) > timing.max_rate_dps:
            return self._invalid(sample_t, available, "RATE_TOO_HIGH")
        if self.config.mode == "law":
            return self._law_reading(sample_t, available, n_true, t_s, lat)
        return self._image_reading(sample_t, available, n_true, r_km, lat, heading, rate_dps,
                                   n_prior_body)

    def _invalid(self, sample_t, available, status, cams=None) -> HorizonReading:
        return HorizonReading(sample_t, available, False, status, np.array([0.0, 0.0, 1.0]),
                              np.full(2, np.nan), np.full((2, 2), np.inf), self.config.architecture,
                              cams or [])

    def _law_reading(self, sample_t, available, n_true, t_s, latitude_deg) -> HorizonReading:
        law = self._law
        dt = 0.0 if self._last_t is None else max(0.0, t_s - self._last_t)
        self._last_t = t_s
        a = math.exp(-dt / law.tau_s) if law.tau_s > 0 else 0.0
        self._gm = a * self._gm + math.sqrt(max(0.0, 1.0 - a * a)) * self.rng.normal(size=2)
        if self.rng.random() > law.availability:
            return self._invalid(sample_t, available, "NO_LIMB")
        bias, sigma_corr = law.at_latitude(latitude_deg)
        white = np.array(law.sigma_white_deg) * self.rng.normal(size=2)
        err = np.radians(bias + white + sigma_corr * self._gm)
        rp = roll_pitch_from_nadir(n_true) + err
        sig = np.radians(np.hypot(np.array(law.sigma_white_deg), sigma_corr))
        return HorizonReading(sample_t, available, True, "OK", nadir_from_roll_pitch(*rp), np.degrees(rp),
                              np.diag(sig ** 2), self.config.architecture)

    def _render(self, index, n_true, r_km, lat, heading, rate_dps, atm_real, atm_cfg):
        cfg = self.config
        truth = SceneTruth(n_true, r_km, lat, heading)
        radiance = render_radiance(cfg.camera, cfg.mounts[index], truth, atm_real, atm_cfg, self.rng)
        return self.units[index].read_frame(radiance, self.rng, math.radians(abs(rate_dps)))

    def _image_reading(self, sample_t, available, n_true, r_km, lat, heading, rate_dps,
                       n_prior_body) -> HorizonReading:
        cfg = self.config
        atm_cfg = cfg.atmosphere
        atm_real = AtmosphereRealization(atm_cfg, self.rng)
        prior_b = np.array([0.0, 0.0, 1.0]) if n_prior_body is None else np.asarray(n_prior_body, float)
        prior_b = prior_b / np.linalg.norm(prior_b)
        frames = [self._render(i, n_true, r_km, lat, heading, rate_dps, atm_real, atm_cfg)
                  for i in range(cfg.n_cameras)]
        return process_frames(frames, cfg, prior_b, r_km, lat, heading, sample_t, available,
                              [u.dead_mask for u in self.units])


def _gamma_for(mount: CameraMount, r_km: float, lat: float, heading: float, limb_h: float) -> float:
    return gamma_from_geometry(r_km, lat, heading, math.radians(mount.azimuth_deg), limb_h)


def process_frames(frames: list[np.ndarray], cfg: HorizonSensorConfig, prior_b: np.ndarray,
                   r_km: float, lat: float, heading: float, sample_t: float = 0.0,
                   available: float = 0.0, dead_masks: list | None = None,
                   dtype=np.float64) -> HorizonReading:
    """ADCS-side + ESP32-side processing of already captured frames -> one HorizonReading."""
    pc = cfg.processing
    dead_masks = dead_masks or [None] * len(frames)
    arch = cfg.architecture
    results: list[HorizonCameraResult] = []

    def cam_prior(i):
        rot = cfg.mounts[i].rotation_body_from_cam()
        return rot, rot.T @ prior_b, _gamma_for(cfg.mounts[i], r_km, lat, heading, pc.assumed_limb_height_km)

    if arch is HorizonArchitecture.FUSED:
        all_rays, all_off, gammas = [], [], []
        for i, frame in enumerate(frames):
            rot, n_c, gamma = cam_prior(i)
            rays, offsets, points, status = gather_limb_rays(
                frame, cfg.camera, pc, n_c, gamma, dtype, dead_masks[i], r_km,
                (cfg.mounts[i].azimuth_deg, cfg.mounts[i].off_nadir_deg))
            results.append(HorizonCameraResult(i, status, len(rays), 0.0, float("nan")))
            if status == "OK":
                all_rays.append(rays @ rot.T)
                all_off.append(offsets if offsets is not None else np.zeros(len(rays)))
                gammas.append(gamma)
        if not all_rays:
            return HorizonReading(sample_t, available, False, "NO_LIMB", prior_b, np.full(2, np.nan),
                                  np.full((2, 2), np.inf), arch, results)
        rays = np.vstack(all_rays)
        gamma0 = float(np.mean(gammas))
        sigma = pc.point_sigma_px * cfg.camera.ifov_rad
        solve_gamma = len(all_rays) == 2
        fit = fit_cone_nadir(rays, prior_b, gamma0, sigma, pc, solve_gamma=solve_gamma,
                             dtype=dtype, gamma_offsets=np.concatenate(all_off))
        fit.arc_deg = _arc_deg(rays, fit.n_cam)
        for r in results:
            r.arc_deg, r.rms_rad = fit.arc_deg, fit.rms_rad
        valid = fit.arc_deg >= pc.min_arc_deg
        status = "OK" if valid else "ARC_TOO_SHORT"
        if valid and len(all_rays) == 1:
            status = "OK_SINGLE_CAMERA"
        return _reading(fit, fit.n_cam, fit.basis, fit.cov_tangent[:2, :2], valid, status, arch, results,
                        sample_t, available, math.degrees(fit.gamma_rad) if solve_gamma else float("nan"))

    estimates = []
    for i, frame in enumerate(frames):
        rot, n_c, gamma = cam_prior(i)
        fit, _ = estimate_nadir(frame, cfg.camera, pc, n_c, gamma, dtype, dead_masks[i], r_km,
                                (cfg.mounts[i].azimuth_deg, cfg.mounts[i].off_nadir_deg))
        results.append(HorizonCameraResult(i, fit.status, fit.n_points, fit.arc_deg, fit.rms_rad))
        if fit.valid:
            estimates.append((rot @ fit.n_cam, rot @ fit.basis, fit.cov_tangent[:2, :2]))
    if not estimates:
        return HorizonReading(sample_t, available, False, results[0].status if results else "NO_LIMB",
                              prior_b, np.full(2, np.nan), np.full((2, 2), np.inf), arch, results)
    n, basis, cov = fuse_nadir_estimates(estimates)
    status = "OK" if len(estimates) == len(frames) else "OK_SINGLE_CAMERA"
    return _reading(None, n, basis, cov, True, status, arch, results, sample_t, available)


def _reading(fit, n, basis, cov, valid, status, arch, results, sample_t, available,
             gamma_deg=float("nan")) -> HorizonReading:
    n = np.asarray(n, dtype=float)
    return HorizonReading(sample_t, available, valid, status, n, np.degrees(roll_pitch_from_nadir(n)),
                          roll_pitch_covariance(n, basis, cov), arch, results, gamma_deg)

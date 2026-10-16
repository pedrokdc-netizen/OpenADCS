"""Synthetic LWIR Earth-limb image generator for the Waveshare Thermal-45 (MI0802) camera.

DESIGN INTENT: simulate the image-acquisition chain (geometry -> atmospheric limb radiance
-> optics blur -> pixel integration -> thermal noise -> floor clipping -> quantisation) so
that the processing chain in ``horizon_processing`` can be run on realistic frames and an
equivalent error law can be extracted (see ``horizon_montecarlo``).

SOURCES (hardware numbers): Meridian MI0802 datasheet v1.0.1 (80x62, 35 um pitch, 8-14 um,
NETD 125 mK @ 1 FPS, noise RMSE 0.2/0.63/1.09 K @ 1/10/30 FPS, 29.3 FPS max, <0.5 %
non-functional pixels, indicative scene range -20..175 C at gain 1) and the Waveshare
Thermal-45 wiki (45 x 34 deg FOV, ESP32-S3).  Atmospheric limb profile: Modenini et al.,
Sensors 2020, 20, 433 (inverse-S radiance, ~38 km apparent limb, ~76 km width, +/-10 km
variation).

ASSUMPTIONS (not datasheet values, flagged for the hardware-validation phase): PSF sigma,
fixed-pattern noise, quantisation step, cloud statistics, atmosphere correlation angle.
This is a NON-flight-proven sensor concept; nothing here is flight validated.
"""

from dataclasses import dataclass, field
import math

import numpy as np

SCHEMA_VERSION = 1

EARTH_A_KM = 6378.137
EARTH_F = 1.0 / 298.257223563
LWIR_BAND_UM = (8.0, 14.0)
_C1 = 1.191042972e8   # W um^4 m^-2 sr^-1  (2 h c^2)
_C2 = 1.4387769e4     # um K


def _band_tables():
    wavelengths = np.linspace(LWIR_BAND_UM[0], LWIR_BAND_UM[1], 301)
    temps = np.arange(120.0, 520.0001, 0.25)
    planck = _C1 / (wavelengths[None, :] ** 5 * np.expm1(_C2 / (wavelengths[None, :] * temps[:, None])))
    integrate = getattr(np, "trapezoid", None) or np.trapz
    radiance = integrate(planck, wavelengths, axis=1)
    return temps, radiance


_T_TABLE, _L_TABLE = _band_tables()
SUN_TEMPERATURE_K = 5772.0
SUN_RADIUS_KM = 695700.0
AU_KM = 149597870.7


def sun_band_irradiance(distance_au: float = 1.0) -> float:
    """In-band (8-14 um) solar irradiance [W m-2] at the spacecraft (blackbody Sun, 5772 K)."""
    wavelengths = np.linspace(LWIR_BAND_UM[0], LWIR_BAND_UM[1], 301)
    planck = _C1 / (wavelengths ** 5 * np.expm1(_C2 / (wavelengths * SUN_TEMPERATURE_K)))
    integrate = getattr(np, "trapezoid", None) or np.trapz
    radiance = float(integrate(planck, wavelengths))
    solid_angle = math.pi * (SUN_RADIUS_KM / (distance_au * AU_KM)) ** 2
    return radiance * solid_angle


def sun_angular_radius_rad(distance_au: float = 1.0) -> float:
    return math.asin(SUN_RADIUS_KM / (distance_au * AU_KM))


def band_radiance(temperature_k):
    """In-band (8-14 um) blackbody radiance [W m-2 sr-1] by table interpolation."""
    return np.interp(temperature_k, _T_TABLE, _L_TABLE)


def band_temperature(radiance):
    return np.interp(radiance, _L_TABLE, _T_TABLE)


def band_radiance_slope(temperature_k):
    """dL/dT [W m-2 sr-1 K-1] of the in-band radiance."""
    return np.interp(temperature_k, _T_TABLE[:-1] + 0.125, np.diff(_L_TABLE) / np.diff(_T_TABLE))


def limb_radius_km(latitude_deg, heading_deg, clock_rad, r_km):
    """Geocentric radius [km] of the ellipsoid at the limb tangent point seen at a clock angle.

    ``clock_rad`` is measured about nadir from body +X (direction of travel) towards +Y.  The
    tangent point lies at central angle acos(R/r) from the sub-satellite point.
    """
    lam = np.arccos(np.clip(EARTH_A_KM * (1 - EARTH_F / 2) / r_km, -1.0, 1.0))
    lat0 = math.radians(latitude_deg)
    az = math.radians(heading_deg) + np.asarray(clock_rad)
    sin_lat = math.sin(lat0) * np.cos(lam) + math.cos(lat0) * np.sin(lam) * np.cos(az)
    return EARTH_A_KM * (1.0 - EARTH_F * sin_lat ** 2)


@dataclass(frozen=True)
class CameraConfig:
    width: int = 80
    height: int = 62
    hfov_deg: float = 45.0           # Waveshare Thermal-45; square pixels assumed
    pixel_pitch_um: float = 35.0     # MI0802 datasheet
    psf_sigma_px: float = 0.6        # ASSUMPTION: 2-element Si lens F#0.8, no MTF data available
    supersample: int = 3
    fps: float = 10.0                # sensor frame rate (<= 29.3 datasheet, 25 Waveshare)
    frames_averaged: int = 1         # temporal averaging done on the ESP32
    floor_k: float = 253.15          # datasheet "indicative scene temperature" lower end (-20 C)
    ceiling_k: float = 448.15        # +175 C at gain 1
    fpn_sigma_k: float = 0.3         # ASSUMPTION: residual fixed-pattern noise after factory NUC
    dead_pixel_fraction: float = 0.005   # datasheet "non-functional detectors < 0.5 %"
    quantization_k: float = 0.1      # ASSUMPTION: 0.1 K output step
    thermal_time_constant_s: float = 0.012   # ASSUMPTION: typical microbolometer 10-15 ms
    noise_reference_k: float = 303.15        # datasheet noise table is at ~30 C scene
    netd_k: float | None = None              # fixed per-frame NETD [K] at the reference scene; None: MI0802 table
    # Sun in the field of view.  The Sun is a point source (0.27 deg radius, ~0.5 px) whose in-band
    # irradiance saturates the pixels it falls on; the scatter terms are ASSUMPTIONS (no lens data).
    sun_scatter_core_fraction: float = 0.02  # ASSUMPTION: share of the solar flux scattered into a halo
    sun_scatter_core_deg: float = 3.0        # ASSUMPTION
    sun_scatter_wide_fraction: float = 0.005 # ASSUMPTION
    sun_scatter_wide_deg: float = 15.0       # ASSUMPTION
    # Housing temperature: factory NUC is valid at ``housing_reference_k``; away from it the offset drifts.
    housing_reference_k: float = 293.15      # ASSUMPTION
    offset_drift_k_per_k: float = 0.05       # ASSUMPTION: apparent scene offset per kelvin of housing change
    fpn_growth_per_k: float = 0.02           # ASSUMPTION: relative FPN growth per kelvin of housing change

    def __post_init__(self):
        if self.width < 8 or self.height < 8 or self.supersample < 1:
            raise ValueError("invalid image geometry")
        if not (0 < self.hfov_deg < 120):
            raise ValueError("hfov_deg must be in (0, 120)")
        if not (0 < self.fps <= 29.3):
            raise ValueError("fps must be in (0, 29.3] (MI0802 maximum)")
        if self.frames_averaged < 1:
            raise ValueError("frames_averaged must be >= 1")
        if not (self.floor_k < self.ceiling_k):
            raise ValueError("floor_k must be below ceiling_k")

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.focal_px))

    @property
    def ifov_rad(self) -> float:
        return 1.0 / self.focal_px

    @property
    def principal_point(self) -> tuple[float, float]:
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def noise_sigma_k(self, scene_k=None):
        """Temporal noise per output frame [K] (datasheet log-interpolated, /sqrt(N) averaging)."""
        if self.netd_k is not None:
            sigma = float(self.netd_k)
        else:
            sigma = float(np.exp(np.interp(math.log(self.fps), np.log([1.0, 10.0, 30.0]),
                                           np.log([0.2, 0.63, 1.09]))))
        sigma /= math.sqrt(self.frames_averaged)
        if scene_k is None:
            return sigma
        ratio = band_radiance_slope(self.noise_reference_k) / band_radiance_slope(scene_k)
        return sigma * ratio


@dataclass(frozen=True)
class CameraMount:
    """Camera boresight in the body frame (nominal nadir = +Z_b, +X_b = velocity direction)."""
    azimuth_deg: float = 0.0     # clock angle of the boresight about nadir, from +X_b towards +Y_b
    off_nadir_deg: float = 66.0  # boresight angle from nadir (limb centred at the design altitude)

    def rotation_body_from_cam(self) -> np.ndarray:
        theta = math.radians(self.off_nadir_deg)
        phi = math.radians(self.azimuth_deg)
        z = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])


@dataclass(frozen=True)
class AtmosphereConfig:
    limb_height_km: float = 38.0        # Modenini 2020 apparent limb height
    width_10_90_km: float = 76.0        # Modenini 2020 nominal width (taken as 10-90 % width)
    common_height_sigma_km: float = 3.0 # scene-wide limb height offset (season/latitude), ASSUMPTION
    local_height_sigma_km: float = 4.0  # along-limb variation, tuned to ~+/-10 km peak as in the paper
    local_width_sigma_km: float = 10.0
    correlation_angle_deg: float = 10.0 # ASSUMPTION: along-limb correlation angle
    warm_temperature_k: float = 285.0   # ASSUMPTION: clear-limb window radiance temperature, equator
    polar_drop_k: float = 35.0
    cloud_mean_k: float = 8.0           # mean cloud-top depression, ASSUMPTION
    cloud_sigma_k: float = 12.0
    texture_sigma_k: float = 2.0
    # Used only when the scene carries Sun information (``SceneTruth.sun_body``):
    seasonal_polar_k: float = 12.0      # ASSUMPTION: winter pole colder / summer pole warmer by this at 90 deg
    night_drop_k: float = 4.0           # ASSUMPTION: night-side limb radiance temperature drop
    terminator_width: float = 0.15      # sin(elevation) half-width of the day/night transition
    enabled: bool = True


OBLIQUITY_DEG = 23.44


@dataclass(frozen=True)
class SceneTruth:
    n_body: np.ndarray      # unit vector body -> Earth centre
    r_km: float             # geocentric distance of the spacecraft
    latitude_deg: float
    heading_deg: float = 0.0
    sun_body: np.ndarray | None = None   # unit vector spacecraft -> Sun, body frame (None: no Sun effects)
    declination_deg: float = 0.0         # solar declination (season); only used with ``sun_body``
    sun_distance_au: float = 1.0

    def __post_init__(self):
        n = np.asarray(self.n_body, dtype=float)
        if n.shape != (3,) or abs(np.linalg.norm(n) - 1.0) > 1e-6:
            raise ValueError("n_body must be a unit 3-vector")
        object.__setattr__(self, "n_body", n)
        if self.sun_body is not None:
            s = np.asarray(self.sun_body, dtype=float)
            if s.shape != (3,) or abs(np.linalg.norm(s) - 1.0) > 1e-6:
                raise ValueError("sun_body must be a unit 3-vector")
            object.__setattr__(self, "sun_body", s)
        if self.r_km <= EARTH_A_KM * (1 - EARTH_F) + 100:
            raise ValueError("r_km is below any supported orbit")


class AtmosphereRealization:
    """One random draw of along-limb atmosphere/cloud state, indexed by clock angle."""

    GRID = 720

    def __init__(self, config: AtmosphereConfig, rng: np.random.Generator):
        self.config = config
        self.clock = np.linspace(-math.pi, math.pi, self.GRID, endpoint=False)
        if not config.enabled:
            self.height_km = np.full(self.GRID, config.limb_height_km)
            self.width_km = np.full(self.GRID, config.width_10_90_km)
            self.cloud_k = np.zeros(self.GRID)
            self.common_offset_km = 0.0
            return
        step = 2 * math.pi / self.GRID
        rho = math.exp(-step / math.radians(config.correlation_angle_deg))

        def circular_gm():
            white = rng.normal(size=self.GRID)
            spectrum = np.fft.rfft(white)
            freq = np.fft.rfftfreq(self.GRID)
            shape = np.sqrt((1 - rho ** 2) / np.abs(1 - rho * np.exp(-2j * np.pi * freq)) ** 2)
            field = np.fft.irfft(spectrum * shape, n=self.GRID)
            return field / max(field.std(), 1e-9)

        self.common_offset_km = float(rng.normal(0.0, config.common_height_sigma_km))
        self.height_km = config.limb_height_km + self.common_offset_km + config.local_height_sigma_km * circular_gm()
        self.width_km = np.clip(config.width_10_90_km + config.local_width_sigma_km * circular_gm(), 20.0, None)
        self.cloud_k = np.clip(config.cloud_mean_k + config.cloud_sigma_k * circular_gm(), 0.0, 90.0)

    def at(self, clock_rad):
        wrapped = (np.asarray(clock_rad) + math.pi) % (2 * math.pi) - math.pi
        idx = (wrapped + math.pi) / (2 * math.pi) * self.GRID
        i0 = np.floor(idx).astype(int) % self.GRID
        i1 = (i0 + 1) % self.GRID
        frac = idx - np.floor(idx)

        def interp(values):
            return values[i0] * (1 - frac) + values[i1] * frac

        return interp(self.height_km), interp(self.width_km), interp(self.cloud_k)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve_separable(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    radius = len(kernel) // 2
    padded = np.pad(image, radius, mode="edge")
    h, w = image.shape
    out = np.zeros_like(image, dtype=float)
    for k, weight in enumerate(kernel):
        out += weight * padded[radius:radius + h, k:k + w]
    padded = np.pad(out, radius, mode="edge")
    result = np.zeros_like(out)
    for k, weight in enumerate(kernel):
        result += weight * padded[k:k + h, radius:radius + w]
    return result


def pixel_rays(config: CameraConfig, supersample: int = 1) -> np.ndarray:
    """Unit rays in the camera frame, shape (H*S, W*S, 3); pixel centres at integer (i, j)."""
    s = supersample
    cx, cy = config.principal_point
    xs = (np.arange(config.width * s) + 0.5) / s - 0.5
    ys = (np.arange(config.height * s) + 0.5) / s - 0.5
    gx, gy = np.meshgrid((xs - cx) / config.focal_px, (ys - cy) / config.focal_px)
    rays = np.stack([gx, gy, np.ones_like(gx)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def render_radiance(config: CameraConfig, mount: CameraMount, truth: SceneTruth,
                    atmosphere: AtmosphereRealization, atm_config: AtmosphereConfig,
                    rng: np.random.Generator) -> np.ndarray:
    """Pixel-integrated, PSF-blurred in-band radiance image [W m-2 sr-1], shape (H, W)."""
    s = config.supersample
    rays_c = pixel_rays(config, s)
    rays_b = rays_c @ mount.rotation_body_from_cam().T
    n = truth.n_body
    cos_psi = np.clip(rays_b @ n, -1.0, 1.0)
    psi = np.arccos(cos_psi)

    ex = np.array([1.0, 0.0, 0.0]) - n[0] * n
    if np.linalg.norm(ex) < 1e-6:
        ex = np.array([0.0, 1.0, 0.0]) - n[1] * n
    e1 = ex / np.linalg.norm(ex)
    e2 = np.cross(n, e1)
    clock = np.arctan2(rays_b @ e2, rays_b @ e1)

    r = truth.r_km
    tangent_radius = limb_radius_km(truth.latitude_deg, truth.heading_deg, clock, r)
    height = np.where(psi < math.pi / 2, r * np.sin(psi) - tangent_radius, r)

    limb_h, width, cloud = atmosphere.at(clock)
    scale = width / (2.0 * math.log(9.0))
    fraction = 1.0 / (1.0 + np.exp(np.clip((height - limb_h) / scale, -60, 60)))

    sin_lat = np.sin(np.radians(truth.latitude_deg)) * 1.0
    lat_t = np.arcsin(np.clip(
        sin_lat * math.cos(math.acos(min(1.0, EARTH_A_KM / r)))
        + math.cos(math.radians(truth.latitude_deg)) * math.sin(math.acos(min(1.0, EARTH_A_KM / r)))
        * np.cos(math.radians(truth.heading_deg) + clock), -1, 1))
    earth_k = atm_config.warm_temperature_k - atm_config.polar_drop_k * np.sin(lat_t) ** 2 - cloud
    if truth.sun_body is not None:
        # season: the winter pole is colder, the summer pole warmer
        winter = -np.sign(lat_t) * math.sin(math.radians(truth.declination_deg)) / math.sin(
            math.radians(OBLIQUITY_DEG))
        earth_k = earth_k - atm_config.seasonal_polar_k * winter * np.sin(lat_t) ** 2
        # day/night: sun elevation at the tangent point
        lam = math.acos(min(1.0, EARTH_A_KM / r))
        tangent_dir = (-math.cos(lam) * n[None, None, :]
                       + math.sin(lam) * (np.cos(clock)[..., None] * e1 + np.sin(clock)[..., None] * e2))
        sin_elevation = tangent_dir @ truth.sun_body
        night = 1.0 / (1.0 + np.exp(np.clip(sin_elevation / atm_config.terminator_width, -60, 60)))
        earth_k = earth_k - atm_config.night_drop_k * night
    if atm_config.enabled:
        earth_k = earth_k + atm_config.texture_sigma_k * rng.normal(size=earth_k.shape)
    earth_k = np.clip(earth_k, 150.0, 330.0)

    radiance = fraction * band_radiance(earth_k) + (1 - fraction) * band_radiance(3.0)
    blurred = convolve_separable(radiance, _gaussian_kernel(config.psf_sigma_px * s))
    h, w = config.height, config.width
    image = blurred.reshape(h, s, w, s).mean(axis=(1, 3))
    if truth.sun_body is not None:
        image = image + sun_radiance_image(config, mount, truth)
    return image


def sun_visibility(truth: SceneTruth) -> float:
    """1 when the Sun disc is fully visible from the spacecraft, 0 when occulted by the Earth."""
    if truth.sun_body is None:
        return 0.0
    earth_angle = math.asin(min(1.0, EARTH_A_KM / truth.r_km))
    nadir_angle = math.acos(float(np.clip(truth.sun_body @ truth.n_body, -1.0, 1.0)))
    disc = sun_angular_radius_rad(truth.sun_distance_au)
    return float(np.clip((nadir_angle - earth_angle) / (2.0 * disc) + 0.5, 0.0, 1.0))


def sun_camera_direction(mount: CameraMount, sun_body: np.ndarray) -> np.ndarray:
    return mount.rotation_body_from_cam().T @ np.asarray(sun_body, dtype=float)


def sun_pixel(config: CameraConfig, sun_cam: np.ndarray) -> tuple[float, float] | None:
    """Pinhole projection of the Sun direction (None when it is behind the image plane)."""
    if sun_cam[2] <= 0.05:
        return None
    cx, cy = config.principal_point
    return cx + config.focal_px * sun_cam[0] / sun_cam[2], cy + config.focal_px * sun_cam[1] / sun_cam[2]


def sun_in_fov(config: CameraConfig, sun_cam: np.ndarray, margin_px: float = 0.0) -> bool:
    pixel = sun_pixel(config, sun_cam)
    return (pixel is not None and -margin_px <= pixel[0] <= config.width - 1 + margin_px
            and -margin_px <= pixel[1] <= config.height - 1 + margin_px)


def sun_radiance_image(config: CameraConfig, mount: CameraMount, truth: SceneTruth) -> np.ndarray:
    """Equivalent in-band radiance [W m-2 sr-1] of the Sun's PSF core and optical scatter per pixel.

    A pixel illuminated by solar irradiance E through a normalised point-spread function psi
    [sr-1] reads like an extended source of radiance E*psi, so the (sub-pixel) Sun disc simply
    adds ``E * psi(theta)``.  Out-of-field Suns still add the scatter terms (stray light).
    """
    visibility = sun_visibility(truth)
    if visibility <= 0.0:
        return 0.0
    sun_cam = sun_camera_direction(mount, truth.sun_body)
    if sun_cam[2] <= -0.2:
        return 0.0
    rays = pixel_rays(config, 1)
    theta = np.arccos(np.clip(rays @ sun_cam, -1.0, 1.0))

    def gaussian(sigma):
        return np.exp(-0.5 * (theta / sigma) ** 2) / (2.0 * math.pi * sigma ** 2)

    f1, f2 = config.sun_scatter_core_fraction, config.sun_scatter_wide_fraction
    psi = ((1.0 - f1 - f2) * gaussian(config.psf_sigma_px * config.ifov_rad)
           + f1 * gaussian(math.radians(config.sun_scatter_core_deg))
           + f2 * gaussian(math.radians(config.sun_scatter_wide_deg)))
    return visibility * sun_band_irradiance(truth.sun_distance_au) * psi


@dataclass
class CameraUnit:
    """Per-unit static hardware imperfections (fixed for the life of one simulated camera)."""
    config: CameraConfig
    seed: int = 0
    fpn_k: np.ndarray = field(init=False)
    dead_mask: np.ndarray = field(init=False)
    dead_value_k: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        shape = (self.config.height, self.config.width)
        self.fpn_k = rng.normal(0.0, self.config.fpn_sigma_k, shape)
        self.dead_mask = rng.random(shape) < self.config.dead_pixel_fraction
        self.dead_value_k = rng.uniform(self.config.floor_k, 330.0, shape)

    def read_frame(self, radiance: np.ndarray, rng: np.random.Generator,
                   body_rate_rad_s: float = 0.0, housing_delta_k: float = 0.0) -> np.ndarray:
        """Radiance -> sensor output [K]: smear, floor clip, noise, FPN, dead pixels, quantisation.

        ``housing_delta_k`` is the camera housing temperature minus the NUC reference: it adds an
        apparent scene offset and grows the fixed-pattern noise (ASSUMED linear coefficients).
        """
        cfg = self.config
        if body_rate_rad_s > 0:
            smear_px = body_rate_rad_s * cfg.thermal_time_constant_s / cfg.ifov_rad
            if smear_px > 0.05:
                # first-order thermal lag along the image motion direction is approximated by an
                # isotropic extra blur (conservative for a rate that is unknown in direction)
                radiance = convolve_separable(radiance, _gaussian_kernel(max(smear_px / 2.0, 0.2)))
        temp = np.clip(band_temperature(radiance), cfg.floor_k, cfg.ceiling_k)
        fpn_scale = 1.0 + cfg.fpn_growth_per_k * abs(housing_delta_k)
        temp = (temp + rng.normal(size=temp.shape) * cfg.noise_sigma_k(temp) + self.fpn_k * fpn_scale
                + cfg.offset_drift_k_per_k * housing_delta_k)
        temp = np.where(self.dead_mask, self.dead_value_k, temp)
        temp = np.clip(temp, cfg.floor_k - 3.0, cfg.ceiling_k)
        if cfg.quantization_k > 0:
            temp = np.round(temp / cfg.quantization_k) * cfg.quantization_k
        return temp

"""Orbit/Sun geometry that drives the horizon sensor model (eclipse, Sun in FOV, season, housing).

Used by the GUI Horizon tab and the scenario Monte Carlo.  Everything here is truth-side geometry;
the sensor model itself (``horizon_sensor``) never sees it except through ``SceneEnvironment``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .horizon_camera import AU_KM, EARTH_A_KM, sun_camera_direction, sun_in_fov, sun_pixel, sun_visibility, SceneTruth
from .horizon_sensor import (HorizonArchitecture, HorizonSensorConfig, HorizonSensorModel, SceneEnvironment,
                             nadir_from_roll_pitch)

HOUSING_SUNLIT_K = 293.15     # ASSUMPTION: camera housing equilibrium in sunlight
HOUSING_ECLIPSE_K = 268.15    # ASSUMPTION: equilibrium after a long eclipse
HOUSING_TAU_S = 1800.0        # ASSUMPTION: housing thermal time constant


def rotation_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def in_eclipse(position_m: np.ndarray, sun_direction: np.ndarray) -> bool:
    """Cylindrical-shadow eclipse test (the umbra/penumbra width is ~0.3 deg of orbit at LEO)."""
    r = float(np.linalg.norm(position_m)) / 1000.0
    along = float(np.dot(position_m / 1000.0, sun_direction))
    return along < 0.0 and math.sqrt(max(r * r - along * along, 0.0)) < EARTH_A_KM


def local_heading_deg(position_m: np.ndarray, velocity_m_s: np.ndarray) -> float:
    """Velocity azimuth from local north (0 = north, 90 = east, 180 = south)."""
    up = position_m / np.linalg.norm(position_m)
    east = np.cross([0.0, 0.0, 1.0], up)
    if np.linalg.norm(east) < 1e-9:
        return 0.0
    east /= np.linalg.norm(east)
    north = np.cross(up, east)
    return math.degrees(math.atan2(float(velocity_m_s @ east), float(velocity_m_s @ north)))


@dataclass(frozen=True)
class SceneGeometry:
    """Truth scene seen by the spacecraft at one sample (body frame = nadir-pointing LVLH)."""
    n_body: np.ndarray
    sun_body: np.ndarray
    r_km: float
    latitude_deg: float
    heading_deg: float
    declination_deg: float
    sun_distance_au: float
    eclipse: bool
    beta_deg: float
    attitude_offset: np.ndarray = field(default_factory=lambda: np.eye(3))

    def environment(self, housing_delta_k: float = 0.0) -> SceneEnvironment:
        return SceneEnvironment(self.sun_body, self.declination_deg, self.sun_distance_au, housing_delta_k)


def scene_from_sample(sample, view, offset_deg: tuple[float, float] = (0.0, 0.0)) -> SceneGeometry:
    """Build the geometry from an ``EnvironmentSample`` and its ``BodyView``.

    ``offset_deg`` = (roll, pitch) true attitude offset of the spacecraft from nadir pointing, so the
    estimator is not handed the truth as its prior.
    """
    position, velocity = sample.position_itrs_m, sample.velocity_itrs_m_s
    r_km = float(np.linalg.norm(position)) / 1000.0
    offset = rotation_x(math.radians(offset_deg[0])) @ rotation_y(math.radians(offset_deg[1]))
    sun = np.asarray(sample.sun_direction_itrs, float)
    sun = sun / np.linalg.norm(sun)
    n_body = offset.T @ (np.asarray(view.earth_center_body_m, float) / np.linalg.norm(view.earth_center_body_m))
    sun_body = offset.T @ np.asarray(view.sun_body_direction, float)
    orbit_normal = np.cross(position, velocity)
    orbit_normal /= np.linalg.norm(orbit_normal)
    return SceneGeometry(
        n_body / np.linalg.norm(n_body), sun_body / np.linalg.norm(sun_body), r_km,
        math.degrees(math.asin(position[2] / np.linalg.norm(position))),
        local_heading_deg(position, velocity),
        math.degrees(math.asin(float(np.clip(sun[2], -1, 1)))),
        float(sample.sun_distance_m) / (AU_KM * 1000.0),
        in_eclipse(position, sun), math.degrees(math.asin(float(np.clip(orbit_normal @ sun, -1, 1)))),
        offset)


def housing_series(eclipse: Sequence[bool], dt_s: float) -> np.ndarray:
    """First-order housing temperature [K] minus the NUC reference, following the eclipse flags."""
    out = np.zeros(len(eclipse))
    temp = HOUSING_ECLIPSE_K if eclipse and eclipse[0] else HOUSING_SUNLIT_K
    a = math.exp(-dt_s / HOUSING_TAU_S)
    for i, ecl in enumerate(eclipse):
        target = HOUSING_ECLIPSE_K if ecl else HOUSING_SUNLIT_K
        temp = target + (temp - target) * a
        out[i] = temp - HOUSING_SUNLIT_K
    return out


def camera_sun_state(config: HorizonSensorConfig, geometry: SceneGeometry) -> list[dict]:
    """Per camera: whether the Sun is in the field of view and its angle from the boresight."""
    truth = SceneTruth(geometry.n_body, geometry.r_km, geometry.latitude_deg, geometry.heading_deg,
                       geometry.sun_body, geometry.declination_deg, geometry.sun_distance_au)
    visible = sun_visibility(truth)
    states = []
    for mount in config.mounts[:config.n_cameras]:
        cam = sun_camera_direction(mount, geometry.sun_body)
        angle = math.degrees(math.acos(float(np.clip(cam[2], -1.0, 1.0))))
        states.append({"in_fov": visible > 0.0 and sun_in_fov(config.camera, cam), "angle_deg": angle,
                       "pixel": sun_pixel(config.camera, cam), "visible": visible})
    return states


@dataclass
class HorizonHistory:
    t_min: np.ndarray
    latitude_deg: np.ndarray
    eclipse: np.ndarray
    sun_in_fov: np.ndarray          # (n, cameras)
    error_deg: np.ndarray           # NaN when no valid reading
    roll_error_deg: np.ndarray
    pitch_error_deg: np.ndarray
    valid: np.ndarray
    status: list[str]
    housing_delta_k: np.ndarray

    def summary(self) -> dict:
        good = self.valid & np.isfinite(self.error_deg)

        def stats(mask):
            e = self.error_deg[mask & good]
            n = int(mask.sum())
            return {"samples": n, "availability": float((mask & good).sum() / n) if n else float("nan"),
                    "rms_deg": float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan"),
                    "p95_deg": float(np.percentile(e, 95)) if e.size else float("nan")}

        polar = np.abs(self.latitude_deg) >= 60.0
        fov = self.sun_in_fov.any(axis=1)
        return {"all": stats(np.ones_like(good)), "sunlit": stats(~self.eclipse),
                "eclipse": stats(self.eclipse), "polar": stats(polar), "sun_in_fov": stats(fov)}


def run_history(samples: Sequence, view_of: Callable, config: HorizonSensorConfig, step_s: float = 60.0,
                offset_deg: tuple[float, float] = (2.0, -1.5), seed: int = 1,
                progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> HorizonHistory:
    """Run the sensor model along the stored samples (decimated to ``step_s``)."""
    times = np.array([(s.time_utc - samples[0].time_utc).total_seconds() for s in samples])
    chosen, last = [], -1e18
    for i, t in enumerate(times):
        if t - last >= step_s - 1e-6:
            chosen.append(i)
            last = t
    geometries = [scene_from_sample(samples[i], view_of(samples[i]), offset_deg) for i in chosen]
    dt = step_s if len(chosen) < 2 else float(np.median(np.diff(times[chosen])))
    housing = housing_series([g.eclipse for g in geometries], dt)
    model = HorizonSensorModel(config, seed=seed)
    n = len(chosen)
    out = HorizonHistory(times[chosen] / 60.0, np.array([g.latitude_deg for g in geometries]),
                         np.array([g.eclipse for g in geometries]), np.zeros((n, config.n_cameras), bool),
                         np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool),
                         [""] * n, housing)
    for k, g in enumerate(geometries):
        if cancelled is not None and cancelled():
            break
        reading = model.measure_now(g.n_body, g.r_km, g.latitude_deg, g.heading_deg, 0.0, None,
                                    t_s=float(times[chosen[k]]), scene=g.environment(housing[k]))
        for j, state in enumerate(camera_sun_state(config, g)):
            out.sun_in_fov[k, j] = state["in_fov"]
        out.status[k] = reading.status
        out.valid[k] = reading.valid
        if reading.valid:
            out.error_deg[k] = math.degrees(math.acos(float(np.clip(reading.n_body @ g.n_body, -1, 1))))
            truth_rp = np.degrees(_roll_pitch(g.n_body))
            out.roll_error_deg[k], out.pitch_error_deg[k] = reading.roll_pitch_deg - truth_rp
        if progress is not None:
            progress(k + 1, n)
    return out


def _roll_pitch(n: np.ndarray) -> np.ndarray:
    return np.array([math.atan2(-n[1], n[2]), math.asin(float(np.clip(n[0], -1, 1)))])

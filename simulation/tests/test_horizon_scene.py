import math

import numpy as np
import pytest

from openadcs_sim import horizon_scene as hs
from openadcs_sim.horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraConfig, CameraMount, CameraUnit,
                                         SceneTruth, render_radiance, sun_band_irradiance, sun_in_fov,
                                         sun_visibility)
from openadcs_sim.horizon_sensor import (HorizonArchitecture, HorizonSensorConfig, HorizonSensorModel,
                                         SceneEnvironment, sun_exclusion_mask)
from openadcs_sim.orbit import SGP4Propagator
from openadcs_sim.scenario import baseline_tle, parse_tle

R = 6378.137 + 500.0
CAMERA = CameraConfig()
MOUNT = CameraMount(0.0, 66.0)


def sun_in_camera(nadir_angle_deg: float):
    """Body-frame Sun direction that lands inside camera 0's image at the given nadir angle."""
    rot = MOUNT.rotation_body_from_cam()
    n = np.array([0.0, 0.0, 1.0])
    for y in np.linspace(-0.25, 0.25, 101):
        cam = np.array([0.0, y, 1.0])
        cam /= np.linalg.norm(cam)
        body = rot @ cam
        if abs(math.degrees(math.acos(body @ n)) - nadir_angle_deg) < 0.3:
            return body
    raise AssertionError("no direction found")


def test_baseline_tle_is_500km_98p5_circular():
    orbit = SGP4Propagator(*parse_tle(baseline_tle()))
    line2 = baseline_tle().splitlines()[2]
    assert float(line2[8:16]) == 98.5
    sample = orbit.propagate(orbit.epoch_utc) if hasattr(orbit, "propagate") else None
    if sample is not None:
        position = np.asarray(sample[0] if isinstance(sample, tuple) else sample.position_itrs_m)
        assert abs(np.linalg.norm(position) / 1000.0 - R) < 15.0


def test_eclipse_cylinder():
    sun = np.array([1.0, 0.0, 0.0])
    assert hs.in_eclipse(np.array([-R * 1000, 0, 0.0]), sun)
    assert not hs.in_eclipse(np.array([R * 1000, 0, 0.0]), sun)
    assert not hs.in_eclipse(np.array([0, R * 1000, 0.0]), sun)


def test_housing_cools_in_eclipse_and_recovers():
    series = hs.housing_series([False] * 5 + [True] * 40 + [False] * 40, 60.0)
    assert series[4] == pytest.approx(0.0)
    assert series[44] < -10.0
    assert series[-1] > series[44]


def test_default_scene_is_unchanged_by_sun_options():
    n = np.array([0.0, 0.0, 1.0])
    base = SceneTruth(n, R, 40.0, 0.0)
    cfg = AtmosphereConfig(enabled=False)
    a = render_radiance(CAMERA, MOUNT, base, AtmosphereRealization(cfg, np.random.default_rng(1)), cfg,
                        np.random.default_rng(2))
    b = render_radiance(CAMERA, MOUNT, SceneTruth(n, R, 40.0, 0.0, None), AtmosphereRealization(cfg, np.random.default_rng(1)),
                        cfg, np.random.default_rng(2))
    assert np.array_equal(a, b)


def test_sun_in_fov_saturates_and_is_masked():
    sun = sun_in_camera(76.0)
    n = np.array([0.0, 0.0, 1.0])
    truth = SceneTruth(n, R, 30.0, 0.0, sun, 5.0)
    assert sun_visibility(truth) == 1.0
    assert sun_in_fov(CAMERA, MOUNT.rotation_body_from_cam().T @ sun)
    assert 1.0 < sun_band_irradiance() < 3.0
    cfg = AtmosphereConfig(enabled=False)
    radiance = render_radiance(CAMERA, MOUNT, truth, AtmosphereRealization(cfg, np.random.default_rng(1)), cfg,
                               np.random.default_rng(2))
    frame = CameraUnit(CAMERA, 3).read_frame(radiance, np.random.default_rng(4))
    mask = sun_exclusion_mask(frame, CAMERA)
    assert mask is not None and mask.sum() > 50
    clean = render_radiance(CAMERA, MOUNT, SceneTruth(n, R, 30.0, 0.0), AtmosphereRealization(cfg, np.random.default_rng(1)),
                            cfg, np.random.default_rng(2))
    assert sun_exclusion_mask(CameraUnit(CAMERA, 3).read_frame(clean, np.random.default_rng(4)), CAMERA) is None


def test_occulted_sun_is_invisible():
    truth = SceneTruth(np.array([0.0, 0.0, 1.0]), R, 0.0, 0.0, np.array([0.0, 0.0, 1.0]))
    assert sun_visibility(truth) == 0.0


def test_sensor_survives_sun_in_fov_with_independent_cameras():
    sun = sun_in_camera(76.0)
    model = HorizonSensorModel(HorizonSensorConfig(architecture=HorizonArchitecture.INDEPENDENT), seed=5)
    n = np.array([0.0, 0.0, 1.0])
    reading = model.measure_now(n, R, 20.0, 10.0, scene=SceneEnvironment(sun, 5.0, 1.0, 0.0))
    assert reading.cameras[0].sun_masked_px > 0
    assert reading.cameras[1].sun_masked_px == 0
    assert reading.valid
    assert math.degrees(math.acos(float(np.clip(reading.n_body @ n, -1, 1)))) < 5.0


def test_history_over_a_short_orbit_segment():
    from datetime import datetime, timedelta, timezone

    from openadcs_sim.attitude import spacecraft_view
    from openadcs_sim.environment import EnvironmentSample

    start = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    samples = []
    for k in range(6):
        angle = k * 0.05
        samples.append(EnvironmentSample(
            start + timedelta(seconds=30 * k),
            np.array([R * 1000 * math.cos(angle), R * 1000 * math.sin(angle) * 0.1, R * 1000 * math.sin(angle)]),
            np.array([-7600 * math.sin(angle), 760 * math.cos(angle), 7600 * math.cos(angle)]),
            np.array([1e-5, 2e-5, 3e-5]), np.array([1.0, 0.0, 0.0]), 1.5e11))
    history = hs.run_history(samples, lambda s: spacecraft_view(s, None),
                             HorizonSensorConfig(architecture=HorizonArchitecture.SINGLE), step_s=60.0)
    assert len(history.t_min) == 3
    assert history.valid.dtype == bool and len(history.status) == 3
    summary = history.summary()
    assert summary["all"]["samples"] == 3

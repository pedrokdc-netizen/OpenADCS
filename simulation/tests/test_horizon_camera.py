import math

import numpy as np
import pytest

from openadcs_sim.horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraConfig, CameraMount,
                                         CameraUnit, SceneTruth, band_radiance, band_temperature,
                                         limb_radius_km, render_radiance)


def nadir(roll_deg=0.0, pitch_deg=0.0):
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    return np.array([math.sin(p), -math.sin(r) * math.cos(p), math.cos(r) * math.cos(p)])


def test_band_radiance_is_monotonic_and_invertible():
    temps = np.array([200.0, 250.0, 290.0, 330.0, 400.0])
    radiance = band_radiance(temps)
    assert np.all(np.diff(radiance) > 0)
    assert band_temperature(radiance) == pytest.approx(temps, abs=0.05)


def test_noise_follows_datasheet_points_and_averaging():
    ref = 303.15
    assert CameraConfig(fps=1.0).noise_sigma_k(ref) == pytest.approx(0.2, rel=0.02)
    assert CameraConfig(fps=10.0).noise_sigma_k(ref) == pytest.approx(0.63, rel=0.02)
    assert CameraConfig(fps=29.3).noise_sigma_k(ref) == pytest.approx(1.09, rel=0.08)
    single = CameraConfig(fps=10.0).noise_sigma_k(ref)
    assert CameraConfig(fps=10.0, frames_averaged=4).noise_sigma_k(ref) == pytest.approx(single / 2, rel=1e-6)


def test_camera_geometry_matches_waveshare_module():
    camera = CameraConfig()
    assert (camera.width, camera.height) == (80, 62)
    assert camera.vfov_deg == pytest.approx(35.6, abs=0.5)
    assert camera.ifov_rad == pytest.approx(math.radians(45.0) / 80, rel=0.1)


def test_config_validation():
    with pytest.raises(ValueError):
        CameraConfig(fps=40.0)
    with pytest.raises(ValueError):
        SceneTruth(np.array([0.0, 0.0, 2.0]), 6871.0, 0.0)


def test_mount_rotation_is_orthonormal_and_boresight_off_nadir():
    for az in (0.0, 90.0, 217.0):
        rot = CameraMount(az, 66.0).rotation_body_from_cam()
        assert rot.T @ rot == pytest.approx(np.eye(3), abs=1e-12)
        assert np.degrees(np.arccos(rot[:, 2] @ [0, 0, 1.0])) == pytest.approx(66.0, abs=1e-9)


def test_limb_radius_is_ellipsoidal():
    equator = float(limb_radius_km(0.0, 0.0, 0.0, 6871.0))
    pole = float(limb_radius_km(89.0, 0.0, 0.0, 6871.0))
    assert 6340.0 < pole < equator < 6385.0


def render(seed=0, scene=False, **camera_kwargs):
    camera = CameraConfig(**camera_kwargs)
    atm = AtmosphereConfig(enabled=scene)
    rng = np.random.default_rng(seed)
    mount = CameraMount(0.0, 66.0)
    radiance = render_radiance(camera, mount, SceneTruth(nadir(), 6871.0, 10.0, 0.0),
                               AtmosphereRealization(atm, rng), atm, rng)
    return camera, radiance, rng


def test_limb_image_has_earth_below_and_cold_space_above():
    camera, radiance, rng = render()
    frame = CameraUnit(camera, 1).read_frame(radiance, rng)
    assert frame.shape == (62, 80)
    assert frame[-6:].mean() > 270.0          # nadir side of the image: warm Earth
    assert frame[:6].mean() < camera.floor_k + 3.0  # space: clipped at the sensor floor


def test_sensor_floor_clips_cold_scenes():
    camera, radiance, rng = render()
    frame = CameraUnit(camera, 1).read_frame(radiance, rng)
    assert frame.min() >= camera.floor_k - 3.0 - 1e-9
    assert frame.max() <= camera.ceiling_k + 1e-9


def test_dead_pixels_fraction_and_determinism():
    camera = CameraConfig()
    a, b = CameraUnit(camera, 5), CameraUnit(camera, 5)
    assert np.array_equal(a.dead_mask, b.dead_mask)
    assert 0.0 < a.dead_mask.mean() < 0.02
    _, radiance, _ = render()
    f1 = a.read_frame(radiance, np.random.default_rng(3))
    f2 = b.read_frame(radiance, np.random.default_rng(3))
    assert np.array_equal(f1, f2)


def test_quantisation_step():
    camera, radiance, rng = render()
    frame = CameraUnit(camera, 1).read_frame(radiance, rng)
    assert np.allclose(frame / camera.quantization_k, np.round(frame / camera.quantization_k), atol=1e-6)


def test_motion_smear_blurs_the_limb():
    camera, radiance, _ = render()
    unit = CameraUnit(CameraConfig(fpn_sigma_k=0.0, dead_pixel_fraction=0.0), 1)
    still = unit.read_frame(radiance, np.random.default_rng(0), 0.0)
    fast = unit.read_frame(radiance, np.random.default_rng(0), math.radians(60.0))
    grad = lambda f: np.abs(np.diff(f, axis=0)).max()  # noqa: E731
    assert grad(fast) < grad(still)


def test_scene_variability_changes_the_image():
    _, plain, _ = render(scene=False)
    _, varied, _ = render(scene=True)
    assert not np.allclose(plain, varied)

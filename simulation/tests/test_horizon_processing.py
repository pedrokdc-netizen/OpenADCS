import math

import numpy as np
import pytest

from openadcs_sim.horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraConfig, CameraMount,
                                         CameraUnit, SceneTruth, render_radiance)
from openadcs_sim.horizon_processing import (ProcessingConfig, conic_nadir, detect_limb, estimate_nadir,
                                             fit_cone_nadir, gamma_from_geometry, limb_offset_table,
                                             pixels_to_rays, repair_dead_pixels)
from openadcs_sim.horizon_sensor import nadir_from_roll_pitch

R_KM = 6871.0
MOUNT = CameraMount(0.0, 66.0)
ROT = MOUNT.rotation_body_from_cam()
CAMERA = CameraConfig()
PROC = ProcessingConfig()


def make_frame(roll=1.0, pitch=-1.5, warm=285.0, seed=0, dtype_noise=True):
    rng = np.random.default_rng(seed)
    atm = AtmosphereConfig(enabled=False, warm_temperature_k=warm, polar_drop_k=0.0)
    n = nadir_from_roll_pitch(math.radians(roll), math.radians(pitch))
    radiance = render_radiance(CAMERA, MOUNT, SceneTruth(n, R_KM, 0.0, 0.0), AtmosphereRealization(atm, rng),
                               atm, rng)
    unit = CameraUnit(CAMERA, seed)
    return unit.read_frame(radiance, rng), unit, n


def solve(frame, unit, dtype=np.float64, config=PROC):
    gamma = gamma_from_geometry(R_KM, 0.0, 0.0, 0.0, config.assumed_limb_height_km)
    return estimate_nadir(frame, CAMERA, config, ROT.T @ np.array([0, 0, 1.0]), gamma, dtype,
                          unit.dead_mask, R_KM)


def test_strong_axis_accuracy_without_scene_variability():
    frame, unit, n = make_frame()
    fit, points = solve(frame, unit)
    assert fit.valid and fit.status == "OK"
    err = np.degrees(ROT @ fit.n_cam - n)
    assert abs(err[0]) < 0.15 and abs(err[1]) < 0.5   # pitch strong axis, roll weak axis
    assert 40 < len(points.pixels) < 200


def test_compensation_removes_plateau_temperature_dependence():
    errors = {}
    for comp in (False, True):
        cfg = ProcessingConfig(compensate_floor_shift=comp)
        errors[comp] = [np.degrees((ROT @ solve(*make_frame(warm=w, seed=2)[:2], config=cfg)[0].n_cam
                                    - make_frame(warm=w, seed=2)[2])[0]) for w in (270.0, 300.0, 330.0)]
    assert np.ptp(errors[True]) < 0.4 * np.ptp(errors[False])
    assert np.ptp(errors[True]) < 0.15


def test_float32_matches_float64_within_tolerance():
    frame, unit, _ = make_frame(seed=4)
    f64, _ = solve(frame, unit)
    f32, _ = solve(frame.astype(np.float32), unit, np.float32)
    assert f32.valid
    assert np.degrees(np.arccos(np.clip(f32.n_cam @ f64.n_cam, -1, 1))) < 0.02


def test_dead_pixels_are_repaired_not_detected_as_edges():
    frame = np.full((62, 80), 300.0)
    frame[:, :] = np.linspace(260.0, 300.0, 62)[:, None]
    mask = np.zeros_like(frame, dtype=bool)
    mask[30, 40] = True
    corrupted = frame.copy()
    corrupted[30, 40] = 440.0
    repaired = repair_dead_pixels(corrupted, mask)
    assert abs(repaired[30, 40] - frame[30, 40]) < 2.0


def test_blank_space_frame_reports_no_limb():
    frame = np.full((62, 80), CAMERA.floor_k)
    fit, points = estimate_nadir(frame, CAMERA, PROC, ROT.T @ np.array([0, 0, 1.0]),
                                 gamma_from_geometry(R_KM, 0.0, 0.0, 0.0, 38.0), r_km=R_KM)
    assert not fit.valid and fit.status in ("NO_LIMB", "TOO_FEW_POINTS")


def test_uniform_warm_frame_has_no_limb():
    frame = np.full((62, 80), 290.0)
    assert len(detect_limb(frame, CAMERA, PROC).pixels) == 0


def test_cone_fit_recovers_exact_synthetic_limb():
    n_true = np.array([0.05, -0.02, 1.0])
    n_true /= np.linalg.norm(n_true)
    gamma = math.radians(69.0)
    basis = np.linalg.svd(np.eye(3) - np.outer(n_true, n_true))[0][:, :2]
    clock = np.radians(np.linspace(-20, 20, 40))
    rays = (math.cos(gamma) * n_true[None] + math.sin(gamma) * (np.cos(clock)[:, None] * basis[:, 0]
                                                                  + np.sin(clock)[:, None] * basis[:, 1]))
    fit = fit_cone_nadir(rays, np.array([0.0, 0.0, 1.0]), gamma, 1e-3, PROC)
    assert fit.n_cam @ n_true == pytest.approx(1.0, abs=1e-9)
    assert fit.rms_rad < 1e-9


def test_conic_fit_is_ill_conditioned_with_one_narrow_camera():
    frame, unit, n = make_frame(seed=3)
    rays = pixels_to_rays(detect_limb(frame, CAMERA, PROC, dead_mask=unit.dead_mask).pixels, CAMERA)
    result = conic_nadir(rays)
    cone, _ = solve(frame, unit)
    cone_err = np.degrees(np.linalg.norm((ROT @ cone.n_cam - n)[:2]))
    conic_err = 1e9 if result is None else np.degrees(np.linalg.norm((ROT @ result[0] - n)[:2]))
    assert conic_err > 5 * cone_err


def test_floor_shift_table_is_monotonic_and_cached():
    temps, offsets = limb_offset_table(CAMERA, PROC, 6875.0)
    assert len(temps) >= 10 and np.all(np.diff(temps) > 0)
    assert offsets[0] < offsets[-1]           # colder plateau -> larger (negative) correction
    assert limb_offset_table(CAMERA, PROC, 6875.0)[0] is temps


def test_limb_height_error_maps_to_about_0p023_deg_per_km():
    frame, unit, n = make_frame(seed=6)
    base, _ = solve(frame, unit)
    biased, _ = solve(frame, unit, config=ProcessingConfig(assumed_limb_height_km=48.0))
    shift = abs(np.degrees((ROT @ biased.n_cam - ROT @ base.n_cam)[0]))
    assert shift == pytest.approx(0.23, abs=0.08)

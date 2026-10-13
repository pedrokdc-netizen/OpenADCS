"""Golden-vector contract for the ESP32-S3 horizon firmware port (no on-target build is possible here)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openadcs_sim.horizon_camera import CameraConfig, CameraMount
from openadcs_sim.horizon_processing import ProcessingConfig, estimate_nadir

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "onboard" / "esp32_horizon" / "golden"


@pytest.fixture(scope="module")
def golden():
    meta = json.loads((GOLDEN / "horizon_golden.json").read_text(encoding="utf-8"))
    raw = np.frombuffer((GOLDEN / "frames.bin").read_bytes(), dtype="<u2")
    h, w = meta["camera"]["height"], meta["camera"]["width"]
    return meta, raw.reshape(-1, h, w)


def run_case(meta, frames, case, dtype):
    camera, processing = CameraConfig(), ProcessingConfig()
    mount = CameraMount(case["mount"]["azimuth_deg"], case["mount"]["off_nadir_deg"])
    rot = mount.rotation_body_from_cam()
    dead = np.zeros((camera.height, camera.width), dtype=bool)
    for r, c in case["dead_pixels_rowcol"]:
        dead[r, c] = True
    frame = (frames[case["frame_index"]].astype(np.float64) / 100.0).astype(dtype)
    fit, points = estimate_nadir(frame, camera, processing, rot.T @ np.array(case["n_prior_body"]),
                                 case["gamma_prior_rad"], dtype, dead, case["r_km"],
                                 (mount.azimuth_deg, mount.off_nadir_deg))
    return fit, points, rot


def test_golden_files_are_consistent(golden):
    meta, frames = golden
    assert frames.shape[0] == len(meta["cases"]) and frames.shape[1:] == (62, 80)
    assert frames.min() >= 20000 and frames.max() <= 45000          # centi-kelvin, plausible
    assert set(c["calibration_table"] for c in meta["cases"]) <= set(meta["calibration_tables"])


def test_float64_reference_reproduces_golden(golden):
    meta, frames = golden
    for case in meta["cases"]:
        fit, points, rot = run_case(meta, frames, case, np.float64)
        exp = case["expected"]
        assert fit.status == exp["status"], case["name"]
        assert len(points.pixels) == exp["n_points"], case["name"]
        if exp["valid"]:
            assert np.allclose(rot @ fit.n_cam, exp["nadir_body"], atol=1e-6), case["name"]


def test_float32_port_stays_inside_tolerances(golden):
    meta, frames = golden
    tol = meta["tolerance"]
    for case in meta["cases"]:
        fit, points, rot = run_case(meta, frames, case, np.float32)
        exp = case["expected"]
        if not exp["valid"]:
            assert not fit.valid, case["name"]
            continue
        assert fit.valid, case["name"]
        assert abs(len(points.pixels) - exp["n_points"]) <= tol["n_points_abs"], case["name"]
        n_ref = np.array(exp["nadir_body"])
        angle = np.degrees(np.arccos(np.clip((rot @ fit.n_cam) @ n_ref, -1, 1)))
        assert angle < tol["nadir_deg"], (case["name"], angle)


def test_generator_is_deterministic(tmp_path):
    out = tmp_path / "golden"
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "make_horizon_golden.py"), "--out", str(out)],
                          capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (out / "frames.bin").read_bytes() == (GOLDEN / "frames.bin").read_bytes()
    assert json.loads((out / "horizon_golden.json").read_text()) == json.loads((GOLDEN / "horizon_golden.json").read_text())


def test_embedded_budget_is_met_by_reference_workload(golden):
    meta, frames = golden
    from openadcs_sim.horizon_sensor import Esp32Config
    esp = Esp32Config()
    worst = max(esp.processing_s(62 * 80, 300, case["expected"]["n_points"] + 20, 8) for case in meta["cases"])
    assert worst * 1e3 < meta["tolerance"]["processing_time_ms_max"]
    assert esp.frame_bytes(CameraConfig()) == 9920

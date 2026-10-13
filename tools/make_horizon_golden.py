"""Generate golden test vectors for the ESP32-S3 horizon firmware port.

    python tools/make_horizon_golden.py [--out onboard/esp32_horizon/golden]

Writes ``frames.bin`` (uint16 little-endian centi-kelvin, row-major, 80x62, cases concatenated) and
``horizon_golden.json`` (inputs, calibration table, expected outputs and tolerances).  The expected
values come from the float64 Python reference chain (``openadcs_sim.horizon_processing``); the C port
must reproduce them within the stated tolerances.  Everything is seeded, so regeneration is
deterministic (checked by ``simulation/tests/test_horizon_golden.py``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation" / "src"))

from openadcs_sim.horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraConfig,  # noqa: E402
                                         CameraMount, CameraUnit, EARTH_A_KM, SceneTruth, render_radiance)
from openadcs_sim.horizon_processing import (ProcessingConfig, estimate_nadir,  # noqa: E402
                                             floor_shift_gamma_offsets, limb_offset_table)
from openadcs_sim.horizon_sensor import _gamma_for, nadir_from_roll_pitch, roll_pitch_from_nadir  # noqa: E402

TOLERANCE = {
    "limb_point_px": 0.05,          # sub-pixel limb point position (float32 vs float64)
    "n_points_abs": 3,              # edge count may differ by tie-breaking in NMS
    "nadir_deg": 0.02,              # fitted nadir direction
    "gamma_offset_rad": 1e-6,
    "processing_time_ms_max": 20.0,  # on-target budget at 240 MHz (ASSUMPTION, see HORIZON_SENSOR.md)
}

# name, camera azimuth, altitude, latitude, heading, roll, pitch, atmosphere enabled, seed
CASES = [
    ("nominal_az0", 0.0, 500.0, 10.0, 30.0, 1.5, -2.0, False, 11),
    ("nominal_az90", 90.0, 500.0, 10.0, 30.0, -2.0, 1.0, False, 12),
    ("midlat_scene_az0", 0.0, 500.0, 45.0, 120.0, 0.5, 2.5, True, 13),
    ("midlat_scene_az90", 90.0, 400.0, -40.0, 250.0, -1.0, -1.5, True, 14),
    ("high_alt_az0", 0.0, 700.0, 20.0, 60.0, 2.0, 0.0, False, 15),
    ("cold_polar_az0", 0.0, 500.0, 80.0, 0.0, 0.0, 0.0, True, 16),
]


def build(out_dir: Path) -> dict:
    camera, processing = CameraConfig(), ProcessingConfig()
    frames, cases, tables = [], [], {}
    for name, az, alt, lat, heading, roll_deg, pitch_deg, scene, seed in CASES:
        mount = CameraMount(az, 66.0)
        rot = mount.rotation_body_from_cam()
        r_km = EARTH_A_KM + alt
        rng = np.random.default_rng(seed)
        atm = AtmosphereConfig(enabled=scene)
        truth_n = nadir_from_roll_pitch(math.radians(roll_deg), math.radians(pitch_deg))
        unit = CameraUnit(camera, seed)
        radiance = render_radiance(camera, mount, SceneTruth(truth_n, r_km, lat, heading),
                                   AtmosphereRealization(atm, rng), atm, rng)
        frame_k = unit.read_frame(radiance, rng)
        centi = np.round(frame_k * 100.0).astype("<u2")
        frame_ref = centi.astype(np.float64) / 100.0
        gamma = _gamma_for(mount, r_km, lat, heading, processing.assumed_limb_height_km)
        n_prior_cam = rot.T @ np.array([0.0, 0.0, 1.0])
        fit, points = estimate_nadir(frame_ref, camera, processing, n_prior_cam, gamma,
                                     dead_mask=unit.dead_mask, r_km=r_km, mount=(az, 66.0))
        key = f"az{az:g}_r{round(r_km / 25.0) * 25.0:g}"
        if key not in tables:
            temps, offsets = limb_offset_table(camera, processing, round(r_km / 25.0) * 25.0, az, 66.0)
            tables[key] = {"plateau_k": temps.round(3).tolist(), "gamma_offset_rad": [float(f"{x:.9g}") for x in offsets]}
        frames.append(centi.reshape(-1))
        order = np.lexsort((points.pixels[:, 1], points.pixels[:, 0]))
        cases.append({
            "name": name, "frame_index": len(frames) - 1,
            "mount": {"azimuth_deg": az, "off_nadir_deg": 66.0},
            "r_km": r_km, "latitude_deg": lat, "heading_deg": heading,
            "truth_roll_pitch_deg": [roll_deg, pitch_deg],
            "n_prior_body": [0.0, 0.0, 1.0], "gamma_prior_rad": float(f"{gamma:.9g}"),
            "calibration_table": key,
            "dead_pixels_rowcol": np.argwhere(unit.dead_mask).tolist(),
            "expected": {
                "status": fit.status, "valid": bool(fit.valid), "n_points": int(len(points.pixels)),
                "limb_points_px_sorted_first16": points.pixels[order][:16].round(4).tolist(),
                "nadir_cam": [float(f"{x:.9g}") for x in fit.n_cam] if fit.valid else None,
                "nadir_body": [float(f"{x:.9g}") for x in (rot @ fit.n_cam)] if fit.valid else None,
                "roll_pitch_deg": roll_pitch_from_nadir(rot @ fit.n_cam).__mul__(180.0 / math.pi).round(5).tolist()
                if fit.valid else None,
                "arc_deg": round(float(fit.arc_deg), 3) if fit.valid else None,
            },
        })
    meta = {
        "format": "frames.bin: uint16 LE centi-kelvin, row-major, 62 rows x 80 columns, cases concatenated",
        "camera": {"width": camera.width, "height": camera.height, "hfov_deg": camera.hfov_deg,
                   "focal_px": camera.focal_px},
        "processing": {k: getattr(processing, k) for k in (
            "smooth_sigma_px", "edge_threshold_sigmas", "dark_side_offset_px", "dark_side_margin_k",
            "gate_deg", "min_points", "assumed_limb_height_km", "point_sigma_px", "iterations", "huber_k",
            "min_arc_deg", "plateau_offset_px")},
        "sensor": {"floor_k": camera.floor_k, "ceiling_k": camera.ceiling_k},
        "tolerance": TOLERANCE, "calibration_tables": tables, "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames.bin").write_bytes(np.concatenate(frames).astype("<u2").tobytes())
    (out_dir / "horizon_golden.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "onboard" / "esp32_horizon" / "golden"))
    args = parser.parse_args(argv)
    meta = build(Path(args.out))
    for case in meta["cases"]:
        e = case["expected"]
        print(f"{case['name']:22s} {e['status']:15s} points={e['n_points']:4d} roll/pitch={e['roll_pitch_deg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

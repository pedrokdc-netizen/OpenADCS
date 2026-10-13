"""Monte Carlo characterisation of the horizon sensor (error law, architectures, sensitivities).

Usage (from the repository root)::

    python -m openadcs_sim.horizon_montecarlo --trials 120 --out simulation/docs/horizon_montecarlo.json

Each trial draws a spacecraft attitude error (roll/pitch uniform +/-3 deg, the sensor prior is the
nominal nadir), an orbit latitude (97.5 deg SSO distribution), a heading, an atmosphere/cloud
realisation and per-camera hardware imperfections.  The *same* rendered frames feed the three
architectures, so the comparison is paired.  A noise-only pass (scene variability off, same
plateau temperatures) isolates the image-sensor + processing contribution; the difference to the
full pass is the scene (limb-height / cloud) contribution that cannot be averaged away.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import dataclass

import numpy as np

from .horizon_camera import (AtmosphereConfig, AtmosphereRealization, CameraConfig, CameraMount,
                             CameraUnit, EARTH_A_KM, SceneTruth, render_radiance)
from .horizon_processing import ProcessingConfig, conic_nadir, fit_cone_nadir, gather_limb_rays
from .horizon_sensor import (HorizonArchitecture, HorizonErrorLaw, HorizonSensorConfig,
                             _gamma_for, nadir_from_roll_pitch, process_frames,
                             roll_pitch_from_nadir)

ARCHITECTURES = (HorizonArchitecture.SINGLE, HorizonArchitecture.INDEPENDENT, HorizonArchitecture.FUSED)
INCLINATION_DEG = 97.5   # sun-synchronous, ASSUMPTION for the latitude distribution


def no_scene_variability(atm: AtmosphereConfig) -> AtmosphereConfig:
    return dataclasses.replace(atm, common_height_sigma_km=0.0, local_height_sigma_km=0.0,
                               cloud_mean_k=0.0, cloud_sigma_k=0.0, texture_sigma_k=0.0)


@dataclass
class TrialSet:
    errors_deg: dict       # architecture value -> (n_valid, 2) roll/pitch errors [deg]
    valid_fraction: dict
    status_counts: dict
    latitudes_deg: dict | None = None   # architecture value -> |latitude| of each valid trial


def run_trials(n_trials: int, altitude_km: float = 500.0, seed: int = 0,
               camera: CameraConfig | None = None, processing: ProcessingConfig | None = None,
               atmosphere: AtmosphereConfig | None = None,
               mounts: tuple[CameraMount, ...] = (CameraMount(0.0, 66.0), CameraMount(90.0, 66.0)),
               attitude_range_deg: float = 3.0) -> TrialSet:
    camera = camera or CameraConfig()
    processing = processing or ProcessingConfig()
    atmosphere = atmosphere or AtmosphereConfig()
    r_km = EARTH_A_KM + altitude_km
    cfgs = {a: HorizonSensorConfig(architecture=a, camera=camera, processing=processing,
                                   atmosphere=atmosphere, mounts=mounts) for a in ARCHITECTURES}
    errors = {a.value: [] for a in ARCHITECTURES}
    counts = {a.value: {} for a in ARCHITECTURES}
    lats = {a.value: [] for a in ARCHITECTURES}
    for k in range(n_trials):
        rng = np.random.default_rng(seed * 100003 + k)
        roll, pitch = np.radians(rng.uniform(-attitude_range_deg, attitude_range_deg, 2))
        n_true = nadir_from_roll_pitch(roll, pitch)
        lat = math.degrees(math.asin(math.sin(math.radians(INCLINATION_DEG))
                                     * math.sin(rng.uniform(0, 2 * math.pi))))
        heading = rng.uniform(0.0, 360.0)
        units = [CameraUnit(camera, seed * 7919 + k * 2 + i) for i in range(2)]
        atm_real = AtmosphereRealization(atmosphere, rng)
        truth = SceneTruth(n_true, r_km, lat, heading)
        frames = [units[i].read_frame(
            render_radiance(camera, mounts[i], truth, atm_real, atmosphere, rng), rng) for i in range(2)]
        for arch in ARCHITECTURES:
            used = frames[:1] if arch is HorizonArchitecture.SINGLE else frames
            reading = process_frames(used, cfgs[arch], np.array([0.0, 0.0, 1.0]), r_km, lat, heading,
                                     dead_masks=[u.dead_mask for u in units][:len(used)])
            counts[arch.value][reading.status] = counts[arch.value].get(reading.status, 0) + 1
            if reading.valid:
                lats[arch.value].append(abs(lat))
                errors[arch.value].append(np.degrees(roll_pitch_from_nadir(reading.n_body)
                                                     - np.array([roll, pitch])))
    arrays = {k: np.array(v).reshape(-1, 2) for k, v in errors.items()}
    return TrialSet(arrays, {k: len(arrays[k]) / n_trials for k in arrays}, counts,
                    {k: np.array(v) for k, v in lats.items()})


def summarise(errors_deg: np.ndarray) -> dict:
    if len(errors_deg) < 2:
        return {"n": int(len(errors_deg))}
    mean = errors_deg.mean(0)
    total = np.linalg.norm(errors_deg, axis=1)
    mad = 1.4826 * np.median(np.abs(errors_deg - np.median(errors_deg, axis=0)), axis=0)
    return {"n": int(len(errors_deg)), "bias_deg": mean.round(4).tolist(),
            "std_deg": errors_deg.std(0, ddof=1).round(4).tolist(),
            "robust_sigma_deg": mad.round(4).tolist(),
            "rms_deg": np.sqrt((errors_deg ** 2).mean(0)).round(4).tolist(),
            "p95_abs_deg": np.percentile(np.abs(errors_deg), 95, axis=0).round(4).tolist(),
            "p95_centred_deg": np.percentile(np.abs(errors_deg - mean), 95, axis=0).round(4).tolist(),
            "pointing_rms_deg": round(float(np.sqrt((total ** 2).mean())), 4),
            "pointing_p95_deg": round(float(np.percentile(total, 95)), 4)}


def _conservative_sigma(summary: dict) -> np.ndarray:
    return np.maximum(np.array(summary["std_deg"]), np.array(summary["p95_centred_deg"]) / 1.96)


def extract_error_law(full: dict, noise_only: dict, availability: float, bands: dict | None = None,
                      tau_s: float = 120.0) -> HorizonErrorLaw:
    """White = sensor/processing noise; correlated = scene term with a p95-based conservative floor."""
    sigma_noise = np.array(noise_only["robust_sigma_deg"])
    corr = np.sqrt(np.maximum(_conservative_sigma(full) ** 2 - sigma_noise ** 2, 0.0))
    rows = []
    edges = {"0-30deg": 30.0, "30-60deg": 60.0, "60-90deg": 90.1}
    for label, summary in (bands or {}).items():
        if "bias_deg" not in summary:
            continue
        c = np.sqrt(np.maximum(_conservative_sigma(summary) ** 2 - sigma_noise ** 2, 0.0))
        rows.append((edges[label], *np.round(summary["bias_deg"], 3).tolist(), *c.round(3).tolist()))
    return HorizonErrorLaw(tuple(np.round(full["bias_deg"], 3)), tuple(sigma_noise.round(3)),
                           tuple(corr.round(3)), tau_s, round(availability, 3), tuple(rows))


def by_latitude(result: TrialSet, arch: str, edges=(0.0, 30.0, 60.0, 90.1)) -> dict:
    errors, lats = result.errors_deg[arch], result.latitudes_deg[arch]
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lats >= lo) & (lats < hi)
        out[f"{lo:g}-{min(hi, 90):g}deg"] = summarise(errors[sel])
    return out


def conic_reference(n_trials: int, altitude_km: float, seed: int) -> dict:
    """Free conic (Modenini sphere case) vs the range-aided cone fit on identical single-camera data."""
    camera, processing = CameraConfig(), ProcessingConfig()
    atm = no_scene_variability(AtmosphereConfig())
    mount = CameraMount(0.0, 66.0)
    rot = mount.rotation_body_from_cam()
    r_km = EARTH_A_KM + altitude_km
    conic_err, cone_err = [], []
    for k in range(n_trials):
        rng = np.random.default_rng(seed * 100003 + k)
        n_true = nadir_from_roll_pitch(*np.radians(rng.uniform(-3, 3, 2)))
        unit = CameraUnit(camera, k)
        real = AtmosphereRealization(atm, rng)
        frame = unit.read_frame(render_radiance(camera, mount, SceneTruth(n_true, r_km, 20.0, 45.0),
                                                real, atm, rng), rng)
        n_prior = rot.T @ np.array([0, 0, 1.0])
        gamma = _gamma_for(mount, r_km, 20.0, 45.0, processing.assumed_limb_height_km)
        rays, offsets, _, status = gather_limb_rays(frame, camera, processing, n_prior, gamma,
                                                    dead_mask=unit.dead_mask, r_km=r_km)
        if status != "OK":
            continue
        result = conic_nadir(rays)
        if result is not None:
            conic_err.append(np.degrees(np.linalg.norm((rot @ result[0] - n_true)[:2])))
        fit = fit_cone_nadir(rays, n_prior, gamma, processing.point_sigma_px * camera.ifov_rad,
                             processing, gamma_offsets=offsets)
        cone_err.append(np.degrees(np.linalg.norm((rot @ fit.n_cam - n_true)[:2])))

    def stat(values):
        v = np.array(values)
        return {"n": int(len(v)), "rms_deg": round(float(np.sqrt((v ** 2).mean())), 3) if len(v) else None,
                "median_deg": round(float(np.median(v)), 3) if len(v) else None}
    return {"free_conic": stat(conic_err), "range_aided_cone": stat(cone_err)}


def run_all(trials: int, seed: int = 1, quick: bool = False) -> dict:
    t0 = time.time()
    camera, atm = CameraConfig(), AtmosphereConfig()
    out: dict = {"trials": trials, "seed": seed, "inclination_deg": INCLINATION_DEG, "baseline": {},
                 "timing_default": {a.value: dataclasses.asdict(HorizonSensorConfig(architecture=a).timing())
                                    for a in ARCHITECTURES}}
    for altitude in (400.0, 500.0, 700.0):
        full = run_trials(trials, altitude, seed, camera, atmosphere=atm)
        noise = run_trials(trials, altitude, seed + 1, camera, atmosphere=no_scene_variability(atm))
        block = {}
        for arch in ARCHITECTURES:
            key = arch.value
            f, n = summarise(full.errors_deg[key]), summarise(noise.errors_deg[key])
            block[key] = {"full": f, "noise_only": n, "valid_fraction_full": full.valid_fraction[key],
                          "valid_fraction_noise": noise.valid_fraction[key], "status_full": full.status_counts[key]}
            block[key]["by_abs_latitude_full"] = by_latitude(full, key)
            if "bias_deg" in f and "bias_deg" in n:
                law = extract_error_law(f, n, full.valid_fraction[key], block[key]["by_abs_latitude_full"])
                block[key]["error_law"] = dataclasses.asdict(law)
        out["baseline"][str(int(altitude))] = block
    if quick:
        out["elapsed_s"] = round(time.time() - t0, 1)
        return out

    half = max(30, trials // 2)
    sens: dict = {}
    two_cam = (HorizonArchitecture.INDEPENDENT, HorizonArchitecture.FUSED)
    for sep in (60.0, 90.0, 120.0, 180.0):
        r = run_trials(trials, 500.0, seed, camera, atmosphere=atm,
                       mounts=(CameraMount(0.0, 66.0), CameraMount(sep, 66.0)))
        sens.setdefault("azimuth_separation_deg", {})[str(int(sep))] = {
            a.value: {**summarise(r.errors_deg[a.value]), "valid_fraction": r.valid_fraction[a.value]}
            for a in two_cam}
    for fps, avg in ((29.0, 1), (10.0, 1), (10.0, 4), (10.0, 10), (1.0, 1)):
        cam = dataclasses.replace(camera, fps=fps, frames_averaged=avg)
        r = run_trials(half, 500.0, seed, cam, atmosphere=no_scene_variability(atm))
        t = HorizonSensorConfig(architecture=HorizonArchitecture.INDEPENDENT, camera=cam).timing()
        sens.setdefault("fps_averaging_noise_only", {})[f"{fps:g}fps_x{avg}"] = {
            "output_rate_hz": round(t.output_rate_hz, 3), "latency_s": round(t.latency_s, 4),
            "max_rate_dps": round(t.max_rate_dps, 2), "noise_sigma_k_at_303K": round(cam.noise_sigma_k(303.15), 3),
            **{a.value: summarise(r.errors_deg[a.value]) for a in ARCHITECTURES}}
    for label, cam_cfg, proc_cfg in (
            ("floor_253K_compensated", camera, ProcessingConfig()),
            ("floor_253K_uncompensated", camera, ProcessingConfig(compensate_floor_shift=False)),
            ("floor_213K_compensated", dataclasses.replace(camera, floor_k=213.15), ProcessingConfig())):
        r = run_trials(half, 500.0, seed, cam_cfg, proc_cfg, atmosphere=atm)
        sens.setdefault("thermal_floor", {})[label] = {
            a.value: {**summarise(r.errors_deg[a.value]), "valid_fraction": r.valid_fraction[a.value]}
            for a in ARCHITECTURES}
    for dh in (-10.0, -5.0, 0.0, 5.0, 10.0):
        r = run_trials(half, 500.0, seed, camera, ProcessingConfig(assumed_limb_height_km=38.0 + dh),
                       no_scene_variability(atm))
        sens.setdefault("limb_height_error_km", {})[f"{dh:+g}"] = {
            a.value: summarise(r.errors_deg[a.value]) for a in ARCHITECTURES}
    out["sensitivity"] = sens
    out["conic_reference"] = {"500": conic_reference(half, 500.0, seed)}
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="baseline only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    result = run_all(args.trials, args.seed, args.quick)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    for alt, block in result["baseline"].items():
        for arch, data in block.items():
            f = data["full"]
            print(f"{alt} km {arch:12s} valid={data['valid_fraction_full']:.2f} std={f.get('std_deg')} "
                  f"bias={f.get('bias_deg')} point_p95={f.get('pointing_p95_deg')}")
    print(f"elapsed {result['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

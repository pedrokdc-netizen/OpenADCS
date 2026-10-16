"""Envelope validation of the Sun-sensor model: baseline, every extreme-corner combination and Monte Carlo.

Every variable parameter has a baseline and a (low, high) extreme.  The run evaluates the six-head array over
a fixed set of random Sun directions (uniform on the sphere, nadir along body +Z, random Earth illumination) and
reports availability and fused-error statistics per case.  ALL extreme ranges are placeholders to be replaced
by measured or vendor data.  Head misalignment is Gaussian in the Monte Carlo runs (``mount_sigma_deg``,
1-sigma per axis) and uses the 3-sigma bound on every axis, random signs, in the corner cases.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .albedo import AlbedoContext
from .sun_sensor_array import SunSensorArray, SunSensorArrayConfig

EARTH_ANGLE_500KM = math.asin(6378137.0 / (6378137.0 + 500e3))
NAMES = ("sigma_px", "veiling_glare", "threshold_fraction", "exposure_peak_fraction", "albedo", "diagonal_fov_deg")


@dataclass(frozen=True)
class EnvelopeSpec:
    # (baseline, low, high)
    sigma_px: tuple = (0.1, 0.05, 0.3)
    veiling_glare: tuple = (0.005, 0.001, 0.05)
    threshold_fraction: tuple = (0.1, 0.05, 0.3)
    exposure_peak_fraction: tuple = (0.8, 0.5, 1.0)
    albedo: tuple = (0.30, 0.20, 0.40)
    # f = 1.78 mm +-5 % (ASSUMED tolerance); low/high FOV come from f high/low
    diagonal_fov_deg: tuple = (120.0, 115.0, 125.0)   # ASSUMED +-5 deg tolerance
    mount_sigma_deg: float = 2.0           # per-axis Gaussian 1-sigma (user); corner bound = 3 sigma
    bias_sigma_px: float = 0.05            # per-head per-axis Gaussian centroid bias 1-sigma (typical calibrated residual)
    sigma_bound: float = 3.0
    sun_directions: int = 400
    monte_carlo_runs: int = 200
    seed: int = 1

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


def _config(values: dict, mount_sigma_deg: float = 0.0, mount_errors_deg=None, bias_sigma_px: float = 0.0,
            seed: int = 0) -> SunSensorArrayConfig:
    return SunSensorArrayConfig(
        centroid_sigma_px=values["sigma_px"], diagonal_fov_deg=values["diagonal_fov_deg"],
        albedo_coefficient=values["albedo"], veiling_glare=values["veiling_glare"],
        threshold_fraction=values["threshold_fraction"], exposure_peak_fraction=values["exposure_peak_fraction"],
        centroid_bias_sigma_px=bias_sigma_px, mount_sigma_deg=mount_sigma_deg,
        mount_errors_deg=mount_errors_deg, seed=seed)


def _directions(count: int, seed: int):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, rng.uniform(0.0, 1.0, count)


def evaluate(config: SunSensorArrayConfig, directions: np.ndarray, illumination: np.ndarray) -> dict:
    """Availability and fused-error statistics of one parameter set over the Sun-direction ensemble."""
    array = SunSensorArray(config)
    nadir = np.array([0.0, 0.0, 1.0])
    errors = []
    for k, (sun, light) in enumerate(zip(directions, illumination)):
        context = AlbedoContext(nadir, EARTH_ANGLE_500KM, config.albedo_coefficient * float(light))
        error = array.measure(float(k), sun, True, context).error_deg
        if error is not None:
            errors.append(error)
    seen = len(errors)
    errors = np.array(errors)
    return {"availability": seen / len(directions),
            "rms_deg": float(np.sqrt(np.mean(errors ** 2))) if seen else math.nan,
            "p95_deg": float(np.percentile(errors, 95)) if seen else math.nan,
            "max_deg": float(errors.max()) if seen else math.nan}


def run_envelope(spec: EnvelopeSpec | None = None, progress=None) -> dict:
    spec = spec or EnvelopeSpec()
    directions, illumination = _directions(spec.sun_directions, spec.seed)
    rng = np.random.default_rng(spec.seed + 1)
    baseline_values = {name: getattr(spec, name)[0] for name in NAMES}
    cases = [("baseline", "baseline", baseline_values, _config(baseline_values))]
    extreme_mount = rng.choice([-1.0, 1.0], size=(6, 3)) * spec.mount_sigma_deg * spec.sigma_bound
    for combo in itertools.product((1, 2), repeat=len(NAMES)):
        values = {name: getattr(spec, name)[c] for name, c in zip(NAMES, combo)}
        label = "".join("L" if c == 1 else "H" for c in combo)
        for mount, bias in itertools.product((False, True), repeat=2):
            # Extreme bias: use the 3-sigma bound as the per-head sigma of a seeded draw, signs fixed by the seed.
            cases.append((label + ("+M" if mount else "") + ("+B" if bias else ""), "corner", values,
                          _config(values, mount_errors_deg=extreme_mount.tolist() if mount else None,
                                  bias_sigma_px=spec.bias_sigma_px * spec.sigma_bound if bias else 0.0,
                                  seed=spec.seed)))
    for index in range(spec.monte_carlo_runs):
        values = {}
        for name in NAMES:
            base, low, high = getattr(spec, name)
            values[name] = float(np.clip(rng.normal(base, (high - low) / (2 * spec.sigma_bound)), low, high))
        cases.append((f"mc{index}", "monte_carlo", values,
                      _config(values, mount_sigma_deg=spec.mount_sigma_deg, bias_sigma_px=spec.bias_sigma_px,
                              seed=int(rng.integers(1, 10 ** 6)))))
    results = []
    for number, (label, kind, values, config) in enumerate(cases):
        results.append({"case": label, "kind": kind, "parameters": values,
                        **evaluate(config, directions, illumination)})
        if progress is not None:
            progress(number + 1, len(cases))
    return {"spec": spec.to_dict(), "cases": results, "summary": summarise(results)}


def summarise(results: list[dict]) -> dict:
    baseline = next(r for r in results if r["kind"] == "baseline")
    summary = {"baseline": {k: baseline[k] for k in ("availability", "rms_deg", "p95_deg", "max_deg")}}
    for kind in ("corner", "monte_carlo"):
        group = [r for r in results if r["kind"] == kind]
        if not group:
            continue
        worst = max(group, key=lambda r: r["rms_deg"])
        summary[kind] = {
            "cases": len(group), "worst_case": worst["case"], "worst_rms_deg": worst["rms_deg"],
            "worst_p95_deg": max(r["p95_deg"] for r in group), "worst_max_deg": max(r["max_deg"] for r in group),
            "min_availability": min(r["availability"] for r in group),
            "best_rms_deg": min(r["rms_deg"] for r in group),
            "rms_p95_over_cases_deg": float(np.percentile([r["rms_deg"] for r in group], 95))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sun-sensor envelope validation (baseline + corners + Monte Carlo)")
    parser.add_argument("--output", type=Path, default=Path("sun_sensor_envelope.json"))
    parser.add_argument("--directions", type=int, default=400)
    parser.add_argument("--runs", type=int, default=200, help="Monte Carlo runs")
    args = parser.parse_args()
    report = run_envelope(EnvelopeSpec(sun_directions=args.directions, monte_carlo_runs=args.runs))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

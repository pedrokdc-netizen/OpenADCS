"""Closed-loop ADCS runs shared by the CLI and the GUI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .assembly import SpacecraftAssembly
from .closed_loop import AnalyticEnvironment, ClosedLoopResult, ClosedLoopSimulation, RealEnvironment


def build_environment(kind: str, tle: tuple[str, str] | None = None, kernels: Path | None = None,
                      start: datetime | None = None):
    """``analytic`` (circular orbit, tilted dipole) or ``sgp4`` (TLE + WMM + Huira Sun)."""
    if kind == "analytic":
        return AnalyticEnvironment()
    if kind != "sgp4":
        raise ValueError("environment must be 'analytic' or 'sgp4'")
    if tle is None or kernels is None or start is None:
        raise ValueError("The SGP4 environment needs a TLE, the kernel directory and a start time.")
    from .common import utc_time
    from .environment import EnvironmentSimulator
    from .magnetic import XYZGeomagModel
    from .orbit import SGP4Propagator
    from .scenario import kernel_paths
    from .sun import HuiraSunModel
    simulator = EnvironmentSimulator(SGP4Propagator(*tle), XYZGeomagModel(), HuiraSunModel(kernel_paths(kernels)))
    return RealEnvironment(simulator, utc_time(start))


def run_closed_loop(assembly: SpacecraftAssembly, env, duration_s: float, dt_s: float = 1.0,
                    initial_rate_rad_s=(0.0, 0.0, 0.0), initial_attitude_offset_deg=(0.0, 0.0, 0.0),
                    seed: int = 0, outage_windows_s: tuple = (), progress: Callable[[float], None] | None = None,
                    directory: Path | None = None) -> ClosedLoopResult:
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(dt_s) or dt_s <= 0 or dt_s > duration_s:
        raise ValueError("Duration and time step must be positive, with the step not longer than the run.")
    kwargs = assembly.closed_loop_kwargs(directory)
    from dataclasses import replace
    if outage_windows_s:
        kwargs["gnss"] = replace(kwargs["gnss"], outage_windows_s=tuple(tuple(w) for w in outage_windows_s))
    sim = ClosedLoopSimulation(
        env, dt_s=dt_s, seed=seed, initial_rate_rad_s=tuple(initial_rate_rad_s),
        initial_attitude_offset_rad=tuple(np.radians(initial_attitude_offset_deg)), **kwargs)
    return sim.run(duration_s, progress)


def summarise(result: ClosedLoopResult, settle_s: float = 0.0) -> dict:
    """Headline numbers; statistics skip the first ``settle_s`` seconds (acquisition)."""
    keep = result.t_s >= settle_s
    if not keep.any():
        keep = np.ones_like(keep)
    modes = [m.value if hasattr(m, "value") else str(m) for m in result.mode]
    return {
        "samples": int(result.t_s.size), "duration_s": float(result.t_s[-1]),
        "pointing_error_deg_rms": float(np.sqrt(np.mean(result.pointing_error_deg[keep] ** 2))),
        "pointing_error_deg_max": float(np.max(result.pointing_error_deg[keep])),
        "estimation_error_deg_rms": float(np.sqrt(np.mean(result.estimation_error_deg[keep] ** 2))),
        "estimation_error_deg_max": float(np.max(result.estimation_error_deg[keep])),
        "final_rate_rad_s": float(result.rate_rad_s[-1]),
        "peak_wheel_momentum_nms": float(np.max(result.wheel_momentum_nms)),
        "first_nominal_s": next((float(t) for t, m in zip(result.t_s, modes) if m.lower() == "nominal"), None),
        "settle_s": float(settle_s),
    }


def sample_dicts(result: ClosedLoopResult, every: int = 1):
    for i in range(0, result.t_s.size, max(1, every)):
        mode = result.mode[i]
        yield {"t_s": float(result.t_s[i]), "mode": mode.value if hasattr(mode, "value") else str(mode),
               "pointing_error_deg": float(result.pointing_error_deg[i]),
               "estimation_error_deg": float(result.estimation_error_deg[i]),
               "rate_rad_s": float(result.rate_rad_s[i]), "wheel_momentum_nms": float(result.wheel_momentum_nms[i]),
               "dipole_am2": np.asarray(result.dipole_am2[i], dtype=float).tolist(), "sunlit": bool(result.sunlit[i])}

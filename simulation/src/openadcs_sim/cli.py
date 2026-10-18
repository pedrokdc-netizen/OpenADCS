import argparse
from datetime import datetime, timedelta
import json
import math
import sys
from pathlib import Path

from .black_sun_sensor import OV5640_DIAGONAL_FOV_DEG
from .common import utc_time
from .environment import EnvironmentSimulator
from .magnetic import XYZGeomagModel
from .orbit import SGP4Propagator
from .spacecraft import SpacecraftConfig, SpacecraftError
from .scenario import kernel_paths, parse_tle
from .sun import HuiraSunModel
from .sun_sensor_array import SunSensorArray, SunSensorArrayConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenADCS ground environment simulation")
    parser.add_argument("--tle", type=Path)
    parser.add_argument("--kernels", type=Path)
    parser.add_argument("--start", help="Timezone-aware ISO 8601 timestamp")
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sun-sensor", action="store_true",
                        help="also run the six-head OV5640 cube Sun sensor on every sample")
    parser.add_argument("--sun-sensor-sigma-px", type=float, default=0.1, help="OV5640 centroid error 1-sigma [px]")
    parser.add_argument("--albedo", type=float, default=0.30, help="Earth Bond albedo for the Sun sensor (0 disables)")
    parser.add_argument("--albedo-leak", type=float, default=1.0, help="albedo pull weight; 0 disables (needs glare above threshold)")
    parser.add_argument("--sun-sensor-diagonal-fov-deg", type=float, default=OV5640_DIAGONAL_FOV_DEG)
    parser.add_argument("--spacecraft", type=Path, help="spacecraft JSON (mass properties or a full assembly); validated and reported on stderr")
    parser.add_argument("--closed-loop", action="store_true",
                        help="run the closed-loop ADCS (sensors, estimator, controller, actuators, dynamics); one JSON line per output sample, summary on stderr")
    parser.add_argument("--environment", choices=("analytic", "sgp4"), default="sgp4",
                        help="closed loop only: analytic circular orbit (no TLE/kernels needed) or SGP4 + WMM + Huira")
    parser.add_argument("--duration-seconds", type=float, default=600.0, help="closed-loop run length")
    parser.add_argument("--dt", type=float, default=1.0, help="closed-loop integration step [s]")
    parser.add_argument("--output-every", type=int, default=10, help="closed loop: print every Nth step")
    parser.add_argument("--initial-offset-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("RX", "RY", "RZ"),
                        help="closed loop: initial attitude offset from nadir [deg]")
    parser.add_argument("--gnss-outage", type=float, nargs=2, action="append", metavar=("START", "END"), default=[],
                        help="closed loop: GNSS outage window in seconds (repeatable)")
    parser.add_argument("--seed", type=int, default=0, help="closed-loop noise seed")
    parser.add_argument("--dynamics", action="store_true", help="torque-driven attitude (gravity gradient) instead of the ideal nadir one")
    parser.add_argument("--initial-rate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("WX", "WY", "WZ"), help="extra initial body rate [rad/s] with --dynamics")
    args = parser.parse_args()
    if not math.isfinite(args.step_seconds) or args.step_seconds <= 0 or args.samples < 1:
        parser.error("Step must be finite and positive; samples must be at least one")
    assembly = None
    if args.spacecraft is not None:
        from .assembly import SpacecraftAssembly
        from .subsystems import SubsystemError
        try:
            assembly = SpacecraftAssembly.load(args.spacecraft)
            assembly.closed_loop_kwargs()
        except (SpacecraftError, SubsystemError, OSError) as error:
            parser.error(str(error))
        craft = assembly.mass_properties
        print(f"spacecraft: {craft.name}, {craft.mass_kg:.4f} kg", file=sys.stderr)
    needs_ephemeris = not (args.closed_loop and args.environment == "analytic")
    if needs_ephemeris and not (args.tle and args.kernels and args.start):
        parser.error("--tle, --kernels and --start are required (except for --closed-loop --environment analytic)")
    if args.closed_loop:
        run_closed_loop_cli(parser, args, assembly)
        return
    start = utc_time(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    lines = parse_tle(args.tle.read_text(encoding="ascii"))
    kernels = kernel_paths(args.kernels)
    array = None
    if args.sun_sensor:
        try:
            array = SunSensorArray(SunSensorArrayConfig(args.sun_sensor_sigma_px, args.sun_sensor_diagonal_fov_deg,
                                                       albedo_coefficient=args.albedo, albedo_leak_fraction=args.albedo_leak))
        except ValueError as error:
            parser.error(str(error))
    attitude_model = None
    if args.dynamics:
        from .dynamics import DynamicsAttitudeModel
        from .spacecraft import pilot1_baseline
        attitude_model = DynamicsAttitudeModel(craft if args.spacecraft is not None else pilot1_baseline(),
                                               initial_rate_body_rad_s=args.initial_rate)
    simulator = EnvironmentSimulator(SGP4Propagator(*lines), XYZGeomagModel(), HuiraSunModel(kernels), array,
                                     attitude_model=attitude_model)
    for index in range(args.samples):
        sample = simulator.sample(start + timedelta(seconds=index * args.step_seconds))
        print(json.dumps(sample.to_dict(), allow_nan=False))


def run_closed_loop_cli(parser, args, assembly) -> None:
    from .adcs_run import build_environment, run_closed_loop, sample_dicts, summarise
    from .assembly import SpacecraftAssembly
    from .subsystems import SubsystemError
    start = None
    tle = kernels = None
    if args.environment == "sgp4":
        start = utc_time(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
        tle, kernels = parse_tle(args.tle.read_text(encoding="ascii")), args.kernels
    try:
        env = build_environment(args.environment, tle, kernels, start)
        result = run_closed_loop(assembly or SpacecraftAssembly(), env, args.duration_seconds, args.dt,
                                 args.initial_rate, args.initial_offset_deg, args.seed,
                                 tuple(tuple(w) for w in args.gnss_outage))
    except (ValueError, SubsystemError, SpacecraftError, OSError) as error:
        parser.error(str(error))
    for row in sample_dicts(result, args.output_every):
        print(json.dumps(row, allow_nan=False))
    print("summary: " + json.dumps(summarise(result, min(600.0, args.duration_seconds / 2))), file=sys.stderr)


if __name__ == "__main__":
    main()

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
    parser.add_argument("--tle", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--start", required=True, help="Timezone-aware ISO 8601 timestamp")
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--sun-sensor", action="store_true",
                        help="also run the six-head OV5640 cube Sun sensor on every sample")
    parser.add_argument("--sun-sensor-sigma-px", type=float, default=0.1, help="OV5640 centroid error 1-sigma [px]")
    parser.add_argument("--albedo", type=float, default=0.30, help="Earth Bond albedo for the Sun sensor (0 disables)")
    parser.add_argument("--albedo-leak", type=float, default=1.0, help="albedo pull weight; 0 disables (needs glare above threshold)")
    parser.add_argument("--sun-sensor-diagonal-fov-deg", type=float, default=OV5640_DIAGONAL_FOV_DEG)
    parser.add_argument("--spacecraft", type=Path, help="spacecraft JSON (mass properties); validated and reported on stderr")
    parser.add_argument("--dynamics", action="store_true", help="torque-driven attitude (gravity gradient) instead of the ideal nadir one")
    parser.add_argument("--initial-rate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("WX", "WY", "WZ"), help="extra initial body rate [rad/s] with --dynamics")
    args = parser.parse_args()
    if not math.isfinite(args.step_seconds) or args.step_seconds <= 0 or args.samples < 1:
        parser.error("Step must be finite and positive; samples must be at least one")
    if args.spacecraft is not None:
        try:
            craft = SpacecraftConfig.load(args.spacecraft)
        except (SpacecraftError, OSError) as error:
            parser.error(str(error))
        print(f"spacecraft: {craft.name}, {craft.mass_kg:.4f} kg", file=sys.stderr)
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


if __name__ == "__main__":
    main()

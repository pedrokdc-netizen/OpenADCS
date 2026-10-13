import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

from .common import utc_time
from .environment import EnvironmentSimulator
from .magnetic import XYZGeomagModel
from .orbit import SGP4Propagator
from .scenario import kernel_paths, parse_tle
from .sun import HuiraSunModel


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenADCS ground environment simulation")
    parser.add_argument("--tle", required=True, type=Path)
    parser.add_argument("--kernels", required=True, type=Path)
    parser.add_argument("--start", required=True, help="Timezone-aware ISO 8601 timestamp")
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if not math.isfinite(args.step_seconds) or args.step_seconds <= 0 or args.samples < 1:
        parser.error("Step must be finite and positive; samples must be at least one")
    start = utc_time(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    lines = parse_tle(args.tle.read_text(encoding="ascii"))
    kernels = kernel_paths(args.kernels)
    simulator = EnvironmentSimulator(SGP4Propagator(*lines), XYZGeomagModel(), HuiraSunModel(kernels))
    for index in range(args.samples):
        sample = simulator.sample(start + timedelta(seconds=index * args.step_seconds))
        print(json.dumps(sample.to_dict(), allow_nan=False))


if __name__ == "__main__":
    main()

"""Standalone Orion B-16 receiver simulation from a TLE and explicit assumptions."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
from pathlib import Path

import astropy.units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, GCRS, ITRS
from astropy.time import Time
import numpy as np

from .common import utc_time
from .orbit import SGP4Propagator
from .orion_b16 import OrionB16, OrionConfig, ReceiverConditions
from .scenario import parse_tle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tle", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--start", required=True, help="Timezone-aware ISO 8601 UTC")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--output", required=True, type=Path, help="JSON-lines ground report")
    parser.add_argument("--binary-output", type=Path, help="0xA8 frames at default 1-second interval")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.binary_output is not None and args.binary_output.resolve() == args.output.resolve():
        parser.error("JSON and binary output paths must differ")
    for destination in (args.output, args.binary_output):
        if destination is not None and destination.resolve() in (args.tle.resolve(), args.profile.resolve()):
            parser.error("Output must not overwrite the TLE or profile")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    config = OrionConfig(**profile["receiver"])
    conditions = ReceiverConditions(**profile["conditions"])
    start = utc_time(datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    receiver = OrionB16(config, start)
    orbit = SGP4Propagator(*parse_tle(args.tle.read_text(encoding="ascii")))
    # Preflight the time range before opening output files.
    orbit.propagate(start)
    orbit.propagate(start + timedelta(seconds=(args.samples - 1) / config.update_rate_hz))
    binary = args.binary_output.open("wb") if args.binary_output is not None else None
    try:
        with args.output.open("w", encoding="utf-8") as output:
            output.write(json.dumps({
                "schema_version": 1, "provenance": config.provenance(),
                "conditions": profile["conditions"],
                "profile_description": profile.get("description", "User-supplied behavioral assumptions"),
                "dynamics_note": "CLI derives GCRS speed; acceleration remains the supplied assumption.",
            }, allow_nan=False) + "\n")
            for index in range(args.samples):
                truth = orbit.propagate(start + timedelta(seconds=index / config.update_rate_hz))
                instant = Time(truth.time_utc)
                representation = CartesianRepresentation(
                    truth.position_itrs_m * u.m,
                    differentials=CartesianDifferential(truth.velocity_itrs_m_s * u.m / u.s),
                )
                inertial = ITRS(representation, obstime=instant).transform_to(GCRS(obstime=instant))
                speed = float(np.linalg.norm(inertial.cartesian.differentials["s"].d_xyz.to_value(u.m / u.s)))
                actual_conditions = replace(conditions, inertial_speed_m_s=speed)
                sample = receiver.sample(truth, actual_conditions)
                record = sample.to_dict()
                record["conditions"] = asdict(actual_conditions)
                record["truth_position_ecef_m"] = truth.position_itrs_m.tolist()
                record["truth_velocity_ecef_m_s"] = truth.velocity_itrs_m_s.tolist()
                output.write(json.dumps(record, allow_nan=False) + "\n")
                if binary is not None and index % config.update_rate_hz == 0:
                    frame = sample.navigation_output()
                    if frame is not None:
                        binary.write(frame)
    finally:
        if binary is not None:
            binary.close()


if __name__ == "__main__":
    main()

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Sequence

import astropy.units as u
from astropy.coordinates import EarthLocation
import numpy as np

from .common import utc_time, vector3
from .environment import EnvironmentSample
from .orbit import SGP4Propagator

KERNEL_NAMES = ("naif0012.tls", "de440s.bsp", "earth_latest_high_prec.bpc")
AU_M = 149597870700.0


def training_tle() -> str:
    """Synthetic near-circular LEO, not the measured orbit of any satellite."""
    lines = (
        "1 99999U 26001A   26282.50000000  .00000000  00000-0  00000-0 0  0001",
        "2 99999  51.6000   0.0000 0010000   0.0000   0.0000 15.50000000    01",
    )
    checked = []
    for line in lines:
        content = line[:68]
        checksum = sum(int(char) if char.isdigit() else int(char == "-") for char in content) % 10
        checked.append(content + str(checksum))
    return "SYNTHETIC LEO - classroom exercise only\n" + "\n".join(checked)


def parse_tle(text: str) -> tuple[str, str]:
    lines = text.strip("\r\n").splitlines()
    if len(lines) == 3 and not lines[0].startswith("1 "):
        lines = lines[1:]
    if len(lines) != 2:
        raise ValueError("Paste two TLE element lines, optionally preceded by the satellite name.")
    return lines[0], lines[1]


def kernel_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(directory / name for name in KERNEL_NAMES)


def sampling_count(duration_seconds: float, step_seconds: float) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("Duration must be a finite, non-negative number of seconds.")
    if not math.isfinite(step_seconds) or step_seconds < 1e-6:
        raise ValueError("Time step must be at least one microsecond.")
    quotient = duration_seconds / step_seconds
    if not math.isfinite(quotient):
        raise ValueError("Duration / time step is too large.")
    nearest = round(quotient)
    if math.isclose(quotient, nearest, rel_tol=0, abs_tol=1e-9):
        quotient = float(nearest)
    return math.floor(quotient) + 1


@dataclass(frozen=True)
class Scenario:
    tle: tuple[str, str]
    kernel_directory: Path
    start_utc: datetime
    duration_seconds: float
    step_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_utc", utc_time(self.start_utc))
        sampling_count(self.duration_seconds, self.step_seconds)
        end = self.end_utc
        if not 2025 <= self.start_utc.year < 2030 or not 2025 <= end.year < 2030:
            raise ValueError("The entire run must fall within WMM2025 coverage (2025-2029).")
        orbit = SGP4Propagator(*self.tle)
        for timestamp in (self.start_utc, end):
            if abs(timestamp - orbit.epoch_utc) > orbit.max_tle_age:
                raise ValueError(
                    "The run extends more than seven days from the TLE epoch. "
                    "Use 'Use TLE epoch' or import a newer TLE."
                )
        missing = [path.name for path in kernel_paths(self.kernel_directory) if not path.is_file()]
        if missing:
            raise ValueError(
                f"Missing kernel files: {', '.join(missing)}. "
                "Select the downloaded kernel folder; see the Guide tab for setup."
            )

    @property
    def sample_count(self) -> int:
        # Include t=0 and all grid points <= duration, not an overshooting final step.
        return sampling_count(self.duration_seconds, self.step_seconds)

    @property
    def end_utc(self) -> datetime:
        return self.start_utc + timedelta(seconds=(self.sample_count - 1) * self.step_seconds)

    def to_dict(self) -> dict[str, object]:
        return {
            "tle": list(self.tle),
            "kernel_directory": str(self.kernel_directory.resolve()),
            "kernel_files": list(KERNEL_NAMES),
            "start_utc": self.start_utc.isoformat(),
            "requested_duration_seconds": self.duration_seconds,
            "step_seconds": self.step_seconds,
            "requested_samples": self.sample_count,
            "last_sample_utc": self.end_utc.isoformat(),
            "models": {"orbit": "SGP4/WGS72", "magnetic": "XYZgeomag/WMM2025", "sun": "Huira/SPICE"},
        }


@dataclass(frozen=True)
class Readout:
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    speed_fixed_km_s: float
    magnetic_magnitude_ut: float
    sun_distance_au: float


def sample_readout(sample: EnvironmentSample) -> Readout:
    location = EarthLocation.from_geocentric(*sample.position_itrs_m, unit=u.m)
    return Readout(
        float(location.lat.to_value(u.deg)), float(location.lon.to_value(u.deg)),
        float(location.height.to_value(u.km)),
        float(np.linalg.norm(sample.velocity_itrs_m_s)) / 1000,
        float(np.linalg.norm(sample.magnetic_field_itrs_t)) * 1e6,
        sample.sun_distance_m / AU_M,
    )


def decode_sample(line: str) -> EnvironmentSample:
    record = json.loads(line)
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("frame") != "ITRS":
        raise ValueError("Simulation worker returned an unsupported environment record.")
    required = (
        "time_utc", "position_m", "velocity_m_s", "magnetic_field_t",
        "sun_direction", "sun_distance_m",
    )
    if any(name not in record for name in required):
        raise ValueError("Simulation worker returned an incomplete environment record.")
    direction = vector3(record["sun_direction"])
    distance = float(record["sun_distance_m"])
    if not math.isfinite(distance) or distance <= 0 or not np.isclose(np.linalg.norm(direction), 1):
        raise ValueError("Simulation worker returned invalid Sun geometry.")
    return EnvironmentSample(
        utc_time(datetime.fromisoformat(record["time_utc"])),
        vector3(record["position_m"]), vector3(record["velocity_m_s"]),
        vector3(record["magnetic_field_t"]), direction, distance,
    )


def export_results(
    destination: Path, scenario: Scenario, samples: Sequence[EnvironmentSample], status: str,
) -> None:
    if not samples:
        raise ValueError("There are no samples to export.")
    if destination.suffix.lower() == ".json":
        with destination.open("w", encoding="utf-8") as output:
            json.dump(
                {"schema_version": 1, "status": status, "scenario": scenario.to_dict(),
                 "samples": [sample.to_dict() for sample in samples]},
                output, indent=2, allow_nan=False,
            )
            output.write("\n")
    elif destination.suffix.lower() == ".csv":
        with destination.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "time_utc", "frame", "run_status", "latitude_deg", "longitude_deg", "altitude_km",
                "x_km", "y_km", "z_km", "vx_fixed_km_s", "vy_fixed_km_s", "vz_fixed_km_s",
                "bx_ut", "by_ut", "bz_ut", "b_magnitude_ut",
                "sun_x_unit", "sun_y_unit", "sun_z_unit", "sun_distance_au",
            ])
            for sample in samples:
                readout = sample_readout(sample)
                writer.writerow([
                    sample.time_utc.isoformat(), "ITRS", status,
                    readout.latitude_deg, readout.longitude_deg, readout.altitude_km,
                    *(sample.position_itrs_m / 1000), *(sample.velocity_itrs_m_s / 1000),
                    *(sample.magnetic_field_itrs_t * 1e6), readout.magnetic_magnitude_ut,
                    *sample.sun_direction_itrs, readout.sun_distance_au,
                ])
    else:
        raise ValueError("Choose a .csv or .json export file.")

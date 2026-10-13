"""Datasheet-based PVT behavior; not a GNSS RF solver or flight-calibrated twin."""

from dataclasses import asdict, dataclass
from datetime import datetime
import math
import struct

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
import numpy as np

from .common import utc_time, vector3
from .orbit import OrbitState
from .sensor_processing import GpsMeasurement, SampleHeader, Status


SUPPORTED_RATES_HZ = (1, 2, 4, 5, 8, 10, 20)
DATASHEET_URL = "https://navspark.mybigcommerce.com/content/Orion_B16_DS.pdf"
PROTOCOL_URL = "https://navspark.mybigcommerce.com/content/AN0037.pdf"
SOURCE_DOCUMENTS = {
    "datasheet": {
        "revision": "1.5, 2022-09-16",
        "sha256": "081e99faf2cf7e27f5043df1dd0c43e0721103bb10e6b53ab08c5c03d3650639",
    },
    "protocol": {
        "revision": "AN0037 1.4.69, 2025-03-26",
        "sha256": "747f89478b095a030634caa7c4029753cbb0d4634b770dbb13a9d697863311ad",
    },
}


@dataclass(frozen=True)
class OrionConfig:
    vertical_sigma_m: float
    velocity_sigma_m_s: float
    position_correlation_s: float
    velocity_correlation_s: float
    seed: int
    horizontal_cep_m: float = 2.0
    update_rate_hz: int = 1
    start_mode: str = "cold"
    hot_start_s: float = 18.0
    warm_start_s: float = 46.0
    cold_start_s: float = 50.0
    reacquisition_s: float = 16.0
    clock_bias_s: float = 0.0
    clock_sigma_s: float = 0.0
    sensor_id: int = 100

    def __post_init__(self) -> None:
        for name in (
            "vertical_sigma_m", "velocity_sigma_m_s", "position_correlation_s",
            "velocity_correlation_s", "horizontal_cep_m", "hot_start_s",
            "warm_start_s", "cold_start_s", "reacquisition_s", "clock_sigma_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.clock_bias_s):
            raise ValueError("clock_bias_s must be finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.update_rate_hz) is not int or self.update_rate_hz not in SUPPORTED_RATES_HZ:
            raise ValueError(f"Orion update rate must be one of {SUPPORTED_RATES_HZ}")
        if self.start_mode not in ("hot", "warm", "cold"):
            raise ValueError("start_mode must be hot, warm or cold")
        SampleHeader(0, self.sensor_id, 0)

    def provenance(self) -> dict[str, object]:
        return {
            "model": "Orion B-16 PVT behavioral model",
            "flight_validated": False,
            "datasheet": DATASHEET_URL,
            "binary_interface": PROTOCOL_URL,
            "source_documents": SOURCE_DOCUMENTS,
            "parameters": asdict(self),
            "assumptions": [
                "2D isotropic Gaussian EN errors: sigma = CEP / sqrt(2 ln(2)).",
                "Vertical/velocity sigmas and correlation times are user assumptions.",
                "Independent ENU position/velocity Gauss-Markov errors, not pseudorange errors.",
                "Published open-sky average TTFF values used as deterministic dwell times.",
                "Warm/hot acquisition conservatively uses the published cold sensitivity.",
                "Continuous qualifying conditions required throughout acquisition.",
                "Satellite count, signal level, DOP, acceleration and geoid are external inputs.",
                "No constellation solver, antenna pattern, lever arm, RF or radiation model.",
                "ITRS truth treated as WGS84 ECEF; no realization-offset correction.",
                "Clock errors are configured PVT errors, not a modeled 1PPS output.",
                "Loss of main power restarts using the configured start mode; no backup SRAM model.",
            ],
        }


@dataclass(frozen=True)
class ReceiverConditions:
    satellites_available: int
    signal_dbm: float | None
    inertial_speed_m_s: float
    acceleration_m_s2: float
    geoid_undulation_m: float
    dop: tuple[float, float, float, float, float]
    power_on: bool = True
    antenna_available: bool = True
    supply_voltage_v: float = 3.3
    temperature_c: float = 20.0

    def __post_init__(self) -> None:
        if type(self.satellites_available) is not int or not 0 <= self.satellites_available <= 255:
            raise ValueError("satellites_available must be an integer in 0..255")
        if self.signal_dbm is not None and not math.isfinite(self.signal_dbm):
            raise ValueError("signal_dbm must be finite or None for signal loss")
        for name in ("inertial_speed_m_s", "acceleration_m_s2"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("geoid_undulation_m", "supply_voltage_v", "temperature_c"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if type(self.power_on) is not bool or type(self.antenna_available) is not bool:
            raise ValueError("Power and antenna conditions must be explicit booleans")
        if len(self.dop) != 5 or not all(math.isfinite(x) and 0 < x <= 655.35 for x in self.dop):
            raise ValueError("Supply GDOP/PDOP/HDOP/VDOP/TDOP in (0, 655.35]")


@dataclass(frozen=True)
class ReceiverSample:
    phase: str
    time_utc: datetime
    measurement: GpsMeasurement
    clock_error_s: float

    def navigation_output(self) -> bytes | None:
        if self.phase in ("POWERED_OFF", "OUT_OF_DOMAIN"):
            return None
        return navigation_frame(self.measurement)

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase, "sample_time_utc": self.time_utc.isoformat(),
            "frame": "WGS84 ECEF", "clock_error_s": self.clock_error_s,
            "measurement": asdict(self.measurement),
        }


class OrionB16:
    def __init__(self, config: OrionConfig, epoch_utc: datetime):
        self.config = config
        self.epoch_utc = utc_time(epoch_utc)
        self._rng = np.random.default_rng(config.seed)
        self._index = 0
        self._phase = "POWERED_OFF"
        self._since_ns: int | None = None
        self._had_fix = False
        self._position_error: np.ndarray | None = None
        self._velocity_error: np.ndarray | None = None

    def _errors(self) -> tuple[np.ndarray, np.ndarray]:
        position_sigma = np.array([
            self.config.horizontal_cep_m / math.sqrt(2 * math.log(2)),
            self.config.horizontal_cep_m / math.sqrt(2 * math.log(2)),
            self.config.vertical_sigma_m,
        ])
        velocity_sigma = np.full(3, self.config.velocity_sigma_m_s)
        for name, sigma, tau in (
            ("_position_error", position_sigma, self.config.position_correlation_s),
            ("_velocity_error", velocity_sigma, self.config.velocity_correlation_s),
        ):
            previous = getattr(self, name)
            phi = math.exp(-1 / (self.config.update_rate_hz * tau)) if tau > 0 else 0.0
            innovation = sigma * self._rng.normal(size=3)
            value = innovation
            if previous is not None and tau > 0:
                value = phi * previous + math.sqrt(
                    -math.expm1(-2 / (self.config.update_rate_hz * tau)),
                ) * innovation
            setattr(self, name, value)
        return self._position_error, self._velocity_error

    def sample(self, truth: OrbitState, conditions: ReceiverConditions) -> ReceiverSample:
        timestamp = utc_time(truth.time_utc)
        expected_ns = self._index * (1_000_000_000 // self.config.update_rate_hz)
        delta = timestamp - self.epoch_utc
        timestamp_ns = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
        if timestamp_ns != expected_ns:
            raise ValueError("Sample each receiver tick in order, starting at epoch_utc")
        position = vector3(truth.position_itrs_m)
        velocity = vector3(truth.velocity_itrs_m_s)
        if np.linalg.norm(position) == 0:
            raise ValueError("GPS truth must have a nonzero Earth-fixed position")
        # UTC -> GPS time uses Astropy's leap-second handling, not a fixed offset.
        gps_seconds = float(Time(timestamp, scale="utc").gps)
        week = math.floor(gps_seconds / 604800)
        tow = gps_seconds - week * 604800
        status = Status.NO_FIX
        domain = (
            3.0 <= conditions.supply_voltage_v <= 3.6
            and -40 <= conditions.temperature_c <= 85
            and conditions.inertial_speed_m_s <= 10_000
            and conditions.acceleration_m_s2 <= 39.2
        )
        threshold = -165 if self._phase == "TRACKING" else (-160 if self._had_fix else -148)
        signal = (
            conditions.antenna_available and conditions.satellites_available >= 4
            and conditions.signal_dbm is not None and conditions.signal_dbm >= threshold
        )
        if not conditions.power_on:
            self._phase, self._since_ns, self._had_fix = "POWERED_OFF", None, False
        elif not domain:
            self._phase, self._since_ns = "OUT_OF_DOMAIN", None
            status |= Status.UNQUALIFIED
        elif not signal:
            self._phase = "REACQUIRING" if self._had_fix else "ACQUIRING"
            self._since_ns = None
        else:
            if self._phase != "TRACKING":
                self._phase = "REACQUIRING" if self._had_fix else "ACQUIRING"
                if self._since_ns is None:
                    self._since_ns = timestamp_ns
                dwell = self.config.reacquisition_s if self._had_fix else getattr(
                    self.config, self.config.start_mode + "_start_s",
                )
                if timestamp_ns - self._since_ns >= dwell * 1e9:
                    self._phase, self._had_fix = "TRACKING", True
            if self._phase == "TRACKING":
                status = Status.OK

        clock_error = 0.0
        if status == Status.OK:
            location = EarthLocation.from_geocentric(*position, unit=u.m)
            lat, lon = location.lat.to_value(u.rad), location.lon.to_value(u.rad)
            enu_to_ecef = np.array([
                [-math.sin(lon), -math.sin(lat) * math.cos(lon), math.cos(lat) * math.cos(lon)],
                [math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat) * math.sin(lon)],
                [0, math.cos(lat), math.sin(lat)],
            ])
            position_error, velocity_error = self._errors()
            measured_position = position + enu_to_ecef @ position_error
            measured_velocity = velocity + enu_to_ecef @ velocity_error
            measured_location = EarthLocation.from_geocentric(*measured_position, unit=u.m)
            clock_error = self.config.clock_bias_s + self.config.clock_sigma_s * self._rng.normal()
            carry, tow = divmod(tow + clock_error, 604800)
            week += int(carry)
            pvt = GpsMeasurement(
                SampleHeader(timestamp_ns, self.config.sensor_id, self._index, status),
                2, conditions.satellites_available, week, tow,
                tuple(float(x) for x in measured_position),
                tuple(float(x) for x in measured_velocity),
                float(measured_location.lat.to_value(u.deg)),
                float(measured_location.lon.to_value(u.deg)),
                float(measured_location.height.to_value(u.m)),
                float(measured_location.height.to_value(u.m)) - conditions.geoid_undulation_m,
                tuple(conditions.dop),
            )
        else:
            self._position_error = self._velocity_error = None
            pvt = GpsMeasurement(
                SampleHeader(timestamp_ns, self.config.sensor_id, self._index, status),
                0, 0, 0, 0.0, None, None, None, None, None, None, None,
            )
        self._index += 1
        return ReceiverSample(self._phase, timestamp, pvt, float(clock_error))


def navigation_frame(measurement: GpsMeasurement) -> bytes:
    """Encode the documented 0xA8 output subset; not command/UART emulation."""
    valid = measurement.header.status == Status.OK
    week = measurement.gps_week
    tow_cs = round(measurement.tow_s * 100)
    carry, tow_cs = divmod(tow_cs, 60_480_000)
    week += carry
    if not 0 <= week <= 65535:
        raise ValueError("GNSS week is outside the binary protocol's range")

    def signed(value: float, scale: float) -> int:
        result = round(value * scale)
        if not -(2**31) <= result < 2**31:
            raise ValueError("Navigation value is outside the signed 32-bit wire range")
        return result

    if valid:
        llh = [
            signed(measurement.latitude_deg, 1e7), signed(measurement.longitude_deg, 1e7),
            signed(measurement.ellipsoid_height_m, 100), signed(measurement.msl_height_m, 100),
        ]
        dop = [round(value * 100) for value in measurement.dop]
        if any(not 0 <= value <= 65535 for value in dop):
            raise ValueError("DOP is outside the binary protocol's range")
        vectors = [
            signed(value, 100) for value in
            (*measurement.position_ecef_m, *measurement.velocity_ecef_m_s)
        ]
    else:
        llh, dop, vectors = [0] * 4, [0] * 5, [0] * 6
    payload = struct.pack(
        ">BBBHIiiiiHHHHHiiiiii",
        0xA8, measurement.fix_mode if valid else 0,
        measurement.satellites_used if valid else 0, week, tow_cs,
        *llh, *dop, *vectors,
    )
    checksum = 0
    for value in payload:
        checksum ^= value
    return b"\xa0\xa1" + len(payload).to_bytes(2, "big") + payload + bytes([checksum]) + b"\r\n"

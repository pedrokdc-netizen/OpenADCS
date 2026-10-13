"""Ground adapters for qualified processing; no sensor/noise truth generator."""

import ctypes as ct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
from pathlib import Path
from typing import Sequence


class Status(IntFlag):
    OK = 0
    INVALID_CONFIG = 1 << 0
    INVALID_INPUT = 1 << 1
    NO_ILLUMINATION = 1 << 2
    OUT_OF_FOV = 1 << 3
    SATURATED = 1 << 4
    DRIVER_FAULT = 1 << 5
    STALE = 1 << 6
    TIME_ORDER = 1 << 7
    CAPACITY = 1 << 8
    UNQUALIFIED = 1 << 9
    ILLUMINATION_UNKNOWN = 1 << 10
    NO_FIX = 1 << 11
    PROTOCOL_ERROR = 1 << 12


class SensorKind(IntEnum):
    SUN = 1
    GYRO = 2
    COARSE_SUN = 3
    IMU = 4
    HORIZON = 5
    GPS = 6


class Illumination(IntEnum):
    UNVERIFIED = 0
    DIRECT_SUN = 1
    DARK = 2


@dataclass(frozen=True)
class SampleHeader:
    timestamp_ns: int
    sensor_id: int
    sequence: int
    status: Status = Status.OK

    def __post_init__(self) -> None:
        _unsigned(self.timestamp_ns, 64, "timestamp_ns")
        _unsigned(self.sensor_id, 32, "sensor_id")
        _unsigned(self.sequence, 32, "sequence")
        _unsigned(self.status, 32, "status")


@dataclass(frozen=True)
class SunConfig:
    active_length_m: Sequence[float]
    mask_height_m: Sequence[float]
    channel_offset: Sequence[float]
    channel_gain: Sequence[float]
    minimum_signal: Sequence[float]
    half_fov_rad: float
    sensor_to_body: Sequence[float]


@dataclass(frozen=True)
class GyroConfig:
    reference_temperature_c: float
    minimum_temperature_c: float
    maximum_temperature_c: float
    bias_polynomial_rad_s: Sequence[float]
    gain: Sequence[float]
    maximum_raw_rate_rad_s: float
    sensor_to_body: Sequence[float]


@dataclass(frozen=True)
class Measurement:
    header: SampleHeader
    kind: SensorKind
    body: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SensorKind):
            raise ValueError("kind must be an explicit SensorKind")
        if self.header.status == Status.OK:
            if self.body is None or len(self.body) != 3:
                raise ValueError("A valid measurement requires three body components")
            if not all(math.isfinite(value) for value in self.body):
                raise ValueError("Valid measurement components must be finite")
        elif self.body is not None:
            raise ValueError("Invalid measurements must not expose a usable vector")


@dataclass(frozen=True)
class Health:
    status: Status
    valid_sun_count: int
    valid_gyro_count: int
    invalid_count: int
    stale_count: int
    future_count: int
    instantaneous_reference_rank: int
    attitude_available: bool
    valid_gps_count: int
    navigation_available: bool


@dataclass(frozen=True)
class GpsMeasurement:
    header: SampleHeader
    fix_mode: int
    satellites_used: int
    gps_week: int
    tow_s: float
    position_ecef_m: tuple[float, float, float] | None
    velocity_ecef_m_s: tuple[float, float, float] | None
    latitude_deg: float | None
    longitude_deg: float | None
    ellipsoid_height_m: float | None
    msl_height_m: float | None
    dop: tuple[float, float, float, float, float] | None

    def __post_init__(self) -> None:
        for name in ("fix_mode", "satellites_used", "gps_week"):
            _unsigned(getattr(self, name), 32, name)
        values = (self.position_ecef_m, self.velocity_ecef_m_s)
        if self.header.status == Status.OK:
            if self.fix_mode not in (2, 3) or not 4 <= self.satellites_used <= 255:
                raise ValueError("A valid GPS record requires a 3D fix and at least four SVs")
            if self.gps_week > 65535 or not math.isfinite(self.tow_s) or not 0 <= self.tow_s < 604800:
                raise ValueError("Invalid GPS week/time of week")
            for vector in values:
                if vector is None or len(vector) != 3 or not all(math.isfinite(x) for x in vector):
                    raise ValueError("Valid GPS records require finite ECEF position and velocity")
            scalars = (self.latitude_deg, self.longitude_deg, self.ellipsoid_height_m, self.msl_height_m)
            if any(x is None or not math.isfinite(x) for x in scalars):
                raise ValueError("Valid GPS records require finite geodetic fields")
            if self.latitude_deg is None or self.longitude_deg is None:
                raise ValueError("Valid GPS records require latitude/longitude")
            if not -90 <= self.latitude_deg <= 90 or not -180 <= self.longitude_deg <= 180:
                raise ValueError("Invalid GPS latitude/longitude")
            if self.dop is None or len(self.dop) != 5 or not all(math.isfinite(x) and x >= 0 for x in self.dop):
                raise ValueError("Valid GPS records require five finite nonnegative DOP fields")
        elif any(value is not None for value in (
            *values, self.latitude_deg, self.longitude_deg, self.ellipsoid_height_m,
            self.msl_height_m, self.dop,
        )):
            raise ValueError("Invalid GPS records must not expose a usable navigation solution")


def _unsigned(value: int, bits: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**bits:
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def _array(values: Sequence[float], count: int) -> ct.Array:
    if len(values) != count:
        raise ValueError(f"Expected exactly {count} components")
    return (ct.c_double * count)(*values)


class _Measurement(ct.Structure):
    _fields_ = [
        ("timestamp_ns", ct.c_uint64), ("sensor_id", ct.c_uint32),
        ("sequence", ct.c_uint32), ("status", ct.c_uint32),
        ("body", ct.c_double * 3),
    ]


class _SunConfig(ct.Structure):
    _fields_ = [
        ("active_length_m", ct.c_double * 2),
        ("mask_height_m", ct.c_double * 2),
        ("channel_offset", ct.c_double * 4),
        ("channel_gain", ct.c_double * 4),
        ("minimum_signal", ct.c_double * 2),
        ("half_fov_rad", ct.c_double),
        ("sensor_to_body", ct.c_double * 9),
    ]


class _SunRaw(ct.Structure):
    _fields_ = [
        ("timestamp_ns", ct.c_uint64), ("sensor_id", ct.c_uint32),
        ("sequence", ct.c_uint32), ("status", ct.c_uint32),
        ("illumination", ct.c_uint32), ("channel", ct.c_double * 4),
    ]


class _GyroConfig(ct.Structure):
    _fields_ = [
        ("reference_temperature_c", ct.c_double),
        ("minimum_temperature_c", ct.c_double),
        ("maximum_temperature_c", ct.c_double),
        ("bias_polynomial_rad_s", ct.c_double * 9),
        ("gain", ct.c_double * 3), ("maximum_raw_rate_rad_s", ct.c_double),
        ("sensor_to_body", ct.c_double * 9),
    ]


class _GyroRaw(ct.Structure):
    _fields_ = [
        ("timestamp_ns", ct.c_uint64), ("sensor_id", ct.c_uint32),
        ("sequence", ct.c_uint32), ("status", ct.c_uint32),
        ("temperature_c", ct.c_double), ("rate_rad_s", ct.c_double * 3),
    ]


class _Slot(ct.Structure):
    _fields_ = [
        ("kind", ct.c_uint32), ("occupied", ct.c_uint32),
        ("measurement", _Measurement),
    ]


class _Gps(ct.Structure):
    _fields_ = [
        ("timestamp_ns", ct.c_uint64), ("sensor_id", ct.c_uint32),
        ("sequence", ct.c_uint32), ("status", ct.c_uint32),
        ("fix_mode", ct.c_uint32), ("satellites_used", ct.c_uint32),
        ("gps_week", ct.c_uint32), ("tow_s", ct.c_double),
        ("position_ecef_m", ct.c_double * 3), ("velocity_ecef_m_s", ct.c_double * 3),
        ("latitude_deg", ct.c_double), ("longitude_deg", ct.c_double),
        ("ellipsoid_height_m", ct.c_double), ("msl_height_m", ct.c_double),
        ("dop", ct.c_double * 5),
    ]


class _GpsSlot(ct.Structure):
    _fields_ = [("occupied", ct.c_uint32), ("measurement", _Gps)]


class _State(ct.Structure):
    _fields_ = [
        ("slots", _Slot * 16), ("gps_slots", _GpsSlot * 4),
        ("maximum_gps_age_ns", ct.c_uint64),
    ]


class _Health(ct.Structure):
    _fields_ = [
        (name, ct.c_uint32) for name in (
            "valid_sun_count", "valid_gyro_count", "invalid_count",
            "stale_count", "future_count", "instantaneous_reference_rank",
            "attitude_available",
            "valid_gps_count", "navigation_available",
        )
    ]


class NativeProcessing:
    """Explicitly load a locally built core, never an implicit library search."""

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path).resolve(strict=True)
        self._library = ct.CDLL(str(self.library_path))
        self._library.openadcs_abi_version.argtypes = []
        self._library.openadcs_abi_version.restype = ct.c_uint32
        if self._library.openadcs_abi_version() != 2:
            raise ValueError("Unsupported OpenADCS processing ABI version")
        signatures = {
            "openadcs_sun_process": [_SunConfig, _SunRaw, _Measurement],
            "openadcs_gyro_process": [_GyroConfig, _GyroRaw, _Measurement],
            "openadcs_accept": [_State, ct.c_uint32, _Measurement],
            "openadcs_health": [
                _State, ct.c_uint64, ct.c_uint64, ct.c_uint64, _Health,
            ],
            "openadcs_init": [_State],
            "openadcs_configure_gps_age": [_State, ct.c_uint64],
            "openadcs_accept_gps": [_State, _Gps],
            "openadcs_gps_decode": [
                ct.POINTER(ct.c_uint8), ct.c_size_t, ct.c_uint64,
                ct.c_uint32, ct.c_uint32, ct.c_uint32, _Gps,
            ],
        }
        for name, arguments in signatures.items():
            function = getattr(self._library, name)
            function.argtypes = [
                ct.POINTER(argument) if issubclass(argument, ct.Structure) else argument
                for argument in arguments
            ]
            function.restype = None if name == "openadcs_init" else ct.c_uint32

    def gps_decode(self, frame: bytes, header: SampleHeader) -> GpsMeasurement:
        data = (ct.c_uint8 * len(frame)).from_buffer_copy(frame)
        output = _Gps()
        self._library.openadcs_gps_decode(
            data, len(frame), header.timestamp_ns, header.sensor_id,
            header.sequence, int(header.status), ct.byref(output),
        )
        valid = output.status == Status.OK
        return GpsMeasurement(
            SampleHeader(output.timestamp_ns, output.sensor_id, output.sequence, Status(output.status)),
            output.fix_mode, output.satellites_used, output.gps_week, output.tow_s,
            (output.position_ecef_m[0], output.position_ecef_m[1], output.position_ecef_m[2]) if valid else None,
            (output.velocity_ecef_m_s[0], output.velocity_ecef_m_s[1], output.velocity_ecef_m_s[2]) if valid else None,
            output.latitude_deg if valid else None, output.longitude_deg if valid else None,
            output.ellipsoid_height_m if valid else None, output.msl_height_m if valid else None,
            (output.dop[0], output.dop[1], output.dop[2], output.dop[3], output.dop[4])
            if valid else None,
        )

    def sun(
        self, config: SunConfig, header: SampleHeader,
        channels: Sequence[float], illumination: Illumination,
    ) -> Measurement:
        if not isinstance(illumination, Illumination):
            raise ValueError("illumination must be an explicit Illumination state")
        native_config = _SunConfig(
            _array(config.active_length_m, 2), _array(config.mask_height_m, 2),
            _array(config.channel_offset, 4), _array(config.channel_gain, 4),
            _array(config.minimum_signal, 2), config.half_fov_rad,
            _array(config.sensor_to_body, 9),
        )
        raw = _SunRaw(
            header.timestamp_ns, header.sensor_id, header.sequence,
            int(header.status), int(illumination), _array(channels, 4),
        )
        output = _Measurement()
        status = Status(self._library.openadcs_sun_process(
            ct.byref(native_config), ct.byref(raw), ct.byref(output),
        ))
        if status == Status.INVALID_CONFIG:
            raise ValueError("Invalid or unqualified Sun-sensor configuration")
        return self._measurement(output, SensorKind.SUN)

    def gyro(
        self, config: GyroConfig, header: SampleHeader,
        rates_rad_s: Sequence[float], temperature_c: float,
    ) -> Measurement:
        native_config = _GyroConfig(
            config.reference_temperature_c, config.minimum_temperature_c,
            config.maximum_temperature_c, _array(config.bias_polynomial_rad_s, 9),
            _array(config.gain, 3), config.maximum_raw_rate_rad_s,
            _array(config.sensor_to_body, 9),
        )
        raw = _GyroRaw(
            header.timestamp_ns, header.sensor_id, header.sequence,
            int(header.status), temperature_c, _array(rates_rad_s, 3),
        )
        output = _Measurement()
        status = Status(self._library.openadcs_gyro_process(
            ct.byref(native_config), ct.byref(raw), ct.byref(output),
        ))
        if status == Status.INVALID_CONFIG:
            raise ValueError("Invalid gyro calibration configuration")
        return self._measurement(output, SensorKind.GYRO)

    @staticmethod
    def _measurement(output: _Measurement, kind: SensorKind) -> Measurement:
        status = Status(output.status)
        return Measurement(
            SampleHeader(output.timestamp_ns, output.sensor_id, output.sequence, status),
            kind, (output.body[0], output.body[1], output.body[2])
            if status == Status.OK else None,
        )


class AdcsInputs:
    """Sensor health/availability only; no hidden attitude estimator or truth."""

    def __init__(self, processing: NativeProcessing, *, maximum_gps_age_ns: int | None = None):
        self.processing = processing
        self._state = _State()
        self.processing._library.openadcs_init(ct.byref(self._state))
        if maximum_gps_age_ns is not None:
            _unsigned(maximum_gps_age_ns, 64, "maximum_gps_age_ns")
            if maximum_gps_age_ns == 0:
                raise ValueError("GPS maximum age must be greater than zero")
            self.processing._library.openadcs_configure_gps_age(
                ct.byref(self._state), maximum_gps_age_ns,
            )

    def submit(self, measurement: Measurement | GpsMeasurement) -> Status:
        if isinstance(measurement, GpsMeasurement):
            if self._state.maximum_gps_age_ns == 0:
                raise ValueError("Configure maximum_gps_age_ns before submitting GPS records")
            header = measurement.header
            native = _Gps(
                header.timestamp_ns, header.sensor_id, header.sequence, int(header.status),
                measurement.fix_mode, measurement.satellites_used, measurement.gps_week,
                measurement.tow_s,
                _array(measurement.position_ecef_m or (0, 0, 0), 3),
                _array(measurement.velocity_ecef_m_s or (0, 0, 0), 3),
                measurement.latitude_deg if measurement.latitude_deg is not None else 0.0,
                measurement.longitude_deg if measurement.longitude_deg is not None else 0.0,
                measurement.ellipsoid_height_m if measurement.ellipsoid_height_m is not None else 0.0,
                measurement.msl_height_m if measurement.msl_height_m is not None else 0.0,
                _array(measurement.dop or (0,) * 5, 5),
            )
            return Status(self.processing._library.openadcs_accept_gps(
                ct.byref(self._state), ct.byref(native),
            ))
        header = measurement.header
        native = _Measurement(
            header.timestamp_ns, header.sensor_id, header.sequence,
            int(header.status), _array(
                measurement.body if measurement.body is not None else (0.0, 0.0, 0.0), 3,
            ),
        )
        return Status(self.processing._library.openadcs_accept(
            ct.byref(self._state), int(measurement.kind), ct.byref(native),
        ))

    def health(
        self, now_ns: int, maximum_sun_age_ns: int, maximum_gyro_age_ns: int,
    ) -> Health:
        _unsigned(maximum_sun_age_ns, 64, "maximum_sun_age_ns")
        _unsigned(maximum_gyro_age_ns, 64, "maximum_gyro_age_ns")
        if maximum_sun_age_ns == 0 or maximum_gyro_age_ns == 0:
            raise ValueError("Sensor maximum ages must be greater than zero")
        output = _Health()
        status = Status(self.processing._library.openadcs_health(
            ct.byref(self._state), _unsigned(now_ns, 64, "now_ns"),
            _unsigned(maximum_sun_age_ns, 64, "maximum_sun_age_ns"),
            _unsigned(maximum_gyro_age_ns, 64, "maximum_gyro_age_ns"),
            ct.byref(output),
        ))
        return Health(
            status, output.valid_sun_count, output.valid_gyro_count,
            output.invalid_count, output.stale_count, output.future_count,
            output.instantaneous_reference_rank, bool(output.attitude_available),
            output.valid_gps_count, bool(output.navigation_available),
        )

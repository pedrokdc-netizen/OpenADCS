"""Subsystem catalogue: engineering-level parameters, validation, builders and a named-config library.

Each subsystem type lists the parameters a systems engineer would choose (turns, wire gauge, core material,
rotor mass...).  ``build`` converts the validated values into the model configuration used by the simulator.
Saved configurations are JSON files ``data/subsystems/<type>/<name>.json`` that the spacecraft window can attach.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCHEMA_VERSION = 1
LIBRARY_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "subsystems"


class SubsystemError(ValueError):
    pass


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    default: Any
    unit: str = ""
    choices: tuple = ()          # non-empty: a drop-down; otherwise a numeric field
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    help: str = ""

    @property
    def is_text(self) -> bool:
        return isinstance(self.default, str) and not self.choices

    def coerce(self, value):
        if self.is_text:
            return str(value).strip() or str(self.default)
        if self.choices:
            if value not in self.choices:
                raise SubsystemError(f"{self.label}: '{value}' is not one of {', '.join(map(str, self.choices))}.")
            return value
        try:
            number = float(str(value).replace(",", "."))
        except ValueError as error:
            raise SubsystemError(f"{self.label}: '{value}' is not a number.") from error
        if not math.isfinite(number):
            raise SubsystemError(f"{self.label}: value must be finite.")
        if self.integer:
            if number != int(number):
                raise SubsystemError(f"{self.label}: must be a whole number.")
            number = int(number)
        if self.minimum is not None and number < self.minimum:
            raise SubsystemError(f"{self.label}: must be at least {self.minimum:g}.")
        if self.maximum is not None and number > self.maximum:
            raise SubsystemError(f"{self.label}: must be at most {self.maximum:g}.")
        return number


def coerce_values(params: tuple[Param, ...], values: dict | None) -> dict:
    values = values or {}
    return {p.key: p.coerce(values.get(p.key, p.default)) for p in params}


@dataclass(frozen=True)
class SubsystemType:
    key: str
    label: str
    description: str
    params: tuple[Param, ...]
    builder: Callable[[dict], Any]
    derived: Callable[[dict], list] | None = None     # [(label, text)] computed read-outs

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def validate(self, values: dict | None) -> dict:
        return coerce_values(self.params, values)

    def build(self, values: dict | None = None):
        return self.builder(self.validate(values))

    def readouts(self, values: dict | None = None) -> list:
        return self.derived(self.validate(values)) if self.derived else []


# ---------------------------------------------------------------------------------------------- wheel
_INERTIA_SHAPES = {"Solid disc (0.5)": 0.5, "Rim-weighted rotor (0.8)": 0.8, "Thin ring (1.0)": 1.0}


def _wheel_inertia(v: dict) -> float:
    return _INERTIA_SHAPES[v["inertia_shape"]] * v["rotor_mass_g"] * 1e-3 * (v["rotor_radius_mm"] * 1e-3) ** 2


def _build_wheel(v: dict):
    from .actuators import ReactionWheelSpec
    max_speed = v["max_speed_rpm"] * 2 * math.pi / 60
    return ReactionWheelSpec(
        name=v["model_name"], mass_kg=v["total_mass_g"] * 1e-3, inertia_kg_m2=_wheel_inertia(v),
        max_torque_nm=v["max_torque_mnm"] * 1e-3, max_speed_rad_s=max_speed,
        coulomb_friction_nm=v["coulomb_friction_unm"] * 1e-6,
        viscous_friction_nm_s=v["viscous_friction_at_max_unm"] * 1e-6 / max_speed)


def _wheel_readouts(v: dict) -> list:
    spec = _build_wheel(v)
    return [("Rotor inertia", f"{spec.inertia_kg_m2 * 1e9:.1f} g mm^2"),
            ("Momentum capacity", f"{spec.max_momentum_nms * 1e3:.3f} mN m s"),
            ("Torque / momentum", f"{spec.max_torque_nm / spec.max_momentum_nms:.3f} 1/s")]


# ---------------------------------------------------------------------------------------------- torquer
CORE_MATERIALS = {   # relative permeability (initial, rod material), density kg/m^3
    "Air (no core)": (1.0, 0.0),
    "Ferrite (soft)": (300.0, 4800.0),
    "Mu-metal / permalloy": (20000.0, 8700.0),
    "Silicon steel": (4000.0, 7650.0),
    "Cobalt-iron (Hiperco)": (10000.0, 8110.0),
}
COPPER_RESISTIVITY_OHM_M, COPPER_DENSITY = 1.68e-8, 8960.0


def _demagnetising_factor(length_m: float, diameter_m: float) -> float:
    m = length_m / diameter_m                  # prolate-spheroid approximation of a rod
    if m <= 1.0001:
        return 1.0 / 3.0
    root = math.sqrt(m * m - 1.0)
    return (m / root * math.log(m + root) - 1.0) / (m * m - 1.0)


def torquer_design(v: dict) -> dict:
    mu_r, density = CORE_MATERIALS[v["core_material"]]
    d_wire, d_core, length = v["wire_diameter_mm"] * 1e-3, v["core_diameter_mm"] * 1e-3, v["core_length_mm"] * 1e-3
    turns = v["turns"]
    per_layer = max(1, int(length // d_wire))
    layers = math.ceil(turns / per_layer)
    mean_diameter = d_core + layers * d_wire
    wire_length = turns * math.pi * mean_diameter
    resistance = COPPER_RESISTIVITY_OHM_M * wire_length / (math.pi * d_wire ** 2 / 4)
    current = min(v["supply_voltage_v"] / resistance, v["current_limit_ma"] * 1e-3)
    mu_eff = mu_r / (1.0 + (mu_r - 1.0) * _demagnetising_factor(length, d_core))
    core_area, coil_area = math.pi * d_core ** 2 / 4, math.pi * mean_diameter ** 2 / 4
    dipole = turns * current * (mu_eff * core_area + (coil_area - core_area))
    mass = COPPER_DENSITY * wire_length * math.pi * d_wire ** 2 / 4 + density * core_area * length
    return {"layers": layers, "resistance_ohm": resistance, "current_a": current, "power_w": current ** 2 * resistance,
            "mu_eff": mu_eff, "dipole_am2": dipole, "mass_kg": mass,
            "voltage_limited": v["supply_voltage_v"] / resistance < v["current_limit_ma"] * 1e-3}


def _build_torquer(v: dict):
    from .actuators import MagnetorquerSpec
    return MagnetorquerSpec(name=v["model_name"], max_dipole_am2=torquer_design(v)["dipole_am2"])


def _torquer_readouts(v: dict) -> list:
    d = torquer_design(v)
    return [("Maximum dipole", f"{d['dipole_am2']:.4f} A m^2"), ("Winding layers", str(d["layers"])),
            ("Resistance", f"{d['resistance_ohm']:.1f} ohm"),
            ("Drive current", f"{d['current_a'] * 1e3:.1f} mA ({'supply' if d['voltage_limited'] else 'limit'}-limited)"),
            ("Power at full dipole", f"{d['power_w']:.3f} W"), ("Effective permeability", f"{d['mu_eff']:.1f}"),
            ("Mass per axis", f"{d['mass_kg'] * 1e3:.1f} g")]


# ---------------------------------------------------------------------------------------------- sun sensor
def _pinhole_fov(v: dict) -> float:
    diagonal = math.hypot(v["width_px"], v["height_px"]) * v["pixel_pitch_um"] * 1e-3
    return math.degrees(2 * math.atan(diagonal / (2 * v["focal_length_mm"])))


def _build_sun(v: dict):
    from .sun_sensor_array import SunSensorArrayConfig
    fov = _pinhole_fov(v) if v["fov_source"] == "Pinhole from focal length" else v["diagonal_fov_deg"]
    return SunSensorArrayConfig(
        centroid_sigma_px=v["centroid_sigma_px"], diagonal_fov_deg=fov, dropout_probability=v["dropout_probability"],
        albedo_coefficient=v["albedo_coefficient"], veiling_glare=v["veiling_glare"],
        threshold_fraction=v["threshold_fraction"], centroid_bias_sigma_px=v["centroid_bias_sigma_px"],
        mount_sigma_deg=v["mount_sigma_deg"])


def _sun_readouts(v: dict) -> list:
    pinhole = _pinhole_fov(v)
    ifov = math.degrees(v["pixel_pitch_um"] * 1e-3 / v["focal_length_mm"])
    return [("Pinhole diagonal FOV", f"{pinhole:.1f} deg"), ("Pixel IFOV (centre)", f"{ifov * 60:.2f} arcmin"),
            ("Centroid accuracy (centre)", f"{ifov * v['centroid_sigma_px'] * 3600:.0f} arcsec 1-sigma")]


# ---------------------------------------------------------------------------------------------- gnss
def _build_gnss(v: dict):
    from .closed_loop import GnssConfig
    return GnssConfig(horizontal_cep_m=v["horizontal_cep_m"], vertical_sigma_m=v["vertical_sigma_m"],
                      velocity_sigma_m_s=v["velocity_sigma_m_s"], position_correlation_s=v["position_correlation_s"],
                      velocity_correlation_s=v["velocity_correlation_s"], reacquisition_s=v["reacquisition_s"])


# ---------------------------------------------------------------------------------------------- magnetometer
def _build_magnetometer(v: dict):
    from .closed_loop import default_magnetometer
    base = default_magnetometer().config
    ones = np.ones(3)
    return replace(base, bias_t=ones * v["bias_nt"] * 1e-9, scale_error=ones * v["scale_error_percent"] * 1e-2,
                   white_noise_sigma_t=ones * v["noise_nt_rms"] * 1e-9, full_scale_t=v["full_scale_ut"] * 1e-6,
                   resolution_t=v["resolution_nt"] * 1e-9, markov_sigma_t=ones * v["markov_sigma_nt"] * 1e-9,
                   markov_time_constant_s=v["markov_time_constant_s"])


# ---------------------------------------------------------------------------------------------- gyro
GYRO_BIAS_PATTERN = np.array([1.0, -1.0, 0.5])


def _build_gyro(v: dict):
    from .closed_loop import default_gyro
    base = default_gyro().config
    ones, deg = np.ones(3), math.pi / 180
    return replace(base, bias_rad_s=GYRO_BIAS_PATTERN * v["turn_on_bias_dps"] * deg,
                   scale_error=ones * v["scale_error_ppm"] * 1e-6,
                   angle_random_walk_rad_s_rthz=ones * v["arw_dps_rthz"] * deg,
                   bias_instability_rad_s=ones * v["bias_instability_dph"] / 3600 * deg,
                   bias_instability_time_constant_s=v["bias_instability_time_constant_s"],
                   rate_random_walk_rad_s2_rthz=ones * v["rrw_dps2_rthz"] * deg,
                   full_scale_rad_s=v["full_scale_dps"] * deg, resolution_rad_s=v["resolution_dps"] * deg)


# ---------------------------------------------------------------------------------------------- horizon
def _build_horizon(v: dict):
    from .horizon_camera import CameraConfig
    from .horizon_sensor import HorizonArchitecture, HorizonSensorConfig
    camera = CameraConfig(width=v["width_px"], height=v["height_px"], hfov_deg=v["hfov_deg"],
                          pixel_pitch_um=v["pixel_pitch_um"], fps=v["frame_rate_hz"],
                          frames_averaged=v["frames_averaged"], fpn_sigma_k=v["fpn_sigma_k"],
                          dead_pixel_fraction=v["dead_pixel_percent"] * 1e-2)
    return HorizonSensorConfig(architecture=HorizonArchitecture(v["architecture"]), camera=camera, mode=v["mode"])


# ---------------------------------------------------------------------------------------------- controller
def _build_controller(v: dict) -> dict:
    deg = math.radians(1.0)
    return {"bandwidth_rad_s": v["bandwidth_rad_s"], "damping": v["damping"], "integral_ratio": v["integral_ratio"],
            "detumble_enter_rad_s": v["detumble_enter_dps"] * deg, "detumble_exit_rad_s": v["detumble_exit_dps"] * deg,
            "nominal_enter_deg": v["nominal_enter_deg"], "nominal_exit_deg": v["nominal_exit_deg"],
            "unload_enter_fraction": v["unload_enter_fraction"], "unload_exit_fraction": v["unload_exit_fraction"],
            "unload_time_constant_s": v["unload_time_constant_s"], "friction_compensation": v["friction_compensation"],
            "wheel_bias_rad_s": v["wheel_bias_rad_s"]}


P = Param
SUBSYSTEM_TYPES: dict[str, SubsystemType] = {t.key: t for t in (
    SubsystemType(
        "reaction_wheel", "Reaction Wheel", "One wheel; the spacecraft window sets how many and their layout.", (
            P("model_name", "Model name", "Custom 10 g wheel"),
            P("total_mass_g", "Total mass (with housing)", 10.0, "g", minimum=0.1),
            P("rotor_mass_g", "Rotor mass", 6.0, "g", minimum=0.1),
            P("rotor_radius_mm", "Rotor outer radius", 6.93, "mm", minimum=1.0),
            P("inertia_shape", "Rotor shape", "Rim-weighted rotor (0.8)", choices=tuple(_INERTIA_SHAPES)),
            P("max_speed_rpm", "Maximum speed", 8500.0, "rpm", minimum=100.0),
            P("max_torque_mnm", "Maximum motor torque", 0.0684, "mN m", minimum=1e-4),
            P("coulomb_friction_unm", "Coulomb friction", 1.368, "uN m", minimum=0.0),
            P("viscous_friction_at_max_unm", "Viscous friction at maximum speed", 2.05, "uN m", minimum=0.0),
        ), _build_wheel, _wheel_readouts),
    SubsystemType(
        "magnetorquer", "Magnetorquer (per axis)", "Wire-wound coil or torque rod; dipole is derived from the winding.", (
            P("model_name", "Model name", "Torque rod 60 mm"),
            P("core_material", "Core material", "Mu-metal / permalloy", choices=tuple(CORE_MATERIALS)),
            P("core_length_mm", "Core length", 60.0, "mm", minimum=5.0),
            P("core_diameter_mm", "Core diameter", 6.0, "mm", minimum=1.0),
            P("wire_diameter_mm", "Wire diameter", 0.16, "mm", minimum=0.02),
            P("turns", "Number of turns", 2400, "", minimum=10, integer=True),
            P("supply_voltage_v", "Supply voltage", 3.3, "V", minimum=0.5),
            P("current_limit_ma", "Driver current limit", 60.0, "mA", minimum=1.0),
        ), _build_torquer, _torquer_readouts),
    SubsystemType(
        "sun_sensor", "Camera Based Sun Sensor", "Six camera heads (one per body face) with centroiding.", (
            P("model_name", "Camera model", "OV5640 (assumed)"),
            P("width_px", "Sensor width", 2592, "px", minimum=64, integer=True),
            P("height_px", "Sensor height", 1944, "px", minimum=64, integer=True),
            P("pixel_pitch_um", "Pixel pitch", 1.4, "um", minimum=0.5),
            P("focal_length_mm", "Focal length", 1.78, "mm", minimum=0.3),
            P("fov_source", "Field of view source", "Specified FOV", choices=("Specified FOV", "Pinhole from focal length")),
            P("diagonal_fov_deg", "Specified diagonal FOV", 120.0, "deg", minimum=5.0, maximum=179.0),
            P("centroid_sigma_px", "Centroid noise (1-sigma)", 0.1, "px", minimum=0.001),
            P("centroid_bias_sigma_px", "Residual calibration bias (1-sigma)", 0.0, "px", minimum=0.0),
            P("threshold_fraction", "Centroid threshold", 0.1, "of full scale", minimum=0.0, maximum=0.9),
            P("veiling_glare", "Lens veiling glare", 0.005, "ratio", minimum=0.0, maximum=0.5),
            P("albedo_coefficient", "Earth albedo seen by the sensor", 0.306, "0..1", minimum=0.0, maximum=1.0),
            P("dropout_probability", "Frame dropout probability", 0.0, "", minimum=0.0, maximum=1.0),
            P("mount_sigma_deg", "Head mounting error (1-sigma)", 0.0, "deg", minimum=0.0),
        ), _build_sun, _sun_readouts),
    SubsystemType(
        "gnss", "GNSS Receiver", "Position/velocity accuracy of the receiver (outages are environmental).", (
            P("model_name", "Receiver model", "Orion B-16 (assumed)"),
            P("horizontal_cep_m", "Horizontal accuracy (CEP)", 2.0, "m", minimum=0.01),
            P("vertical_sigma_m", "Vertical accuracy (1-sigma)", 3.0, "m", minimum=0.01),
            P("velocity_sigma_m_s", "Velocity accuracy (1-sigma)", 0.1, "m/s", minimum=0.0001),
            P("position_correlation_s", "Position error correlation time", 10.0, "s", minimum=0.1),
            P("velocity_correlation_s", "Velocity error correlation time", 2.0, "s", minimum=0.1),
            P("reacquisition_s", "Re-acquisition time after outage", 16.0, "s", minimum=0.0),
        ), _build_gnss),
    SubsystemType(
        "magnetometer", "Magnetometer", "Three-axis magnetometer; its mounting is set in the spacecraft window.", (
            P("model_name", "Model name", "Generic MEMS magnetometer"),
            P("bias_nt", "Zero-field offset (per axis)", 100.0, "nT", minimum=0.0),
            P("scale_error_percent", "Scale-factor error", 0.5, "%", minimum=0.0),
            P("noise_nt_rms", "Noise per sample (rms)", 100.0, "nT", minimum=0.0),
            P("markov_sigma_nt", "Slow drift (1-sigma)", 0.0, "nT", minimum=0.0),
            P("markov_time_constant_s", "Slow drift time constant", 0.0, "s", minimum=0.0),
            P("resolution_nt", "Resolution (0 = ideal)", 0.0, "nT", minimum=0.0),
            P("full_scale_ut", "Full scale", 1000.0, "uT", minimum=1.0),
        ), _build_magnetometer),
    SubsystemType(
        "horizon_sensor", "IR Camera Horizon Sensor", "Thermal cameras that observe the Earth limb.", (
            P("model_name", "Model name", "Thermal-45 / MI0802 (assumed)"),
            P("architecture", "Architecture", "independent", choices=("single", "independent", "fused")),
            P("mode", "Simulation fidelity", "law", choices=("law", "image")),
            P("width_px", "Detector width", 80, "px", minimum=8, integer=True),
            P("height_px", "Detector height", 62, "px", minimum=8, integer=True),
            P("hfov_deg", "Horizontal FOV", 45.0, "deg", minimum=5.0, maximum=170.0),
            P("pixel_pitch_um", "Pixel pitch", 35.0, "um", minimum=1.0),
            P("frame_rate_hz", "Frame rate", 10.0, "Hz", minimum=0.1),
            P("frames_averaged", "Frames averaged", 1, "", minimum=1, integer=True),
            P("fpn_sigma_k", "Fixed-pattern noise", 0.3, "K", minimum=0.0),
            P("dead_pixel_percent", "Dead pixels", 0.5, "%", minimum=0.0, maximum=50.0),
        ), _build_horizon),
    SubsystemType(
        "gyro", "Gyroscope / IMU", "Three-axis rate gyro; its mounting is set in the spacecraft window.", (
            P("model_name", "Model name", "Generic MEMS gyro"),
            P("arw_dps_rthz", "Angle random walk", 0.005, "deg/s/rtHz", minimum=0.0),
            P("bias_instability_dph", "Bias instability", 0.0, "deg/h", minimum=0.0),
            P("bias_instability_time_constant_s", "Bias instability time constant", 0.0, "s", minimum=0.0),
            P("rrw_dps2_rthz", "Rate random walk", 0.0, "deg/s^2/rtHz", minimum=0.0),
            P("turn_on_bias_dps", "Turn-on bias magnitude", 0.05, "deg/s", minimum=0.0),
            P("scale_error_ppm", "Scale-factor error", 500.0, "ppm", minimum=0.0),
            P("full_scale_dps", "Full scale", 573.0, "deg/s", minimum=1.0),
            P("resolution_dps", "Resolution (0 = ideal)", 0.0, "deg/s", minimum=0.0),
        ), _build_gyro),
    SubsystemType(
        "controller", "ADCS Controller (OBC tuning)", "Flight-software gains and mode thresholds.", (
            P("model_name", "Tuning name", "Baseline tuning"),
            P("bandwidth_rad_s", "Control bandwidth", 0.08, "rad/s", minimum=0.001),
            P("damping", "Damping ratio", 0.8, "", minimum=0.1),
            P("integral_ratio", "Integral zero / bandwidth", 0.2, "", minimum=0.0),
            P("detumble_enter_dps", "Detumble entry rate", 5.73, "deg/s", minimum=0.1),
            P("detumble_exit_dps", "Detumble exit rate", 1.72, "deg/s", minimum=0.05),
            P("nominal_enter_deg", "Nominal mode entry (1-sigma)", 2.0, "deg", minimum=0.1),
            P("nominal_exit_deg", "Nominal mode exit (1-sigma)", 6.0, "deg", minimum=0.1),
            P("unload_enter_fraction", "Wheel unloading start", 0.25, "of one wheel", minimum=0.01),
            P("unload_exit_fraction", "Wheel unloading stop", 0.10, "of one wheel", minimum=0.0),
            P("unload_time_constant_s", "Unloading time constant", 150.0, "s", minimum=1.0),
            P("friction_compensation", "Friction feed-forward", 0.8, "0..1", minimum=0.0, maximum=1.0),
            P("wheel_bias_rad_s", "Wheel null-space bias speed", 500.0, "rad/s", minimum=0.0),
        ), _build_controller),
)}
# ---------------------------------------------------------------------------------------------- library
def _safe(name: str) -> str:
    cleaned = re.sub(r"[^\w .()-]+", "_", name.strip()).strip(" .")
    if not cleaned:
        raise SubsystemError("Enter a name for the configuration.")
    return cleaned


def library_path(type_key: str, name: str, directory: Path | None = None) -> Path:
    return (directory or LIBRARY_DIRECTORY) / type_key / f"{_safe(name)}.json"


def save_subsystem(type_key: str, name: str, values: dict, directory: Path | None = None) -> Path:
    path = library_path(type_key, name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_subsystem_file(path, type_key, name, values)


def write_subsystem_file(path: Path, type_key: str, name: str, values: dict) -> Path:
    payload = {"schema_version": SCHEMA_VERSION, "kind": "subsystem", "type": type_key, "name": name.strip(),
               "parameters": get_type(type_key).validate(values)}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return Path(path)


def read_subsystem_file(path: Path) -> tuple[str, str, dict]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubsystemError(f"Cannot read {Path(path).name}: {error}") from error
    if not isinstance(payload, dict) or payload.get("kind") != "subsystem":
        raise SubsystemError("This is not a subsystem configuration file.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SubsystemError(f"Unsupported subsystem schema version {payload.get('schema_version')}.")
    subsystem = get_type(payload.get("type"))
    return subsystem.key, str(payload.get("name", Path(path).stem)), subsystem.validate(payload.get("parameters"))


def load_subsystem(type_key: str, name: str, directory: Path | None = None) -> dict:
    path = library_path(type_key, name, directory)
    if not path.is_file():
        raise SubsystemError(f"No saved {get_type(type_key).label} named '{name}'.")
    key, _, values = read_subsystem_file(path)
    if key != type_key:
        raise SubsystemError(f"'{name}' is a {key} configuration, not {type_key}.")
    return values


def list_saved(type_key: str, directory: Path | None = None) -> list[str]:
    folder = (directory or LIBRARY_DIRECTORY) / type_key
    names = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            names.append(read_subsystem_file(path)[1])
        except SubsystemError:
            continue
    return names


def get_type(type_key) -> SubsystemType:
    try:
        return SUBSYSTEM_TYPES[type_key]
    except KeyError as error:
        raise SubsystemError(f"Unknown subsystem type '{type_key}'.") from error


def build_saved(type_key: str, name: str | None, directory: Path | None = None):
    """Model configuration for a saved subsystem (``None`` name = the built-in defaults)."""
    subsystem = get_type(type_key)
    values = subsystem.defaults() if name is None else load_subsystem(type_key, name, directory)
    return subsystem.builder(subsystem.validate(values))

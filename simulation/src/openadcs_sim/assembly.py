"""Assembled spacecraft: mass properties + spacecraft-level variables + attached subsystem configurations.

Spacecraft-level variables are what the integrator decides when the vehicle is assembled (wheel layout, sensor
mounting, outer dimensions, optical and magnetic properties).  Environment settings (field-model noise, atmosphere,
GNSS outages, orbit) are not part of it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import subsystems
from .spacecraft import SpacecraftConfig, SpacecraftError, pilot1_baseline

SCHEMA_VERSION = 2
KIND = "spacecraft_assembly"
DEFAULT_SLOT = "(built-in default)"
NONE_SLOT = "(not fitted)"
# slot -> may be left unfitted
SLOTS = {"reaction_wheel": False, "magnetorquer": False, "sun_sensor": False, "gnss": False,
         "magnetometer": False, "gyro": False, "horizon_sensor": True, "controller": False}

P = subsystems.Param
VARIABLE_GROUPS: dict[str, tuple[P, ...]] = {
    "Structure and surfaces": (
        P("box_x_mm", "Body length (X)", 200.0, "mm", minimum=10.0),
        P("box_y_mm", "Body width (Y)", 100.0, "mm", minimum=10.0),
        P("box_z_mm", "Body height (Z)", 100.0, "mm", minimum=10.0),
        P("drag_coefficient", "Drag coefficient", 2.2, "", minimum=1.0, maximum=4.0),
        P("specular_reflectivity", "Specular reflectivity", 0.10, "0..1", minimum=0.0, maximum=1.0),
        P("diffuse_reflectivity", "Diffuse reflectivity", 0.20, "0..1", minimum=0.0, maximum=1.0),
        P("residual_dipole_x_am2", "Residual magnetic moment X", 0.003, "A m^2"),
        P("residual_dipole_y_am2", "Residual magnetic moment Y", -0.002, "A m^2"),
        P("residual_dipole_z_am2", "Residual magnetic moment Z", 0.004, "A m^2"),
    ),
    "Reaction wheel layout": (
        P("wheel_layout", "Wheel layout", "Pyramid (4 wheels)", choices=("Pyramid (4 wheels)", "Orthogonal (3 wheels)")),
        P("pyramid_skew_deg", "Pyramid skew angle", 54.7356, "deg", minimum=5.0, maximum=85.0),
    ),
    "Gyroscope mounting": tuple(
        P(f"gyro_{axis}_deg", f"Mounting {axis}", 0.0, "deg", minimum=-180.0, maximum=180.0)
        for axis in ("roll", "pitch", "yaw")),
    "Magnetometer mounting": tuple(
        P(f"magnetometer_{axis}_deg", f"Mounting {axis}", 0.0, "deg", minimum=-180.0, maximum=180.0)
        for axis in ("roll", "pitch", "yaw")),
    "Horizon camera mounting": (
        P("horizon_cam1_azimuth_deg", "Camera 1 azimuth about nadir", 0.0, "deg", minimum=-360.0, maximum=360.0),
        P("horizon_cam1_off_nadir_deg", "Camera 1 boresight off nadir", 66.0, "deg", minimum=0.0, maximum=120.0),
        P("horizon_cam2_azimuth_deg", "Camera 2 azimuth about nadir", 90.0, "deg", minimum=-360.0, maximum=360.0),
        P("horizon_cam2_off_nadir_deg", "Camera 2 boresight off nadir", 66.0, "deg", minimum=0.0, maximum=120.0),
    ),
}
VARIABLES: tuple[P, ...] = tuple(p for group in VARIABLE_GROUPS.values() for p in group)


def euler_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class SpacecraftAssembly:
    mass_properties: SpacecraftConfig = field(default_factory=pilot1_baseline)
    variables: dict = field(default_factory=lambda: {p.key: p.default for p in VARIABLES})
    attached: dict = field(default_factory=lambda: {slot: DEFAULT_SLOT for slot in SLOTS})

    def __post_init__(self):
        object.__setattr__(self, "variables", subsystems.coerce_values(VARIABLES, self.variables))
        attached = {slot: DEFAULT_SLOT for slot in SLOTS}
        for slot, name in dict(self.attached).items():
            if slot not in SLOTS:
                raise SpacecraftError(f"Unknown subsystem slot '{slot}'.")
            if name == NONE_SLOT and not SLOTS[slot]:
                raise SpacecraftError(f"The {subsystems.get_type(slot).label} cannot be left unfitted.")
            attached[slot] = name
        object.__setattr__(self, "attached", attached)

    # ------------------------------------------------------------------ persistence
    def to_dict(self, directory: Path | None = None, embed: bool = True) -> dict:
        record = {"schema_version": SCHEMA_VERSION, "kind": KIND, "mass_properties": self.mass_properties.to_dict(),
                  "variables": dict(self.variables), "attached": dict(self.attached)}
        if embed:
            record["subsystems"] = {}
            for slot, name in self.attached.items():
                if name not in (DEFAULT_SLOT, NONE_SLOT):
                    record["subsystems"][slot] = {"name": name,
                                                  "parameters": subsystems.load_subsystem(slot, name, directory)}
        return record

    @classmethod
    def from_dict(cls, record: dict, directory: Path | None = None) -> "SpacecraftAssembly":
        """Accepts assembly files and legacy mass-properties-only files; embedded subsystems are added to the library."""
        if not isinstance(record, dict):
            raise SpacecraftError("The spacecraft file must contain a JSON object.")
        if record.get("kind") != KIND:
            return cls(mass_properties=SpacecraftConfig.from_dict(record))
        if record.get("schema_version") != SCHEMA_VERSION:
            raise SpacecraftError(f"Unsupported spacecraft assembly schema version {record.get('schema_version')}.")
        try:
            variables = subsystems.coerce_values(VARIABLES, record.get("variables"))
            attached = dict(record.get("attached", {}))
            for slot, entry in dict(record.get("subsystems", {})).items():
                known = subsystems.list_saved(slot, directory)
                if entry["name"] not in known:
                    subsystems.save_subsystem(slot, entry["name"], entry["parameters"], directory)
        except (subsystems.SubsystemError, KeyError, TypeError) as error:
            raise SpacecraftError(f"Invalid spacecraft assembly: {error}") from error
        return cls(SpacecraftConfig.from_dict(record["mass_properties"]), variables, attached)

    def save(self, path: Path | str, directory: Path | None = None) -> None:
        try:
            text = json.dumps(self.to_dict(directory), indent=2)
        except subsystems.SubsystemError as error:
            raise SpacecraftError(str(error)) from error
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str, directory: Path | None = None) -> "SpacecraftAssembly":
        try:
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SpacecraftError(f"Cannot read {Path(path).name}: {error}") from error
        return cls.from_dict(record, directory)

    # ------------------------------------------------------------------ models
    def _config(self, slot: str, directory: Path | None):
        name = self.attached[slot]
        try:
            return subsystems.build_saved(slot, None if name == DEFAULT_SLOT else name, directory)
        except subsystems.SubsystemError as error:
            raise SpacecraftError(str(error)) from error

    def closed_loop_kwargs(self, directory: Path | None = None) -> dict:
        """Keyword arguments for ``ClosedLoopSimulation`` that realise this assembly."""
        from .actuators import ReactionWheelArray, pyramid_axes
        from .adcs_control import AdcsConfig
        from .disturbances import DisturbanceConfig
        from .horizon_camera import CameraMount
        from .horizon_sensor import HorizonArchitecture
        v = self.variables
        axes = pyramid_axes(v["pyramid_skew_deg"]) if v["wheel_layout"].startswith("Pyramid") else np.eye(3)
        wheels = ReactionWheelArray(self._config("reaction_wheel", directory), axes)
        adcs = replace(AdcsConfig(np.array(self.mass_properties.inertia_kg_m2), wheels,
                                  magnetorquer=self._config("magnetorquer", directory)),
                       **self._config("controller", directory))
        gyro = self._config("gyro", directory)
        mag = self._config("magnetometer", directory)
        gyro = replace(gyro, sensor_from_body=euler_matrix(v["gyro_roll_deg"], v["gyro_pitch_deg"], v["gyro_yaw_deg"]).T)
        mag = replace(mag, sensor_from_body=euler_matrix(
            v["magnetometer_roll_deg"], v["magnetometer_pitch_deg"], v["magnetometer_yaw_deg"]).T)
        horizon = None
        if self.attached["horizon_sensor"] != NONE_SLOT:
            horizon = self._config("horizon_sensor", directory)
            mounts = (CameraMount(v["horizon_cam1_azimuth_deg"], v["horizon_cam1_off_nadir_deg"]),
                      CameraMount(v["horizon_cam2_azimuth_deg"], v["horizon_cam2_off_nadir_deg"]))
            horizon = replace(horizon, mounts=mounts[:1] if horizon.architecture is HorizonArchitecture.SINGLE else mounts)
        disturbances = replace(
            DisturbanceConfig(), dimensions_m=(v["box_x_mm"] * 1e-3, v["box_y_mm"] * 1e-3, v["box_z_mm"] * 1e-3),
            drag_coefficient=v["drag_coefficient"], specular_reflectivity=v["specular_reflectivity"],
            diffuse_reflectivity=v["diffuse_reflectivity"],
            residual_dipole_am2=(v["residual_dipole_x_am2"], v["residual_dipole_y_am2"], v["residual_dipole_z_am2"]))
        return {"spacecraft": self.mass_properties, "wheels": wheels, "adcs_config": adcs,
                "gyro_config": gyro, "magnetometer_config": mag, "sun_config": self._config("sun_sensor", directory),
                "horizon_config": horizon, "use_horizon": horizon is not None,
                "gnss": self._config("gnss", directory), "disturbances": disturbances}

    def simulation(self, env, directory: Path | None = None, **options):
        from .closed_loop import ClosedLoopSimulation
        return ClosedLoopSimulation(env, **{**self.closed_loop_kwargs(directory), **options})

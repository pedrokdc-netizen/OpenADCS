"""Spacecraft mass properties: validated model, JSON configuration files and CAD-import helper.

SI units everywhere in memory and in the JSON file (kg, m, kg m^2).  The inertia tensor is the true
symmetric inertia matrix about the centre of mass, in the BODY axes.  The centre of mass is expressed
in the same axes relative to the CAD/geometric origin.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
G_MM2_TO_KG_M2 = 1e-9          # 1 g mm^2 = 1e-3 kg * 1e-6 m^2
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "spacecraft"


class SpacecraftError(ValueError):
    pass


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise SpacecraftError(f"{name} must be 3 finite numbers")
    return array


@dataclass(frozen=True)
class SpacecraftConfig:
    name: str
    mass_kg: float
    inertia_kg_m2: np.ndarray                       # 3x3 about the centre of mass, body axes
    com_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    surface_area_m2: float | None = None
    volume_m3: float | None = None
    notes: str = ""

    def __post_init__(self):
        if not self.name.strip():
            raise SpacecraftError("Spacecraft name is required")
        if not (math.isfinite(self.mass_kg) and self.mass_kg > 0):
            raise SpacecraftError("Mass must be positive and finite")
        inertia = np.asarray(self.inertia_kg_m2, dtype=float)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
            raise SpacecraftError("Inertia must be a finite 3x3 matrix")
        if not np.allclose(inertia, inertia.T, rtol=1e-6, atol=1e-12 * max(1.0, np.abs(inertia).max())):
            raise SpacecraftError("Inertia matrix must be symmetric")
        inertia = 0.5 * (inertia + inertia.T)
        moments = np.linalg.eigvalsh(inertia)
        if moments[0] <= 0:
            raise SpacecraftError("Inertia matrix must be positive definite")
        a, b, c = moments
        if a + b < c * (1 - 1e-9):
            raise SpacecraftError("Principal moments violate the triangle inequality (physically impossible body)")
        for label, value in (("surface area", self.surface_area_m2), ("volume", self.volume_m3)):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise SpacecraftError(f"{label} must be positive when given")
        object.__setattr__(self, "inertia_kg_m2", inertia)
        object.__setattr__(self, "com_m", _vector(self.com_m, "Centre of mass"))

    @property
    def principal_moments_kg_m2(self) -> np.ndarray:
        return np.linalg.eigvalsh(self.inertia_kg_m2)

    @property
    def principal_axes(self) -> np.ndarray:
        """Columns are the principal axes in body coordinates (ascending moment)."""
        return np.linalg.eigh(self.inertia_kg_m2)[1]

    @property
    def inertia_inverse(self) -> np.ndarray:
        return np.linalg.inv(self.inertia_kg_m2)

    def to_dict(self) -> dict:
        record = {"schema_version": SCHEMA_VERSION, "name": self.name, "mass_kg": self.mass_kg,
                  "com_m": self.com_m.tolist(), "inertia_kg_m2": self.inertia_kg_m2.tolist(),
                  "frame": "body axes; inertia about the centre of mass", "notes": self.notes}
        if self.surface_area_m2 is not None:
            record["surface_area_m2"] = self.surface_area_m2
        if self.volume_m3 is not None:
            record["volume_m3"] = self.volume_m3
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "SpacecraftConfig":
        if not isinstance(record, dict):
            raise SpacecraftError("Spacecraft file must contain a JSON object")
        if record.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise SpacecraftError(f"Unsupported spacecraft schema_version {record.get('schema_version')!r}")
        try:
            return cls(str(record["name"]), float(record["mass_kg"]), np.array(record["inertia_kg_m2"], dtype=float),
                       np.array(record.get("com_m", [0.0, 0.0, 0.0]), dtype=float),
                       None if record.get("surface_area_m2") is None else float(record["surface_area_m2"]),
                       None if record.get("volume_m3") is None else float(record["volume_m3"]),
                       str(record.get("notes", "")))
        except KeyError as error:
            raise SpacecraftError(f"Spacecraft file is missing {error.args[0]!r}") from error
        except (TypeError, ValueError) as error:
            if isinstance(error, SpacecraftError):
                raise
            raise SpacecraftError(f"Invalid spacecraft value: {error}") from error

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "SpacecraftConfig":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as error:
            raise SpacecraftError(f"{Path(path).name} is not valid JSON: {error}") from error


def from_cad(name: str, mass_g: float, com_mm, inertia_g_mm2, positive_tensor_notation: bool = True,
             surface_area_mm2: float | None = None, volume_mm3: float | None = None,
             notes: str = "") -> SpacecraftConfig:
    """Build a config from CAD mass properties (g, mm, g mm^2), inertia about the centre of mass.

    SolidWorks' "positive tensor notation" lists products of inertia, so the true inertia matrix has the
    off-diagonal terms negated (verified against the reported principal moments).
    """
    inertia = np.array(inertia_g_mm2, dtype=float) * G_MM2_TO_KG_M2
    if positive_tensor_notation:
        inertia = 2 * np.diag(np.diag(inertia)) - inertia
    return SpacecraftConfig(
        name, mass_g / 1000.0, inertia, np.asarray(com_mm, dtype=float) / 1000.0,
        None if surface_area_mm2 is None else surface_area_mm2 * 1e-6,
        None if volume_mm3 is None else volume_mm3 * 1e-9, notes)


def cad_values(config: SpacecraftConfig, positive_tensor_notation: bool = True) -> dict:
    """Inverse of ``from_cad``: mass [g], COM [mm], inertia [g mm^2] (and optional area/volume)."""
    inertia = config.inertia_kg_m2 / G_MM2_TO_KG_M2
    if positive_tensor_notation:
        inertia = 2 * np.diag(np.diag(inertia)) - inertia
    return {"mass_g": config.mass_kg * 1000.0, "com_mm": config.com_m * 1000.0, "inertia_g_mm2": inertia,
            "surface_area_mm2": None if config.surface_area_m2 is None else config.surface_area_m2 * 1e6,
            "volume_mm3": None if config.volume_m3 is None else config.volume_m3 * 1e9}


def pilot1_baseline() -> SpacecraftConfig:
    """Pilot1_V2 CAD mass properties (user-supplied baseline; CAD axes are ASSUMED to be the body axes)."""
    return from_cad(
        "Pilot1_V2 (baseline)", 500.0, (-27.32, -136.62, -10.20),
        [[877507.45, 2559.92, 203.76], [2559.92, 2072830.46, 20741.53], [203.76, 20741.53, 2481596.41]],
        surface_area_mm2=340702.48, volume_mm3=149613.57,
        notes="Pilot1_V2_cadmodel_sw, mass overridden to 500 g. CAD output axes assumed equal to the body axes. "
              "Inertia about the centre of mass.")

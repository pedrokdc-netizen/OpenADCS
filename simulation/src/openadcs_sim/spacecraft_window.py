"""Separate 'Spacecraft definition' window: edit mass properties, load/save the JSON configuration."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import numpy as np

from .spacecraft import (DEFAULT_DIRECTORY, SpacecraftConfig, SpacecraftError, cad_values, from_cad,
                         pilot1_baseline)

AXES = ("x", "y", "z")


class SpacecraftWindow(tk.Toplevel):
    """Non-modal editor.  Entries use CAD units (g, mm, g mm^2); the JSON file and the simulation use SI."""

    def __init__(self, parent, config: SpacecraftConfig, on_apply: Callable[[SpacecraftConfig], None]):
        super().__init__(parent)
        self.title("OpenADCS | Spacecraft definition")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply
        self.name = tk.StringVar()
        self.mass = tk.StringVar()
        self.area = tk.StringVar()
        self.volume = tk.StringVar()
        self.notes = tk.StringVar()
        self.com = [tk.StringVar() for _ in range(3)]
        self.inertia = [[tk.StringVar() for _ in range(3)] for _ in range(3)]
        self.positive_tensor = tk.BooleanVar(value=True)
        self.message = tk.StringVar()
        self.summary = tk.StringVar()
        self._build()
        self.show(config)
        self.positive_tensor.trace_add("write", lambda *_: self._convention_changed())
        self._convention = self.positive_tensor.get()

    def _build(self) -> None:
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Define the spacecraft mass properties. Enter CAD values (grams, millimetres, "
                             "g mm^2); the saved JSON and the simulation use SI.", wraplength=520).grid(
            row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
        ttk.Label(body, text="Name").grid(row=1, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.name, width=38).grid(row=1, column=1, columnspan=4, sticky="we", pady=2)
        ttk.Label(body, text="Mass (g)").grid(row=2, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.mass, width=12).grid(row=2, column=1, sticky="w", pady=2)
        ttk.Label(body, text="Centre of mass (mm)  x, y, z").grid(row=3, column=0, sticky="w")
        for column, variable in enumerate(self.com):
            ttk.Entry(body, textvariable=variable, width=12).grid(row=3, column=1 + column, sticky="w", pady=2, padx=(0, 4))
        ttk.Label(body, text="Inertia about the centre of mass, body axes (g mm^2)").grid(
            row=4, column=0, columnspan=5, sticky="w", pady=(10, 2))
        for column, axis in enumerate(AXES):
            ttk.Label(body, text=axis.upper()).grid(row=5, column=1 + column)
        for row, axis in enumerate(AXES):
            ttk.Label(body, text=axis.upper()).grid(row=6 + row, column=0, sticky="e")
            for column in range(3):
                ttk.Entry(body, textvariable=self.inertia[row][column], width=14).grid(
                    row=6 + row, column=1 + column, padx=(0, 4), pady=1)
        ttk.Checkbutton(body, text="Off-diagonals are CAD 'positive tensor' products of inertia (SolidWorks)",
                        variable=self.positive_tensor).grid(row=9, column=0, columnspan=5, sticky="w", pady=(4, 0))
        ttk.Label(body, text="Surface area (mm^2, optional)").grid(row=10, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(body, textvariable=self.area, width=14).grid(row=10, column=1, sticky="w", pady=(8, 0))
        ttk.Label(body, text="Volume (mm^3, optional)").grid(row=11, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.volume, width=14).grid(row=11, column=1, sticky="w")
        ttk.Label(body, text="Notes").grid(row=12, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.notes, width=38).grid(row=12, column=1, columnspan=4, sticky="we", pady=2)
        ttk.Label(body, textvariable=self.summary, font=("Consolas", 9), justify=tk.LEFT).grid(
            row=13, column=0, columnspan=5, sticky="w", pady=(8, 0))
        ttk.Label(body, textvariable=self.message, foreground="#b91c1c", wraplength=520).grid(
            row=14, column=0, columnspan=5, sticky="w", pady=(4, 0))
        buttons = ttk.Frame(body)
        buttons.grid(row=15, column=0, columnspan=5, sticky="we", pady=(10, 0))
        for text, command in (("Load JSON...", self.load), ("Save JSON...", self.save),
                              ("Reset to Pilot1 baseline", lambda: self.show(pilot1_baseline()))):
            ttk.Button(buttons, text=text, command=command).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Apply to simulation", command=self.apply).pack(side=tk.RIGHT, padx=(0, 6))

    # ------------------------------------------------------------------ conversion
    def show(self, config: SpacecraftConfig) -> None:
        values = cad_values(config, self.positive_tensor.get())
        self.name.set(config.name)
        self.mass.set(f"{values['mass_g']:.6g}")
        for variable, value in zip(self.com, values["com_mm"]):
            variable.set(f"{value:.6g}")
        for i in range(3):
            for j in range(3):
                self.inertia[i][j].set(f"{values['inertia_g_mm2'][i][j]:.8g}")
        self.area.set("" if values["surface_area_mm2"] is None else f"{values['surface_area_mm2']:.8g}")
        self.volume.set("" if values["volume_mm3"] is None else f"{values['volume_mm3']:.8g}")
        self.notes.set(config.notes)
        self.message.set("")
        self._summarise(config)

    def _convention_changed(self) -> None:
        # The tensor entries flip sign on the off-diagonals when the convention changes.
        if self.positive_tensor.get() == getattr(self, "_convention", self.positive_tensor.get()):
            return
        try:
            for i, j in ((0, 1), (0, 2), (1, 2)):
                value = -float(self.inertia[i][j].get())
                self.inertia[i][j].set(f"{value:.8g}")
                self.inertia[j][i].set(f"{value:.8g}")
        except ValueError:
            pass
        self._convention = self.positive_tensor.get()

    def read_config(self) -> SpacecraftConfig:
        try:
            number = lambda var, label: float(var.get().replace(",", "."))  # noqa: E731
            inertia = [[number(self.inertia[i][j], "inertia") for j in range(3)] for i in range(3)]
            optional = lambda var: None if not var.get().strip() else float(var.get().replace(",", "."))  # noqa: E731
            return from_cad(self.name.get(), number(self.mass, "mass"), [number(v, "com") for v in self.com], inertia,
                            self.positive_tensor.get(), optional(self.area), optional(self.volume), self.notes.get())
        except SpacecraftError:
            raise
        except ValueError as error:
            raise SpacecraftError("Every numeric field must contain a number.") from error

    def _summarise(self, config: SpacecraftConfig) -> None:
        moments = config.principal_moments_kg_m2
        self.summary.set(
            f"SI: mass {config.mass_kg:.4f} kg | principal moments "
            f"{moments[0]:.4e}, {moments[1]:.4e}, {moments[2]:.4e} kg m^2")

    # ------------------------------------------------------------------ actions
    def apply(self) -> SpacecraftConfig | None:
        try:
            config = self.read_config()
        except SpacecraftError as error:
            self.message.set(str(error))
            return None
        self.message.set("")
        self._summarise(config)
        self.on_apply(config)
        return config

    def save(self) -> None:
        try:
            config = self.read_config()
        except SpacecraftError as error:
            self.message.set(str(error))
            return
        DEFAULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            parent=self, title="Save spacecraft configuration", defaultextension=".json",
            initialdir=str(DEFAULT_DIRECTORY), initialfile="spacecraft.json", filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            config.save(Path(filename))
        except OSError as error:
            messagebox.showerror("Cannot save", str(error), parent=self)
            return
        self.message.set("")
        self.on_apply(config)

    def load(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self, title="Load spacecraft configuration",
            initialdir=str(DEFAULT_DIRECTORY if DEFAULT_DIRECTORY.is_dir() else Path.cwd()),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if filename:
            self.load_path(Path(filename))

    def load_path(self, path: Path) -> bool:
        try:
            config = SpacecraftConfig.load(path)
        except (SpacecraftError, OSError) as error:
            self.message.set(str(error))
            return False
        self.show(config)
        self.on_apply(config)
        return True

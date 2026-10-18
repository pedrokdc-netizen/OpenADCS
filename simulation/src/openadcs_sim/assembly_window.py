"""Spacecraft configuration window: assembly variables, attached subsystems, import/export."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from . import subsystems
from .assembly import DEFAULT_SLOT, NONE_SLOT, SLOTS, VARIABLE_GROUPS, SpacecraftAssembly
from .spacecraft import DEFAULT_DIRECTORY, SpacecraftConfig, SpacecraftError
from .subsystem_window import ParamForm, SubsystemEditor


class AssemblyWindow(tk.Toplevel):
    def __init__(self, parent, assembly: SpacecraftAssembly, on_apply: Callable[[SpacecraftAssembly], None],
                 directory: Path | None = None):
        super().__init__(parent)
        self.title("Spacecraft configuration")
        self.on_apply, self.directory = on_apply, directory
        self.mass_properties: SpacecraftConfig = assembly.mass_properties
        self.mass_window = None
        self.editors: list[SubsystemEditor] = []
        body = ttk.Frame(self, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(body)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Spacecraft:").pack(side=tk.LEFT)
        self.name = tk.StringVar()
        ttk.Entry(header, textvariable=self.name, width=28).pack(side=tk.LEFT, padx=6)
        ttk.Button(header, text="Import...", command=self.import_file).pack(side=tk.LEFT)
        ttk.Button(header, text="Export...", command=self.export_file).pack(side=tk.LEFT, padx=6)

        mass = ttk.LabelFrame(body, text="Mass properties", padding=6)
        mass.pack(fill=tk.X, pady=8)
        self.mass_summary = tk.StringVar()
        ttk.Label(mass, textvariable=self.mass_summary, font=("Consolas", 9), justify=tk.LEFT).pack(side=tk.LEFT)
        ttk.Button(mass, text="Edit mass, CoM, inertia...", command=self.edit_mass).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(body)
        notebook.pack(fill=tk.BOTH, expand=True)
        self.forms: dict[str, ParamForm] = {}
        for group, params in VARIABLE_GROUPS.items():
            frame = ttk.Frame(notebook, padding=8)
            form = ParamForm(frame, params, assembly.variables)
            form.pack(fill=tk.X)
            notebook.add(frame, text=group)
            self.forms[group] = form

        self.slot_vars: dict[str, tk.StringVar] = {}
        self.slot_boxes: dict[str, ttk.Combobox] = {}
        sub_frame = ttk.Frame(notebook, padding=8)
        notebook.add(sub_frame, text="Subsystems")
        sub_frame.columnconfigure(1, weight=1)
        ttk.Label(sub_frame, text="Attach saved subsystem configurations. Edit them in their own window.",
                  foreground="#555555", wraplength=460).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        for row, slot in enumerate(SLOTS, start=1):
            ttk.Label(sub_frame, text=subsystems.get_type(slot).label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            variable = tk.StringVar(value=DEFAULT_SLOT)
            box = ttk.Combobox(sub_frame, textvariable=variable, state="readonly", width=34)
            box.grid(row=row, column=1, sticky="we", pady=3)
            ttk.Button(sub_frame, text="Edit...", command=lambda s=slot: self.edit_subsystem(s)).grid(
                row=row, column=2, padx=(6, 0))
            self.slot_vars[slot], self.slot_boxes[slot] = variable, box

        self.message = tk.StringVar()
        ttk.Label(body, textvariable=self.message, foreground="#b91c1c", wraplength=520).pack(anchor="w", pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Apply", command=self.apply).pack(side=tk.RIGHT, padx=6)
        ttk.Button(buttons, text="Reset to baseline", command=lambda: self.show(SpacecraftAssembly())).pack(side=tk.RIGHT)
        self.show(assembly)

    # ------------------------------------------------------------------ state
    def refresh_slots(self) -> None:
        for slot, box in self.slot_boxes.items():
            options = [DEFAULT_SLOT] + ([NONE_SLOT] if SLOTS[slot] else []) + subsystems.list_saved(slot, self.directory)
            box["values"] = options
            if self.slot_vars[slot].get() not in options:
                self.slot_vars[slot].set(DEFAULT_SLOT)

    def show(self, assembly: SpacecraftAssembly) -> None:
        self.mass_properties = assembly.mass_properties
        self.name.set(assembly.mass_properties.name)
        for form in self.forms.values():
            form.set(assembly.variables)
        self.refresh_slots()
        for slot, name in assembly.attached.items():
            if name not in self.slot_boxes[slot]["values"]:
                self.message.set(f"Saved {subsystems.get_type(slot).label} '{name}' was not found; using the default.")
                name = DEFAULT_SLOT
            self.slot_vars[slot].set(name)
        self._summarise()

    def _summarise(self) -> None:
        m = self.mass_properties.principal_moments_kg_m2
        self.mass_summary.set(f"{self.mass_properties.mass_kg * 1e3:.1f} g | principal moments "
                              f"{m[0]:.3e}, {m[1]:.3e}, {m[2]:.3e} kg m^2")

    def read_assembly(self) -> SpacecraftAssembly:
        variables: dict = {}
        for form in self.forms.values():
            variables.update(form.values())
        name = self.name.get().strip() or self.mass_properties.name
        mass = SpacecraftConfig.from_dict({**self.mass_properties.to_dict(), "name": name})
        return SpacecraftAssembly(mass, variables, {slot: var.get() for slot, var in self.slot_vars.items()})

    # ------------------------------------------------------------------ actions
    def edit_mass(self) -> None:
        from .spacecraft_window import SpacecraftWindow
        if self.mass_window is not None and self.mass_window.winfo_exists():
            self.mass_window.lift()
            return
        self.mass_window = SpacecraftWindow(self, self.mass_properties, self._mass_applied)

    def _mass_applied(self, config: SpacecraftConfig) -> None:
        self.mass_properties = config
        self.name.set(config.name)
        self._summarise()

    def edit_subsystem(self, slot: str) -> SubsystemEditor:
        editor = SubsystemEditor(self, slot, self.directory, lambda key, name: self._subsystem_saved(key, name))
        chosen = self.slot_vars[slot].get()
        if chosen in subsystems.list_saved(slot, self.directory):
            editor.saved_choice.set(chosen)
            editor.load_selected()
        self.editors.append(editor)
        return editor

    def _subsystem_saved(self, slot: str, name: str) -> None:
        self.refresh_slots()
        self.slot_vars[slot].set(name)

    def apply(self) -> SpacecraftAssembly | None:
        try:
            assembly = self.read_assembly()
            assembly.closed_loop_kwargs(self.directory)   # surfaces missing or invalid subsystem configs
        except (subsystems.SubsystemError, SpacecraftError, ValueError) as error:
            self.message.set(str(error))
            return None
        self.message.set("")
        self.on_apply(assembly)
        return assembly

    def export_file(self) -> None:
        assembly = self.apply()
        if assembly is None:
            return
        DEFAULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        filename = filedialog.asksaveasfilename(
            parent=self, title="Export spacecraft configuration", defaultextension=".json",
            initialdir=str(DEFAULT_DIRECTORY), initialfile="spacecraft_assembly.json", filetypes=[("JSON", "*.json")])
        if filename:
            self.export_path(Path(filename))

    def export_path(self, path: Path) -> bool:
        assembly = self.apply()
        if assembly is None:
            return False
        try:
            assembly.save(path, self.directory)
        except (SpacecraftError, OSError) as error:
            self.message.set(str(error))
            return False
        return True

    def import_file(self) -> None:
        filename = filedialog.askopenfilename(
            parent=self, title="Import spacecraft configuration",
            initialdir=str(DEFAULT_DIRECTORY if DEFAULT_DIRECTORY.is_dir() else Path.cwd()),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if filename:
            self.import_path(Path(filename))

    def import_path(self, path: Path) -> bool:
        try:
            assembly = SpacecraftAssembly.load(path, self.directory)
        except (SpacecraftError, subsystems.SubsystemError, OSError) as error:
            self.message.set(str(error))
            return False
        self.message.set("")
        self.show(assembly)
        return True

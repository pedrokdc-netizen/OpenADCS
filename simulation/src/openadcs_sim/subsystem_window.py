"""Subsystem configuration windows: a catalogue list and a form-driven editor for each subsystem type."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from . import subsystems
from .subsystems import Param, SubsystemError, SubsystemType


class ParamForm(ttk.Frame):
    """Labelled rows: drop-down for choice parameters, entry for numbers and names."""

    def __init__(self, parent, params: tuple[Param, ...], values: dict | None = None,
                 on_change: Callable[[], None] | None = None):
        super().__init__(parent)
        self.params, self.on_change = params, on_change
        self.variables: dict[str, tk.StringVar] = {}
        self.widgets: dict[str, tk.Widget] = {}
        self.columnconfigure(1, weight=1)
        for row, param in enumerate(params):
            ttk.Label(self, text=param.label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            variable = tk.StringVar(value=str(param.default))
            if param.choices:
                widget = ttk.Combobox(self, textvariable=variable, values=list(param.choices), state="readonly", width=28)
                widget.bind("<<ComboboxSelected>>", lambda _e: self._changed())
            else:
                widget = ttk.Entry(self, textvariable=variable, width=30)
                widget.bind("<KeyRelease>", lambda _e: self._changed())
            widget.grid(row=row, column=1, sticky="we", pady=2)
            ttk.Label(self, text=param.unit, foreground="#555555", width=12).grid(row=row, column=2, sticky="w", padx=(6, 0))
            self.variables[param.key], self.widgets[param.key] = variable, widget
        if values:
            self.set(values)

    def _changed(self) -> None:
        if self.on_change:
            self.on_change()

    def set(self, values: dict) -> None:
        for param in self.params:
            if param.key in values:
                value = values[param.key]
                self.variables[param.key].set(f"{value:.10g}" if isinstance(value, float) else str(value))

    def raw(self) -> dict:
        return {key: variable.get() for key, variable in self.variables.items()}

    def values(self) -> dict:
        return subsystems.coerce_values(self.params, self.raw())


class SubsystemEditor(tk.Toplevel):
    """Edit one subsystem type and keep named configurations in the library."""

    def __init__(self, parent, type_key: str, directory: Path | None = None,
                 on_saved: Callable[[str, str], None] | None = None):
        super().__init__(parent)
        self.subsystem: SubsystemType = subsystems.get_type(type_key)
        self.directory, self.on_saved = directory, on_saved
        self.title(f"{self.subsystem.label} configuration")
        self.resizable(False, True)
        body = ttk.Frame(self, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text=self.subsystem.label, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(body, text=self.subsystem.description, wraplength=480, foreground="#555555").pack(anchor="w", pady=(0, 8))

        saved = ttk.LabelFrame(body, text="Saved configurations", padding=6)
        saved.pack(fill=tk.X)
        self.saved_choice = tk.StringVar()
        self.saved_box = ttk.Combobox(saved, textvariable=self.saved_choice, state="readonly", width=34)
        self.saved_box.pack(side=tk.LEFT)
        ttk.Button(saved, text="Load", command=self.load_selected).pack(side=tk.LEFT, padx=6)
        ttk.Button(saved, text="Import file...", command=self.import_file).pack(side=tk.LEFT)

        self.form = ParamForm(body, self.subsystem.params, on_change=self.refresh)
        self.form.pack(fill=tk.X, pady=8)
        self.readout = tk.StringVar()
        ttk.Label(body, textvariable=self.readout, font=("Consolas", 9), justify=tk.LEFT).pack(anchor="w")
        self.message = tk.StringVar()
        ttk.Label(body, textvariable=self.message, foreground="#b91c1c", wraplength=480).pack(anchor="w", pady=(4, 0))

        save_row = ttk.Frame(body)
        save_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(save_row, text="Save as:").pack(side=tk.LEFT)
        self.save_name = tk.StringVar(value=self.form.variables["model_name"].get() if "model_name" in self.form.variables else "")
        ttk.Entry(save_row, textvariable=self.save_name, width=30).pack(side=tk.LEFT, padx=6)
        ttk.Button(save_row, text="Save", command=self.save).pack(side=tk.LEFT)
        ttk.Button(save_row, text="Export file...", command=self.export_file).pack(side=tk.LEFT, padx=6)
        ttk.Button(save_row, text="Defaults", command=self.reset).pack(side=tk.RIGHT)
        self.refresh_saved()
        self.refresh()

    # ------------------------------------------------------------------ helpers
    def refresh_saved(self) -> None:
        names = subsystems.list_saved(self.subsystem.key, self.directory)
        self.saved_box["values"] = names
        if names and self.saved_choice.get() not in names:
            self.saved_choice.set(names[0])

    def refresh(self) -> None:
        try:
            lines = [f"{label}: {text}" for label, text in self.subsystem.readouts(self.form.values())]
        except SubsystemError as error:
            self.readout.set("")
            self.message.set(str(error))
            return
        except Exception as error:   # derived read-outs must never break the form
            self.readout.set("")
            self.message.set(f"Cannot evaluate these values: {error}")
            return
        self.message.set("")
        self.readout.set("\n".join(lines))

    def show(self, values: dict, name: str | None = None) -> None:
        self.form.set(values)
        if name is not None:
            self.save_name.set(name)
        self.refresh()

    # ------------------------------------------------------------------ actions
    def reset(self) -> None:
        self.show(self.subsystem.defaults())

    def load_selected(self) -> bool:
        name = self.saved_choice.get()
        if not name:
            self.message.set("There is no saved configuration to load.")
            return False
        try:
            self.show(subsystems.load_subsystem(self.subsystem.key, name, self.directory), name)
        except SubsystemError as error:
            self.message.set(str(error))
            return False
        return True

    def save(self) -> Path | None:
        try:
            values = self.form.values()
            self.subsystem.build(values)
            path = subsystems.save_subsystem(self.subsystem.key, self.save_name.get(), values, self.directory)
        except (SubsystemError, ValueError) as error:
            self.message.set(str(error))
            return None
        except OSError as error:
            messagebox.showerror("Cannot save", str(error), parent=self)
            return None
        self.refresh_saved()
        self.saved_choice.set(self.save_name.get().strip())
        self.message.set("")
        if self.on_saved:
            self.on_saved(self.subsystem.key, self.save_name.get().strip())
        return path

    def export_file(self) -> None:
        filename = filedialog.asksaveasfilename(parent=self, title="Export subsystem configuration",
                                                defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not filename:
            return
        try:
            values = self.form.values()
            self.subsystem.build(values)
            subsystems.write_subsystem_file(Path(filename), self.subsystem.key, self.save_name.get() or "unnamed", values)
        except (SubsystemError, ValueError, OSError) as error:
            self.message.set(str(error))

    def import_file(self) -> None:
        filename = filedialog.askopenfilename(parent=self, title="Import subsystem configuration",
                                              filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if filename:
            self.import_path(Path(filename))

    def import_path(self, path: Path) -> bool:
        try:
            key, name, values = subsystems.read_subsystem_file(path)
            if key != self.subsystem.key:
                raise SubsystemError(f"This file holds a {subsystems.get_type(key).label}, not a {self.subsystem.label}.")
        except SubsystemError as error:
            self.message.set(str(error))
            return False
        self.show(values, name)
        return True


class SubsystemListWindow(tk.Toplevel):
    """Catalogue of subsystem types; choosing one opens its editor."""

    def __init__(self, parent, directory: Path | None = None, on_saved: Callable[[str, str], None] | None = None):
        super().__init__(parent)
        self.title("Spacecraft subsystems")
        self.directory, self.on_saved = directory, on_saved
        self.editors: dict[str, SubsystemEditor] = {}
        body = ttk.Frame(self, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Choose a subsystem to configure", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.keys = list(subsystems.SUBSYSTEM_TYPES)
        self.listbox = tk.Listbox(body, height=len(self.keys), exportselection=False, width=44)
        for key in self.keys:
            self.listbox.insert(tk.END, subsystems.SUBSYSTEM_TYPES[key].label)
        self.listbox.pack(fill=tk.X, pady=8)
        self.listbox.selection_set(0)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.open_selected())
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._describe())
        self.description = tk.StringVar()
        ttk.Label(body, textvariable=self.description, wraplength=360, foreground="#555555").pack(anchor="w")
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(buttons, text="Configure...", command=self.open_selected).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        self._describe()

    def _describe(self) -> None:
        selection = self.listbox.curselection()
        if selection:
            key = self.keys[selection[0]]
            saved = len(subsystems.list_saved(key, self.directory))
            self.description.set(f"{subsystems.SUBSYSTEM_TYPES[key].description}  ({saved} saved)")

    def open_selected(self, key: str | None = None) -> SubsystemEditor | None:
        if key is None:
            selection = self.listbox.curselection()
            if not selection:
                return None
            key = self.keys[selection[0]]
        editor = self.editors.get(key)
        if editor is not None and editor.winfo_exists():
            editor.lift()
            return editor
        editor = SubsystemEditor(self, key, self.directory, self._saved)
        self.editors[key] = editor
        return editor

    def _saved(self, key: str, name: str) -> None:
        self._describe()
        if self.on_saved:
            self.on_saved(key, name)

"""Horizon sensor tab: simulated IR frames, estimate vs truth and a pointing-error history."""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import horizon_scene as hs
from .horizon_sensor import HorizonArchitecture, HorizonSensorConfig, HorizonSensorModel

ARCHITECTURE_LABELS = {
    "1 camera": HorizonArchitecture.SINGLE,
    "2 independent cameras": HorizonArchitecture.INDEPENDENT,
    "2 fused cameras": HorizonArchitecture.FUSED,
}
SUN_COLOR = "#facc15"
EMPTY_TEXT = "Run a simulation, then select a time on the timeline."


class HorizonPanel(ttk.Frame):
    """Needs the stored samples (``set_samples``) and the selected one (``show_sample``)."""

    def __init__(self, parent, view_of: Callable):
        super().__init__(parent)
        self.view_of = view_of
        self.samples: list = []
        self.sample = None
        self.history: hs.HorizonHistory | None = None
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._marker = None
        self._shown = None

        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(fill=tk.X)
        self.architecture = tk.StringVar(value="2 independent cameras")
        self.roll = tk.StringVar(value="2.0")
        self.pitch = tk.StringVar(value="-1.5")
        self.step = tk.StringVar(value="60")
        ttk.Label(bar, text="Architecture").pack(side=tk.LEFT)
        box = ttk.Combobox(bar, textvariable=self.architecture, values=list(ARCHITECTURE_LABELS),
                           state="readonly", width=22)
        box.pack(side=tk.LEFT, padx=(4, 12))
        box.bind("<<ComboboxSelected>>", lambda _: self._settings_changed())
        for label, variable in (("True roll offset (deg)", self.roll), ("pitch (deg)", self.pitch),
                                ("History step (s)", self.step)):
            ttk.Label(bar, text=label).pack(side=tk.LEFT)
            entry = ttk.Entry(bar, textvariable=variable, width=6)
            entry.pack(side=tk.LEFT, padx=(4, 10))
            entry.bind("<Return>", lambda _: self._settings_changed())
        self.compute_button = ttk.Button(bar, text="Compute history", command=self.compute_history)
        self.compute_button.pack(side=tk.LEFT)
        self.status = tk.StringVar(value=EMPTY_TEXT)
        ttk.Label(self, textvariable=self.status, padding=(8, 0)).pack(fill=tk.X)

        self.figure = Figure(figsize=(8, 6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False).pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty()

    # ------------------------------------------------------------------ inputs
    def sensor_config(self) -> HorizonSensorConfig:
        return HorizonSensorConfig(architecture=ARCHITECTURE_LABELS[self.architecture.get()])

    def offset(self) -> tuple[float, float]:
        try:
            return float(self.roll.get()), float(self.pitch.get())
        except ValueError:
            return 0.0, 0.0

    def set_samples(self, samples) -> None:
        self.cancel()
        self.samples = list(samples)
        self.history = None
        self._shown = None
        if not self.samples:
            self.sample = None
            self.status.set(EMPTY_TEXT)
            self._draw_empty()

    def clear(self) -> None:
        self.set_samples([])

    def _settings_changed(self) -> None:
        self.history = None
        self._shown = None
        if self.sample is not None:
            self.show_sample(self.sample)

    # ------------------------------------------------------------------ selected sample
    def show_sample(self, sample) -> None:
        if sample is self._shown:
            return
        self.sample = sample
        self._shown = sample
        cfg = self.sensor_config()
        geometry = hs.scene_from_sample(sample, self.view_of(sample), self.offset())
        delta = self._housing_for(sample)
        model = HorizonSensorModel(cfg, seed=1)
        reading = model.measure_now(geometry.n_body, geometry.r_km, geometry.latitude_deg,
                                    geometry.heading_deg, scene=geometry.environment(delta))
        self._draw(cfg, geometry, reading, model.last_frames, hs.camera_sun_state(cfg, geometry), delta)

    def _housing_for(self, sample) -> float:
        history = self.history
        if history is None or not self.samples:
            return 0.0
        t = (sample.time_utc - self.samples[0].time_utc).total_seconds() / 60.0
        return float(np.interp(t, history.t_min, history.housing_delta_k))

    def _draw_empty(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot()
        axis.text(0.5, 0.5, EMPTY_TEXT, ha="center", va="center", transform=axis.transAxes, color="#64748b")
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _draw(self, cfg, geometry, reading, frames, sun_states, housing_delta) -> None:
        figure = self.figure
        figure.clear()
        n = len(frames)
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.9), width_ratios=(1.0, 1.0, 1.25))
        axes = []
        for i, frame in enumerate(frames):
            axis = figure.add_subplot(grid[0, i])
            axes.append(axis)
            image = axis.imshow(frame, cmap="inferno", vmin=cfg.camera.floor_k - 3, vmax=290.0,
                                interpolation="nearest")
            state = sun_states[i]
            result = reading.cameras[i] if i < len(reading.cameras) else None
            if state["in_fov"] and state["pixel"] is not None:
                axis.plot([state["pixel"][0]], [state["pixel"][1]], marker="*", color=SUN_COLOR,
                          markersize=16, markeredgecolor="black", linestyle="None", label="Sun")
                axis.legend(loc="lower right", fontsize=7)
            text = f"Camera {i} (az {cfg.mounts[i].azimuth_deg:.0f} deg)"
            if result is not None:
                text += f"\n{result.status}, {result.n_points} pts"
                if result.sun_masked_px:
                    text += f", Sun mask {result.sun_masked_px} px"
            axis.set_title(text, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(image, ax=axes, location="bottom", label="Scene temperature (K, 8-14 um)",
                        shrink=0.7, aspect=40, pad=0.02)
        readout = figure.add_subplot(grid[0, 2])
        readout.set_axis_off()
        readout.text(0.0, 1.0, self._readout_text(cfg, geometry, reading, sun_states, housing_delta),
                     va="top", family="monospace", fontsize=7, transform=readout.transAxes)
        self._history_axes(grid)
        self.canvas.draw_idle()

    @staticmethod
    def _readout_text(cfg, geometry, reading, sun_states, housing_delta) -> str:
        truth = np.degrees(hs._roll_pitch(geometry.n_body))
        lines = [f"Architecture : {cfg.architecture.value}",
                 f"Latitude     : {geometry.latitude_deg:7.2f} deg  (heading {geometry.heading_deg:5.1f})",
                 f"Eclipse      : {'YES' if geometry.eclipse else 'no'}   beta {geometry.beta_deg:5.1f} deg",
                 f"Housing dT   : {housing_delta:+6.1f} K"]
        for i, state in enumerate(sun_states):
            lines.append(f"Sun cam {i}    : {'IN FOV' if state['in_fov'] else 'out'} "
                         f"({state['angle_deg']:.0f} deg off boresight)")
        lines.append(f"Truth roll/pitch : {truth[0]:6.2f} {truth[1]:6.2f} deg")
        if reading.valid:
            err = reading.roll_pitch_deg - truth
            sigma = np.degrees(np.sqrt(np.diag(reading.cov_roll_pitch_rad2)))
            point = math.degrees(math.acos(float(np.clip(reading.n_body @ geometry.n_body, -1, 1))))
            lines += [f"Est.  roll/pitch : {reading.roll_pitch_deg[0]:6.2f} {reading.roll_pitch_deg[1]:6.2f} deg",
                      f"Error            : {err[0]:6.2f} {err[1]:6.2f} deg  (|e| {point:.2f})",
                      f"Reported 1-sigma : {sigma[0]:6.2f} {sigma[1]:6.2f} deg",
                      f"Status           : {reading.status}"]
        else:
            lines += ["Estimate         : NONE", f"Status           : {reading.status}"]
        lines.append("Yaw is unobservable from the horizon.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ history
    def _history_axes(self, grid) -> None:
        err = self.figure.add_subplot(grid[1, :])
        self._marker = None
        history = self.history
        if history is None:
            err.text(0.5, 0.5, "Click 'Compute history' to run the sensor over the whole run.",
                     ha="center", va="center", transform=err.transAxes, color="#64748b")
            err.set_axis_off()
            return
        t = history.t_min
        err.plot(t, history.error_deg, ".-", color="#2563eb", label="Pointing error (valid)")
        gaps = ~history.valid
        if gaps.any():
            err.plot(t[gaps], np.zeros(gaps.sum()), "x", color="#dc2626", label="No estimate")
        self._shade(err, t, history.eclipse, "#94a3b8", "Eclipse")
        sun = history.sun_in_fov.any(axis=1)
        if sun.any():
            top = float(np.nanmax(history.error_deg)) * 1.05 if np.isfinite(history.error_deg).any() else 1.0
            err.plot(t[sun], np.full(sun.sum(), top), "*", color=SUN_COLOR, markeredgecolor="black",
                     markersize=11, linestyle="None", label="Sun in FOV")
        err.set(xlabel="Elapsed time (min)", ylabel="Pointing error (deg)")
        err.set_ylim(bottom=-0.3)
        err.grid(alpha=0.25)
        lat = err.twinx()
        lat.plot(t, history.latitude_deg, color="#16a34a", alpha=0.5, linewidth=1)
        lat.set_ylabel("Latitude (deg)", color="#16a34a")
        err.legend(loc="upper left", fontsize=7, ncol=4)
        err.set_title("  |  ".join(f"{k}: p95 {v['p95_deg']:.1f} deg, avail {100 * v['availability']:.0f}%"
                                   for k, v in history.summary().items() if v["samples"]), fontsize=8)
        if self.sample is not None and self.samples:
            minute = (self.sample.time_utc - self.samples[0].time_utc).total_seconds() / 60.0
            self._marker = err.axvline(minute, color="#f97316", linestyle=":", linewidth=1.5)

    @staticmethod
    def _shade(axis, t, mask, color, label) -> None:
        if not mask.any() or len(t) < 2:
            return
        half = 0.5 * float(np.median(np.diff(t)))
        edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
        for k, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            axis.axvspan(t[a] - half, t[b - 1] + half, color=color, alpha=0.25, label=label if k == 0 else None)

    def compute_history(self) -> None:
        if not self.samples or (self._thread is not None and self._thread.is_alive()):
            return
        try:
            step = max(5.0, float(self.step.get()))
        except ValueError:
            step = 60.0
        cfg, offset, samples = self.sensor_config(), self.offset(), list(self.samples)
        self._cancel.clear()
        self.compute_button.state(["disabled"])
        self.status.set("Computing horizon-sensor history ...")

        def work() -> None:
            try:
                result = hs.run_history(samples, self.view_of, cfg, step, offset,
                                        progress=lambda i, n: self._queue.put(("progress", i, n)),
                                        cancelled=self._cancel.is_set)
                self._queue.put(("done", result, None))
            except Exception as error:   # shown in the status line instead of killing the GUI
                self._queue.put(("error", str(error), None))

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self.after(100, self._poll)

    def _poll(self) -> None:
        finished = False
        while True:
            try:
                kind, a, b = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.status.set(f"Computing horizon-sensor history ... {a}/{b}")
            elif kind == "done":
                self.history, finished = a, True
                self.status.set(f"History ready: {len(a.t_min)} captures.")
            else:
                finished = True
                self.status.set(f"History failed: {a}")
        if finished:
            self.compute_button.state(["!disabled"])
            self._shown = None
            if self.sample is not None:
                self.show_sample(self.sample)
        elif self._thread is not None and self._thread.is_alive():
            self.after(150, self._poll)

    def cancel(self) -> None:
        self._cancel.set()

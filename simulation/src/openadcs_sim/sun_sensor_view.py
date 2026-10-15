"""Sun tab panel: six-head OV5640 cube Sun sensor coverage, per-head status and fused error."""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import sun_sensor_array as ssa
from .albedo import context_from_view
from .horizon_scene import in_eclipse

EMPTY_TEXT = "Run a simulation, then select a time on the timeline."
STATUS_COLOR = {"OK": "#15803d", "ECLIPSE": "#475569", "OUT_OF_FOV": "#94a3b8"}


class SunSensorPanel(ttk.Frame):
    """Needs the stored samples (``set_samples``) and the selected one (``show_sample``)."""

    def __init__(self, parent, view_of: Callable):
        super().__init__(parent)
        self.view_of = view_of
        self.samples: list = []
        self.sample = None
        self.history: ssa.SunSensorHistory | None = None
        self._queue: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._shown = None
        self._coverage = None

        ttk.Label(self, text="Sun sensor (OV5640): 6 heads, one per cube face, 120 deg diagonal",
                  font=("TkDefaultFont", 9, "bold"), padding=(6, 2)).pack(fill=tk.X)
        bar = ttk.Frame(self, padding=(6, 2))
        bar.pack(fill=tk.X)
        self.sigma = tk.StringVar(value="0.1")
        self.albedo = tk.StringVar(value="0.30")
        self.leak = tk.StringVar(value="1.0")
        self.step = tk.StringVar(value="60")
        for label, variable in (("Centroid error 1-sigma (px)", self.sigma), ("Albedo (0=off)", self.albedo), ("Albedo leak", self.leak),
                                  ("History step (s)", self.step)):
            ttk.Label(bar, text=label).pack(side=tk.LEFT)
            entry = ttk.Entry(bar, textvariable=variable, width=6)
            entry.pack(side=tk.LEFT, padx=(4, 10))
            entry.bind("<Return>", lambda _: self._settings_changed())
        self.compute_button = ttk.Button(bar, text="Compute history", command=self.compute_history)
        self.compute_button.pack(side=tk.LEFT)
        self.status = tk.StringVar(value=EMPTY_TEXT)
        ttk.Label(self, textvariable=self.status, padding=(8, 0)).pack(fill=tk.X)

        self.figure = Figure(figsize=(8, 3.6), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().configure(height=260)
        self.canvas.get_tk_widget().pack(fill=tk.X)
        self._draw_empty()

    # ------------------------------------------------------------------ inputs
    def sensor_config(self) -> ssa.SunSensorArrayConfig:
        try:
            return ssa.SunSensorArrayConfig(centroid_sigma_px=float(self.sigma.get()),
                                            albedo_coefficient=float(self.albedo.get()),
                                            albedo_leak_fraction=float(self.leak.get()))
        except ValueError:
            self.status.set("Invalid setting (sigma > 0, albedo and leak in [0, 1]); using defaults.")
            return ssa.SunSensorArrayConfig()

    def step_s(self) -> float:
        try:
            return max(1.0, float(self.step.get()))
        except ValueError:
            return 60.0

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
        self.cancel()
        self.history = None
        self._shown = None
        self._coverage = None
        if self.sample is not None:
            self.show_sample(self.sample)

    # ------------------------------------------------------------------ selected sample
    def show_sample(self, sample) -> None:
        if sample is self._shown:
            return
        self.sample = sample
        self._shown = sample
        cfg = self.sensor_config()
        sunlit = not in_eclipse(sample.position_itrs_m, sample.sun_direction_itrs)
        view = self.view_of(sample)
        measurement = ssa.measure_once(cfg, view.sun_body_direction, sunlit,
                                       context_from_view(sample, view, cfg.albedo_coefficient))
        if self._coverage is None:
            self._coverage = ssa.coverage_grid(cfg)
        self._draw(cfg, measurement)

    # ------------------------------------------------------------------ history
    def compute_history(self) -> None:
        if not self.samples or (self._thread is not None and self._thread.is_alive()):
            return
        cfg, step = self.sensor_config(), self.step_s()
        samples = list(self.samples)
        self._cancel.clear()
        self.compute_button.state(["disabled"])
        self.status.set("Computing Sun-sensor history...")

        def work():
            try:
                result = ssa.run_history(samples, self.view_of, cfg, step,
                                         progress=lambda i, n: self._queue.put(("progress", (i, n))),
                                         cancelled=self._cancel.is_set)
                self._queue.put(("done", result))
            except Exception as error:  # surfaced in the status line
                self._queue.put(("error", str(error)))

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self.after(100, self._poll)

    def cancel(self) -> None:
        self._cancel.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.compute_button.state(["!disabled"])

    def _poll(self) -> None:
        finished = False
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    self.status.set(f"Computing Sun-sensor history... {payload[0]}/{payload[1]}")
                elif kind == "done":
                    self.history = payload
                    finished = True
                elif kind == "error":
                    self.status.set(f"Sun-sensor history failed: {payload}")
                    finished = True
        except queue.Empty:
            pass
        if finished:
            self.compute_button.state(["!disabled"])
            self._shown = None
            if self.history is not None:
                s = self.history.summary()
                if "rms_deg" in s:
                    self.status.set(f"History: sunlit availability {s['availability_sunlit']:.1%}, "
                                    f"RMS {s['rms_deg']:.4f} deg, p95 {s['p95_deg']:.4f} deg, "
                                    f">=2 heads {s['multi_head_fraction']:.1%}")
                else:
                    self.status.set("History: no sunlit samples.")
            if self.sample is not None:
                self.show_sample(self.sample)
        elif self._thread is not None and self._thread.is_alive():
            self.after(150, self._poll)

    # ------------------------------------------------------------------ drawing
    def _draw_empty(self) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot()
        axis.text(0.5, 0.5, EMPTY_TEXT, ha="center", va="center", transform=axis.transAxes, color="#64748b")
        axis.set_axis_off()
        self.canvas.draw_idle()

    def _draw(self, cfg, measurement) -> None:
        figure = self.figure
        figure.clear()
        grid = figure.add_gridspec(1, 3, width_ratios=(1.3, 1.1, 1.2))

        az, el, count = self._coverage
        axis = figure.add_subplot(grid[0, 0])
        mesh = axis.pcolormesh(az, el, count, cmap="Greens", vmin=0, vmax=3, shading="auto")
        figure.colorbar(mesh, ax=axis, ticks=[0, 1, 2, 3], label="heads seeing direction", shrink=0.8)
        sun = measurement.sun_true_body
        sun_az = math.degrees(math.atan2(sun[1], sun[0]))
        sun_el = math.degrees(math.asin(max(-1.0, min(1.0, sun[2]))))
        axis.plot([sun_az], [sun_el], marker="*", color="#f59e0b", markeredgecolor="black", markersize=14)
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)
        axis.set_xlabel("Body azimuth (deg)")
        axis.set_ylabel("Body elevation (deg)")
        axis.set_title("Coverage (star = Sun)", fontsize=9)

        table = figure.add_subplot(grid[0, 1])
        table.set_axis_off()
        table.set_title("Heads", fontsize=9)
        for row, (name, reading) in enumerate(zip(ssa.HEAD_NAMES, measurement.heads)):
            y = 1 - row * 0.1
            centroid = "" if reading.centroid_px is None else f" ({reading.centroid_px[0]:.0f}, {reading.centroid_px[1]:.0f}) px"
            table.text(0.0, y, f"{name}", fontsize=9, family="monospace", transform=table.transAxes, va="top")
            alb = "" if reading.albedo_ratio <= 0 else f" alb {reading.albedo_ratio:.2f}"
            table.text(0.15, y, f"{reading.status}{centroid}{alb}", fontsize=9, family="monospace",
                       color=STATUS_COLOR.get(reading.status, "#b45309"), transform=table.transAxes, va="top")
        error = measurement.error_deg
        fused = measurement.fused
        summary = (f"Valid heads: {measurement.valid_heads}/6\n"
                   + ("Fused: " + ", ".join(f"{v:+.4f}" for v in fused.sun_body) + "\n" if fused.valid else "Fused: none\n")
                   + (f"Error: {error:.4f} deg" if error is not None else
                      ("Error: n/a (eclipse)" if not measurement.sunlit else "Error: n/a")))
        table.text(0.0, 0.32, summary, fontsize=9, family="monospace", transform=table.transAxes, va="top")

        history = figure.add_subplot(grid[0, 2])
        history.set_title("Fused error vs time", fontsize=9)
        h = self.history
        if h is None:
            history.text(0.5, 0.5, "Click 'Compute history'", ha="center", va="center",
                         transform=history.transAxes, color="#64748b")
            history.set_axis_off()
        else:
            history.plot(h.t_min, h.error_deg, color="#0f766e", linewidth=1.0)
            history.set_yscale("log")
            history.set_xlabel("Elapsed time (min)")
            history.set_ylabel("Error (deg)")
            history.grid(True, alpha=0.3)
            for start, stop in _runs(~h.sunlit, h.t_min):
                history.axvspan(start, stop, color="#64748b", alpha=0.25)
            if self.samples:
                now = (self.sample.time_utc - self.samples[0].time_utc).total_seconds() / 60.0
                history.axvline(now, color="#dc2626", linewidth=1.0)
            twin = history.twinx()
            twin.step(h.t_min, h.valid_heads, where="post", color="#2563eb", linewidth=0.8, alpha=0.7)
            twin.set_ylim(0, 6)
            twin.set_ylabel("Valid heads", color="#2563eb")
        self.canvas.draw_idle()


def _runs(mask: np.ndarray, t: np.ndarray):
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = t[i]
        if not flag and start is not None:
            yield start, t[i]
            start = None
    if start is not None:
        yield start, t[-1]

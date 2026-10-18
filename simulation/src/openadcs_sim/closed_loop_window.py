"""Closed-loop ADCS window: runs the assembled spacecraft's sensors, estimator, controller and actuators."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Callable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .adcs_run import build_environment, run_closed_loop, summarise
from .assembly import SpacecraftAssembly

ENVIRONMENTS = {"Analytic orbit (fast, no files needed)": "analytic",
                "SGP4 TLE + WMM + Huira Sun (from the main window)": "sgp4"}


class ClosedLoopWindow(tk.Toplevel):
    def __init__(self, parent, get_assembly: Callable[[], SpacecraftAssembly],
                 get_ephemeris: Callable[[], tuple] | None = None):
        super().__init__(parent)
        self.title("Closed-loop ADCS simulation")
        self.get_assembly, self.get_ephemeris = get_assembly, get_ephemeris
        self.events: queue.Queue = queue.Queue()
        self.thread: threading.Thread | None = None
        self.result = None
        self._closed = False

        side = ttk.Frame(self, padding=8)
        side.pack(side=tk.LEFT, fill=tk.Y)
        self.vars: dict[str, tk.StringVar] = {}

        def row(label, key, default, unit=""):
            frame = ttk.Frame(side)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=24).pack(side=tk.LEFT)
            self.vars[key] = tk.StringVar(value=str(default))
            ttk.Entry(frame, textvariable=self.vars[key], width=10).pack(side=tk.LEFT)
            ttk.Label(frame, text=unit).pack(side=tk.LEFT, padx=4)

        ttk.Label(side, text="Environment").pack(anchor="w")
        self.environment = tk.StringVar(value=next(iter(ENVIRONMENTS)))
        ttk.Combobox(side, textvariable=self.environment, values=list(ENVIRONMENTS), state="readonly", width=44).pack(fill=tk.X)
        row("Duration", "duration", 1800, "s")
        row("Time step", "dt", 1.0, "s")
        row("Initial rate X Y Z", "rate", "0.02 -0.02 0.01", "rad/s")
        row("Initial offset X Y Z", "offset", "20 -15 30", "deg")
        row("GNSS outages", "outages", "", "s a-b, c-d")
        row("Noise seed", "seed", 0)
        self.spacecraft_label = tk.StringVar()
        ttk.Label(side, textvariable=self.spacecraft_label, wraplength=330, foreground="#555555").pack(anchor="w", pady=6)
        self.run_button = ttk.Button(side, text="Run closed loop", command=self.start)
        self.run_button.pack(fill=tk.X)
        self.progress = ttk.Progressbar(side, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=6)
        self.message = tk.StringVar()
        ttk.Label(side, textvariable=self.message, wraplength=330, foreground="#b91c1c").pack(anchor="w")
        self.summary = tk.StringVar()
        ttk.Label(side, textvariable=self.summary, font=("Consolas", 9), justify=tk.LEFT).pack(anchor="w", pady=6)

        self.figure = Figure(figsize=(7.5, 6.5), tight_layout=True)
        self.axes = self.figure.subplots(4, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.refresh_label()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._poll)

    def _close(self) -> None:
        self._closed = True
        self.destroy()

    def refresh_label(self) -> None:
        assembly = self.get_assembly()
        custom = sum(name != "(built-in default)" for name in assembly.attached.values())
        self.spacecraft_label.set(f"Spacecraft: {assembly.mass_properties.name} ({custom} saved subsystem(s) attached; "
                                  "change them in Spacecraft > Configure spacecraft...)")

    # ------------------------------------------------------------------ inputs
    def _numbers(self, key: str, count: int) -> tuple:
        try:
            values = tuple(float(x.replace(",", ".")) for x in self.vars[key].get().split())
        except ValueError as error:
            raise ValueError(f"Enter {count} numbers for '{key}'.") from error
        if len(values) != count:
            raise ValueError(f"Enter {count} numbers for '{key}'.")
        return values

    def _outages(self) -> tuple:
        text = self.vars["outages"].get().strip()
        windows = []
        for part in filter(None, (p.strip() for p in text.split(","))):
            try:
                start, end = (float(x) for x in part.split("-"))
            except ValueError as error:
                raise ValueError("GNSS outages must look like '600-900, 1500-1700'.") from error
            if not 0 <= start < end:
                raise ValueError("Each GNSS outage needs 0 <= start < end.")
            windows.append((start, end))
        return tuple(windows)

    def read_parameters(self) -> dict:
        try:
            parameters = dict(duration_s=float(self.vars["duration"].get()), dt_s=float(self.vars["dt"].get()),
                              initial_rate_rad_s=self._numbers("rate", 3), initial_attitude_offset_deg=self._numbers("offset", 3),
                              seed=int(self.vars["seed"].get()), outage_windows_s=self._outages())
        except ValueError as error:
            raise ValueError(str(error) if "Enter" in str(error) or "GNSS" in str(error) else
                             "Duration, step and seed must be numbers.") from error
        return parameters

    # ------------------------------------------------------------------ run
    def start(self) -> bool:
        if self.thread is not None and self.thread.is_alive():
            return False
        try:
            parameters = self.read_parameters()
            assembly = self.get_assembly()
            kind = ENVIRONMENTS[self.environment.get()]
            ephemeris = self.get_ephemeris() if kind == "sgp4" and self.get_ephemeris else (None, None, None)
            if kind == "sgp4" and ephemeris[0] is None:
                raise ValueError("The SGP4 environment is not available here.")
            assembly.closed_loop_kwargs()
        except Exception as error:
            self.message.set(str(error))
            return False
        self.message.set("")
        self.summary.set("")
        self.progress["value"] = 0
        self.run_button.configure(state=tk.DISABLED)
        self.refresh_label()

        def work():
            try:
                env = build_environment(kind, *ephemeris)
                result = run_closed_loop(assembly, env, progress=lambda f: self.events.put(("progress", f)), **parameters)
                self.events.put(("done", result))
            except Exception as error:
                self.events.put(("error", str(error)))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        return True

    def _poll(self) -> None:
        if self._closed:
            return
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self.progress["value"] = payload
                elif kind == "done":
                    self.show_result(payload)
                else:
                    self.message.set(payload)
                    self.run_button.configure(state=tk.NORMAL)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def show_result(self, result) -> None:
        self.result = result
        self.progress["value"] = 1.0
        self.run_button.configure(state=tk.NORMAL)
        stats = summarise(result, min(600.0, float(result.t_s[-1]) / 2))
        self.summary.set(
            f"Pointing error  rms {stats['pointing_error_deg_rms']:.2f}  max {stats['pointing_error_deg_max']:.2f} deg\n"
            f"Estimate error  rms {stats['estimation_error_deg_rms']:.2f}  max {stats['estimation_error_deg_max']:.2f} deg\n"
            f"Peak wheel momentum {stats['peak_wheel_momentum_nms'] * 1e3:.3f} mN m s\n"
            f"First NOMINAL at {stats['first_nominal_s']} s (statistics skip the first {stats['settle_s']:.0f} s)")
        t = result.t_s / 60.0
        ax = self.axes
        for a in ax:
            a.clear()
            a.grid(True, alpha=0.3)
        ax[0].plot(t, result.pointing_error_deg, label="true pointing error")
        ax[0].plot(t, result.estimation_error_deg, label="estimation error")
        ax[0].set_ylabel("deg")
        ax[0].legend(loc="upper right")
        ax[1].plot(t, result.rate_rad_s)
        ax[1].set_ylabel("body rate [rad/s]")
        ax[1].set_yscale("log")
        ax[2].plot(t, result.wheel_momentum_nms * 1e3)
        ax[2].set_ylabel("wheel h [mN m s]")
        ax[3].plot(t, abs(result.dipole_am2).max(axis=1))
        ax[3].set_ylabel("max |dipole| [A m^2]")
        ax[3].set_xlabel("time [min]")
        for index, sunlit in enumerate(result.sunlit):
            if not sunlit and (index == 0 or result.sunlit[index - 1]):
                end = index
                while end < len(result.sunlit) and not result.sunlit[end]:
                    end += 1
                for a in ax:
                    a.axvspan(t[index], t[min(end, len(t) - 1)], color="0.85", zorder=0)
        self.canvas.draw_idle()

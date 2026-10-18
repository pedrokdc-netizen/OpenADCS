from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import traceback

from .environment import EnvironmentSample
from .gui_runner import FinishedEvent, RunWorker, SampleEvent
from .orbit import SGP4Propagator
from .playback import PlaybackClock
from .scenario import KERNEL_NAMES, Readout, Scenario, export_results, parse_tle, sampling_count, training_tle, baseline_tle
from .assembly import SpacecraftAssembly
from .spacecraft import pilot1_baseline

GUIDE = """QUICK START
1. Import a satellite TLE (.tle or .txt), or paste its two element lines.
   For a classroom exercise, use "Load synthetic LEO". It is NOT a real satellite.
2. Select the folder containing the three NAIF kernel files.
3. Choose a UTC start time. "Use TLE epoch" is the safest starting point.
4. Set the run duration in minutes and the sampling interval in seconds.
5. Click Run simulation. Use Stop to retain a clearly labelled partial result.
6. Replay/inspect results with the Animation toolbar; export CSV or JSON.

ANIMATION AND WINDOWS
After a run, use shared forward/reverse playback, pause, rewind/end or sample steps.
60x means 60 simulation seconds per real second. Choose or enter a positive speed.
Scrubbing the timeline pauses playback; all views share one UTC cursor.
States use the recorded sample at/before that UTC, with no physics interpolation.
High playback speeds may skip displayed samples; exports still retain every sample.
Playback stops at the endpoints; forward at end restarts, reverse at start reverses
from the end. Partial results remain labelled partial during replay.
The Window menu hides/reopens individual graphics or all graphics without losing
results or camera/zoom. Samples, Guide and Diagnostics are always available.
System architecture is available before a run: blocks show implementation status
and arrows show exchanged variables/units. Fit gives an overview; use 100%/zoom,
scrollbars and block/link selection to inspect contracts and source/boilerplates.
It is a static interface map, NOT live MCU/bus activity or an enabled ADCS loop.

INPUTS AND SAMPLING
A TLE (Two-Line Element set) describes a satellite's mean orbit at its epoch.
Keep both fixed-width lines and their checksum digits unchanged. The satellite
name above the lines is optional. A smaller time step samples the model more
frequently; it does not improve the underlying SGP4/TLE orbit accuracy.
Samples include the start and every grid point not exceeding the duration.
If duration is not divisible by step, the last sample precedes the requested end.
All samples remain in memory; very long, finely sampled runs can use substantial RAM.

READING THE RESULTS
ITRS is an Earth-fixed coordinate frame: Z points north, X toward the equator
at the Greenwich meridian, and Y completes a right-handed system.
Geodetic latitude, longitude and altitude use the WGS84 ellipsoid.
Earth-fixed speed is NOT inertial orbital speed.
Magnetic components are in microtesla (1 microtesla = 1000 nanotesla).
The Sun vector points from spacecraft toward the Sun; its length is one.
One astronomical unit (AU) is 149,597,870.7 km.
Ground-track jumps at +/-180 deg longitude are split, not drawn across the map.
The 3D trajectory view is Earth-fixed, not an inertial Kepler ellipse.
The Spacecraft view follows the selected sample, with a spacecraft-fixed origin.
Body +Z points to Earth's center, +X follows horizontal inertial velocity, and
+Y completes the right-handed frame. This ideal reference has no ADCS dynamics.
The attitude arrow is the +Z boresight; a full rotation matrix carries orientation.
Velocity is inertial translational velocity expressed in body axes, not spin rate.
Local 3U CubeSat dimensions are in meters; vector lengths are schematic.
The 3D trajectory shows a textured Earth with Sun, B, nadir and velocity vectors.
Plots show up to 3000 points for responsiveness; exports retain every sample.

MODEL LIMITS
Orbit: Vallado SGP4 / WGS72, limited here to +/-7 days from the TLE epoch.
Magnetics: XYZgeomag / WMM2025, years 2025-2029 and altitude -1 to 850 km.
Sun: Huira scene with geometric SPICE ephemerides (ITRF93 realization of ITRS).
The GUI run has no eclipses, irradiance, radiation forces or attitude dynamics.
Sun/gyro processing and standalone Orion B-16 PVT modeling exist separately;
they are not automatically executed by the GUI's environment run.
Architecture PLACEHOLDERS are disabled boilerplates, not working digital models.
A Sun vector during Earth shadow does NOT mean the satellite is illuminated.
IERS and SPICE coverage warnings must be considered in accuracy studies.
Read diagnostics after each run; warnings are preserved, not suppressed.

KERNEL SETUP (normally performed once by the lab/instructor)
Download using the repository command:
  python simulation\\tools\\fetch_kernels.py
Then select simulation\\data\\kernels, containing:
  naif0012.tls                   Leap-second time conversions
  de440s.bsp                     Planetary / Sun ephemerides
  earth_latest_high_prec.bpc     Earth orientation (rolling time coverage)
The rolling Earth kernel is not valid for all dates; coverage errors are reported.
Archive exact kernel and Earth-orientation files with scientific experiments.

EXPORTS
CSV uses student-friendly km, km/s, microtesla, degrees and AU, with units in
column names. JSON preserves the simulator's SI-valued environment records
and includes the scenario and completion status. Stopped or failed runs remain
labelled as partial; do not treat them as completed datasets.
"""


class SimulationApp:
    def __init__(self, root: tk.Tk):
        from .plots import PlotPanel

        self.root = root
        root.title("OpenADCS | Ground Simulation Lab")
        root.geometry("1320x880")
        root.minsize(1080, 760)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.report_callback_exception = self.callback_error
        self.worker: RunWorker | None = None
        self.result_scenario: Scenario | None = None
        self.result_status = "not run"
        self.samples: list[EnvironmentSample] = []
        self.readouts: list[Readout] = []
        self.closing = False
        self.stopping = False
        self.last_plot = 0.0
        self.last_body_plot = 0.0
        self.playback = PlaybackClock()
        self._animation_timer: str | None = None
        self.speed_value = tk.StringVar(value="60x")
        self.timeline_value = tk.DoubleVar(value=0)
        self.animation_status = tk.StringVar(value="Run a simulation to enable playback.")
        self.animation_time = tk.StringVar(value="Animation UTC: -- | Selected sample UTC: --")
        self.start_value = tk.StringVar(value=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.duration_value = tk.StringVar(value="95")
        self.step_value = tk.StringVar(value="10")
        self.kernel_value = tk.StringVar(value=str(Path("simulation") / "data" / "kernels"))
        self.status_value = tk.StringVar(value="Ready. Import a TLE or load the synthetic classroom exercise.")
        self.tle_info = tk.StringVar(value="TLE epoch: not loaded")
        self.kernel_info = tk.StringVar()
        self.schedule_info = tk.StringVar()
        self.sample_info = tk.StringVar(value="No samples yet")
        self.summary = [tk.StringVar(value="--") for _ in range(4)]
        self.details = tk.StringVar(value="Run a simulation to inspect a sample.")
        self.input_widgets: list[tk.Widget] = []
        self.spacecraft = pilot1_baseline()
        self.spacecraft_window = None
        self.assembly = SpacecraftAssembly(self.spacecraft)
        self.assembly_window = None
        self.subsystem_window = None
        self.closed_loop_window = None
        self.spacecraft_info = tk.StringVar(value=f"{self.spacecraft.name} | {self.spacecraft.mass_kg * 1000:g} g")

        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        style.configure("Value.TLabel", font=("Segoe UI", 16, "bold"), foreground="#1d4ed8")
        style.configure("Run.TButton", font=("Segoe UI", 11, "bold"), padding=(12, 8))
        shell = ttk.Frame(root, padding=16)
        shell.pack(fill=tk.BOTH, expand=True)
        ttk.Label(shell, text="Ground Simulation Lab", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            shell, text="Orbit, geomagnetic field and Sun geometry | UTC time | ITRS data + spacecraft-fixed view",
            foreground="#475569",
        ).pack(anchor=tk.W, pady=(2, 12))
        body = ttk.Panedwindow(shell, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        control_host = ttk.Frame(body, width=390)
        control_host.pack_propagate(False)
        control_canvas = tk.Canvas(control_host, width=370, highlightthickness=0)
        control_scroll = ttk.Scrollbar(control_host, orient=tk.VERTICAL, command=control_canvas.yview)
        control_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_canvas.configure(yscrollcommand=control_scroll.set)
        controls = ttk.Frame(control_canvas, padding=(0, 0, 12, 0))
        control_window = control_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.bind("<Configure>", lambda event: control_canvas.itemconfigure(control_window, width=event.width))
        self.control_canvas = control_canvas
        results = ttk.Frame(body)
        body.add(control_host, weight=0)
        body.add(results, weight=1)
        self._build_controls(controls)
        cards = ttk.Frame(results)
        cards.pack(fill=tk.X, pady=(0, 10))
        for index, title in enumerate(("Altitude (km)", "Earth-fixed speed (km/s)", "|B| (microtesla)", "Sun distance (AU)")):
            card = ttk.LabelFrame(cards, text=title, padding=8)
            card.grid(row=0, column=index, sticky="nsew", padx=(0, 6))
            cards.columnconfigure(index, weight=1)
            ttk.Label(card, textvariable=self.summary[index], style="Value.TLabel").pack(anchor=tk.W)
        notebook = ttk.Notebook(results)
        notebook.pack(fill=tk.BOTH, expand=True)
        self.plotter = PlotPanel(notebook)
        notebook.select(self.plotter.spacecraft_tab)
        inspector = ttk.Frame(notebook, padding=16)
        notebook.add(inspector, text="Samples")
        ttk.Label(inspector, textvariable=self.sample_info, font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(inspector, text="Use the shared Animation timeline below to inspect stored samples.").pack(anchor=tk.W, pady=12)
        ttk.Label(inspector, textvariable=self.details, font=("Consolas", 11), justify=tk.LEFT).pack(anchor=tk.W)
        guide = ttk.Frame(notebook)
        notebook.add(guide, text="Guide")
        self._text_panel(guide, GUIDE)
        diagnostics = ttk.Frame(notebook)
        notebook.add(diagnostics, text="Diagnostics")
        self.log = self._text_panel(diagnostics, "Run warnings and errors will appear here.\n")
        self.notebook = notebook
        self.diagnostics_tab = diagnostics
        self._build_window_menu()
        self._build_animation_toolbar(shell)
        bottom = ttk.Frame(shell)
        # Bottom-anchored widgets are packed before the expanding body so they are never clipped.
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=(12, 0), before=body)
        animation = next(w for w in shell.winfo_children() if str(w.cget("text") if "text" in w.keys() else "") .startswith("Animation"))
        animation.pack_configure(side=tk.BOTTOM, before=body)
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_value, wraplength=1200).pack(anchor=tk.W, pady=(5, 0))
        self.kernel_value.trace_add("write", lambda *_: self.check_kernels())
        self.duration_value.trace_add("write", lambda *_: self.update_schedule())
        self.step_value.trace_add("write", lambda *_: self.update_schedule())
        self.check_kernels()
        self.update_schedule()
        root.after(100, self.poll)

    def _build_window_menu(self) -> None:
        from .plots import VIEW_TITLES

        menu = tk.Menu(self.root)
        self.window_menu = tk.Menu(menu, tearoff=False)
        self.view_enabled: dict[str, tk.BooleanVar] = {}
        for key, title in VIEW_TITLES.items():
            variable = tk.BooleanVar(value=True)
            self.view_enabled[key] = variable
            self.window_menu.add_checkbutton(
                label=title, variable=variable, command=lambda name=key: self.toggle_view(name),
            )
        self.window_menu.add_separator()
        self.window_menu.add_command(label="Open all visualizations", command=lambda: self.set_all_views(True))
        self.window_menu.add_command(label="Close all visualizations", command=lambda: self.set_all_views(False))
        self.window_menu.add_command(label="Close active visualization", command=self.close_active_view)
        self.close_view_menu_index = self.window_menu.index(tk.END)
        self.window_menu.configure(postcommand=self.update_window_menu)
        menu.add_cascade(label="Window", menu=self.window_menu)
        spacecraft_menu = tk.Menu(menu, tearoff=False)
        spacecraft_menu.add_command(label="Configure spacecraft...", command=self.open_assembly_window)
        spacecraft_menu.add_command(label="Configure subsystems...", command=self.open_subsystem_window)
        spacecraft_menu.add_command(label="Run closed-loop ADCS...", command=self.open_closed_loop_window)
        spacecraft_menu.add_separator()
        spacecraft_menu.add_command(label="Mass properties only...", command=self.open_spacecraft_window)
        menu.add_cascade(label="Spacecraft", menu=spacecraft_menu)
        self.root.configure(menu=menu)

    def open_spacecraft_window(self) -> None:
        from .spacecraft_window import SpacecraftWindow

        window = self.spacecraft_window
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            window.show(self.spacecraft)
            return
        self.spacecraft_window = SpacecraftWindow(self.root, self.spacecraft, self.set_spacecraft)

    def open_assembly_window(self) -> None:
        from .assembly_window import AssemblyWindow

        window = self.assembly_window
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            return
        self.assembly_window = AssemblyWindow(self.root, self.assembly, self.set_assembly)

    def open_closed_loop_window(self) -> None:
        from .closed_loop_window import ClosedLoopWindow

        window = self.closed_loop_window
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            window.refresh_label()
            return
        self.closed_loop_window = ClosedLoopWindow(self.root, lambda: self.assembly, self.closed_loop_ephemeris)

    def closed_loop_ephemeris(self) -> tuple:
        """TLE, kernel directory and start time currently entered in the main window."""
        return (parse_tle(self.tle_text.get("1.0", tk.END)), Path(self.kernel_value.get()).resolve(),
                datetime.fromisoformat(self.start_value.get().strip().replace("Z", "+00:00")))

    def open_subsystem_window(self) -> None:
        from .subsystem_window import SubsystemListWindow

        window = self.subsystem_window
        if window is not None and window.winfo_exists():
            window.deiconify()
            window.lift()
            return
        self.subsystem_window = SubsystemListWindow(self.root)

    def set_assembly(self, assembly) -> None:
        self.assembly = assembly
        self.set_spacecraft(assembly.mass_properties)

    def set_spacecraft(self, config) -> None:
        self.spacecraft = config
        self.assembly = replace(self.assembly, mass_properties=config)
        self.spacecraft_info.set(f"{config.name} | {config.mass_kg * 1000:g} g")
        self.apply_attitude_model()

    def apply_attitude_model(self) -> None:
        """Ideal nadir reference, or rigid-body dynamics (gravity gradient) with the current spacecraft inertia."""
        if self.dynamics_enabled.get():
            from .dynamics import DynamicsAttitudeModel
            self.plotter.attitude_model = DynamicsAttitudeModel(self.spacecraft)
        else:
            self.plotter.attitude_model = None
        self.plotter.refresh_active()

    def toggle_view(self, key: str) -> None:
        self.plotter.set_visible(key, self.view_enabled[key].get())

    def set_all_views(self, visible: bool) -> None:
        for key, variable in self.view_enabled.items():
            variable.set(visible)
            self.plotter.set_visible(key, visible)
        if visible:
            self.notebook.select(self.plotter.spacecraft_tab)

    def close_active_view(self) -> None:
        key = self.plotter.active_view()
        if key is not None:
            self.view_enabled[key].set(False)
            self.plotter.set_visible(key, False)

    def update_window_menu(self) -> None:
        state = tk.NORMAL if self.plotter.active_view() is not None else tk.DISABLED
        self.window_menu.entryconfigure(self.close_view_menu_index, state=state)

    def _build_animation_toolbar(self, parent: ttk.Frame) -> None:
        animation = ttk.LabelFrame(parent, text="Animation (recorded results)", padding=8)
        animation.pack(fill=tk.X, pady=(10, 0))
        actions = ttk.Frame(animation)
        actions.pack(fill=tk.X)
        self.animation_buttons: dict[str, ttk.Button] = {}
        for key, title, command in (
            ("rewind", "Rewind", lambda: self.jump_playback(False)),
            ("back", "Step -", lambda: self.step_playback(-1)),
            ("reverse", "Play reverse", lambda: self.play(-1)),
            ("pause", "Pause", self.pause_playback),
            ("forward", "Play forward", lambda: self.play(1)),
            ("next", "Step +", lambda: self.step_playback(1)),
            ("end", "End", lambda: self.jump_playback(True)),
        ):
            button = ttk.Button(actions, text=title, command=command, state=tk.DISABLED)
            button.pack(side=tk.LEFT, padx=(0, 4))
            self.animation_buttons[key] = button
        ttk.Label(actions, text="Speed:").pack(side=tk.LEFT, padx=(10, 3))
        self.speed_entry = ttk.Combobox(
            actions, textvariable=self.speed_value, width=7,
            values=("1x", "10x", "30x", "60x", "120x", "300x", "600x", "1200x", "3600x"),
            state=tk.DISABLED,
        )
        self.speed_entry.pack(side=tk.LEFT)
        self.speed_entry.bind("<<ComboboxSelected>>", lambda _: self.apply_speed())
        self.speed_entry.bind("<Return>", lambda _: self.apply_speed())
        self.speed_entry.bind("<FocusOut>", lambda _: self.apply_speed())
        for key, title, factor in (("slower", "Slower", .5), ("faster", "Faster", 2)):
            button = ttk.Button(actions, text=title, command=lambda multiplier=factor: self.change_speed(multiplier),
                                state=tk.DISABLED)
            button.pack(side=tk.LEFT, padx=(4, 0))
            self.animation_buttons[key] = button
        self.slider = ttk.Scale(animation, from_=0, to=1, variable=self.timeline_value,
                                command=self.seek_playback, state=tk.DISABLED)
        self.slider.pack(fill=tk.X, pady=(8, 3))
        row = ttk.Frame(animation)
        row.pack(fill=tk.X)
        ttk.Label(row, textvariable=self.animation_time).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self.animation_status).pack(side=tk.RIGHT)

    def update_animation_controls(self) -> None:
        ready = self.worker is None and bool(self.playback.timestamps)
        playable = ready and self.playback.duration > 0
        for key, button in self.animation_buttons.items():
            enabled = playable if key in ("forward", "reverse") else ready
            if key == "pause":
                enabled = ready and self.playback.playing
            button.configure(state=tk.NORMAL if enabled else tk.DISABLED)
        self.slider.configure(state=tk.NORMAL if ready else tk.DISABLED)
        self.speed_entry.configure(state=tk.NORMAL if ready else tk.DISABLED)
        if not ready:
            self.animation_status.set("Run a simulation to enable playback.")
        else:
            mode = ("Playing forward" if self.playback.direction == 1 else "Playing reverse") if self.playback.playing else "Paused"
            partial = " | partial data" if self.result_status != "completed" else ""
            self.animation_status.set(f"{mode} | {self.playback.rate:g}x{partial}")

    def _cancel_animation_timer(self) -> None:
        if self._animation_timer is not None:
            self.root.after_cancel(self._animation_timer)
            self._animation_timer = None

    def display_clock(self) -> None:
        if not self.playback.timestamps:
            return
        index = self.playback.index
        self.timeline_value.set(self.playback.elapsed)
        self.animation_time.set(
            f"Animation UTC: {self.playback.time_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Sample UTC: {self.playback.timestamps[index].strftime('%H:%M:%S')} ({index + 1:,}/{len(self.samples):,})"
        )
        self.inspect_sample(str(index), sync_clock=False)
        self.update_animation_controls()

    def play(self, direction: int = 1) -> None:
        if self.worker is not None or self.playback.duration <= 0:
            return
        if not self.apply_speed():
            return
        self._cancel_animation_timer()
        self.playback.play(direction, time.monotonic())
        self.display_clock()
        self._animation_timer = self.root.after(50, self.animation_tick)

    def animation_tick(self) -> None:
        self._animation_timer = None
        if self.closing or self.worker is not None or not self.playback.playing:
            return
        self.playback.advance(time.monotonic())
        self.display_clock()
        if self.playback.playing:
            self._animation_timer = self.root.after(50, self.animation_tick)

    def pause_playback(self) -> None:
        self._cancel_animation_timer()
        if self.playback.timestamps:
            self.playback.pause(time.monotonic())
            self.display_clock()

    def apply_speed(self) -> bool:
        try:
            rate = float(self.speed_value.get().strip().lower().removesuffix("x"))
            self.playback.set_rate(rate, time.monotonic())
        except ValueError as error:
            self.speed_value.set(f"{self.playback.rate:g}x")
            messagebox.showerror("Invalid playback speed", str(error), parent=self.root)
            return False
        self.speed_value.set(f"{rate:g}x")
        if self.playback.timestamps:
            self.display_clock()
        return True

    def change_speed(self, factor: float) -> None:
        if self.apply_speed():
            self.speed_value.set(f"{self.playback.rate * factor:g}x")
            self.apply_speed()

    def seek_playback(self, value: str) -> None:
        if self.worker is not None or not self.playback.timestamps:
            return
        self._cancel_animation_timer()
        self.playback.pause(time.monotonic())
        self.playback.seek(float(value), time.monotonic())
        self.display_clock()

    def step_playback(self, direction: int) -> None:
        if self.worker is not None or not self.playback.timestamps:
            return
        self._cancel_animation_timer()
        self.playback.step(direction, time.monotonic())
        self.display_clock()

    def jump_playback(self, end: bool) -> None:
        if self.worker is not None or not self.playback.timestamps:
            return
        self._cancel_animation_timer()
        self.playback.pause(time.monotonic())
        self.playback.seek(self.playback.duration if end else 0, time.monotonic())
        self.display_clock()

    def _button(self, parent: tk.Widget, text: str, command) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        self.input_widgets.append(button)
        return button

    def _build_controls(self, parent: ttk.Frame) -> None:
        craft_frame = ttk.LabelFrame(parent, text="Spacecraft", padding=10)
        craft_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(craft_frame, textvariable=self.spacecraft_info, wraplength=240).pack(side=tk.LEFT)
        ttk.Button(craft_frame, text="Define...", command=self.open_spacecraft_window).pack(side=tk.RIGHT)
        self.dynamics_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(craft_frame, text="Torque-driven attitude (dynamics)", variable=self.dynamics_enabled,
                        command=self.apply_attitude_model).pack(side=tk.BOTTOM, anchor=tk.W)
        tle_frame = ttk.LabelFrame(parent, text="1  Satellite orbit", padding=10)
        tle_frame.pack(fill=tk.X)
        row = ttk.Frame(tle_frame)
        row.pack(fill=tk.X)
        self._button(row, "Import TLE file...", self.import_tle).pack(side=tk.LEFT)
        self._button(row, "Load synthetic LEO", self.load_training).pack(side=tk.RIGHT)
        self._button(row, "Load baseline 500 km / 98.5 deg", self.load_baseline).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Label(tle_frame, text="Paste TLE lines below; the satellite name is optional.", wraplength=340).pack(anchor=tk.W, pady=6)
        self.tle_text = tk.Text(tle_frame, height=5, width=42, wrap=tk.NONE, font=("Consolas", 9), undo=True)
        self.tle_text.pack(fill=tk.X)
        scrollbar = ttk.Scrollbar(tle_frame, orient=tk.HORIZONTAL, command=self.tle_text.xview)
        scrollbar.pack(fill=tk.X)
        self.tle_text.configure(xscrollcommand=scrollbar.set)
        self.input_widgets.append(self.tle_text)
        ttk.Label(tle_frame, textvariable=self.tle_info, wraplength=340).pack(anchor=tk.W, pady=6)
        self._button(tle_frame, "Validate TLE / show epoch", self.validate_tle).pack(fill=tk.X)

        timing = ttk.LabelFrame(parent, text="2  Simulation timeline", padding=10)
        timing.pack(fill=tk.X, pady=10)
        ttk.Label(timing, text="Start time (UTC, ISO 8601)").pack(anchor=tk.W)
        entry = ttk.Entry(timing, textvariable=self.start_value)
        entry.pack(fill=tk.X, pady=(3, 6))
        self.input_widgets.append(entry)
        self._button(timing, "Use TLE epoch", self.use_epoch).pack(fill=tk.X, pady=(0, 8))
        grid = ttk.Frame(timing)
        grid.pack(fill=tk.X)
        for index, (label, variable) in enumerate((("Duration (min)", self.duration_value), ("Time step (s)", self.step_value))):
            ttk.Label(grid, text=label).grid(row=0, column=index, sticky=tk.W)
            entry = ttk.Entry(grid, textvariable=variable, width=15)
            entry.grid(row=1, column=index, sticky="ew", padx=(0, 8), pady=4)
            grid.columnconfigure(index, weight=1)
            self.input_widgets.append(entry)
        ttk.Label(timing, textvariable=self.schedule_info, wraplength=340).pack(anchor=tk.W, pady=4)

        kernels = ttk.LabelFrame(parent, text="3  Ephemeris data", padding=10)
        kernels.pack(fill=tk.X)
        entry = ttk.Entry(kernels, textvariable=self.kernel_value)
        entry.pack(fill=tk.X)
        self.input_widgets.append(entry)
        self._button(kernels, "Choose kernel folder...", self.browse_kernels).pack(fill=tk.X, pady=6)
        ttk.Label(kernels, textvariable=self.kernel_info, wraplength=340).pack(anchor=tk.W)
        ttk.Label(
            parent, text="WMM2025: 2025-2029, altitude -1 to 850 km.\n"
            "Sun vector is geometric; eclipse and radiation forces are not modeled.",
            wraplength=360, foreground="#92400e",
        ).pack(anchor=tk.W, pady=12)
        self.run_button = ttk.Button(parent, text="Run simulation", command=self.run, style="Run.TButton")
        self.run_button.pack(fill=tk.X)
        self.stop_button = ttk.Button(parent, text="Stop (keep partial results)", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=6)
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        self.csv_button = ttk.Button(row, text="Export CSV...", command=lambda: self.export(".csv"), state=tk.DISABLED)
        self.csv_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.json_button = ttk.Button(row, text="Export JSON...", command=lambda: self.export(".json"), state=tk.DISABLED)
        self.json_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _text_panel(self, parent: ttk.Frame, content: str) -> tk.Text:
        text = tk.Text(parent, wrap=tk.WORD, font=("Segoe UI", 10), padx=14, pady=12)
        scrollbar = ttk.Scrollbar(parent, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert("1.0", content)
        text.configure(state=tk.DISABLED)
        return text

    def log_message(self, content: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, content + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def callback_error(self, kind, error, tb) -> None:
        self._cancel_animation_timer()
        self.playback.pause(time.monotonic())
        self.log_message("".join(traceback.format_exception(kind, error, tb)))
        if self.worker is not None:
            self.worker.cancel()
            self.root.after(100, self.poll)
        messagebox.showerror("Interface error", f"{error}\n\nDetails are in Diagnostics.", parent=self.root)

    def current_tle(self) -> SGP4Propagator:
        model = SGP4Propagator(*parse_tle(self.tle_text.get("1.0", tk.END)))
        self.tle_info.set(f"TLE epoch: {model.epoch_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        return model

    def validate_tle(self) -> None:
        try:
            self.current_tle()
        except (ValueError, OverflowError) as error:
            self.tle_info.set("TLE needs correction")
            messagebox.showerror("Invalid TLE", str(error), parent=self.root)

    def use_epoch(self) -> None:
        try:
            self.start_value.set(self.current_tle().epoch_utc.isoformat())
        except (ValueError, OverflowError) as error:
            messagebox.showerror("Cannot use TLE epoch", str(error), parent=self.root)

    def import_tle(self) -> None:
        filename = filedialog.askopenfilename(parent=self.root, title="Import satellite TLE",
                                             filetypes=[("TLE / text files", "*.tle *.txt"), ("All files", "*.*")])
        if not filename:
            return
        try:
            text = Path(filename).read_text(encoding="ascii")
            SGP4Propagator(*parse_tle(text))
        except (OSError, ValueError, OverflowError) as error:
            messagebox.showerror("Cannot import TLE", str(error), parent=self.root)
            return
        self.tle_text.delete("1.0", tk.END)
        self.tle_text.insert("1.0", text)
        self.current_tle()
        self.status_value.set("TLE imported. Choose a UTC start near its epoch.")

    def load_training(self) -> None:
        self.tle_text.delete("1.0", tk.END)
        self.tle_text.insert("1.0", training_tle())
        self.use_epoch()
        self.status_value.set("Synthetic classroom orbit loaded. NOT an observed or operational satellite.")

    def load_baseline(self) -> None:
        self.tle_text.delete("1.0", tk.END)
        self.tle_text.insert("1.0", baseline_tle())
        self.use_epoch()
        self.status_value.set("Baseline 500 km / 98.5 deg circular orbit loaded (synthetic, not exactly sun-synchronous).")

    def browse_kernels(self) -> None:
        directory = filedialog.askdirectory(parent=self.root, title="Choose NAIF kernel folder")
        if directory:
            self.kernel_value.set(directory)

    def check_kernels(self) -> None:
        directory = Path(self.kernel_value.get())
        missing = [name for name in KERNEL_NAMES if not (directory / name).is_file()]
        self.kernel_info.set("All three kernel files found." if not missing else "Missing: " + ", ".join(missing))

    def update_schedule(self) -> None:
        try:
            duration, step = float(self.duration_value.get()) * 60, float(self.step_value.get())
            count = sampling_count(duration, step)
            self.schedule_info.set(f"{count:,} samples, including start. Last sample at {(count - 1) * step / 60:g} min.")
        except (ValueError, OverflowError):
            self.schedule_info.set("Enter a non-negative duration and a positive time step.")

    def set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        for widget in self.input_widgets:
            widget.configure(state=state)
        self.run_button.configure(state=state)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        can_export = not running and bool(self.samples)
        for button in (self.csv_button, self.json_button):
            button.configure(state=tk.NORMAL if can_export else tk.DISABLED)
        self.update_animation_controls()

    def run(self) -> None:
        if self.worker is not None:
            return
        try:
            self.current_tle()
            scenario = Scenario(
                parse_tle(self.tle_text.get("1.0", tk.END)), Path(self.kernel_value.get()).resolve(),
                datetime.fromisoformat(self.start_value.get().strip().replace("Z", "+00:00")),
                float(self.duration_value.get()) * 60, float(self.step_value.get()),
            )
        except (ValueError, OverflowError, OSError) as error:
            messagebox.showerror("Check simulation inputs", str(error), parent=self.root)
            return
        if scenario.sample_count > 100000 and not messagebox.askyesno(
            "Large simulation", f"This run retains {scenario.sample_count:,} samples in memory.\n"
            "It may use substantial RAM and take a long time. Continue?", parent=self.root,
        ):
            return
        self.pause_playback()
        self.playback.load(())
        self.timeline_value.set(0)
        self.animation_time.set("Animation UTC: -- | Selected sample UTC: --")
        self.samples.clear()
        self.readouts.clear()
        self.result_scenario = scenario
        self.result_status = "running"
        self.stopping = False
        self.plotter.clear()
        for variable in self.summary:
            variable.set("--")
        self.details.set("Waiting for the first sample...")
        self.sample_info.set("Waiting for the first sample...")
        self.progress["value"] = 0
        self.last_plot = time.monotonic()
        self.last_body_plot = time.monotonic()
        self.set_running(True)
        self.log_message(f"\nRun started: {scenario.start_utc.isoformat()} | {scenario.sample_count} samples")
        self.status_value.set("Starting physics worker. Initial ephemeris / IERS loading may take a moment...")
        self.worker = RunWorker(scenario)
        try:
            self.worker.start()
        except RuntimeError as error:
            self.worker = None
            self.result_status = "failed"
            self.set_running(False)
            self.status_value.set(f"Could not start simulation worker: {error}")
            messagebox.showerror("Cannot start worker", str(error), parent=self.root)

    def stop(self) -> None:
        if self.worker is not None:
            self.stopping = True
            self.worker.cancel()
            self.stop_button.configure(state=tk.DISABLED)
            self.status_value.set("Stopping worker; received samples will remain available as partial results.")

    def poll(self) -> None:
        worker = self.worker
        finished = None
        added = False
        if worker is not None:
            # Bound each poll so a fast producer cannot starve the Tk event loop.
            for _ in range(128):
                try:
                    event = worker.events.get_nowait()
                except queue.Empty:
                    break
                if isinstance(event, SampleEvent):
                    self.samples.append(event.sample)
                    self.readouts.append(event.readout)
                    added = True
                elif isinstance(event, FinishedEvent):
                    finished = event
                    break
            if added:
                count = len(self.samples)
                self.inspect_sample(str(count - 1))
                if self.result_scenario is not None:
                    total = self.result_scenario.sample_count
                    self.progress["value"] = count * 100 / total
                    phase = "Stopping" if self.stopping else "Running"
                    self.status_value.set(f"{phase}: {count:,} / {total:,} samples received.")
                if not self.closing and time.monotonic() - self.last_plot >= 2:
                    self.plotter.render(self.samples, self.readouts)
                    self.last_plot = time.monotonic()
            if finished is not None:
                # Keep the worker reference until its reader thread has fully returned.
                self.result_status = finished.status
                if worker.thread.is_alive():
                    self.root.after(25, lambda: self.finish_when_joined(worker, finished))
                    return
                self.finish_when_joined(worker, finished)
                return
        if self.closing and self.worker is None:
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def finish_when_joined(self, worker: RunWorker, finished: FinishedEvent) -> None:
        if worker.thread.is_alive():
            self.root.after(25, lambda: self.finish_when_joined(worker, finished))
            return
        self.worker = None
        if self.closing:
            self.root.destroy()
            return
        self.set_running(False)
        self.plotter.render(self.samples, self.readouts)
        self.playback.load([sample.time_utc for sample in self.samples])
        self.slider.configure(to=max(1e-6, self.playback.duration))
        self.display_clock()
        self.update_animation_controls()
        label = "Completed" if finished.status == "completed" else f"{finished.status.capitalize()} (partial results)"
        warning = " Review Diagnostics." if finished.diagnostics else ""
        self.status_value.set(f"{label}: {len(self.samples):,} samples.{warning}")
        self.log_message(f"{label}: {len(self.samples)} samples.\n{finished.diagnostics}")
        if finished.status == "failed":
            self.notebook.select(self.diagnostics_tab)
            messagebox.showerror(
                "Simulation failed",
                "The run did not complete. Received samples are labelled partial.\n\n"
                + finished.diagnostics[-2000:], parent=self.root,
            )
        self.root.after(100, self.poll)

    def inspect_sample(self, value: str, sync_clock: bool = True) -> None:
        if not self.samples:
            return
        index = max(0, min(len(self.samples) - 1, round(float(value))))
        sample, row = self.samples[index], self.readouts[index]
        if sync_clock and self.worker is None and self.playback.timestamps:
            self._cancel_animation_timer()
            self.playback.pause(time.monotonic())
            self.playback.seek(self.playback.offsets[index], time.monotonic())
            self.timeline_value.set(self.playback.elapsed)
            self.animation_time.set(
                f"Animation UTC: {sample.time_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Sample UTC: {sample.time_utc.strftime('%H:%M:%S')} ({index + 1:,}/{len(self.samples):,})"
            )
            self.update_animation_controls()
        if self.worker is None or time.monotonic() - self.last_body_plot >= 2:
            self.plotter.render_selected(sample)
            self.last_body_plot = time.monotonic()
        for variable, number, digits in zip(
            self.summary,
            (row.altitude_km, row.speed_fixed_km_s, row.magnetic_magnitude_ut, row.sun_distance_au),
            (2, 3, 3, 6),
        ):
            variable.set(f"{number:.{digits}f}")
        self.sample_info.set(f"Sample {index + 1:,} / {len(self.samples):,} | {sample.time_utc.isoformat()}")

        def components(vector) -> str:
            return "  ".join(f"{component: .6f}" for component in vector)

        self.details.set(
            f"Frame: ITRS (Earth-fixed)\n\n"
            f"Latitude:  {row.latitude_deg: .5f} deg\nLongitude: {row.longitude_deg: .5f} deg\n"
            f"Altitude:  {row.altitude_km: .3f} km (WGS84 geodetic)\n\n"
            f"Position [X, Y, Z] (km)\n{components(sample.position_itrs_m / 1000)}\n\n"
            f"Velocity [X, Y, Z] (Earth-fixed km/s)\n{components(sample.velocity_itrs_m_s / 1000)}\n\n"
            f"Magnetic field [X, Y, Z] (microtesla)\n{components(sample.magnetic_field_itrs_t * 1e6)}\n\n"
            f"Sun direction [X, Y, Z] (unit vector)\n{components(sample.sun_direction_itrs)}\n\n"
            "Sun vector points spacecraft -> Sun; illumination is not modeled."
        )

    def export(self, suffix: str) -> None:
        if self.result_scenario is None or not self.samples or self.worker is not None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root, title="Export simulation results", defaultextension=suffix,
            initialfile=f"openadcs_{self.result_status}{suffix}",
            filetypes=[("CSV spreadsheet" if suffix == ".csv" else "JSON dataset", f"*{suffix}")],
        )
        if not filename:
            return
        try:
            export_results(Path(filename), self.result_scenario, self.samples, self.result_status, self.spacecraft)
        except (OSError, ValueError) as error:
            messagebox.showerror("Export failed", str(error), parent=self.root)
            return
        self.status_value.set(f"Exported {len(self.samples):,} samples ({self.result_status}) to {filename}")

    def close(self) -> None:
        if self.worker is not None:
            if not messagebox.askyesno("Close simulation?", "Stop the running simulation and close the window?", parent=self.root):
                return
            self.closing = True
            self.pause_playback()
            self.stop()
        else:
            self.pause_playback()
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        SimulationApp(root)
    except ModuleNotFoundError as error:
        messagebox.showerror(
            "GUI dependency missing",
            f"{error}\n\nInstall the GUI extra: python -m pip install -e \".[gui]\"",
            parent=root,
        )
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()

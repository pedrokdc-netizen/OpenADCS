import tkinter as tk
from tkinter import ttk
from typing import Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from .attitude import AttitudeModel, BodyView, spacecraft_view
from .environment import EnvironmentSample
from .scenario import Readout, sample_readout
from .spacecraft_plot import BodyRenderer
from . import earth_view
from .view_tools import enable_wheel_zoom

COLORS = ("#2563eb", "#16a34a", "#db2777")
VIEW_TITLES = {
    "spacecraft": "Spacecraft",
    "ground": "Ground track",
    "trajectory": "3D trajectory",
    "magnetic": "Magnetics",
    "sun": "Sun",
    "horizon": "Horizon sensor",
    "architecture": "System architecture",
}


def ground_track(longitudes: np.ndarray, latitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert breaks instead of drawing a false segment across the date line."""
    breaks = np.flatnonzero(np.abs(np.diff(longitudes)) > 180) + 1
    return np.insert(longitudes, breaks, np.nan), np.insert(latitudes, breaks, np.nan)


class PlotPanel:
    def __init__(self, notebook: ttk.Notebook, attitude_model: AttitudeModel | None = None):
        self.notebook = notebook
        self.attitude_model = attitude_model
        self.selected_sample: EnvironmentSample | None = None
        self._body_sample: EnvironmentSample | None = None
        self._body_view: BodyView | None = None
        self._body_renderers: dict[str, BodyRenderer] = {}
        self._rendered_samples: dict[str, EnvironmentSample] = {}
        self.tabs: dict[str, ttk.Frame] = {}
        self.view_figures: dict[str, Figure] = {}
        self.view_canvases: dict[str, FigureCanvasTkAgg] = {}
        self.toolbars: dict[str, NavigationToolbar2Tk] = {}
        self.popout_buttons: dict[str, ttk.Button] = {}
        self.windows: dict[str, tk.Toplevel] = {}
        self.markers = {}
        self.start_time = None
        self.figures: list[Figure] = []
        self.canvases: list[FigureCanvasTkAgg] = []
        for key, title in VIEW_TITLES.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            self.tabs[key] = frame
            if key == "architecture":
                from .architecture_view import ArchitecturePanel

                self.architecture_panel = ArchitecturePanel(frame)
                self.architecture_panel.pack(fill=tk.BOTH, expand=True)
                continue
            if key == "horizon":
                from .horizon_view import HorizonPanel

                self.horizon_panel = HorizonPanel(frame, self._horizon_view)
                self.horizon_panel.pack(fill=tk.BOTH, expand=True)
                continue
            if key == "sun":
                from .sun_sensor_view import SunSensorPanel

                self.sun_sensor_panel = SunSensorPanel(frame, self._horizon_view)
                self.sun_sensor_panel.pack(side=tk.BOTTOM, fill=tk.X)
            figure = Figure(figsize=(7, 4.5), dpi=100,
                            layout=None if key in ("spacecraft", "trajectory") else "constrained")
            canvas = FigureCanvasTkAgg(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.view_figures[key] = figure
            self.view_canvases[key] = canvas
            self.toolbars[key] = toolbar
            enable_wheel_zoom(figure)
            button = ttk.Button(frame, text="Pop out", command=lambda k=key: self.toggle_popout(k))
            button.pack(side=tk.TOP, anchor=tk.E, padx=4, pady=2)
            self.popout_buttons[key] = button
            if key == "trajectory":
                figure.canvas.mpl_connect("motion_notify_event", self._refresh_scene_if_dragging)
                figure.canvas.mpl_connect("button_release_event", self._refresh_scene_if_dragging)
            if key != "spacecraft":
                self.figures.append(figure)
                self.canvases.append(canvas)
        self.spacecraft_figure = self.view_figures["spacecraft"]
        self.spacecraft_canvas = self.view_canvases["spacecraft"]
        self.spacecraft_tab = self.tabs["spacecraft"]
        notebook.bind("<<NotebookTabChanged>>", lambda _: self.refresh_active(), add="+")
        self.clear()

    def _refresh_scene_if_dragging(self, event) -> None:
        scene = self.markers.get("trajectory_scene")
        if scene is not None and (event.name == "button_release_event" or event.button is not None):
            scene.refresh()
            event.canvas.draw_idle()

    def _mount_canvas(self, key: str, master) -> None:
        figure = self.view_figures[key]
        old = self.view_canvases[key]
        old.get_tk_widget().destroy()
        self.toolbars[key].destroy()
        canvas = FigureCanvasTkAgg(figure, master=master)
        toolbar = NavigationToolbar2Tk(canvas, master, pack_toolbar=False)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.view_canvases[key], self.toolbars[key] = canvas, toolbar
        if figure in self.figures:
            self.canvases[self.figures.index(figure)] = canvas
        if key == "spacecraft":
            self.spacecraft_canvas = canvas
        canvas.draw_idle()

    def toggle_popout(self, key: str) -> None:
        if key in self.windows:
            self.dock(key)
        else:
            self.popout(key)

    def popout(self, key: str) -> None:
        if key in self.windows:
            self.windows[key].lift()
            return
        window = tk.Toplevel(self.tabs[key])
        window.title(f"OpenADCS - {VIEW_TITLES[key]}")
        window.geometry("1100x760")
        window.minsize(420, 320)
        window.protocol("WM_DELETE_WINDOW", lambda: self.dock(key))
        self.windows[key] = window
        self._mount_canvas(key, window)
        self.popout_buttons[key].configure(text="Dock")

    def dock(self, key: str) -> None:
        window = self.windows.pop(key, None)
        if window is None:
            return
        # the figure keeps its artists and state; only the Tk canvas moves
        self._mount_canvas(key, self.tabs[key])
        window.destroy()
        self.popout_buttons[key].configure(text="Pop out")

    def _horizon_view(self, sample: EnvironmentSample) -> BodyView:
        return spacecraft_view(sample, self.attitude_model)   # no shared cache: also called from a thread

    def active_view(self) -> str | None:
        selected = self.notebook.select()
        return next((key for key, frame in self.tabs.items() if str(frame) == selected), None)

    def is_visible(self, key: str) -> bool:
        return self.notebook.tab(self.tabs[key], "state") != "hidden"

    def set_visible(self, key: str, visible: bool) -> None:
        if visible:
            self.notebook.add(self.tabs[key])
            self.notebook.select(self.tabs[key])
            self.refresh_active()
        else:
            self.notebook.hide(self.tabs[key])

    def refresh_active(self) -> None:
        key = self.active_view()
        if key is None or key == "architecture":
            return
        if key == "horizon":
            if self.selected_sample is not None:
                self.horizon_panel.show_sample(self.selected_sample)
            return
        if key == "sun" and self.selected_sample is not None:
            self.sun_sensor_panel.show_sample(self.selected_sample)
        if self.selected_sample is not None and key == "spacecraft":
            if self._rendered_samples.get(key) is not self.selected_sample:
                if self._body_sample is not self.selected_sample:
                    self._body_view = spacecraft_view(self.selected_sample, self.attitude_model)
                    self._body_sample = self.selected_sample
                if self._body_view is None:
                    raise RuntimeError("Selected spacecraft sample has no body-frame state")
                renderer = self._body_renderers.get(key)
                if renderer is None:
                    renderer = BodyRenderer(self.view_figures[key])
                    self._body_renderers[key] = renderer
                renderer.update(self.selected_sample, self._body_view)
                self._rendered_samples[key] = self.selected_sample
        self.view_canvases[key].draw_idle()

    def render_selected(self, sample: EnvironmentSample) -> None:
        if sample is self.selected_sample:
            return
        self.selected_sample = sample
        if self.start_time is not None:
            elapsed = (sample.time_utc - self.start_time).total_seconds() / 60
            ground = self.markers.get("ground")
            if ground is not None:
                row = sample_readout(sample)
                ground[0].set_data([row.longitude_deg], [row.latitude_deg])
                ground[1].set_data([elapsed], [row.altitude_km])
                ground[2].set_position((row.longitude_deg + 3, row.latitude_deg + 3))
                ground[3].set_data(*earth_view.footprint(row.longitude_deg, row.latitude_deg, row.altitude_km))
            trajectory = self.markers.get("trajectory")
            if trajectory is not None:
                point = sample.position_itrs_m / 1000
                self.markers["trajectory_scene"].set_sample(sample, point)
            for key in ("magnetic", "sun"):
                if key in self.markers:
                    self.markers[key].set_xdata([elapsed, elapsed])
        self.refresh_active()

    def clear(self) -> None:
        self.selected_sample = None
        self._body_sample = None
        self._body_view = None
        self._body_renderers.clear()
        self._rendered_samples.clear()
        self.markers.clear()
        self.start_time = None
        self.horizon_panel.clear()
        self.sun_sensor_panel.clear()
        for key, figure in self.view_figures.items():
            figure.clear()
            body = key in ("spacecraft", "trajectory", "ground")
            figure.set_facecolor("#0f172a" if body else "white")
            axis = figure.add_subplot()
            if body:
                axis.set_facecolor("#0f172a")
                figure.set_facecolor(earth_view.SPACE if key != "spacecraft" else "#0f172a")
            axis.text(
                0.5, 0.5, "Configure a scenario, then click Run simulation.",
                ha="center", va="center", transform=axis.transAxes, color="#e2e8f0" if body else "#64748b",
            )
            axis.set_axis_off()
        self.refresh_active()

    def render(self, samples: Sequence[EnvironmentSample], readouts: Sequence[Readout]) -> None:
        if not samples:
            self.clear()
            return
        self.start_time = samples[0].time_utc
        self.horizon_panel.set_samples(samples)
        self.sun_sensor_panel.set_samples(samples)
        indices = np.unique(np.linspace(0, len(samples) - 1, min(len(samples), 3000), dtype=int))
        chosen = [samples[index] for index in indices]
        derived = [readouts[index] for index in indices]
        elapsed = np.array([(sample.time_utc - samples[0].time_utc).total_seconds() / 60 for sample in chosen])
        position = np.array([sample.position_itrs_m / 1000 for sample in chosen])
        magnetic = np.array([sample.magnetic_field_itrs_t * 1e6 for sample in chosen])
        sun = np.array([sample.sun_direction_itrs for sample in chosen])
        marker = "o" if len(chosen) == 1 else None

        figure = self.figures[0]
        figure.clear()
        figure.set_facecolor(earth_view.SPACE)
        track, altitude = figure.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        longitude, latitude = ground_track(
            np.array([row.longitude_deg for row in derived]),
            np.array([row.latitude_deg for row in derived]),
        )
        image = earth_view.texture()
        if image is not None:
            track.imshow(image, extent=(-180, 180, -90, 90), aspect="auto", zorder=0)
        else:
            track.set_facecolor("#1e3a5f")
        track.set_xticks(range(-180, 181, 30))
        track.set_yticks(range(-90, 91, 30))
        track.grid(color="white", alpha=0.35, linewidth=0.6)
        track.plot(longitude, latitude, color="#facc15", linewidth=1.6, zorder=3)
        track.scatter([derived[0].longitude_deg], [derived[0].latitude_deg], color=COLORS[1], zorder=4, label="Start")
        track_marker, = track.plot([derived[-1].longitude_deg], [derived[-1].latitude_deg], color="#ef4444",
                                   marker="o", markeredgecolor="white", linestyle="None", zorder=5, label="Spacecraft")
        label = track.annotate("Pilot1", (derived[-1].longitude_deg + 3, derived[-1].latitude_deg + 3),
                               color="white", fontsize=9, zorder=6)
        footprint_line, = track.plot(*earth_view.footprint(derived[-1].longitude_deg, derived[-1].latitude_deg,
                                                           derived[-1].altitude_km),
                                     color="#38bdf8", linewidth=1.4, zorder=4, label="Horizon footprint")
        track.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (deg)", ylabel="Latitude (deg)",
                  title="Ground track (2D)")
        track.legend(loc="lower left", fontsize=8)
        altitude.plot(elapsed, [row.altitude_km for row in derived], color=COLORS[0], marker=marker)
        altitude_marker, = altitude.plot([elapsed[-1]], [derived[-1].altitude_km],
                                         color=COLORS[2], marker="o", linestyle="None")
        self.markers["ground"] = (track_marker, altitude_marker, label, footprint_line)
        altitude.set(xlabel="Elapsed time (min)", ylabel="Geodetic altitude (km)")
        altitude.grid(alpha=0.25)
        altitude.set_facecolor("white")
        for axis in (track, altitude):
            axis.tick_params(colors="#e2e8f0")
            axis.xaxis.label.set_color("#e2e8f0")
            axis.yaxis.label.set_color("#e2e8f0")
            axis.title.set_color("#e2e8f0")

        figure = self.figures[1]
        figure.clear()
        figure.set_facecolor(earth_view.SPACE)
        earth_view.draw_stars(figure)
        axis = figure.add_subplot(projection="3d")
        axis.set_position((0, 0, 1, 1))
        axis.set_facecolor("none")
        limit = max(float(np.max(np.abs(position))) * 1.1, 7000)
        earth_view.draw_globe(axis)
        scene = earth_view.TrajectoryScene(axis, position, limit)
        self.markers["trajectory"] = scene.marker
        self.markers["trajectory_scene"] = scene
        for artist in (*axis.collections, *axis.lines):
            artist.set_clip_path(None)   # 3D axes are square; clip to the whole figure instead
            artist.set_clip_box(figure.bbox)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        figure.text(0.5, 0.97, "3D trajectory (Earth-fixed ITRS)", color="#e2e8f0", ha="center", va="top")
        axis.set_axis_off()
        axis.set_box_aspect((1, 1, 1))

        for figure, values, title, ylabel in (
            (self.figures[2], magnetic, "XYZgeomag / WMM2025 in ITRS", "Magnetic field (microtesla)"),
            (self.figures[3], sun, "Geometric spacecraft-to-Sun unit vector in ITRS", "Direction component (unitless)"),
        ):
            figure.clear()
            axis = figure.add_subplot()
            for index, name in enumerate(("X", "Y", "Z")):
                axis.plot(elapsed, values[:, index], color=COLORS[index], label=name, marker=marker)
            if figure is self.figures[2]:
                axis.plot(elapsed, np.linalg.norm(values, axis=1), color="#334155", linestyle="--", label="Magnitude")
            else:
                axis.set_ylim(-1.05, 1.05)
            axis.set(xlabel="Elapsed time (min)", ylabel=ylabel, title=title)
            axis.legend(loc="upper right")
            axis.grid(alpha=0.25)
            key = "magnetic" if figure is self.figures[2] else "sun"
            self.markers[key] = axis.axvline(elapsed[-1], color="#f97316", linestyle=":", linewidth=1.5)
        if self.selected_sample is not None:
            selected = self.selected_sample
            self.selected_sample = None
            self.render_selected(selected)
        else:
            self.render_selected(samples[-1])

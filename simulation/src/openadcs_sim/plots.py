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

COLORS = ("#2563eb", "#16a34a", "#db2777")
VIEW_TITLES = {
    "spacecraft": "Spacecraft",
    "earth": "Earth context",
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
                            layout=None if key in ("spacecraft", "earth") else "constrained")
            canvas = FigureCanvasTkAgg(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.view_figures[key] = figure
            self.view_canvases[key] = canvas
            if key not in ("spacecraft", "earth"):
                self.figures.append(figure)
                self.canvases.append(canvas)
        self.spacecraft_figure = self.view_figures["spacecraft"]
        self.spacecraft_canvas = self.view_canvases["spacecraft"]
        self.spacecraft_tab = self.tabs["spacecraft"]
        self.earth_figure = self.view_figures["earth"]
        self.earth_tab = self.tabs["earth"]
        notebook.bind("<<NotebookTabChanged>>", lambda _: self.refresh_active(), add="+")
        self.clear()

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
        if self.selected_sample is not None and key in ("spacecraft", "earth"):
            if self._rendered_samples.get(key) is not self.selected_sample:
                if self._body_sample is not self.selected_sample:
                    self._body_view = spacecraft_view(self.selected_sample, self.attitude_model)
                    self._body_sample = self.selected_sample
                if self._body_view is None:
                    raise RuntimeError("Selected spacecraft sample has no body-frame state")
                renderer = self._body_renderers.get(key)
                if renderer is None:
                    renderer = BodyRenderer(self.view_figures[key], earth_context=key == "earth")
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
            trajectory = self.markers.get("trajectory")
            if trajectory is not None:
                point = sample.position_itrs_m / 1000
                trajectory.set_data_3d([point[0]], [point[1]], [point[2]])
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
            body = key in ("spacecraft", "earth")
            figure.set_facecolor("#0f172a" if body else "white")
            axis = figure.add_subplot()
            if body:
                axis.set_facecolor("#0f172a")
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
        track, altitude = figure.subplots(2, 1)
        longitude, latitude = ground_track(
            np.array([row.longitude_deg for row in derived]),
            np.array([row.latitude_deg for row in derived]),
        )
        track.plot(longitude, latitude, color=COLORS[0], linewidth=1.3)
        track.scatter([derived[0].longitude_deg], [derived[0].latitude_deg], color=COLORS[1], label="Start")
        track_marker, = track.plot([derived[-1].longitude_deg], [derived[-1].latitude_deg],
                                   color=COLORS[2], marker="o", linestyle="None", label="Current time")
        track.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (deg)", ylabel="Latitude (deg)",
                  title="WGS84 sub-satellite ground track (coordinate grid, not a basemap)")
        track.legend(loc="upper right", fontsize=8)
        altitude.plot(elapsed, [row.altitude_km for row in derived], color=COLORS[0], marker=marker)
        altitude_marker, = altitude.plot([elapsed[-1]], [derived[-1].altitude_km],
                                         color=COLORS[2], marker="o", linestyle="None")
        self.markers["ground"] = (track_marker, altitude_marker)
        altitude.set(xlabel="Elapsed time (min)", ylabel="Geodetic altitude (km)")
        for axis in (track, altitude):
            axis.grid(alpha=0.25)

        figure = self.figures[1]
        figure.clear()
        axis = figure.add_subplot(projection="3d")
        azimuth, polar = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 20))
        radius = 6371
        axis.plot_surface(
            radius * np.cos(azimuth) * np.sin(polar),
            radius * np.sin(azimuth) * np.sin(polar),
            radius * np.cos(polar), color="#93c5fd", alpha=0.25, linewidth=0,
        )
        axis.plot(*position.T, color=COLORS[0], linewidth=1.4)
        trajectory_marker, = axis.plot([position[-1, 0]], [position[-1, 1]], [position[-1, 2]],
                                       color=COLORS[2], marker="o", linestyle="None")
        self.markers["trajectory"] = trajectory_marker
        limit = max(float(np.max(np.abs(position))) * 1.1, 7000)
        axis.set(xlabel="ITRS X (km)", ylabel="ITRS Y (km)", zlabel="ITRS Z (km)",
                 xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
                 title="Earth-fixed trajectory; Earth sphere is illustrative")
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

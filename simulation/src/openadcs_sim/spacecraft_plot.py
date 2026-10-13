from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .attitude import AttitudeModel, BodyView, earth_surface_body_m, spacecraft_view, unit_vector
from .environment import EnvironmentSample

VECTOR_COLORS = ("#22d3ee", "#4ade80", "#fde047", "#f472b6")
VECTOR_NAMES = ("Inertial velocity", "Magnetic field", "Spacecraft -> Sun", "Attitude boresight (+Z)")


def box_faces(low: tuple[float, float, float], high: tuple[float, float, float]) -> list:
    x, y, z = low
    a, b, c = high
    vertices = np.array([
        [x, y, z], [a, y, z], [a, b, z], [x, b, z],
        [x, y, c], [a, y, c], [a, b, c], [x, b, c],
    ])
    return [vertices[indices] for indices in (
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
    )]


def draw_cubesat(axis) -> None:
    axis.add_collection3d(Poly3DCollection(
        box_faces((-.05, -.05, -.17), (.05, .05, .17)),
        facecolors=["#475569", "#e2e8f0", "#d4a93a", "#64748b", "#94a3b8", "#d4a93a"],
        edgecolors="#e2e8f0", linewidths=.8, alpha=.95,
    ))
    for sign in (-1, 1):
        y_min, y_max = sorted((sign * .08, sign * .38))
        axis.add_collection3d(Poly3DCollection(
            box_faces((-.09, y_min, -.025), (.09, y_max, -.015)),
            facecolors="#172554", edgecolors="#93c5fd", linewidths=.6,
        ))
        for y in np.linspace(y_min, y_max, 7):
            axis.plot([-.09, .09], [y, y], [-.014, -.014], color="#94a3b8", linewidth=.4)
        axis.plot([0, 0], [sign * .05, sign * .08], [-.02, -.02], color="#d4a93a", linewidth=2)
    axis.scatter([0], [0], [.18], color="#f472b6", s=18)


def style_body_axis(axis, units: str) -> None:
    axis.set_facecolor("#0f172a")
    axis.set(xlabel=f"Body X ({units})", ylabel=f"Body Y ({units})", zlabel=f"Body Z ({units})")
    axis.tick_params(colors="#cbd5e1", labelsize=7)
    for coordinate in (axis.xaxis, axis.yaxis, axis.zaxis):
        coordinate.label.set_color("#e2e8f0")
        coordinate.set_pane_color((.06, .09, .16, 1))
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=20, azim=-55, roll=180)


class BodyRenderer:
    """Keep axes and static geometry alive so playback preserves camera/zoom."""

    def __init__(self, figure: Figure, earth_context: bool = False):
        self.figure = figure
        self.earth_context = earth_context
        figure.clear()
        figure.set_facecolor("#0f172a")
        self.axis = figure.add_subplot(projection="3d")
        style_body_axis(self.axis, "km" if earth_context else "m")
        self.header = figure.suptitle("", color="#f8fafc", fontsize=10)
        self.dynamic_artists = []
        self.auto_limits = None
        if earth_context:
            self.axis.scatter(0, 0, 0, color="#f8fafc", s=28)
            self.axis.text(0, 0, -500, "Spacecraft origin", color="#e2e8f0", fontsize=8)
            self.center_line, = self.axis.plot([], [], [], color="#f472b6", linestyle="--", linewidth=1)
            self.center_marker, = self.axis.plot([], [], [], color="#93c5fd", marker="o", linestyle="None")
            self.axis.set_title(
                "WGS84 Earth at physical scale\nBody origin at spacecraft; marker not to scale",
                color="#e2e8f0", fontsize=10, pad=28,
            )
        else:
            draw_cubesat(self.axis)
            for index, label in enumerate(("Body +X", "Body +Y", "Body +Z")):
                endpoint = np.eye(3)[index] * .43
                self.axis.plot([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
                               color="#94a3b8", linestyle=":", linewidth=.8)
                self.axis.text(*endpoint, label, color="#cbd5e1", fontsize=8)
            self.axis.set(xlim=(-.8, .8), ylim=(-.8, .8), zlim=(-.8, .8))
            self.axis.set_title("3U CubeSat + deployable panels\nDirection arrows use schematic lengths",
                                color="#e2e8f0", fontsize=10, pad=28)
            figure.legend(
                handles=[Line2D([0], [0], color=color, linewidth=2, label=name)
                         for name, color in zip(VECTOR_NAMES, VECTOR_COLORS)],
                loc="lower center", ncol=2, labelcolor="#e2e8f0",
                facecolor="#0f172a", edgecolor="#334155", fontsize=9,
            )
        figure.subplots_adjust(left=.04, right=.94, top=.73, bottom=.18)

    def update(self, sample: EnvironmentSample, view: BodyView) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()
        self.header.set_text(
            f"{sample.time_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | {view.attitude.model}"
        )
        if self.earth_context:
            surface = earth_surface_body_m(view) / 1000
            self.dynamic_artists.append(self.axis.plot_surface(
                *np.moveaxis(surface, -1, 0), color="#0284c7", alpha=.85,
                edgecolor="#075985", linewidth=.15, rstride=1, cstride=1,
            ))
            center = view.earth_center_body_m / 1000
            self.center_line.set_data_3d([0, center[0]], [0, center[1]], [0, center[2]])
            self.center_marker.set_data_3d([center[0]], [center[1]], [center[2]])
            limits = (self.axis.get_xlim(), self.axis.get_ylim(), self.axis.get_zlim())
            if self.auto_limits is None or limits == self.auto_limits:
                points = np.vstack((surface.reshape(-1, 3), np.zeros((1, 3))))
                lower, upper = np.min(points, axis=0), np.max(points, axis=0)
                midpoint = (lower + upper) / 2
                span = float(np.max(upper - lower)) * .56
                self.axis.set(xlim=(midpoint[0] - span, midpoint[0] + span),
                              ylim=(midpoint[1] - span, midpoint[1] + span),
                              zlim=(midpoint[2] - span, midpoint[2] + span))
                self.auto_limits = (self.axis.get_xlim(), self.axis.get_ylim(), self.axis.get_zlim())
        else:
            directions = (
                unit_vector(view.velocity_body_m_s), unit_vector(view.magnetic_body_t),
                view.sun_body_direction, view.attitude_body_direction,
            )
            for index, (direction, color, label) in enumerate(zip(directions, VECTOR_COLORS, ("v", "B", "Sun", "Attitude"))):
                endpoint = direction * (.72 if index == 3 else .6)
                self.dynamic_artists.append(self.axis.quiver(
                    0, 0, 0, *endpoint, color=color, arrow_length_ratio=.15, linewidth=2,
                ))
                self.dynamic_artists.append(self.axis.text(*(endpoint * 1.08), label, color=color, fontsize=8))


def render_spacecraft(
    figure: Figure, sample: EnvironmentSample, attitude_model: AttitudeModel | None = None,
) -> BodyView:
    view = spacecraft_view(sample, attitude_model)
    BodyRenderer(figure).update(sample, view)
    return view


def render_earth_context(
    figure: Figure, sample: EnvironmentSample, attitude_model: AttitudeModel | None = None,
) -> BodyView:
    view = spacecraft_view(sample, attitude_model)
    BodyRenderer(figure, earth_context=True).update(sample, view)
    return view

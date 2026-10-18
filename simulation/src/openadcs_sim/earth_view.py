"""STK-style Earth drawing helpers: textured globe, world map, footprint and vectors."""
from functools import lru_cache
from pathlib import Path

import numpy as np

TEXTURE = Path(__file__).resolve().parents[2] / "assets" / "earth" / "earth_2048.jpg"
SPACE = "#02040a"
EARTH_RADIUS_KM = 6371.0
VECTOR_STYLES = (   # key, label, colour
    ("sun", "Sun", "#facc15"),
    ("magnetic", "B field", "#38bdf8"),
    ("nadir", "Nadir", "#4ade80"),
    ("velocity", "Velocity", "#f472b6"),
)


@lru_cache(maxsize=1)
def texture() -> np.ndarray | None:
    """Equirectangular Earth image (rows north to south) or None when the asset is missing."""
    if not TEXTURE.exists():
        return None
    from PIL import Image

    return np.asarray(Image.open(TEXTURE).convert("RGB"), dtype=float) / 255


def sample_texture(longitude_deg: np.ndarray, latitude_deg: np.ndarray) -> np.ndarray:
    image = texture()
    if image is None:
        return np.full(np.shape(longitude_deg) + (3,), (0.15, 0.35, 0.6))
    rows, columns = image.shape[:2]
    column = np.clip(((longitude_deg + 180) / 360 * columns).astype(int), 0, columns - 1)
    row = np.clip(((90 - latitude_deg) / 180 * rows).astype(int), 0, rows - 1)
    return image[row, column]


def face_colors(columns: int, rows: int) -> np.ndarray:
    """Area-averaged texture, one colour per globe face (fewer, smoother polygons than point sampling)."""
    image = texture()
    if image is None:
        return np.full((rows, columns, 3), (0.15, 0.35, 0.6))
    from PIL import Image

    small = Image.fromarray((image * 255).astype(np.uint8)).resize((columns, rows), Image.BOX)
    return np.asarray(small, dtype=float) / 255


def draw_globe(axis, radius_km: float = EARTH_RADIUS_KM, longitudes: int = 96, latitudes: int = 48) -> None:
    # every 3D redraw repaints all faces (~1 s at 28k faces), so keep the mesh coarse
    lam, phi = np.meshgrid(np.radians(np.linspace(-180, 180, longitudes + 1)),
                           np.radians(np.linspace(90, -90, latitudes + 1)))
    axis.plot_surface(
        radius_km * np.cos(phi) * np.cos(lam), radius_km * np.cos(phi) * np.sin(lam), radius_km * np.sin(phi),
        facecolors=face_colors(longitudes, latitudes), shade=False, linewidth=0, antialiased=False,
        rstride=1, cstride=1, zorder=5,
    )


def draw_stars(figure, count: int = 250, seed: int = 7) -> None:
    """Fixed 2D star backdrop behind the (transparent) 3D axes, so stars never overlap the globe."""
    rng = np.random.default_rng(seed)
    backdrop = figure.add_axes((0, 0, 1, 1), zorder=-1)
    backdrop.set_facecolor(SPACE)
    backdrop.scatter(rng.random(count), rng.random(count), s=rng.uniform(0.3, 2.0, count), c="white", alpha=0.8)
    backdrop.set(xlim=(0, 1), ylim=(0, 1))
    backdrop.set_axis_off()


def body_vectors(sample) -> dict[str, np.ndarray]:
    """Unit vectors in ITRS: Sun, magnetic field, nadir and velocity."""
    def unit(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    return {
        "sun": unit(sample.sun_direction_itrs),
        "magnetic": unit(sample.magnetic_field_itrs_t),
        "nadir": unit(-sample.position_itrs_m),
        "velocity": unit(sample.velocity_itrs_m_s),
    }


def footprint(longitude_deg: float, latitude_deg: float, altitude_km: float, points: int = 181
              ) -> tuple[np.ndarray, np.ndarray]:
    """Horizon-limited visibility circle on a spherical Earth (lon, lat in degrees, NaN-split at the date line)."""
    angle = np.arccos(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + max(altitude_km, 0.0)))
    bearing = np.linspace(0, 2 * np.pi, points)
    lat0, lon0 = np.radians(latitude_deg), np.radians(longitude_deg)
    lat = np.arcsin(np.sin(lat0) * np.cos(angle) + np.cos(lat0) * np.sin(angle) * np.cos(bearing))
    lon = lon0 + np.arctan2(np.sin(bearing) * np.sin(angle) * np.cos(lat0),
                            np.cos(angle) - np.sin(lat0) * np.sin(lat))
    lon = (np.degrees(lon) + 180) % 360 - 180
    lat = np.degrees(lat)
    breaks = np.flatnonzero(np.abs(np.diff(lon)) > 180) + 1
    return np.insert(lon, breaks, np.nan), np.insert(lat, breaks, np.nan)


class TrajectoryScene:
    """Orbit, marker and vectors over the globe.

    matplotlib does not depth-sort lines against surfaces, so all 3D artists use explicit z-orders and the
    parts of the orbit/vectors hidden behind the Earth are masked from the current camera direction.
    """
    SEGMENT_POINTS = 16

    def __init__(self, axis, position: np.ndarray, limit_km: float):
        axis.computed_zorder = False
        self.axis, self.position, self.limit = axis, position, limit_km
        self.orbit, = axis.plot([], [], [], color="#facc15", linewidth=1.5, zorder=10)
        self.marker, = axis.plot([position[-1, 0]], [position[-1, 1]], [position[-1, 2]], color="#ef4444",
                                 marker="o", markeredgecolor="white", linestyle="None", zorder=12)
        self.vectors = {}
        for key, label, color in VECTOR_STYLES:
            line, = axis.plot([], [], [], color=color, linewidth=2.2, zorder=11)
            self.vectors[key] = (line, axis.text(0, 0, 0, label, color=color, fontsize=8, zorder=13))
        self.point = position[-1]
        self.sample = None
        self.refresh()

    def camera(self) -> np.ndarray:
        elevation, azimuth = np.radians(self.axis.elev), np.radians(self.axis.azim)
        return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth),
                         np.sin(elevation)])

    def occluded(self, points: np.ndarray) -> np.ndarray:
        camera = self.camera()
        along = points @ camera
        perpendicular = np.linalg.norm(points - along[:, None] * camera, axis=1)
        return (along < 0) & (perpendicular < EARTH_RADIUS_KM * 0.999)

    def set_sample(self, sample, point: np.ndarray) -> None:
        self.sample, self.point = sample, np.asarray(point, dtype=float)
        self.marker.set_data_3d([point[0]], [point[1]], [point[2]])
        self.refresh()

    def refresh(self) -> None:
        points = np.where(self.occluded(self.position)[:, None], np.nan, self.position)
        self.orbit.set_data_3d(points[:, 0], points[:, 1], points[:, 2])
        self.marker.set_visible(not self.occluded(self.point[None])[0])
        if self.sample is None:
            return
        length = 0.4 * self.limit
        unit = body_vectors(self.sample)
        fractions = np.linspace(0, 1, self.SEGMENT_POINTS)[:, None]
        for key, (line, text) in self.vectors.items():
            segment = self.point + unit[key] * length * fractions
            segment = np.where(self.occluded(segment)[:, None], np.nan, segment)
            line.set_data_3d(segment[:, 0], segment[:, 1], segment[:, 2])
            tip = self.point + unit[key] * length * 1.12
            text.set_position_3d(tip)
            text.set_visible(not self.occluded(tip[None])[0])

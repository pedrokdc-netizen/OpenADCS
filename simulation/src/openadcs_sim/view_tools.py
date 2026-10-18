"""Mouse-wheel zoom for matplotlib figures, 2D (zoom about the cursor) and 3D (camera zoom)."""
import numpy as np

ZOOM_STEP = 1.2
ZOOM_RANGE = (0.25, 12.0)


def _three_d_axes(figure):
    return next((axis for axis in figure.axes if hasattr(axis, "get_proj")), None)


def zoom_3d(axis, factor: float) -> None:
    zoom = float(np.clip(getattr(axis, "_oa_zoom", 1.0) * factor, *ZOOM_RANGE))
    axis._oa_zoom = zoom
    axis.set_box_aspect((1, 1, 1), zoom=zoom)


def zoom_2d(axis, x: float, y: float, factor: float) -> None:
    (x0, x1), (y0, y1) = axis.get_xlim(), axis.get_ylim()
    axis.set_xlim(x - (x - x0) / factor, x + (x1 - x) / factor)
    axis.set_ylim(y - (y - y0) / factor, y + (y1 - y) / factor)


def on_scroll(event) -> None:
    factor = ZOOM_STEP if event.button == "up" else 1 / ZOOM_STEP
    figure = event.canvas.figure
    axis3d = _three_d_axes(figure)
    if axis3d is not None:
        zoom_3d(axis3d, factor)
    elif event.inaxes is not None and event.xdata is not None:
        zoom_2d(event.inaxes, event.xdata, event.ydata, factor)
    else:
        return
    event.canvas.draw_idle()


def enable_wheel_zoom(figure) -> None:
    """Callbacks live on the figure, so they survive moving it to a pop-out window."""
    if not getattr(figure, "_oa_zoom_connected", False):
        figure.canvas.mpl_connect("scroll_event", on_scroll)
        figure._oa_zoom_connected = True

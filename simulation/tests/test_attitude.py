from datetime import datetime, timezone

import numpy as np
import pytest

from openadcs_sim.attitude import (
    AttitudeState, NadirPointingAttitude, WGS84_A_M, WGS84_B_M,
    earth_surface_body_m, spacecraft_view, unit_vector,
)
from openadcs_sim.common import vector3
from openadcs_sim.environment import EnvironmentSample

TIME = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


@pytest.fixture
def sample():
    return EnvironmentSample(
        TIME, vector3([6778137, 0, 0]), vector3([0, 7000, 0]),
        vector3([1e-5, 2e-5, 3e-5]), unit_vector(vector3([1, 2, 3])), 1.5e11,
    )


def test_nadir_reference_uses_velocity_horizontal_projection():
    attitude = NadirPointingAttitude().orientation(TIME, vector3([1, 0, 0]), vector3([2, 3, 0]))
    np.testing.assert_allclose(attitude.body_to_gcrs, [[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    assert np.linalg.det(attitude.body_to_gcrs) == pytest.approx(1)
    with pytest.raises(ValueError, match="non-zero"):
        NadirPointingAttitude().orientation(TIME, vector3([1, 0, 0]), vector3([2, 0, 0]))


@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)), np.eye(2), np.diag([1, 1, -1]),
    np.eye(3) * 2, np.eye(3) * float("nan"),
])
def test_invalid_attitude_not_silently_replaced(matrix):
    with pytest.raises(ValueError, match="right-handed orthonormal"):
        AttitudeState(matrix)


def test_rotation_is_copied_and_immutable():
    source = np.eye(3)
    state = AttitudeState(source)
    source[0, 0] = 2
    assert state.body_to_gcrs[0, 0] == 1
    with pytest.raises(ValueError):
        state.body_to_gcrs[0, 0] = 2


def test_body_origin_nadir_directions_and_inertial_velocity(sample):
    view = spacecraft_view(sample)
    np.testing.assert_allclose(view.itrs_to_body.T @ view.itrs_to_body, np.eye(3), atol=1e-12)
    assert np.linalg.det(view.itrs_to_body) == pytest.approx(1)
    np.testing.assert_allclose(view.earth_center_body_m, [0, 0, 6778137], atol=1e-7)
    np.testing.assert_allclose(view.attitude_body_direction, [0, 0, 1])
    assert view.velocity_body_m_s[0] > 7400
    np.testing.assert_allclose(view.velocity_body_m_s[1:], [0, 0], atol=1e-3)
    assert np.linalg.norm(view.magnetic_body_t) == pytest.approx(np.linalg.norm(sample.magnetic_field_itrs_t))
    assert np.linalg.norm(view.sun_body_direction) == pytest.approx(1)
    np.testing.assert_allclose(view.magnetic_body_t, view.itrs_to_body @ sample.magnetic_field_itrs_t)
    np.testing.assert_allclose(view.sun_body_direction, view.itrs_to_body @ sample.sun_direction_itrs)


def test_earth_radii_surface_equation_and_actual_altitude(sample):
    view = spacecraft_view(sample)
    surface = earth_surface_body_m(view)
    fixed = (surface - view.earth_center_body_m) @ view.itrs_to_body
    equation = (fixed[..., 0] / WGS84_A_M) ** 2 + (fixed[..., 1] / WGS84_A_M) ** 2 + (fixed[..., 2] / WGS84_B_M) ** 2
    np.testing.assert_allclose(equation, 1, atol=1e-12)
    assert np.max(fixed[..., 0]) == pytest.approx(6378137)
    assert np.max(fixed[..., 2]) == pytest.approx(6356752.314245179)
    closest = view.itrs_to_body @ vector3([WGS84_A_M, 0, 0]) + view.earth_center_body_m
    np.testing.assert_allclose(closest, [0, 0, 400000], atol=1e-7)
    with pytest.raises(ValueError, match="mesh"):
        earth_surface_body_m(view, longitude_count=2)


def test_external_attitude_provider_receives_time_and_is_used(sample):
    class External:
        def orientation(self, time_utc, position_gcrs_m, velocity_gcrs_m_s):
            assert time_utc == TIME
            assert np.linalg.norm(position_gcrs_m) == pytest.approx(6778137)
            return AttitudeState(np.eye(3), "Future ADCS input")

    view = spacecraft_view(sample, External())
    assert view.attitude.model == "Future ADCS input"
    np.testing.assert_array_equal(view.attitude.body_to_gcrs, np.eye(3))
    assert abs(view.earth_center_body_m[0]) > 1e6
    assert abs(view.earth_center_body_m[2]) < 1e5


def test_plot_uses_separate_scales_and_is_renderable(sample):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from openadcs_sim.spacecraft_plot import BodyRenderer, render_spacecraft

    figure = Figure(figsize=(12, 6))
    canvas = FigureCanvasAgg(figure)
    view = render_spacecraft(figure, sample)
    canvas.draw()
    local, = figure.axes
    assert local.get_xlabel() == "Body X (m)"
    assert local.get_xlim() == (-.8, .8)
    assert "nadir" in view.attitude.model
    assert len(figure.legends[0].texts) == 4
    renderer = BodyRenderer(figure)
    renderer.update(sample, view)
    renderer.axis.view_init(elev=35, azim=15, roll=8)
    renderer.axis.set_xlim(-.4, .4)
    axis = renderer.axis
    renderer.update(sample, view)
    assert renderer.axis is axis
    assert (axis.elev, axis.azim, axis.roll) == (35, 15, 8)
    assert axis.get_xlim() == (-.4, .4)

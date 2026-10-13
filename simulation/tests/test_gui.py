from pathlib import Path
from datetime import datetime, timedelta, timezone
import time

import numpy as np
import pytest

pytest.importorskip("matplotlib")
tk = pytest.importorskip("tkinter")

from openadcs_sim.gui import SimulationApp
from openadcs_sim.environment import EnvironmentSample
from openadcs_sim.plots import ground_track
from openadcs_sim.scenario import KERNEL_NAMES, sample_readout


@pytest.fixture(scope="module")
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def app(tk_root):
    root = tk_root
    interface = SimulationApp(root)
    try:
        yield interface
    finally:
        if interface.worker is not None:
            interface.worker.cancel()
            deadline = time.monotonic() + 10
            while interface.worker is not None and time.monotonic() < deadline:
                root.update()
                time.sleep(0.01)
        owners = {}
        widgets = [root]
        while widgets:
            widget = widgets.pop()
            widgets.extend(widget.winfo_children())
            owners.update({command: widget for command in widget._tclCommands or ()})
        for timer in root.tk.call("after", "info"):
            script = root.tk.call("after", "info", timer)[0]
            command = root.tk.splitlist(script)[0]
            owner = owners.get(command)
            if owner is not None:
                owner.after_cancel(timer)
            else:
                root.tk.call("after", "cancel", timer)
        for child in root.winfo_children():
            child.destroy()


def test_gui_defaults_and_training_controls(app):
    assert len(app.notebook.tabs()) == 10
    assert str(app.stop_button["state"]) == "disabled"
    assert str(app.csv_button["state"]) == "disabled"
    assert "571 samples" in app.schedule_info.get()
    app.load_training()
    assert app.start_value.get() == "2026-10-09T12:00:00+00:00"
    assert "SYNTHETIC" in app.tle_text.get("1.0", tk.END)
    assert "NOT an observed" in app.status_value.get()
    app.step_value.set("0")
    assert "positive time step" in app.schedule_info.get()
    app.duration_value.set("0.005")
    app.step_value.set("0.1")
    assert "4 samples" in app.schedule_info.get()


def test_ground_track_breaks_the_dateline():
    x, y = ground_track(np.array([179., -179., -178.]), np.array([10., 11., 12.]))
    assert np.isnan(x[1]) and np.isnan(y[1])
    np.testing.assert_array_equal(x[[0, 2, 3]], [179, -179, -178])


def test_architecture_available_before_run_and_window_menu_preserves_state(app):
    panel = app.plotter.architecture_panel
    assert app.plotter.is_visible("architecture")
    assert len(app.plotter.figures) == 4
    assert "architecture" not in app.plotter.view_figures
    panel.select_module("adcs")
    panel.set_zoom(1.25)
    items = panel.canvas.find_all()
    details = panel.details.get("1.0", tk.END)
    assert "PARTIAL" in details
    assert "attitude_estimator.py" in details
    app.notebook.select(app.plotter.tabs["architecture"])
    app.plotter.refresh_active()
    app.close_active_view()
    assert not app.plotter.is_visible("architecture")
    app.window_menu.invoke(list(app.view_enabled).index("architecture"))
    assert app.plotter.active_view() == "architecture"
    app.plotter.clear()
    assert panel.scale == 1.25
    assert panel.canvas.find_all() == items
    assert panel.details.get("1.0", tk.END) == details


def test_architecture_zoom_fit_and_selection(app):
    from openadcs_sim.architecture import CONNECTIONS, MODULES

    panel = app.plotter.architecture_panel
    assert len(panel.canvas.find_withtag("module_box")) == len(MODULES)
    assert len(panel.canvas.find_withtag("connection_label")) == len(CONNECTIONS)
    panel.fit(900, 500)
    assert panel.fit_mode
    from openadcs_sim.architecture import DIAGRAM_SIZE
    assert panel.scale == pytest.approx(min(888 / DIAGRAM_SIZE[0], 488 / DIAGRAM_SIZE[1]))
    panel.set_zoom(2)
    assert not panel.fit_mode
    assert panel.zoom_text.get() == "200%"
    panel.select_connection("gps_adcs")
    assert "ECEF PVT" in panel.details.get("1.0", tk.END)
    assert "Available API or offline path" in panel.details.get("1.0", tk.END)
    panel.select_connection("adcs_actuators")
    assert "PLANNED / not connected" in panel.details.get("1.0", tk.END)
    panel.module_choice.current(next(index for index, module in enumerate(MODULES) if module.id == "obc"))
    panel._choose_module(None)
    assert panel.scale == 1
    assert panel.selected_module == "obc"
    assert "PLACEHOLDER" in panel.details.get("1.0", tk.END)
    with pytest.raises(ValueError, match="finite and positive"):
        panel.set_zoom(float("nan"))


def test_architecture_text_is_inside_drawing_and_blocks_at_readable_zoom(app):
    from openadcs_sim.architecture import DIAGRAM_SIZE, MODULES

    panel = app.plotter.architecture_panel
    for zoom in (0.25, 0.4, 0.75, 1.0, 2.0):
        panel.set_zoom(zoom)
        for item in panel.canvas.find_all():
            bounds = panel.canvas.bbox(item)
            assert bounds is not None
            assert bounds[0] >= 0 and bounds[1] >= 0
            assert bounds[2] <= DIAGRAM_SIZE[0] * zoom
            assert bounds[3] <= DIAGRAM_SIZE[1] * zoom
        for module in MODULES:
            left, top, right, bottom = (value * zoom for value in module.bounds)
            text_bounds = []
            for item in panel.canvas.find_withtag(f"module:{module.id}"):
                if panel.canvas.type(item) == "text":
                    bounds = panel.canvas.bbox(item)
                    assert bounds[0] >= left and bounds[1] >= top
                    assert bounds[2] <= right and bounds[3] <= bottom
                    text_bounds.append(bounds)
            for first, second in zip(text_bounds, text_bounds[1:]):
                assert first[3] <= second[1]
        if zoom >= 1:
            for label in panel.canvas.find_withtag("connection_label"):
                bounds = panel.canvas.bbox(label)
                for module in MODULES:
                    left, top, right, bottom = (value * zoom for value in module.bounds)
                    assert (bounds[2] <= left or bounds[0] >= right or
                            bounds[3] <= top or bounds[1] >= bottom)


@pytest.fixture
def recorded_app(app):
    start = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    for index in range(4):
        angle = index * .01
        sample = EnvironmentSample(
            start + timedelta(seconds=10 * index),
            np.array([6778137 * np.cos(angle), 6778137 * np.sin(angle), 0]),
            np.array([-7000 * np.sin(angle), 7000 * np.cos(angle), 0]),
            np.array([1e-5, 2e-5, 3e-5]), np.array([1., 0., 0.]), 1.5e11,
        )
        app.samples.append(sample)
        app.readouts.append(sample_readout(sample))
    app.result_status = "completed"
    app.plotter.render(app.samples, app.readouts)
    app.playback.load([sample.time_utc for sample in app.samples])
    app.slider.configure(to=app.playback.duration)
    app.display_clock()
    return app


def test_shared_timeline_controls_and_recorded_state(recorded_app):
    app = recorded_app
    app.seek_playback("15")
    assert app.playback.elapsed == 15
    assert app.playback.index == 1
    assert app.plotter.selected_sample is app.samples[1]
    assert "Sample 2 / 4" in app.sample_info.get()
    assert "12:00:15" in app.animation_time.get()
    assert "12:00:10" in app.animation_time.get()
    ground, altitude = app.plotter.markers["ground"]
    assert ground.get_xdata()[0] == app.readouts[1].longitude_deg
    assert altitude.get_xdata()[0] == pytest.approx(10 / 60)
    for key in ("magnetic", "sun"):
        assert app.plotter.markers[key].get_xdata() == [10 / 60, 10 / 60]
    x, y, z = app.plotter.markers["trajectory"].get_data_3d()
    np.testing.assert_allclose([x[0], y[0], z[0]], app.samples[1].position_itrs_m / 1000)
    app.step_playback(1)
    assert app.playback.index == 2
    app.step_playback(-1)
    assert app.playback.index == 1
    app.jump_playback(True)
    assert app.playback.index == 3
    app.jump_playback(False)
    assert app.playback.index == 0
    app.result_status = "cancelled"
    app.update_animation_controls()
    assert "partial data" in app.animation_status.get()


def test_architecture_persists_through_render_and_playback(recorded_app):
    app = recorded_app
    panel = app.plotter.architecture_panel
    panel.select_module("gps")
    panel.set_zoom(1.5)
    original = panel.canvas.find_all()
    app.plotter.set_visible("architecture", True)
    app.seek_playback("20")
    app.plotter.render(app.samples, app.readouts)
    assert panel.scale == 1.5
    assert panel.canvas.find_all() == original
    assert panel.selected_module == "gps"
    assert "NOT flight-calibrated" in panel.details.get("1.0", tk.END)


def test_animation_timer_speed_pause_scrub_and_endpoint(recorded_app, monkeypatch):
    app = recorded_app
    wall = [100.]
    monkeypatch.setattr("openadcs_sim.gui.time.monotonic", lambda: wall[0])
    app.play()
    timer = app._animation_timer
    assert timer is not None
    assert app.playback.playing
    assert str(app.animation_buttons["pause"]["state"]) == "normal"
    app.root.update_idletasks()
    assert app.playback.playing  # Programmatic slider updates must not invoke scrub.
    wall[0] = 100.25
    app.pause_playback()
    assert app.playback.elapsed == 15
    assert app._animation_timer is None
    assert timer not in app.root.tk.call("after", "info")
    app.speed_value.set("120x")
    assert app.apply_speed()
    app.change_speed(.5)
    assert app.playback.rate == 60
    app.change_speed(2)
    assert app.playback.rate == 120
    app.play(-1)
    app.seek_playback("12")
    assert not app.playback.playing
    assert app._animation_timer is None
    app.play()
    app._cancel_animation_timer()
    wall[0] = 101.
    app.animation_tick()
    assert app.playback.index == 3
    assert not app.playback.playing
    assert app._animation_timer is None
    assert str(app.animation_buttons["pause"]["state"]) == "disabled"


def test_window_menu_hides_reopens_at_shared_time_preserving_camera(recorded_app):
    app = recorded_app
    renderer = app.plotter._body_renderers["spacecraft"]
    renderer.axis.view_init(elev=31, azim=18)
    renderer.axis.set_xlim(-.4, .4)
    app.window_menu.invoke(0)
    assert not app.plotter.is_visible("spacecraft")
    assert not app.view_enabled["spacecraft"].get()
    app.seek_playback("20")
    assert app.plotter._rendered_samples["spacecraft"] is app.samples[0]
    app.window_menu.invoke(0)
    assert app.plotter.active_view() == "spacecraft"
    assert app.plotter._rendered_samples["spacecraft"] is app.samples[2]
    assert app.plotter._body_renderers["spacecraft"] is renderer
    assert (renderer.axis.elev, renderer.axis.azim) == (31, 18)
    assert renderer.axis.get_xlim() == (-.4, .4)
    app.close_active_view()
    assert not app.plotter.is_visible("spacecraft")
    app.set_all_views(False)
    assert all(not app.plotter.is_visible(key) for key in app.view_enabled)
    visible_titles = [app.notebook.tab(tab, "text") for tab in app.notebook.tabs()
                      if app.notebook.tab(tab, "state") != "hidden"]
    assert visible_titles == ["Samples", "Guide", "Diagnostics"]
    app.set_all_views(True)
    assert all(app.plotter.is_visible(key) for key in app.view_enabled)
    assert app.playback.index == 2
    assert len(app.samples) == 4


def test_invalid_playback_speed_is_reported(recorded_app, monkeypatch):
    errors = []
    monkeypatch.setattr("openadcs_sim.gui.messagebox.showerror", lambda *args, **kwargs: errors.append(args))
    recorded_app.speed_value.set("NaN")
    recorded_app.play()
    assert errors
    assert not recorded_app.playback.playing
    assert recorded_app._animation_timer is None
    assert recorded_app.speed_value.get() == "60x"


def test_close_stops_animation(recorded_app, monkeypatch):
    app = recorded_app
    destroyed = []
    monkeypatch.setattr(app.root, "destroy", lambda: destroyed.append(True))
    app.play()
    timer = app._animation_timer
    app.close()
    assert destroyed
    assert not app.playback.playing
    assert app._animation_timer is None
    assert timer not in app.root.tk.call("after", "info")


def test_new_run_stops_and_clears_recorded_playback(recorded_app, monkeypatch, tmp_path):
    app = recorded_app
    for name in KERNEL_NAMES:
        (tmp_path / name).touch()
    app.load_training()
    app.kernel_value.set(str(tmp_path))
    started = []
    monkeypatch.setattr("openadcs_sim.gui.RunWorker.start", lambda _: started.append(True))
    app.play()
    timer = app._animation_timer
    app.run()
    assert started
    assert app.worker is not None
    assert not app.playback.playing
    assert app._animation_timer is None
    assert timer not in app.root.tk.call("after", "info")
    assert not app.playback.timestamps
    assert not app.samples
    assert str(app.animation_buttons["forward"]["state"]) == "disabled"
    assert str(app.slider["state"]) == "disabled"
    app.worker = None


def test_earth_camera_survives_sample_updates(recorded_app):
    app = recorded_app
    app.plotter.set_visible("earth", True)
    renderer = app.plotter._body_renderers["earth"]
    renderer.axis.view_init(elev=42, azim=30)
    renderer.axis.set_xlim(-3000, 3000)
    limits = (renderer.axis.get_xlim(), renderer.axis.get_ylim(), renderer.axis.get_zlim())
    app.seek_playback("20")
    assert app.plotter._body_renderers["earth"] is renderer
    assert (renderer.axis.elev, renderer.axis.azim) == (42, 30)
    assert (renderer.axis.get_xlim(), renderer.axis.get_ylim(), renderer.axis.get_zlim()) == limits


def test_invalid_inputs_do_not_start_worker(app, monkeypatch):
    errors = []
    monkeypatch.setattr("openadcs_sim.gui.messagebox.showerror", lambda *args, **kwargs: errors.append(args))
    app.run()
    assert app.worker is None
    assert errors
    assert not app.samples


def test_failed_worker_is_visible_and_unlocks_inputs(app, monkeypatch, tmp_path):
    errors = []
    monkeypatch.setattr("openadcs_sim.gui.messagebox.showerror", lambda *args, **kwargs: errors.append(args))
    for name in KERNEL_NAMES:
        (tmp_path / name).touch()
    app.load_training()
    app.kernel_value.set(str(tmp_path))
    app.duration_value.set("0")
    app.run()
    deadline = time.monotonic() + 40
    while app.worker is not None and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert app.worker is None
    assert app.result_status == "failed"
    assert errors
    assert "partial results" in app.status_value.get()
    assert app.notebook.select() == str(app.diagnostics_tab)
    assert str(app.run_button["state"]) == "normal"
    assert str(app.json_button["state"]) == "disabled"
    assert "Worker exited" in app.log.get("1.0", tk.END)


@pytest.mark.integration
def test_gui_stop_preserves_partial_results(app, monkeypatch):
    directory = Path("simulation/data/kernels").resolve()
    if not all((directory / name).is_file() for name in KERNEL_NAMES):
        pytest.skip("Download kernels first")
    errors = []
    monkeypatch.setattr("openadcs_sim.gui.messagebox.showerror", lambda *args, **kwargs: errors.append(args))
    app.load_training()
    app.kernel_value.set(str(directory))
    app.duration_value.set("60")
    app.step_value.set("1")
    app.run()
    deadline = time.monotonic() + 40
    stopped = False
    while app.worker is not None and time.monotonic() < deadline:
        app.root.update()
        if app.samples and not stopped:
            app.stop()
            stopped = True
        time.sleep(0.01)
    assert stopped
    assert app.worker is None
    assert not errors
    assert app.result_status == "cancelled"
    assert 0 < len(app.samples) < 3601
    assert "partial results" in app.status_value.get()
    assert str(app.json_button["state"]) == "normal"
    original = app.result_scenario
    app.duration_value.set("5")
    assert app.result_scenario is original


@pytest.mark.integration
def test_gui_runs_real_physics_and_restores_controls(app, monkeypatch):
    directory = Path("simulation/data/kernels").resolve()
    if not all((directory / name).is_file() for name in KERNEL_NAMES):
        pytest.skip("Download kernels first")
    errors = []
    monkeypatch.setattr("openadcs_sim.gui.messagebox.showerror", lambda *args, **kwargs: errors.append(args))
    app.load_training()
    app.kernel_value.set(str(directory))
    app.duration_value.set("0.5")
    app.step_value.set("10")
    app.run()
    assert app.worker is not None
    assert str(app.run_button["state"]) == "disabled"
    deadline = time.monotonic() + 40
    while app.worker is not None and time.monotonic() < deadline:
        app.root.update()
        time.sleep(0.01)
    assert app.worker is None
    assert not errors
    assert app.result_status == "completed"
    assert len(app.samples) == 4
    assert str(app.run_button["state"]) == "normal"
    assert str(app.csv_button["state"]) == "normal"
    assert app.progress["value"] == 100
    app.inspect_sample("0")
    assert "Sample 1 / 4" in app.sample_info.get()
    assert "Earth-fixed km/s" in app.details.get()
    assert len(app.plotter.figures[0].axes) == 2
    assert len(app.plotter.figures[1].axes) == 1
    assert len(app.plotter.spacecraft_figure.axes) == 1
    app.notebook.select(app.plotter.earth_tab)
    app.root.update_idletasks()
    app.plotter.refresh_active()
    assert len(app.plotter.earth_figure.axes) == 1
    assert app.plotter.selected_sample is app.samples[0]
    app.inspect_sample("3")
    assert app.plotter.selected_sample is app.samples[3]

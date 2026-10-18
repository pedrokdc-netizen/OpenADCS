import json

import numpy as np
import pytest

from openadcs_sim import subsystems
from openadcs_sim.assembly import DEFAULT_SLOT, NONE_SLOT, SpacecraftAssembly, euler_matrix
from openadcs_sim.closed_loop import AnalyticEnvironment
from openadcs_sim.spacecraft import SpacecraftError, pilot1_baseline


@pytest.mark.parametrize("key", list(subsystems.SUBSYSTEM_TYPES))
def test_every_type_builds_from_defaults_and_round_trips(key, tmp_path):
    subsystem = subsystems.SUBSYSTEM_TYPES[key]
    assert subsystem.build() is not None
    subsystems.save_subsystem(key, "Unit Test 1", subsystem.defaults(), tmp_path)
    assert subsystems.list_saved(key, tmp_path) == ["Unit Test 1"]
    assert subsystems.load_subsystem(key, "Unit Test 1", tmp_path) == subsystem.validate(subsystem.defaults())


def test_default_wheel_reproduces_scaled_rw0003():
    from openadcs_sim.actuators import scaled_rw0003
    built, reference = subsystems.SUBSYSTEM_TYPES["reaction_wheel"].build(), scaled_rw0003()
    assert built.inertia_kg_m2 == pytest.approx(reference.inertia_kg_m2, rel=1e-3)
    assert built.max_speed_rad_s == pytest.approx(reference.max_speed_rad_s, rel=1e-6)
    assert built.viscous_friction_nm_s == pytest.approx(reference.viscous_friction_nm_s, rel=1e-2)


def test_magnetorquer_dipole_follows_winding_physics():
    torquer = subsystems.SUBSYSTEM_TYPES["magnetorquer"]
    base = torquer.defaults()
    design = subsystems.torquer_design(base)
    assert design["dipole_am2"] == pytest.approx(0.2, rel=0.1)
    air = subsystems.torquer_design({**base, "core_material": "Air (no core)"})
    assert air["dipole_am2"] < 0.1 * design["dipole_am2"]
    assert torquer.build(base).max_dipole_am2 == pytest.approx(design["dipole_am2"])
    limited = subsystems.torquer_design({**base, "current_limit_ma": 10.0})
    assert limited["dipole_am2"] < design["dipole_am2"] and not limited["voltage_limited"]


def test_validation_messages():
    wheel = subsystems.SUBSYSTEM_TYPES["reaction_wheel"]
    with pytest.raises(subsystems.SubsystemError, match="not a number"):
        wheel.validate({"max_speed_rpm": "fast"})
    with pytest.raises(subsystems.SubsystemError, match="at least"):
        wheel.validate({"rotor_mass_g": -1})
    with pytest.raises(subsystems.SubsystemError, match="not one of"):
        wheel.validate({"inertia_shape": "cube"})
    with pytest.raises(subsystems.SubsystemError, match="whole number"):
        subsystems.SUBSYSTEM_TYPES["magnetorquer"].validate({"turns": 10.5})
    assert wheel.validate({"rotor_mass_g": "6,5"})["rotor_mass_g"] == 6.5
    with pytest.raises(subsystems.SubsystemError, match="name"):
        subsystems.save_subsystem("gnss", "  ", {}, None)


def test_file_checks(tmp_path):
    path = subsystems.save_subsystem("gnss", "A", {"horizontal_cep_m": 5.0}, tmp_path)
    assert subsystems.read_subsystem_file(path)[2]["horizontal_cep_m"] == 5.0
    path.write_text(json.dumps({"kind": "other"}), encoding="utf-8")
    with pytest.raises(subsystems.SubsystemError, match="not a subsystem"):
        subsystems.read_subsystem_file(path)
    with pytest.raises(subsystems.SubsystemError, match="No saved"):
        subsystems.load_subsystem("gnss", "missing", tmp_path)


def test_assembly_round_trip_embeds_subsystems(tmp_path):
    library, other = tmp_path / "lib", tmp_path / "other"
    subsystems.save_subsystem("gnss", "Receiver 5 m", {"horizontal_cep_m": 5.0}, library)
    assembly = SpacecraftAssembly(variables={"wheel_layout": "Orthogonal (3 wheels)", "gyro_yaw_deg": 30},
                                  attached={"gnss": "Receiver 5 m", "horizon_sensor": NONE_SLOT})
    path = tmp_path / "sc.json"
    assembly.save(path, library)
    loaded = SpacecraftAssembly.load(path, other)
    assert loaded.attached["gnss"] == "Receiver 5 m" and loaded.attached["horizon_sensor"] == NONE_SLOT
    assert loaded.variables["wheel_layout"] == "Orthogonal (3 wheels)"
    assert subsystems.list_saved("gnss", other) == ["Receiver 5 m"]


def test_legacy_mass_file_loads_as_default_assembly(tmp_path):
    path = tmp_path / "old.json"
    pilot1_baseline().save(path)
    loaded = SpacecraftAssembly.load(path, tmp_path)
    assert loaded.mass_properties.mass_kg == pilot1_baseline().mass_kg
    assert set(loaded.attached.values()) == {DEFAULT_SLOT}


def test_assembly_rejects_bad_input(tmp_path):
    with pytest.raises(SpacecraftError):
        SpacecraftAssembly(attached={"gyro": NONE_SLOT})
    with pytest.raises(SpacecraftError):
        SpacecraftAssembly(attached={"nonsense": DEFAULT_SLOT})
    with pytest.raises(subsystems.SubsystemError):
        SpacecraftAssembly(variables={"box_x_mm": -5})
    with pytest.raises(SpacecraftError, match="No saved"):
        SpacecraftAssembly(attached={"gnss": "missing"}).closed_loop_kwargs(tmp_path)


def test_assembly_drives_the_closed_loop():
    plain = SpacecraftAssembly().closed_loop_kwargs()
    assert plain["wheels"].axes.shape == (3, 4) and plain["use_horizon"]
    custom = SpacecraftAssembly(
        variables={"wheel_layout": "Orthogonal (3 wheels)", "gyro_yaw_deg": 90.0, "box_x_mm": 300},
        attached={"horizon_sensor": NONE_SLOT}).closed_loop_kwargs()
    assert custom["wheels"].axes.shape == (3, 3) and not custom["use_horizon"] and custom["horizon_config"] is None
    assert custom["disturbances"].dimensions_m[0] == pytest.approx(0.3)
    assert np.allclose(custom["gyro_config"].sensor_from_body, euler_matrix(0, 0, 90).T)
    sim = SpacecraftAssembly().simulation(AnalyticEnvironment(), initial_rate_rad_s=(0.02, 0.0, 0.0))
    assert sim.horizon is not None and sim.wheels.axes.shape == (3, 4)


def test_single_horizon_architecture_uses_one_camera(tmp_path):
    subsystems.save_subsystem("horizon_sensor", "one", {"architecture": "single"}, tmp_path)
    kwargs = SpacecraftAssembly(attached={"horizon_sensor": "one"}).closed_loop_kwargs(tmp_path)
    assert len(kwargs["horizon_config"].mounts) == 1


# ------------------------------------------------------------------------------------------------ windows
tk = pytest.importorskip("tkinter")


@pytest.fixture(scope="module")
def root():
    try:
        window = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    window.withdraw()
    yield window
    window.destroy()


def test_subsystem_editor_save_load_and_errors(root, tmp_path):
    from openadcs_sim.subsystem_window import SubsystemListWindow
    saved = []
    window = SubsystemListWindow(root, tmp_path, lambda key, name: saved.append((key, name)))
    editor = window.open_selected("magnetorquer")
    assert window.open_selected("magnetorquer") is editor
    editor.form.variables["turns"].set("3000")
    editor.refresh()
    assert "Maximum dipole" in editor.readout.get()
    editor.save_name.set("Rod 3000 turns")
    assert editor.save() is not None and saved == [("magnetorquer", "Rod 3000 turns")]
    editor.reset()
    assert editor.form.variables["turns"].get() == "2400"
    assert editor.load_selected()
    assert editor.form.variables["turns"].get() == "3000"
    editor.form.variables["turns"].set("abc")
    editor.refresh()
    assert "not a number" in editor.message.get()
    assert editor.save() is None
    window.destroy()


def test_assembly_window_import_export_and_apply(root, tmp_path):
    from openadcs_sim.assembly_window import AssemblyWindow
    subsystems.save_subsystem("gnss", "Receiver 5 m", {"horizontal_cep_m": 5.0}, tmp_path)
    applied = []
    window = AssemblyWindow(root, SpacecraftAssembly(), applied.append, tmp_path)
    assert "Receiver 5 m" in window.slot_boxes["gnss"]["values"]
    assert NONE_SLOT in window.slot_boxes["horizon_sensor"]["values"] and NONE_SLOT not in window.slot_boxes["gyro"]["values"]
    window.slot_vars["gnss"].set("Receiver 5 m")
    window.forms["Reaction wheel layout"].variables["wheel_layout"].set("Orthogonal (3 wheels)")
    path = tmp_path / "export.json"
    assert window.export_path(path) and applied[-1].attached["gnss"] == "Receiver 5 m"
    window.show(SpacecraftAssembly())
    assert window.slot_vars["gnss"].get() == DEFAULT_SLOT
    assert window.import_path(path)
    assert window.slot_vars["gnss"].get() == "Receiver 5 m"
    assert window.forms["Reaction wheel layout"].variables["wheel_layout"].get() == "Orthogonal (3 wheels)"
    window.forms["Structure and surfaces"].variables["box_x_mm"].set("oops")
    assert window.apply() is None and "not a number" in window.message.get()
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert not window.import_path(bad) and "Cannot read" in window.message.get()
    window.destroy()


def test_closed_loop_run_helpers_and_window(root):
    from openadcs_sim.adcs_run import build_environment, run_closed_loop, sample_dicts, summarise
    from openadcs_sim.closed_loop_window import ClosedLoopWindow
    result = run_closed_loop(SpacecraftAssembly(), build_environment("analytic"), 300.0, 1.0,
                             (0.02, -0.02, 0.01), (20, -15, 30))
    stats = summarise(result)
    assert stats["samples"] == 300 and stats["pointing_error_deg_max"] < 60
    assert len(list(sample_dicts(result, 100))) == 3
    with pytest.raises(ValueError):
        run_closed_loop(SpacecraftAssembly(), build_environment("analytic"), -1.0)
    with pytest.raises(ValueError, match="needs a TLE"):
        build_environment("sgp4")
    window = ClosedLoopWindow(root, SpacecraftAssembly)
    window.vars["duration"].set("120")
    window.vars["outages"].set("10-20")
    assert window.read_parameters()["outage_windows_s"] == ((10.0, 20.0),)
    window.vars["rate"].set("1 2")
    assert not window.start() and "3 numbers" in window.message.get()
    window.vars["rate"].set("0.01 0 0")
    assert window.start()
    window.thread.join(60)
    kind, payload = None, None
    while True:
        kind, payload = window.events.get(timeout=5)
        if kind != "progress":
            break
    assert kind == "done"
    window.show_result(payload)
    assert "Pointing error" in window.summary.get()
    window.destroy()

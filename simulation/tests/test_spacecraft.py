import json

import numpy as np
import pytest

from openadcs_sim.spacecraft import DEFAULT_DIRECTORY, SpacecraftConfig, SpacecraftError, from_cad, pilot1_baseline


def test_baseline_matches_cad_principal_moments():
    craft = pilot1_baseline()
    assert craft.mass_kg == pytest.approx(0.5)
    expected = np.array([877501.93, 2071786.21, 2482646.18]) * 1e-9
    assert craft.principal_moments_kg_m2 == pytest.approx(expected, rel=1e-6)


def test_positive_tensor_flips_off_diagonals():
    craft = pilot1_baseline()
    assert craft.inertia_kg_m2[0, 1] == pytest.approx(-2559.92e-9)
    assert craft.inertia_kg_m2[1, 2] == pytest.approx(-20741.53e-9)


def test_json_round_trip(tmp_path):
    craft = pilot1_baseline()
    path = tmp_path / "craft.json"
    craft.save(path)
    loaded = SpacecraftConfig.load(path)
    assert loaded.name == craft.name
    assert np.allclose(loaded.inertia_kg_m2, craft.inertia_kg_m2)
    assert np.allclose(loaded.com_m, craft.com_m)


def test_shipped_baseline_file_loads():
    loaded = SpacecraftConfig.load(DEFAULT_DIRECTORY / "pilot1_v2.json")
    assert np.allclose(loaded.inertia_kg_m2, pilot1_baseline().inertia_kg_m2)


def test_validation_errors(tmp_path):
    good = pilot1_baseline().to_dict()
    for key, value in (("mass_kg", -1), ("inertia_kg_m2", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                       ("inertia_kg_m2", [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]),
                       ("inertia_kg_m2", [[1, 0, 0], [0, 1, 0], [0, 0, 5]]),
                       ("schema_version", 99)):
        bad = dict(good, **{key: value})
        with pytest.raises(SpacecraftError):
            SpacecraftConfig.from_dict(bad)
    missing = dict(good)
    del missing["mass_kg"]
    with pytest.raises(SpacecraftError):
        SpacecraftConfig.from_dict(missing)
    broken = tmp_path / "bad.json"
    broken.write_text("{not json")
    with pytest.raises(SpacecraftError):
        SpacecraftConfig.load(broken)


def test_from_cad_negative_mass_rejected():
    with pytest.raises(SpacecraftError):
        from_cad("x", -5, [0, 0, 0], np.eye(3) * 100)

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import astropy.units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, EarthLocation, ITRS, TEME
from astropy.time import Time
import numpy as np
import pytest
import spiceypy as spice

from openadcs_sim import EnvironmentSimulator, HuiraSunModel, SGP4Propagator, XYZGeomagModel
from openadcs_sim.common import utc_time, vector3
from openadcs_sim.orbit import OrbitState
from openadcs_sim.sun import SunState

# Vallado's published satellite 00005 verification case.
LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
NOW = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


def test_sgp4_vallado_reference_in_teme():
    model = SGP4Propagator(LINE1, LINE2)
    state = model.propagate(model.epoch_utc)
    fixed = ITRS(
        CartesianRepresentation(
            state.position_itrs_m * u.m,
            differentials=CartesianDifferential(state.velocity_itrs_m_s * u.m / u.s),
        ),
        obstime=Time(state.time_utc),
    )
    teme = fixed.transform_to(TEME(obstime=Time(state.time_utc)))
    np.testing.assert_allclose(
        teme.cartesian.xyz.to_value(u.km),
        [7022.46529266, -1400.08296755, 0.03995155], atol=0.001, rtol=0,
    )
    np.testing.assert_allclose(
        teme.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s),
        [1.893841015, 6.405893759, 4.534807250], atol=1e-6, rtol=0,
    )


def test_invalid_and_stale_tle():
    with pytest.raises(ValueError):
        SGP4Propagator(LINE1[:-1] + "0", LINE2)
    with pytest.raises(ValueError, match="different satellites"):
        changed = LINE2[:2] + "00006" + LINE2[7:68]
        checksum = sum(int(c) if c.isdigit() else int(c == "-") for c in changed) % 10
        SGP4Propagator(LINE1, changed + str(checksum))
    model = SGP4Propagator(LINE1, LINE2)
    with pytest.raises(ValueError, match="TLE age"):
        model.propagate(model.epoch_utc + timedelta(days=8))
    with pytest.raises(ValueError, match="timezone-aware"):
        model.propagate(datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="69-character"):
        SGP4Propagator(LINE1[:-1] + " ", LINE2)


def test_sgp4_failure_is_explicit():
    model = SGP4Propagator(LINE1, LINE2)

    class DecayedSatellite:
        def sgp4(self, jd1, jd2):
            return 6, (0, 0, 0), (0, 0, 0)

    model._satellite = DecayedSatellite()
    with pytest.raises(RuntimeError, match="SGP4 failed.*decayed"):
        model.propagate(model.epoch_utc)


def test_utc_normalization():
    assert utc_time(NOW.astimezone(timezone(timedelta(hours=-3)))) == NOW


@pytest.mark.parametrize("value", [[1, 2], [1, 2, float("nan")], [[1], [2], [3]]])
def test_invalid_vector(value):
    with pytest.raises(ValueError):
        vector3(value)


def test_vectors_are_copied_and_read_only():
    original = np.array([1.0, 2.0, 3.0])
    result = vector3(original)
    original[0] = 99
    assert result[0] == 1
    with pytest.raises(ValueError):
        result[0] = 2


def test_wmm2025_reference():
    # NOAA WMM2025 test values: decimal year 2025.0, 89 N, -121 E, 28 km.
    position = EarthLocation.from_geodetic(-121 * u.deg, 89 * u.deg, 28000 * u.m)
    xyz = vector3([part.to_value(u.m) for part in position.geocentric])
    field = XYZGeomagModel().field(datetime(2025, 1, 1, tzinfo=timezone.utc), xyz)
    lat, lon = np.deg2rad([89, -121])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    down = np.array([-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)])
    ned_nt = np.array([north @ field, east @ field, down @ field]) * 1e9
    np.testing.assert_allclose(ned_nt, [-255.388723, -1482.460628, 56194.288771], atol=0.5, rtol=0)


@pytest.mark.parametrize("year", [2024, 2030])
def test_wmm_epoch_rejected(year):
    with pytest.raises(ValueError, match="WMM2025"):
        XYZGeomagModel().field(datetime(year, 1, 1, tzinfo=timezone.utc), vector3([6778137, 0, 0]))


@pytest.mark.parametrize("altitude", [-2000, 851000])
def test_wmm_altitude_rejected(altitude):
    with pytest.raises(ValueError, match="altitude"):
        XYZGeomagModel().field(NOW, vector3([6378137 + altitude, 0, 0]))


@pytest.mark.parametrize("altitude", [0, 400000, 850000])
def test_wmm_altitudes_supported(altitude):
    field = XYZGeomagModel().field(NOW, vector3([6378137 + altitude, 0, 0]))
    assert 1e-6 < np.linalg.norm(field) < 1e-4


def test_last_wmm_day_and_poles():
    timestamp = datetime(2029, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    for z in [6356752.314245, -6356752.314245]:
        assert np.all(np.isfinite(XYZGeomagModel().field(timestamp, vector3([0, 0, z]))))


def test_environment_composition_and_serialization():
    position = vector3([6778137, 0, 0])
    velocity = vector3([0, 7000, 0])

    class Orbit:
        def propagate(self, timestamp):
            return OrbitState(timestamp, position, velocity)

    class Sun:
        def sample(self, timestamp, spacecraft):
            np.testing.assert_array_equal(spacecraft, position)
            assert timestamp == NOW
            return SunState(vector3([1, 0, 0]), 1.5e11)

    sample = EnvironmentSimulator(Orbit(), XYZGeomagModel(), Sun()).sample(NOW)
    data = json.loads(json.dumps(sample.to_dict(), allow_nan=False))
    assert data["frame"] == "ITRS"
    assert data["schema_version"] == 1
    assert data["position_m"] == position.tolist()
    assert data["velocity_m_s"] == velocity.tolist()
    assert data["sun_direction"] == [1, 0, 0]
    assert data["sun_distance_m"] == 1.5e11
    assert len(data["magnetic_field_t"]) == 3


def test_missing_kernels_fail():
    with pytest.raises(ValueError, match="Supply"):
        HuiraSunModel([])
    with pytest.raises(FileNotFoundError):
        HuiraSunModel([Path("does-not-exist.bsp")])


@pytest.mark.integration
def test_real_huira_spice_environment():
    directory = Path(os.environ.get("OPENADCS_KERNELS", "simulation/data/kernels"))
    kernels = [directory / name for name in [
        "naif0012.tls", "de440s.bsp", "earth_latest_high_prec.bpc",
    ]]
    if not all(path.is_file() for path in kernels):
        pytest.skip("Download kernels using simulation/tools/fetch_kernels.py")
    model = HuiraSunModel(kernels)
    state = model.sample(NOW, vector3([6778137, 0, 0]))
    et = spice.str2et(NOW.strftime("%Y-%m-%dT%H:%M:%S"))
    sun_j2000_km, _ = spice.spkpos("SUN", et, "J2000", "NONE", "EARTH")
    expected = spice.pxform("J2000", "ITRF93", et) @ sun_j2000_km * 1000 - [6778137, 0, 0]
    np.testing.assert_allclose(
        state.direction_itrs, expected / np.linalg.norm(expected), rtol=0, atol=1e-14,
    )
    assert state.distance_m == pytest.approx(np.linalg.norm(expected), rel=1e-14)
    assert np.linalg.norm(state.direction_itrs) == pytest.approx(1, abs=1e-14)
    assert 1.45e11 < state.distance_m < 1.53e11
    assert model.sun.valid()
    huira_position = model.spacecraft_frame.get_static_position()
    np.testing.assert_allclose(
        [huira_position.x, huira_position.y, huira_position.z], [6778137, 0, 0], atol=1e-6,
    )
    opposite = model.sample(NOW, vector3([-6778137, 0, 0]))
    assert not np.array_equal(state.direction_itrs, opposite.direction_itrs)


@pytest.mark.integration
def test_cli_real_models(tmp_path):
    directory = Path(os.environ.get("OPENADCS_KERNELS", "simulation/data/kernels")).resolve()
    if not (directory / "de440s.bsp").is_file():
        pytest.skip("Download SPICE kernels first")
    # Synthetic LEO elements for pipeline testing, NOT a current operational TLE.
    line1 = LINE1[:18] + "26282.50000000" + LINE1[32:68]
    line2 = LINE2[:26] + "0010000" + LINE2[33:52] + "15.50000000" + LINE2[63:68]

    def checksum(line):
        return line + str(sum(int(c) if c.isdigit() else int(c == "-") for c in line) % 10)

    tle = tmp_path / "synthetic.tle"
    tle.write_text(checksum(line1) + "\n" + checksum(line2) + "\n", encoding="ascii")
    process = subprocess.run(
        [
            sys.executable, "-m", "openadcs_sim.cli",
            "--tle", str(tle), "--kernels", str(directory),
            "--start", NOW.isoformat(), "--step-seconds", "10", "--samples", "3",
        ],
        check=True, capture_output=True, text=True,
    )
    records = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(records) == 3
    for index, record in enumerate(records):
        assert datetime.fromisoformat(record["time_utc"]) == NOW + timedelta(seconds=index * 10)
        assert record["frame"] == "ITRS"
        assert 6.7e6 < np.linalg.norm(record["position_m"]) < 7e6
        assert 1e-6 < np.linalg.norm(record["magnetic_field_t"]) < 1e-4
        assert np.linalg.norm(record["sun_direction"]) == pytest.approx(1, abs=1e-14)
    assert records[0]["position_m"] != records[-1]["position_m"]

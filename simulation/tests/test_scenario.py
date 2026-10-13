import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import time

import numpy as np
import pytest

from openadcs_sim.environment import EnvironmentSample
from openadcs_sim.common import vector3
from openadcs_sim.gui_runner import FinishedEvent, RunWorker, SampleEvent
from openadcs_sim.scenario import (
    KERNEL_NAMES, Scenario, decode_sample, export_results, parse_tle, sample_readout,
    sampling_count, training_tle,
)

START = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


@pytest.fixture
def scenario(tmp_path):
    for name in KERNEL_NAMES:
        (tmp_path / name).touch()
    return Scenario(parse_tle(training_tle()), tmp_path, START, 60, 10)


@pytest.fixture
def sample():
    return EnvironmentSample(
        START, vector3([6778137, 0, 0]), vector3([0, 7000, 0]),
        vector3([1e-5, 2e-5, 3e-5]), vector3([1, 0, 0]), 149597870700,
    )


@pytest.mark.parametrize("duration,step,count", [
    (0, 10, 1), (60, 10, 7), (61, 10, 7), (5, 10, 1), (0.3, 0.1, 4),
])
def test_sampling_grid(duration, step, count):
    assert sampling_count(duration, step) == count


@pytest.mark.parametrize("duration,step", [
    (-1, 1), (float("nan"), 1), (float("inf"), 1),
    (60, 0), (60, -1), (60, float("nan")), (60, 1e-8),
])
def test_invalid_sampling(duration, step):
    with pytest.raises(ValueError):
        sampling_count(duration, step)


def test_tle_parsing_and_training_epoch(scenario):
    assert scenario.sample_count == 7
    assert scenario.end_utc == START + timedelta(seconds=60)
    assert parse_tle(training_tle()) == parse_tle("\n".join(parse_tle(training_tle())) + "\n")
    with pytest.raises(ValueError, match="two TLE"):
        parse_tle("not a TLE")


def test_scenario_coverage_and_missing_data(scenario, tmp_path):
    with pytest.raises(ValueError, match="seven days"):
        Scenario(scenario.tle, scenario.kernel_directory, START + timedelta(days=8), 0, 10)
    with pytest.raises(ValueError, match="2025-2029"):
        Scenario(scenario.tle, scenario.kernel_directory, START.replace(year=2030), 0, 10)
    with pytest.raises(ValueError, match="timezone-aware"):
        Scenario(scenario.tle, scenario.kernel_directory, START.replace(tzinfo=None), 0, 10)
    with pytest.raises(ValueError, match="Missing kernel"):
        Scenario(scenario.tle, tmp_path / "missing", START, 0, 10)


def test_readout_units_and_worker_decode(sample):
    row = sample_readout(sample)
    assert row.latitude_deg == pytest.approx(0)
    assert row.longitude_deg == pytest.approx(0)
    assert row.altitude_km == pytest.approx(400)
    assert row.speed_fixed_km_s == pytest.approx(7)
    assert row.magnetic_magnitude_ut == pytest.approx(np.sqrt(1400))
    assert row.sun_distance_au == 1
    restored = decode_sample(json.dumps(sample.to_dict()))
    np.testing.assert_array_equal(restored.position_itrs_m, sample.position_itrs_m)
    for record in ({}, {"frame": "TEME", "schema_version": 1}):
        with pytest.raises(ValueError):
            decode_sample(json.dumps(record))
    bad = sample.to_dict()
    bad["sun_direction"] = [0, 0, 0]
    with pytest.raises(ValueError, match="Sun geometry"):
        decode_sample(json.dumps(bad))


def test_exports_keep_units_scenario_and_partial_status(scenario, sample, tmp_path):
    json_path, csv_path = tmp_path / "results.json", tmp_path / "results.csv"
    export_results(json_path, scenario, [sample], "cancelled")
    record = json.loads(json_path.read_text())
    assert record["status"] == "cancelled"
    assert record["scenario"]["tle"] == list(scenario.tle)
    assert record["samples"][0]["position_m"][0] == 6778137
    export_results(csv_path, scenario, [sample], "failed")
    with csv_path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["run_status"] == "failed"
    assert row["frame"] == "ITRS"
    assert float(row["x_km"]) == pytest.approx(6778.137)
    assert float(row["by_ut"]) == pytest.approx(20)
    assert float(row["altitude_km"]) == pytest.approx(400)
    assert float(row["sun_distance_au"]) == 1
    with pytest.raises(ValueError, match="no samples"):
        export_results(json_path, scenario, [], "failed")
    with pytest.raises(ValueError, match=".csv or .json"):
        export_results(tmp_path / "results.txt", scenario, [sample], "completed")


def collect(worker):
    events = []
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            event = worker.events.get(timeout=0.1)
        except queue.Empty:
            continue
        events.append(event)
        if isinstance(event, FinishedEvent):
            worker.thread.join(timeout=5)
            assert not worker.thread.is_alive()
            return events
    worker.cancel()
    raise AssertionError("Simulation worker did not finish in 40 seconds")


def test_worker_reports_invalid_kernels(scenario):
    worker = RunWorker(scenario)
    worker.start()
    events = collect(worker)
    assert events[-1].status == "failed"
    assert events[-1].diagnostics


def test_worker_can_cancel_during_startup(scenario):
    worker = RunWorker(scenario)
    worker.cancel()
    worker.start()
    events = collect(worker)
    assert events[-1].status == "cancelled"


@pytest.mark.integration
def test_worker_real_models():
    directory = Path("simulation/data/kernels").resolve()
    if not all((directory / name).is_file() for name in KERNEL_NAMES):
        pytest.skip("Download kernels first")
    scenario = Scenario(parse_tle(training_tle()), directory, START, 20, 10)
    worker = RunWorker(scenario)
    worker.start()
    events = collect(worker)
    samples = [event for event in events if isinstance(event, SampleEvent)]
    assert len(samples) == 3
    assert events[-1].status == "completed"
    assert samples[-1].sample.time_utc == START + timedelta(seconds=20)
    assert 300 < samples[0].readout.altitude_km < 500

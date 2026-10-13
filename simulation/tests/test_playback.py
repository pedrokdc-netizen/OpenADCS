from datetime import datetime, timedelta, timezone

import pytest

from openadcs_sim.playback import PlaybackClock

START = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)


@pytest.fixture
def clock():
    playback = PlaybackClock()
    playback.load([START + timedelta(seconds=value) for value in (0, 10, 20, 30)])
    return playback


def test_acceleration_uses_wall_clock_not_frame_count(clock):
    clock.play(1, 100)
    assert clock.advance(100.25) == 1
    assert clock.elapsed == 15
    assert clock.playing
    assert clock.advance(101) == 3
    assert clock.elapsed == 30
    assert not clock.playing
    assert clock.time_utc == START + timedelta(seconds=30)


def test_reverse_play_start_at_end_and_stop_at_start(clock):
    clock.play(-1, 0)
    assert clock.elapsed == 30
    assert clock.advance(.25) == 1
    assert clock.elapsed == 15
    assert clock.advance(1) == 0
    assert not clock.playing


def test_pause_resume_does_not_skip_paused_wall_time(clock):
    clock.play(1, 0)
    clock.pause(.1)
    assert clock.elapsed == pytest.approx(6)
    clock.play(1, 100)
    clock.advance(100.1)
    assert clock.elapsed == pytest.approx(12)


def test_rate_change_preserves_elapsed_time(clock):
    clock.play(1, 0)
    clock.set_rate(10, .1)
    assert clock.elapsed == pytest.approx(6)
    clock.advance(.5)
    assert clock.elapsed == pytest.approx(10)
    clock.play(-1, .5)
    clock.advance(.7)
    assert clock.elapsed == pytest.approx(8)


def test_step_seek_and_nonuniform_recorded_samples(clock):
    clock.seek(19, 0)
    assert clock.index == 1
    assert clock.step(1, 0) == 2
    assert clock.elapsed == 20
    assert clock.step(-1, 0) == 1
    clock.load([START, START + timedelta(seconds=3), START + timedelta(seconds=18)])
    clock.seek(10, 0)
    assert clock.index == 1
    clock.seek(999, 0)
    assert clock.index == 2
    clock.seek(-1, 0)
    assert clock.index == 0


def test_new_dataset_resets_clock_and_disallows_one_sample_playback(clock):
    clock.play(1, 0)
    clock.load([START])
    assert not clock.playing
    assert clock.elapsed == 0
    assert clock.index == 0
    with pytest.raises(ValueError, match="two recorded"):
        clock.play(1, 0)
    clock.load(())
    with pytest.raises(ValueError, match="No recorded"):
        _ = clock.index


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_speed_rejected(clock, rate):
    with pytest.raises(ValueError, match="finite positive"):
        clock.set_rate(rate, 0)


def test_invalid_timestamps_and_monotonic_clock(clock):
    with pytest.raises(ValueError, match="strictly increasing"):
        clock.load([START, START])
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.load([START.replace(tzinfo=None)])
    clock.play(1, 10)
    with pytest.raises(ValueError, match="monotonic"):
        clock.advance(9)

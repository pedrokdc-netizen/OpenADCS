from bisect import bisect_right
from datetime import datetime, timedelta
import math
from typing import Sequence

from .common import utc_time


class PlaybackClock:
    """Wall-clock-based replay of recorded timestamps, independent of render FPS."""

    def __init__(self) -> None:
        self.timestamps: tuple[datetime, ...] = ()
        self.offsets: tuple[float, ...] = ()
        self.elapsed = 0.0
        self.rate = 60.0
        self.direction = 1
        self.playing = False
        self._last_wall: float | None = None

    def load(self, timestamps: Sequence[datetime]) -> None:
        times = tuple(utc_time(value) for value in timestamps)
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("Playback timestamps must be strictly increasing")
        self.timestamps = times
        self.offsets = tuple((value - times[0]).total_seconds() for value in times)
        self.elapsed = 0
        self.playing = False
        self._last_wall = None

    @property
    def duration(self) -> float:
        return self.offsets[-1] if self.offsets else 0.0

    @property
    def index(self) -> int:
        if not self.offsets:
            raise ValueError("No recorded samples are loaded")
        return max(0, bisect_right(self.offsets, self.elapsed) - 1)

    @property
    def time_utc(self) -> datetime:
        if not self.timestamps:
            raise ValueError("No recorded samples are loaded")
        return self.timestamps[0] + timedelta(seconds=self.elapsed)

    def advance(self, now: float) -> int:
        if not math.isfinite(now):
            raise ValueError("Playback wall time must be finite")
        if self.playing and self._last_wall is not None:
            delta = now - self._last_wall
            if delta < 0:
                raise ValueError("Playback requires a monotonic clock")
            self.elapsed = min(self.duration, max(0, self.elapsed + delta * self.rate * self.direction))
            self._last_wall = now
            if (self.direction == 1 and self.elapsed >= self.duration) or (self.direction == -1 and self.elapsed <= 0):
                self.playing = False
                self._last_wall = None
        return self.index

    def play(self, direction: int, now: float) -> None:
        if direction not in (-1, 1):
            raise ValueError("Playback direction must be forward or reverse")
        if self.duration <= 0:
            raise ValueError("Playback needs at least two recorded timestamps")
        if not math.isfinite(now):
            raise ValueError("Playback wall time must be finite")
        if self.playing:
            self.advance(now)
        if direction == 1 and self.elapsed >= self.duration:
            self.elapsed = 0
        elif direction == -1 and self.elapsed <= 0:
            self.elapsed = self.duration
        self.direction = direction
        self.playing = True
        self._last_wall = now

    def pause(self, now: float) -> None:
        if self.playing:
            self.advance(now)
        self.playing = False
        self._last_wall = None

    def set_rate(self, rate: float, now: float) -> None:
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("Playback speed must be a finite positive multiplier")
        if self.playing:
            self.advance(now)
        self.rate = rate

    def seek(self, elapsed: float, now: float) -> None:
        if not math.isfinite(elapsed) or not math.isfinite(now):
            raise ValueError("Playback seek times must be finite")
        self.elapsed = min(self.duration, max(0, elapsed))
        if self.playing:
            self._last_wall = now
            if (self.direction == 1 and self.elapsed >= self.duration) or (self.direction == -1 and self.elapsed <= 0):
                self.playing = False
                self._last_wall = None

    def step(self, delta: int, now: float) -> int:
        self.pause(now)
        index = max(0, min(len(self.timestamps) - 1, self.index + delta))
        self.seek(self.offsets[index], now)
        return index

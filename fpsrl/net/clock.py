"""Client-side playback clock for remote players.

Snapshots over the internet arrive in bursts: TCP and the route both add jitter. Rebasing the
playback time on every arrival (`server_time + time_since_arrival`) makes the clock lurch with
that jitter, and a fixed short delay runs the buffer dry whenever a packet is late, so remote
players freeze and then jump. This clock instead:

* estimates the host clock from the *least delayed* recent snapshot (the delay floor), so one
  late packet does not move it,
* measures jitter as how much later than that floor recent snapshots arrived, and keeps the
  playback point `interval + jitter` behind the newest data (bounded by `MIN_DELAY`/`MAX_DELAY`),
* runs smoothly at real speed and steers toward that target by at most ±`MAX_SKEW` in rate,
  only snapping on a large jump (host reset, long stall).
"""
from __future__ import annotations

import collections


class InterpClock:
    WINDOW = 2.0  # seconds of arrival history for the delay floor and jitter
    MIN_DELAY = 0.035
    MAX_DELAY = 0.15  # the host rewinds hits at most 0.25 s, leave room for the one-way latency
    MAX_SKEW = 0.08  # playback rate stays within 1 +- this while catching up / backing off
    SNAP = 0.25

    def __init__(self):
        self.samples: collections.deque = collections.deque()  # (local time, server - local offset)
        self.interval = 1.0 / 30.0
        self.delay = 0.06
        self.time: float | None = None  # playback time in host seconds
        self.now = 0.0
        self.last_server = None
        self.last_local = None

    def reset(self):
        self.samples.clear()
        self.time = None
        self.last_server = self.last_local = None

    def on_snapshot(self, server_time: float, now: float):
        if self.last_server is not None and server_time < self.last_server - 0.5:
            self.reset()  # the host restarted the match: its clock went back to 0
        if self.last_server is not None and server_time > self.last_server:
            gap = server_time - self.last_server
            self.interval += (min(gap, 0.2) - self.interval) * 0.05
        self.last_server, self.last_local = server_time, now
        self.samples.append((now, server_time - now))
        while self.samples and self.samples[0][0] < now - self.WINDOW:
            self.samples.popleft()

    def _floor(self):
        return max(o for _, o in self.samples)

    def jitter(self) -> float:
        """95th percentile of how late recent snapshots were compared with the fastest one."""
        if len(self.samples) < 4:
            return 0.02
        best = self._floor()
        late = sorted(best - o for _, o in self.samples)
        return late[min(len(late) - 1, int(len(late) * 0.95))]

    def target(self, now: float) -> float | None:
        if not self.samples:
            return None
        self.delay = min(self.MAX_DELAY, max(self.MIN_DELAY, self.interval + self.jitter() * 1.1 + 0.005))
        return now + self._floor() - self.delay

    def advance(self, now: float) -> float | None:
        """Playback time for this frame (host seconds), or None before the first snapshot."""
        tgt = self.target(now)
        if tgt is None:
            return None
        if self.time is None or abs(tgt - self.time) > self.SNAP:
            self.time, self.now = tgt, now
            return self.time
        dt = max(0.0, now - self.now)
        self.now = now
        err = tgt - self.time
        rate = 1.0 + max(-self.MAX_SKEW, min(self.MAX_SKEW, err * 4.0))
        self.time += dt * rate
        return self.time

    def peek(self, now: float) -> float:
        """Best current playback time without advancing (e.g. for the lag compensation stamp)."""
        if self.time is not None:
            return self.time
        tgt = self.target(now)
        return tgt if tgt is not None else (self.last_server or 0.0)

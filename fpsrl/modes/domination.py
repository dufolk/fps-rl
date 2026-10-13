"""Domination: the points open one at a time, best of three.

After a preparation phase (everyone frozen) a random point opens. While exactly one team stands
in it, that team scores a point every `tick` seconds no matter how many are inside; both teams
inside contest it and nobody scores. The first team to `target` wins the point, or the higher
score when its `point_time` runs out (a tie wins nothing). The second point (random among the
rest) opens `gap` seconds later, the last one right after the second ends. Two points win the
match, so 2:0 ends it early.
"""
from __future__ import annotations

import math

import numpy as np

from .. import config as C
from ..geometry import poly_center
from .base import GameMode

PREP, LIVE, GAP, OVER = "prep", "live", "gap", "over"


class Domination(GameMode):
    name = "dom"
    spawn_protect = 0.0       # the prep phase already blocks damage; invuln is for respawns
    respawn_protect = 3.0
    DEFAULTS = {"prep_time": 10.0, "point_time": 220.0, "tick": 0.9, "target": 100, "gap": 3.0,
                "wins_needed": 2, "respawn_time": 5.0, "regen_delay": 4.0, "regen_rate": 25.0}

    def reset(self):
        self.points = self.w.map.objectives.get("points", [])
        self.centers = [poly_center(p["poly"]) for p in self.points]
        n = len(self.points)
        self.order = [int(i) for i in self.w.rng.permutation(n)]
        self.stage = 0
        self.active = -1
        self.cap = [0, 0]        # score of the open point
        self.acc = 0.0           # time the sole occupant has held it since its last point
        self.acc_team = -1
        self.t0 = 0.0            # when the open point opened
        self.result = [-1] * n   # winner per point: team, -1 not played / nobody
        self.played = [False] * n
        self.wins = [0, 0]
        self.present = (0, 0)
        self.phase = PREP
        self.phase_end = self.p["prep_time"]
        self.w.cfg.respawn_time = self.p["respawn_time"]
        super().reset()
        for a in self.w.agents:
            a.frozen = True

    # ------------------------------------------------------------------ flow
    def _open(self, i: int):
        self.active = i
        self.cap = [0, 0]
        self.acc, self.acc_team = 0.0, -1
        self.t0 = self.w.time
        self.phase = LIVE
        self.w._emit("point_active", point=i, name=self.points[i]["name"], stage=self.stage)

    def _finish(self, team: int, reason: str):
        w = self.w
        i = self.active
        self.result[i] = team
        self.played[i] = True
        if team >= 0:
            self.wins[team] += 1
        w._emit("point_captured", point=i, team=team, name=self.points[i]["name"], reason=reason,
                cap=list(self.cap), wins=list(self.wins))
        self.active = -1
        self.present = (0, 0)
        self.stage += 1
        if max(self.wins) >= self.p["wins_needed"] or self.stage >= len(self.order):
            self.phase = OVER
            w.done = True
            w.winner = -1 if self.wins[0] == self.wins[1] else int(np.argmax(self.wins))
        elif self.stage == 1:
            self.phase = GAP
            self.phase_end = w.time + self.p["gap"]
        else:
            self._open(self.order[self.stage])

    def pre_step(self, dt: float):
        w = self.w
        if self.phase == PREP:
            if w.time >= self.phase_end:
                for a in w.agents:
                    a.frozen = False
                if self.order:
                    self._open(self.order[0])
            return
        if self.phase == GAP:
            if w.time >= self.phase_end:
                self._open(self.order[self.stage])
            return
        if self.phase != LIVE:
            return
        z = self.points[self.active]
        n = [0, 0]
        for a in w.agents:
            if a.alive and self.in_zone(a.pos, z):
                n[a.team] += 1
        self.present = (n[0], n[1])
        if bool(n[0]) != bool(n[1]):
            team = 0 if n[0] else 1
            if self.acc_team != team:
                self.acc, self.acc_team = 0.0, team
            self.acc += dt
            while self.acc >= self.p["tick"] - 1e-9:
                self.acc -= self.p["tick"]
                self.cap[team] += 1
            if self.cap[team] >= self.p["target"]:
                self.cap[team] = int(self.p["target"])
                self._finish(team, "capture")
                return
        else:
            self.acc, self.acc_team = 0.0, -1
        if w.time - self.t0 >= self.p["point_time"]:
            c0, c1 = self.cap
            self._finish(-1 if c0 == c1 else (0 if c0 > c1 else 1), "time")

    def post_step(self, dt: float):
        # out of combat, health comes back on its own; a fresh hit restarts the delay
        delay, rate = self.p["regen_delay"], self.p["regen_rate"]
        now = self.w.time
        for a in self.w.agents:
            if a.alive and a.hp < C.MAX_HP and now - a.last_damage_time >= delay:
                a.hp = min(float(C.MAX_HP), a.hp + rate * dt)

    def can_respawn(self, a) -> bool:
        return self.phase != OVER

    def time_left(self) -> float:
        """Seconds left in the current phase (prep / gap countdown, or the open point's clock)."""
        if self.phase == LIVE:
            return max(0.0, self.p["point_time"] - (self.w.time - self.t0))
        if self.phase in (PREP, GAP):
            return max(0.0, self.phase_end - self.w.time)
        return 0.0

    # ------------------------------------------------------------------ info
    def score(self) -> list:
        return list(self.wins)

    def state(self) -> dict:
        return {"phase": self.phase, "phase_end": round(self.phase_end, 3), "order": list(self.order),
                "stage": self.stage, "active": self.active, "cap": list(self.cap), "acc": round(self.acc, 3),
                "acc_team": self.acc_team, "t0": round(self.t0, 3), "result": list(self.result),
                "played": list(self.played), "wins": list(self.wins), "present": list(self.present)}

    def apply_state(self, st: dict):
        if not st:
            return
        self.phase = st["phase"]
        self.phase_end = st["phase_end"]
        self.order = list(st["order"])
        self.stage = st["stage"]
        self.active = st["active"]
        self.cap = list(st["cap"])
        self.acc = st["acc"]
        self.acc_team = st["acc_team"]
        self.t0 = st["t0"]
        self.result = list(st["result"])
        self.played = list(st["played"])
        self.wins = list(st["wins"])
        self.present = tuple(st["present"])

    def obs(self, a) -> np.ndarray:
        """Per point (max 3): state (1 open, -1 decided, 0 waiting), score lead toward own team
        (open: cap difference / target, decided: +1 won / -1 lost / 0 nobody), distance, bearing sin."""
        o = np.zeros(12, dtype=np.float32)
        yr = math.radians(a.yaw)
        own = a.team
        for i in range(min(3, len(self.points))):
            c = self.centers[i]
            dx, dy = c[0] - a.pos[0], c[1] - a.pos[1]
            dist = math.hypot(dx, dy)
            if i == self.active:
                o[i * 4 + 0] = 1.0
                o[i * 4 + 1] = (self.cap[own] - self.cap[1 - own]) / self.p["target"]
            elif self.played[i]:
                o[i * 4 + 0] = -1.0
                r = self.result[i]
                o[i * 4 + 1] = 0.0 if r < 0 else (1.0 if r == own else -1.0)
            o[i * 4 + 2] = min(dist / 60.0, 1.0)
            o[i * 4 + 3] = math.sin(math.atan2(dy, dx) - yr) if dist > 1e-3 else 0.0
        return o

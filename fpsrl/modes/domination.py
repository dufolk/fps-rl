"""Domination: capture polygon zones, owned zones tick points, first to the score limit wins."""
from __future__ import annotations

import math

import numpy as np

from ..geometry import poly_center
from .base import GameMode


class Domination(GameMode):
    name = "dom"
    DEFAULTS = {"score_limit": 200, "capture_time": 6.0, "points_per_sec": 1.0, "respawn_time": 5.0}

    def reset(self):
        self.points = self.w.map.objectives.get("points", [])
        self.centers = [poly_center(p["poly"]) for p in self.points]
        n = len(self.points)
        self.owner = [-1] * n
        self.progress = [0.0] * n  # -1 (team 1) .. +1 (team 0)
        self.present = [(0, 0)] * n
        self.pts = [0.0, 0.0]
        self.w.cfg.respawn_time = self.p["respawn_time"]
        super().reset()

    def pre_step(self, dt: float):
        w = self.w
        for i, z in enumerate(self.points):
            n0 = n1 = 0
            for a in w.agents:
                if a.alive and self.in_zone(a.pos, z):
                    if a.team == 0:
                        n0 += 1
                    else:
                        n1 += 1
            self.present[i] = (n0, n1)
            if n0 and n1 or not (n0 or n1):
                if not (n0 or n1) and self.owner[i] == -1:
                    self.progress[i] *= math.exp(-0.25 * dt)  # abandoned partial captures decay
                continue
            team = 0 if n0 else 1
            sign = 1.0 if team == 0 else -1.0
            rate = min(2.0, 1.0 + 0.5 * (max(n0, n1) - 1)) / self.p["capture_time"]
            before = self.progress[i]
            self.progress[i] = float(np.clip(before + sign * rate * dt, -1.0, 1.0))
            if self.owner[i] not in (-1, team) and before * sign < 0 <= self.progress[i] * sign:
                w._emit("point_neutral", point=i, team=self.owner[i])
                self.owner[i] = -1
            if self.owner[i] != team and abs(self.progress[i]) >= 1.0:
                self.owner[i] = team
                w._emit("point_captured", point=i, team=team, name=z["name"])
        for o in self.owner:
            if o >= 0:
                self.pts[o] += self.p["points_per_sec"] * dt

    def post_step(self, dt: float):
        w = self.w
        lim = self.p["score_limit"]
        if max(self.pts) >= lim:
            w.done = True
            w.winner = int(np.argmax(self.pts))
        elif w.cfg.time_limit and w.time >= w.cfg.time_limit:
            w.done = True
            w.winner = -1 if int(self.pts[0]) == int(self.pts[1]) else int(np.argmax(self.pts))

    def spawn_points(self, a):
        """Prefer spawns far from enemy-held points."""
        sp = self.spawn_points_team(a.team)
        enemy = [self.centers[i] for i, o in enumerate(self.owner) if o not in (-1, a.team)]
        if not enemy:
            return sp
        scored = sorted(sp, key=lambda s: -min(np.hypot(s[0] - c[0], s[1] - c[1]) for c in enemy))
        return scored[:max(3, len(scored) // 2)]

    def score(self) -> list:
        return [int(self.pts[0]), int(self.pts[1])]

    def state(self) -> dict:
        return {"owner": list(self.owner), "progress": [round(p, 3) for p in self.progress],
                "pts": [round(p, 2) for p in self.pts], "present": [list(p) for p in self.present]}

    def apply_state(self, st: dict):
        if not st:
            return
        self.owner = list(st.get("owner", self.owner))
        self.progress = list(st.get("progress", self.progress))
        self.pts = list(st.get("pts", self.pts))
        self.present = [tuple(p) for p in st.get("present", self.present)]

    def obs(self, a) -> np.ndarray:
        """Per point (max 3): relative owner, progress toward own team, distance, bearing sin."""
        o = np.zeros(12, dtype=np.float32)
        yr = math.radians(a.yaw)
        for i in range(min(3, len(self.points))):
            own = self.owner[i]
            c = self.centers[i]
            dx, dy = c[0] - a.pos[0], c[1] - a.pos[1]
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx) - yr
            o[i * 4 + 0] = 0.0 if own == -1 else (1.0 if own == a.team else -1.0)
            o[i * 4 + 1] = self.progress[i] * (1.0 if a.team == 0 else -1.0)
            o[i * 4 + 2] = min(dist / 60.0, 1.0)
            o[i * 4 + 3] = math.sin(ang) if dist > 1e-3 else 0.0
        return o

"""Elimination (歼灭): round based team wipe, the only game mode of the experiment environment.

Phases: freeze (can look, cannot move or shoot) -> live -> over -> next round. A round ends when one
team is wiped out; on the clock the side with more players alive wins, equal numbers are a draw.
Every round everyone respawns with the full fixed loadout (no economy). Sides swap after
`half_rounds`; the first team to `win_rounds` wins the match.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

ATT, DEF = 0, 1  # map sides (index into map.spawns)
_ROUND_UID = itertools.count(1)
PHASES = ("freeze", "live", "over")


class Elimination:
    name = "elim"
    DEFAULTS = {"freeze_time": 5.0, "round_time": 115.0, "post_time": 5.0, "win_rounds": 13, "half_rounds": 12}

    def __init__(self, world):
        self.w = world
        self.p = {**self.DEFAULTS, **(world.cfg.mode_params or {})}
        self.swapped = 0

    # ------------------------------------------------------------------ sides
    def side(self, team: int) -> int:
        return team ^ self.swapped

    def team_of_side(self, side: int) -> int:
        return side ^ self.swapped

    def members(self, team: int, alive_only=False):
        return [a for a in self.w.agents if a.team == team and (a.alive or not alive_only)]

    def spawn_points_team(self, team: int):
        return list(self.w.map.spawns[self.side(team)])

    # ------------------------------------------------------------------ rounds
    def reset(self):
        self.round_no = 0
        self.wins = [0, 0]
        self.swapped = 0
        self.phase = "over"
        self.phase_end = 0.0
        self.last_result = None
        self.start_round()

    def start_round(self):
        w = self.w
        w.clear_round()
        self.round_no += 1
        self.round_uid = next(_ROUND_UID)  # unique across matches, for per-round caches
        for team in (0, 1):
            members = self.members(team)
            sp = self.spawn_points_team(team)
            order = w.rng.permutation(len(sp))
            for k, a in enumerate(members):
                w._spawn_at(a, sp[order[k % len(sp)]])
                a.frozen = True
        self.phase = "freeze"
        self.phase_end = w.time + self.p["freeze_time"]
        w._emit("round_start", round=self.round_no, wins=list(self.wins), swapped=self.swapped)

    def begin_live(self):
        for a in self.w.agents:
            a.frozen = False
        self.phase = "live"
        self.phase_end = self.w.time + self.p["round_time"]
        self.w._emit("round_live", round=self.round_no)

    def pre_step(self, dt: float):
        w = self.w
        if self.phase == "freeze" and w.time >= self.phase_end:
            self.begin_live()
        elif self.phase == "over" and w.time >= self.phase_end and not w.done:
            self.next_round()

    def post_step(self, dt: float):
        w = self.w
        if self.phase != "live" or w.done:
            return
        att_alive = self.members(self.team_of_side(ATT), alive_only=True)
        def_alive = self.members(self.team_of_side(DEF), alive_only=True)
        if not def_alive and not att_alive:
            self.end_round(-1, "draw")
        elif not def_alive:
            self.end_round(ATT, "elimination")
        elif not att_alive:
            self.end_round(DEF, "elimination")
        elif w.time >= self.phase_end:
            if len(att_alive) > len(def_alive):
                self.end_round(ATT, "time")
            elif len(def_alive) > len(att_alive):
                self.end_round(DEF, "time")
            else:
                self.end_round(-1, "draw")

    def end_round(self, side: int, reason: str):
        w = self.w
        team = self.team_of_side(side) if side in (ATT, DEF) else -1
        if team >= 0:
            self.wins[team] += 1
        self.phase = "over"
        self.phase_end = w.time + self.p["post_time"]
        self.last_result = {"team": team, "side": side, "reason": reason}
        w._emit("round_end", team=team, side=side, reason=reason, wins=list(self.wins), round=self.round_no)
        if team >= 0 and self.wins[team] >= self.p["win_rounds"]:
            w.done = True
            w.winner = team
        elif self.round_no >= 2 * self.p["half_rounds"]:
            w.done = True
            w.winner = -1 if self.wins[0] == self.wins[1] else int(np.argmax(self.wins))

    def next_round(self):
        if self.round_no == self.p["half_rounds"]:
            self.swapped ^= 1
            self.w._emit("side_swap", swapped=self.swapped)
        self.start_round()

    # ------------------------------------------------------------------ info
    def score(self) -> list:
        return list(self.wins)

    def time_left(self) -> float:
        return max(0.0, self.phase_end - self.w.time)

    def state(self) -> dict:
        return {"phase": self.phase, "left": round(self.time_left(), 2), "round": self.round_no,
                "wins": list(self.wins), "swapped": self.swapped, "result": self.last_result}

    def obs(self, a) -> np.ndarray:
        """[side, phase one-hot(3), time left, own wins, enemy wins, round, allies alive, enemies alive,
        zone margin, bearing to safe zone center]"""
        o = np.zeros(12, dtype=np.float32)
        o[0] = 1.0 if self.side(a.team) == ATT else -1.0
        o[1 + PHASES.index(self.phase)] = 1.0
        o[4] = min(self.time_left() / 120.0, 1.0)
        target = max(1, self.p["win_rounds"])
        o[5] = self.wins[a.team] / target
        o[6] = self.wins[1 - a.team] / target
        o[7] = min(self.round_no / (2.0 * self.p["half_rounds"]), 1.0)
        n = max(1, self.w.cfg.team_size)
        o[8] = len(self.members(a.team, alive_only=True)) / n
        o[9] = len(self.members(1 - a.team, alive_only=True)) / n
        if self.w.zone:
            cx, cy = self.w.zone["center"]
            r = max(1.0, float(self.w.zone["radius"]))
            dx, dy = cx - a.pos[0], cy - a.pos[1]
            o[10] = float(np.clip((r - math.hypot(dx, dy)) / r, -1.0, 1.0))
            o[11] = math.sin(math.atan2(dy, dx) - math.radians(a.yaw))
        return o

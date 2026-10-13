"""Game mode interface. `World` owns one mode and delegates round flow, spawning,
loadouts, objectives and scoring to it."""
from __future__ import annotations

import numpy as np

from .. import config as C
from ..geometry import point_in_poly


class GameMode:
    name = "tdm"
    free_weapons = True  # primaries chosen freely from the menu (no economy)
    spawn_protect = C.SPAWN_PROTECT
    move_scale = 1.0  # multiplied into run speed; Siege is slower than the other modes
    DEFAULTS: dict = {}

    def __init__(self, world):
        self.w = world
        self.p = {**self.DEFAULTS, **(world.cfg.mode_params or {})}
        self.swapped = 0

    # ------------------------------------------------------------------ sides
    def side(self, team: int) -> int:
        """Map side (index into map.spawns / buy zones): 0 = attackers on objective maps."""
        return team ^ self.swapped

    def team_of_side(self, side: int) -> int:
        return side ^ self.swapped

    def members(self, team: int, alive_only=False):
        return [a for a in self.w.agents if a.team == team and (a.alive or not alive_only)]

    # ------------------------------------------------------------------ lifecycle
    def reset(self):
        for team in (0, 1):
            self.w.spawn_side(self.spawn_points_team(team), self.members(team))

    def pre_step(self, dt: float):
        pass

    def post_step(self, dt: float):
        self.w.tdm_check_end()

    def can_respawn(self, a) -> bool:
        return True

    def spawn_points_team(self, team: int):
        return list(self.w.map.spawns[self.side(team)])

    def spawn_points(self, a):
        return self.spawn_points_team(a.team)

    def give_loadout(self, a):
        self.w.default_loadout(a)

    def on_kill(self, victim, attacker, weapon: str, headshot: bool):
        pass

    # ------------------------------------------------------------------ interaction
    def use(self, a) -> bool:
        """F key (edge). Objective actions first, then gadget interactions (reinforce / barricade)."""
        return self.w.gadgets.use(a)

    def finish_busy(self, a, kind: str, data):
        self.w.gadgets.finish_busy(a, kind, data)

    def item_fire(self, a, ws):
        pass

    def buy_price(self, a, name: str):
        return None

    def can_buy_now(self, a) -> bool:
        return False

    # ------------------------------------------------------------------ info
    def score(self) -> list:
        return list(self.w.team_kills)

    def state(self) -> dict:
        return {}

    def apply_state(self, st: dict):
        pass

    def obs(self, a) -> np.ndarray:
        """Mode-specific observation block (fixed size 12) for RL agents."""
        return np.zeros(12, dtype=np.float32)

    def hud(self, team: int) -> dict:
        return {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def in_zone(pos, zone) -> bool:
        if not point_in_poly(pos[0], pos[1], zone["poly"]):
            return False
        return zone.get("zmin", -1e9) - 0.3 <= pos[2] <= zone.get("zmax", 1e9)


class TeamDeathmatch(GameMode):
    name = "tdm"

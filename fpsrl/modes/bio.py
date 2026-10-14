"""Classic infection: everyone starts human, then a mother zombie spreads it by knife.

Humans win if any of them are still human when the clock runs out. Zombies win when the last human
is infected. Nobody respawns during a round.
"""
from __future__ import annotations

import numpy as np

from ..weapons import MELEE, SLOT_MELEE
from .base import GameMode


class Biohazard(GameMode):
    name = "bio"
    free_weapons = True
    spawn_protect = 0.0
    DEFAULTS = {"infect_delay": 20.0, "round_time": 180.0, "post_time": 5.0, "mother_count": 1,
                "mother_hp": 6000.0, "zombie_hp": 2500.0, "mother_speed": 1.15, "zombie_speed": 1.25,
                "wins_needed": 5,
                # CF-style knockback: horizontal m/s added per point of bullet damage on a zombie
                "knockback": 0.045, "mother_knockback": 0.5, "knockback_max": 14.0, "knockback_time": 0.35}
    KNOCK_CATEGORY = {"sniper": 1.5, "rifle": 1.0, "heavy": 1.0, "smg": 0.85, "pistol": 0.7}
    SHOTGUN_KNOCK = 1.6

    def reset(self):
        self.round_no = 0
        self.wins = [0, 0]
        self.phase = "over"
        self.phase_end = 0.0
        self.round_end = 0.0
        self.w.cfg.respawn = False
        self.start_round()

    def _spots(self):
        sp = list(self.w.map.spawns.get(0, [])) + list(self.w.map.spawns.get(1, []))
        return sp or [(0.0, 0.0, 0.0, 0.0)]

    def start_round(self):
        w = self.w
        w.clear_round()
        self.round_no += 1
        for a in w.agents:
            a.team = 0
            a.role = "human"
            a.speed_mult = 1.0
        w.spawn_side(self._spots(), w.agents)
        self.phase = "prep"
        self.phase_end = w.time + self.p["infect_delay"]
        self.round_end = w.time + self.p["round_time"]
        w._emit("round_start", round=self.round_no, wins=list(self.wins))

    def _turn(self, a, mother: bool, attacker: int = -1):
        from ..sim import WeaponState
        a.team = 1
        a.role = "mother" if mother else "zombie"
        a.hp = float(self.p["mother_hp"] if mother else self.p["zombie_hp"])
        a.speed_mult = float(self.p["mother_speed"] if mother else self.p["zombie_speed"])
        a.armor = 0.0
        a.helmet = False
        a.inv = {SLOT_MELEE: WeaponState(MELEE)}
        a.slot = SLOT_MELEE
        a.ads = False
        a.throw_at = -1.0
        a.cook_until = -1.0
        a.reload_until = 0.0
        self.w._emit("infected", agent=a.id, attacker=attacker, mother=bool(mother))

    def _choose_mothers(self):
        alive = [a for a in self.w.agents if a.alive and a.role == "human"]
        if len(alive) < 2:
            return
        n = min(int(self.p["mother_count"]), len(alive) - 1)
        pick = np.atleast_1d(self.w.rng.choice(len(alive), size=n, replace=False))
        for i in pick:
            self._turn(alive[int(i)], mother=True)

    def try_infect(self, attacker, victim) -> bool:
        if self.phase != "live" or not victim.alive or not attacker.alive:
            return False
        if getattr(attacker, "role", "human") == "human" or getattr(victim, "role", "human") != "human":
            return False
        self._turn(victim, mother=False, attacker=attacker.id)
        return True

    def on_damage(self, victim, attacker, amount, direction, weapon, kind):
        """Bullets shove zombies away from the shooter; the mother resists half of it."""
        if self.phase != "live" or kind != "bullet" or direction is None or attacker is None:
            return
        if getattr(attacker, "role", "human") != "human" or getattr(victim, "role", "human") == "human":
            return
        if not victim.alive or victim.hp <= 0.0:
            return
        d = np.asarray(direction, dtype=np.float64)[:2]
        n = float(np.hypot(*d))
        if n < 1e-6:
            return
        from ..weapons import WEAPONS
        spec = WEAPONS.get(weapon)
        mult = 1.0
        if spec is not None:
            mult = self.SHOTGUN_KNOCK if spec.pellets > 1 else self.KNOCK_CATEGORY.get(spec.category, 1.0)
        dv = amount * self.p["knockback"] * mult
        if victim.role == "mother":
            dv *= self.p["mother_knockback"]
        vxy = victim.vel[:2] + d / n * dv
        cap = float(self.p["knockback_max"])
        sp = float(np.hypot(*vxy))
        if sp > cap:
            vxy *= cap / sp
        victim.vel[0], victim.vel[1] = vxy
        lift = min(dv * 0.12, 2.5)
        if lift > 0.3:
            victim.vel[2] = max(victim.vel[2], lift)
            victim.on_ground = False
        victim.knock_until = max(victim.knock_until, self.w.time + self.p["knockback_time"])

    def _finish(self, team: int, reason: str):
        w = self.w
        if self.phase == "over":
            return
        self.wins[team] += 1
        self.phase = "over"
        self.phase_end = w.time + self.p["post_time"]
        w._emit("round_end", team=team, reason=reason, wins=list(self.wins))
        if self.wins[team] >= self.p["wins_needed"]:
            w.done = True
            w.winner = team

    def pre_step(self, dt: float):
        w = self.w
        if self.phase in ("prep", "live") and w.time >= self.round_end:
            humans = [a for a in w.agents if a.alive and getattr(a, "role", "human") == "human"]
            self._finish(0 if humans else 1, "time" if humans else "infected")
            return
        if self.phase == "prep" and w.time >= self.phase_end:
            self._choose_mothers()
            self.phase = "live"
            w._emit("round_live", round=self.round_no)
        elif self.phase == "live":
            if not any(a.alive and getattr(a, "role", "human") == "human" for a in w.agents):
                self._finish(1, "infected")
        elif self.phase == "over" and w.time >= self.phase_end and not w.done:
            self.start_round()

    def can_respawn(self, a) -> bool:
        return False

    def score(self) -> list:
        return list(self.wins)

    def time_left(self) -> float:
        if self.phase in ("prep", "live"):
            return max(0.0, self.round_end - self.w.time)
        return 0.0

    def state(self) -> dict:
        return {"phase": self.phase, "wins": list(self.wins), "round": self.round_no,
                "round_end": round(self.round_end, 3)}

    def apply_state(self, st: dict):
        if not st:
            return
        self.phase = st["phase"]
        self.wins = list(st["wins"])
        self.round_no = st["round"]
        self.round_end = st["round_end"]

    def hud(self, team: int) -> dict:
        humans = sum(a.alive and getattr(a, "role", "human") == "human" for a in self.w.agents)
        zombies = sum(a.alive and getattr(a, "role", "human") != "human" for a in self.w.agents)
        return {"humans": humans, "zombies": zombies, "phase": self.phase}

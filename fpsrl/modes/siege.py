"""Siege (R6 style): one life, prep phase with drones / reinforcements, defenders hold a pair of
sites, attackers plant a defuser. First to 4 rounds (overtime to 5 at 3:3)."""
from __future__ import annotations

import numpy as np

from ..operators import ATTACKERS, DEFENDERS, OPERATORS
from ..weapons import SLOT_MELEE, SLOT_PISTOL, SLOT_PRIMARY, WEAPONS
from .rounds import ATT, DEF, RoundMode


class Siege(RoundMode):
    name = "r6"
    bomb_item = "defuser"
    START_PHASE = "prep"
    DEFAULTS = {**RoundMode.DEFAULTS, "prep_time": 45.0, "round_time": 180.0, "bomb_time": 45.0,
                "plant_time": 7.0, "defuse_time": 7.0, "win_rounds": 4, "swap_every": 3,
                "reinforce_per_player": 2, "drones_per_player": 2}

    def on_match_start(self):
        ops = (self.w.cfg.mode_params or {}).get("operators", {})
        self.ops = {}
        k_att = k_def = 0
        for a in self.w.agents:
            want = ops.get(a.id, ops.get(str(a.id), {})) or {}
            att = want.get("attack") or ATTACKERS[k_att % len(ATTACKERS)]
            dfn = want.get("defend") or DEFENDERS[k_def % len(DEFENDERS)]
            k_att += 1
            k_def += 1
            self.ops[a.id] = {"attack": att, "defend": dfn}

    def set_operator(self, a, name: str) -> bool:
        if name not in OPERATORS:
            return False
        o = OPERATORS[name]
        self.ops[a.id]["attack" if o.side == 0 else "defend"] = name
        return True

    def choose_operator(self, a, name: str, primary: str | None = None) -> bool:
        """Player pick from the operator menu: applies at once during prep, otherwise next round."""
        o = OPERATORS.get(name)
        if o is None or o.side != self.side(a.team):
            return False
        self.set_operator(a, name)
        if primary in o.primaries:
            a.loadout["primary"] = primary
        if self.phase == "prep" and a.alive:
            self.give_loadout(a)
            a.busy = None
            a.slot = -1  # force a switch event so viewers redraw the new weapon
            self.w.switch_weapon(a, self.w.best_slot(a))
            a.last_slot = SLOT_PISTOL if a.slot == SLOT_PRIMARY else SLOT_MELEE
            self.w._emit("operator", agent=a.id, name=name)
        return True

    def operator_of(self, a):
        side = self.side(a.team)
        return OPERATORS[self.ops[a.id]["attack" if side == ATT else "defend"]]

    def start_duration(self) -> float:
        return self.p["prep_time"]

    def before_spawn(self, fresh: bool):
        sets = self.w.map.objectives.get("site_sets") or [list(range(len(self.sites)))]
        self.site_set = int(self.w.rng.integers(len(sets)))
        self.active_sites = list(sets[self.site_set])
        n_def = len(self.members(self.team_of_side(DEF)))
        self.reinforce_left = self.p["reinforce_per_player"] * n_def
        for a in self.w.agents:
            a.inv = {}  # everyone gets a fresh operator loadout every round

    def spawn_points_team(self, team: int):
        side = self.side(team)
        if side == DEF:
            ds = self.w.map.objectives.get("defender_spawns", {})
            sp = ds.get(self.site_set, ds.get(str(self.site_set)))
            if sp:
                return list(sp)
            centers = [self.site_centers[i] for i in self.active_sites]
            allsp = list(self.w.map.spawns[DEF])
            allsp.sort(key=lambda s: min(np.hypot(s[0] - c[0], s[1] - c[1]) for c in centers))
            return allsp[:max(5, len(allsp) // 3)]
        return list(self.w.map.spawns[ATT])

    def after_spawn(self, fresh: bool):
        for a in self.w.agents:
            a.frozen = self.side(a.team) == ATT  # attackers drone during prep, defenders set up
            a.operator = self.operator_of(a).name
        self.w.gadgets.round_start(self.p["drones_per_player"])
        self.give_bomb()

    def give_loadout(self, a):
        from ..sim import WeaponState
        op = self.operator_of(a)
        a.operator = op.name
        inv = {SLOT_MELEE: WeaponState("knife"), SLOT_PISTOL: WeaponState(op.secondary)}
        if op.primaries:
            want = a.loadout.get("primary")
            prim = want if want in op.primaries else op.primaries[0]
            inv[SLOT_PRIMARY] = WeaponState(prim)
            a.primary = prim
        for g in op.generics:
            inv[WEAPONS[g].slot] = WeaponState(g)
        inv[WEAPONS[op.gadget].slot] = WeaponState(op.gadget)
        a.inv = inv
        a.armor = 0.0

    def refill(self, a):
        pass

    def should_swap(self, played: int) -> bool:
        return played % self.p["swap_every"] == 0

    def on_round_end(self, team, side, reason):
        target = self.p["win_rounds"]
        if self.win_target == target and self.wins[0] == self.wins[1] == target - 1:
            self.win_target = target + 1  # overtime
            self.w._emit("overtime")

    def detonate(self):
        self.w._emit("defuser_complete", pos=self.bomb["pos"])
        self.end_round(ATT, "defuser")

    def state(self) -> dict:
        st = super().state()
        st["reinforce"] = self.reinforce_left
        st["ops"] = {str(k): v for k, v in self.ops.items()}
        return st

    def apply_state(self, st: dict):
        super().apply_state(st)
        if st:
            self.reinforce_left = st.get("reinforce", getattr(self, "reinforce_left", 0))
            if "ops" in st:
                self.ops = {int(k): v for k, v in st["ops"].items()}

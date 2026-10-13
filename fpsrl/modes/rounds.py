"""Round based objective modes: CS style bomb defusal (with economy) and the shared round flow
reused by the Siege (R6) mode."""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..geometry import poly_center
from ..weapons import SECONDARY, SLOT_BOMB, SLOT_MELEE, SLOT_PISTOL, SLOT_PRIMARY, WEAPONS
from .base import GameMode

ATT, DEF = 0, 1  # sides
_ROUND_UID = itertools.count(1)


class RoundMode(GameMode):
    """Phases: start (freeze / prep) -> live -> planted -> over -> next round."""
    free_weapons = False
    spawn_protect = 0.0
    bomb_item = "c4"
    START_PHASE = "freeze"
    DEFAULTS = {"freeze_time": 5.0, "round_time": 115.0, "bomb_time": 40.0, "post_time": 5.0,
                "plant_time": 3.2, "defuse_time": 10.0, "kit_defuse_time": 5.0,
                "win_rounds": 13, "half_rounds": 12, "bomb_radius": 20.0, "bomb_damage": 500.0}

    def reset(self):
        self.round_no = 0
        self.wins = [0, 0]
        self.swapped = 0
        self.win_target = self.p["win_rounds"]
        self.sites = self.w.map.objectives.get("sites", [])
        self.site_centers = [poly_center(s["poly"]) for s in self.sites]
        self.active_sites = list(range(len(self.sites)))
        self.phase = "over"
        self.phase_end = 0.0
        self.bomb = None  # dict(pos, state: carried|dropped|planted, carrier, site, t, defuser)
        self.last_result = None
        self._hint_at = {}
        self.w.cfg.respawn = False
        self.on_match_start()
        self.start_round(fresh=True)

    def on_match_start(self):
        pass

    # ------------------------------------------------------------------ rounds
    def start_round(self, fresh: bool = False):
        w = self.w
        survivors = {a.id for a in w.agents if a.alive} if not fresh else set()
        w.clear_round()
        self.round_no += 1
        self.round_uid = next(_ROUND_UID)  # unique across matches, for per-round caches
        self.bomb = None
        self.before_spawn(fresh)
        for team in (0, 1):
            members = self.members(team)
            sp = self.spawn_points_team(team)
            order = w.rng.permutation(len(sp))
            for k, a in enumerate(members):
                keep = a.id in survivors
                w._spawn_at(a, sp[order[k % len(sp)]], keep_inventory=keep)
                if keep:
                    self.refill(a)
                a.frozen = True
        self.after_spawn(fresh)
        self.phase = self.START_PHASE
        self.phase_end = w.time + self.start_duration()
        w._emit("round_start", round=self.round_no, wins=list(self.wins), swapped=self.swapped)

    def start_duration(self) -> float:
        return self.p["freeze_time"]

    def before_spawn(self, fresh: bool):
        pass

    def after_spawn(self, fresh: bool):
        self.give_bomb()

    def refill(self, a):
        for ws in a.inv.values():
            if ws.spec.kind == "gun":
                ws.mag = ws.spec.mag
                ws.reserve = ws.spec.reserve

    def give_bomb(self):
        att = self.members(self.team_of_side(ATT))
        if not att or not self.bomb_item:
            return
        carrier = att[int(self.w.rng.integers(len(att)))]
        self.w.give(carrier, self.bomb_item)
        self.bomb = {"state": "carried", "carrier": carrier.id, "pos": carrier.pos.tolist(), "site": -1,
                     "t": 0.0}

    def begin_live(self):
        for a in self.w.agents:
            a.frozen = False
        self.phase = "live"
        self.phase_end = self.w.time + self.p["round_time"]
        self.w._emit("round_live", round=self.round_no)

    def pre_step(self, dt: float):
        w = self.w
        if self.phase in ("freeze", "prep") and w.time >= self.phase_end:
            self.begin_live()
        elif self.phase == "over" and w.time >= self.phase_end and not w.done:
            self.next_round()
        b = self.bomb
        if b is None:
            return
        if b["state"] == "carried":
            c = w.agents[b["carrier"]]
            b["pos"] = c.pos.tolist()
        elif b["state"] == "dropped" and self.phase in ("live", "freeze", "prep"):
            for a in w.agents:
                if a.alive and self.side(a.team) == ATT and np.hypot(a.pos[0] - b["pos"][0],
                                                                      a.pos[1] - b["pos"][1]) < 0.9 \
                        and abs(a.pos[2] - b["pos"][2]) < 1.2:
                    w.give(a, self.bomb_item)
                    b.update(state="carried", carrier=a.id)
                    w._emit("bomb_pickup", agent=a.id)
                    break

    def post_step(self, dt: float):
        w = self.w
        if self.phase in ("over", "freeze", "prep") or w.done:
            return
        att_alive = [a for a in self.members(self.team_of_side(ATT)) if a.alive]
        def_alive = [a for a in self.members(self.team_of_side(DEF)) if a.alive]
        planted = self.phase == "planted"
        if planted and w.time >= self.phase_end:
            self.detonate()
            return
        if not def_alive:
            self.end_round(ATT, "elimination")
        elif not att_alive and not planted:
            self.end_round(DEF, "elimination")
        elif self.phase == "live" and w.time >= self.phase_end:
            self.on_timeout()

    def on_timeout(self):
        self.end_round(DEF, "time")

    def detonate(self):
        w = self.w
        b = self.bomb
        pos = np.array(b["pos"]) + np.array([0.0, 0.0, 0.3])
        w._emit("bomb_explode", pos=b["pos"])
        planter = w.agents[b.get("planter", 0)]
        R = self.p["bomb_radius"]
        for v in w.agents:
            if not v.alive:
                continue
            dist = float(np.linalg.norm(v.pos - pos))
            if dist < R:
                w.apply_damage(v, self.p["bomb_damage"] * (1.0 - dist / R) ** 1.6, planter, self.bomb_item,
                               1, None, "bomb")
        w.destruct.blast(pos, 3.0, 5000.0)
        self.end_round(ATT, "bomb")

    def end_round(self, side: int, reason: str):
        w = self.w
        team = self.team_of_side(side) if side in (ATT, DEF) else -1
        if team >= 0:
            self.wins[team] += 1
        self.phase = "over"
        self.phase_end = w.time + self.p["post_time"]
        self.last_result = {"team": team, "side": side, "reason": reason}
        for a in w.agents:
            a.busy = None
        w._emit("round_end", team=team, side=side, reason=reason, wins=list(self.wins), round=self.round_no)
        self.on_round_end(team, side, reason)
        if team >= 0 and self.wins[team] >= self.win_target:
            w.done = True
            w.winner = team

    def on_round_end(self, team, side, reason):
        pass

    def next_round(self):
        played = self.round_no
        fresh = False
        if self.should_swap(played):
            self.swapped ^= 1
            fresh = True
            self.on_swap()
            self.w._emit("side_swap", swapped=self.swapped)
        self.start_round(fresh=fresh)

    def should_swap(self, played: int) -> bool:
        return played == self.p["half_rounds"]

    def on_swap(self):
        pass

    # ------------------------------------------------------------------ rules
    def can_respawn(self, a) -> bool:
        return False

    def on_kill(self, victim, attacker, weapon, headshot):
        b = self.bomb
        if b is not None and b["state"] == "carried" and b["carrier"] == victim.id:
            b.update(state="dropped", carrier=-1, pos=victim.pos.tolist())
            victim.inv.pop(SLOT_BOMB, None)
            self.w._emit("bomb_drop", agent=victim.id, pos=b["pos"])

    def site_at(self, pos) -> int:
        for i in self.active_sites:
            if self.in_zone(pos, self.sites[i]):
                return i
        return -1

    def item_fire(self, a, ws):
        self.try_plant(a)

    def try_plant(self, a) -> bool:
        if self.phase != "live" or self.side(a.team) != ATT or not a.on_ground:
            return False
        if SLOT_BOMB not in a.inv or a.inv[SLOT_BOMB].name != self.bomb_item:
            return False
        site = self.site_at(a.pos)
        if site < 0:
            if self.w.time - self._hint_at.get(a.id, -9.0) > 1.5:
                self._hint_at[a.id] = self.w.time
                self.w._emit("hint", agent=a.id, text="not_in_site")
            return False
        self.w.start_busy(a, "plant", self.p["plant_time"], site)
        return True

    def defuse_time(self, a) -> float:
        return self.p["defuse_time"]

    def try_defuse(self, a) -> bool:
        b = self.bomb
        if self.phase != "planted" or b is None or self.side(a.team) != DEF:
            return False
        if np.hypot(a.pos[0] - b["pos"][0], a.pos[1] - b["pos"][1]) > 1.5 or abs(a.pos[2] - b["pos"][2]) > 1.4:
            return False
        self.w.start_busy(a, "defuse", self.defuse_time(a), None)
        return True

    def use(self, a) -> bool:
        if self.try_defuse(a) or self.try_plant(a):
            return True
        return super().use(a)

    def finish_busy(self, a, kind, data):
        w = self.w
        if kind == "plant":
            if self.phase != "live":
                return
            a.inv.pop(SLOT_BOMB, None)
            if a.slot == SLOT_BOMB:
                a.slot = -1
                w.switch_weapon(a, w.best_slot(a))
            self.bomb = {"state": "planted", "carrier": -1, "pos": a.pos.tolist(), "site": data,
                         "t": w.time, "planter": a.id}
            self.phase = "planted"
            self.phase_end = w.time + self.p["bomb_time"]
            w._emit("bomb_planted", agent=a.id, site=data, pos=a.pos.tolist())
            self.on_plant(a)
        elif kind == "defuse":
            if self.phase != "planted":
                return
            self.bomb["state"] = "defused"
            w._emit("bomb_defused", agent=a.id)
            self.on_defuse(a)
            self.end_round(DEF, "defuse")
        else:
            super().finish_busy(a, kind, data)

    def on_plant(self, a):
        pass

    def on_defuse(self, a):
        pass

    # ------------------------------------------------------------------ info
    def score(self) -> list:
        return list(self.wins)

    def time_left(self) -> float:
        return max(0.0, self.phase_end - self.w.time)

    def state(self) -> dict:
        b = self.bomb
        return {"phase": self.phase, "left": round(self.time_left(), 2), "round": self.round_no,
                "wins": list(self.wins), "swapped": self.swapped, "sites": list(self.active_sites),
                "bomb": None if b is None else {"state": b["state"], "pos": [round(v, 2) for v in b["pos"]],
                                                "carrier": b["carrier"], "site": b["site"]},
                "result": self.last_result}

    def apply_state(self, st: dict):
        if not st:
            return
        self.phase = st.get("phase", self.phase)
        self.phase_end = self.w.time + st.get("left", 0.0)
        self.round_no = st.get("round", self.round_no)
        self.wins = list(st.get("wins", self.wins))
        self.swapped = st.get("swapped", self.swapped)
        self.active_sites = list(st.get("sites", self.active_sites))
        self.bomb = st.get("bomb")
        self.last_result = st.get("result")

    def obs(self, a) -> np.ndarray:
        """[side, phase one-hot(3), time left, bomb state(2), bomb dist, bomb bearing, site dist,
        site bearing, carrying]"""
        o = np.zeros(12, dtype=np.float32)
        o[0] = 1.0 if self.side(a.team) == ATT else -1.0
        o[1 + {"freeze": 0, "prep": 0, "live": 1, "planted": 2}.get(self.phase, 0)] = 1.0
        o[4] = min(self.time_left() / 120.0, 1.0)
        yr = math.radians(a.yaw)
        b = self.bomb
        if b is not None:
            o[5] = 1.0 if b["state"] == "planted" else 0.0
            o[6] = 1.0 if b["state"] == "dropped" else 0.0
            dx, dy = b["pos"][0] - a.pos[0], b["pos"][1] - a.pos[1]
            o[7] = min(math.hypot(dx, dy) / 80.0, 1.0)
            o[8] = math.sin(math.atan2(dy, dx) - yr)
            o[11] = 1.0 if b["state"] == "carried" and b["carrier"] == a.id else 0.0
        if self.active_sites:
            c = min((self.site_centers[i] for i in self.active_sites),
                    key=lambda c: math.hypot(c[0] - a.pos[0], c[1] - a.pos[1]))
            dx, dy = c[0] - a.pos[0], c[1] - a.pos[1]
            o[9] = min(math.hypot(dx, dy) / 80.0, 1.0)
            o[10] = math.sin(math.atan2(dy, dx) - yr)
        return o


class BombDefusal(RoundMode):
    name = "bomb"
    economy = True  # buy menu + money (Elimination reuses this)
    SPECIAL_ITEMS = {"kevlar", "kevlar_helmet", "defuser"}
    DEFAULTS = {**RoundMode.DEFAULTS, "freeze_time": 15.0, "post_time": 7.0,
                "win_rounds": 16, "half_rounds": 15, "max_rounds": 30,
                "start_money": 800, "max_money": 16000, "win_reward": 3250,
                "loss_base": 1400, "loss_step": 500, "loss_max": 3400, "plant_reward": 800,
                "objective_win_reward": 3500, "defuse_reward": 300, "buy_time": 20.0,
                "team_kill_penalty": 300}
    GEAR_PRICES = {"kevlar": 650, "kevlar_helmet": 1000, "defuser": 400}
    # Existing weapons stand in for the closest CS:GO side-specific equivalents.
    SIDE_WEAPONS = {
        ATT: {"glock", "deagle", "dualies", "mp5", "uzi", "ak47", "awm", "scar20", "m249", "nova"},
        DEF: {"glock", "deagle", "dualies", "mp5", "mp9", "m4a4", "m4a1", "famas",
              "awm", "scar20", "m249", "nova"},
    }

    def on_match_start(self):
        # Since the 2019 economy update, each half starts at loss-count 1:
        # the pistol-round loser receives $1900.
        self.loss_streak = [1, 1]
        for a in self.w.agents:
            a.money = self.p["start_money"]

    def on_swap(self):
        self.loss_streak = [1, 1]
        for a in self.w.agents:
            a.money = self.p["start_money"]
            a.inv = {}

    def give_loadout(self, a):
        from ..sim import WeaponState
        pistol = SECONDARY
        a.inv = {SLOT_PISTOL: WeaponState(pistol), SLOT_MELEE: WeaponState("knife")}
        a.loadout = {"primary": None, "secondary": pistol}

    def after_spawn(self, fresh: bool):
        super().after_spawn(fresh)
        # CS:GO's buy window starts at round start, so freeze time consumes part of it.
        self.buy_end = self.w.time + self.p["buy_time"]

    def can_buy_now(self, a) -> bool:
        if not a.alive or self.phase not in ("freeze", "live"):
            return False
        if self.w.time > self.buy_end:
            return False
        zones = self.w.map.buy_zones.get(self.side(a.team))
        if zones:
            return any(self.in_zone(a.pos, {"poly": z}) for z in zones)
        sp = self.w.map.spawns[self.side(a.team)]
        return min(np.hypot(a.pos[0] - s[0], a.pos[1] - s[1]) for s in sp) < 9.0

    def buy_price(self, a, name: str):
        if not self.can_buy_now(a):
            return None
        spec = WEAPONS[name]
        if name == "defuser":
            return None
        if spec.kind not in ("gun", "grenade") or spec.price < 0:
            return None
        if spec.kind == "gun" and name not in self.SIDE_WEAPONS[self.side(a.team)]:
            return None
        cur = a.inv.get(spec.slot)
        if spec.kind == "grenade":
            total = sum(ws.mag for ws in a.inv.values() if ws.spec.kind == "grenade")
            max_carry = int(spec.extra.get("max_carry", 1))
            if total >= 4 or (cur is not None and (cur.name != name or cur.mag >= max_carry)):
                return None
        elif cur is not None and cur.name == name:
            return None
        if name == "molotov" and self.side(a.team) == DEF:
            return 600  # incendiary grenade price
        return spec.price

    def buy_special(self, a, name: str) -> bool:
        price = self.gear_price(a, name)
        if price is None or a.money < price:
            return False
        if name == "defuser":
            a.money -= price
            self.w.give(a, "defuser")
        elif name == "kevlar":
            a.money -= price
            a.armor = 0.2
        elif name == "kevlar_helmet":
            a.money -= price
            a.armor = 0.2
            a.helmet = True
        else:
            return False
        self.w._emit("buy", agent=a.id, weapon=name, price=price)
        return True

    def gear_price(self, a, name: str):
        if not self.can_buy_now(a):
            return None
        if name == "defuser":
            return 400 if self.side(a.team) == DEF and SLOT_BOMB not in a.inv else None
        if name == "kevlar":
            return 650 if a.armor <= 0 else None
        if name == "kevlar_helmet":
            return None if a.helmet else (350 if a.armor > 0 else 1000)
        return None

    def defuse_time(self, a) -> float:
        w = a.inv.get(SLOT_BOMB)
        return self.p["kit_defuse_time"] if w is not None and w.name == "defuser" else self.p["defuse_time"]

    def on_kill(self, victim, attacker, weapon, headshot):
        super().on_kill(victim, attacker, weapon, headshot)
        if victim.inv.get(SLOT_BOMB) is not None and victim.inv[SLOT_BOMB].name == "defuser":
            victim.inv.pop(SLOT_BOMB, None)
        if attacker is None:
            return
        if attacker.id == victim.id or attacker.team == victim.team:
            self._pay(attacker, -self.p["team_kill_penalty"])
        else:
            reward = WEAPONS[weapon].kill_reward if weapon in WEAPONS else 300
            self._pay(attacker, reward)

    def on_plant(self, a):
        self._pay(a, 300)

    def on_defuse(self, a):
        self._pay(a, self.p["defuse_reward"])

    def on_round_end(self, team, side, reason):
        loser = 1 - team
        self.loss_streak[team] = max(0, self.loss_streak[team] - 1)
        bonus = self.p["loss_base"] + self.p["loss_step"] * self.loss_streak[loser]
        bonus = min(bonus, self.p["loss_max"])
        self.loss_streak[loser] = min(self.loss_streak[loser] + 1, 4)
        planted = self.bomb is not None and self.bomb["state"] in ("planted", "defused")
        win_reward = self.p["objective_win_reward"] if reason in ("bomb", "defuse") else self.p["win_reward"]
        for a in self.w.agents:
            if a.team == team:
                self._pay(a, win_reward)
            else:
                if reason == "time" and a.alive and self.side(a.team) == ATT:
                    continue  # attackers who survive a lost round on time get nothing
                plant_bonus = self.p["plant_reward"] if planted and self.side(a.team) == ATT else 0
                self._pay(a, bonus + plant_bonus)
        # the round winner keeps playing, dead players lose their guns
        for a in self.w.agents:
            if not a.alive:
                a.inv = {}
        if self.round_no >= self.p["max_rounds"] and self.wins[0] == self.wins[1]:
            self.w.done = True
            self.w.winner = -1

    def _pay(self, a, amount):
        a.money = int(np.clip(a.money + amount, 0, self.p["max_money"]))

    def state(self) -> dict:
        st = super().state()
        st["buy_left"] = round(max(0.0, self.buy_end - self.w.time), 2) \
            if self.phase in ("freeze", "live") else 0.0
        return st


class Elimination(BombDefusal):
    """Team elimination on the bomb maps: the same rounds and economy, with no bomb to plant.
    Wipe the other team; on the clock, the side with more players left wins, and a tie is a draw."""

    name = "elim"
    bomb_item = ""

    def buy_price(self, a, name: str):
        if name == "defuser":
            return None
        return super().buy_price(a, name)

    def buy_special(self, a, name: str) -> bool:
        if name == "defuser":
            return False
        return super().buy_special(a, name)

    def on_timeout(self):
        att = [a for a in self.members(self.team_of_side(ATT)) if a.alive]
        dfn = [a for a in self.members(self.team_of_side(DEF)) if a.alive]
        if len(att) > len(dfn):
            self.end_round(ATT, "time")
        elif len(dfn) > len(att):
            self.end_round(DEF, "time")
        else:
            self.end_round(-1, "draw")

    def on_round_end(self, team, side, reason):
        if team >= 0:
            super().on_round_end(team, side, reason)
            return
        for a in self.w.agents:
            self._pay(a, self.p["loss_base"])
            if not a.alive:
                a.inv = {}

"""Headless, deterministic game simulation.

This is the single source of truth for gameplay. The Panda3D client, scripted
bots and the RL environments all drive the same `World.step(actions)`.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .geometry import aabb_overlap_mask, raycast_boxes, segments_blocked, segments_hit_spheres
from .maps import load_map
from .weapons import (LOADOUT_BY_KEY, MELEE, SECONDARY, SLOT_FIRE, SLOT_FLASH, SLOT_FRAG, SLOT_MELEE,
                      SLOT_PISTOL, SLOT_PRIMARY, SLOT_SMOKE, WEAPONS)

TEAM_NAMES = ("BL", "GR")
LEAN_DIST = 0.38
LEAN_ROLL = 10.0
LEAN_SPEED = 6.0


@dataclass
class Action:
    """Per-tick input of one agent.

    move_*: [-1, 1]; look_*: degrees to add this tick (yaw positive = turn left,
    pitch positive = look up); buttons are "held" states; switch: 0 keeps the
    current weapon, 1..7 selects a slot; lean: -1 left, 0, +1 right; use: plant /
    defuse / reinforce / barricade (hold); gadget: toggle drone / camera view.
    """
    move_forward: float = 0.0
    move_right: float = 0.0
    look_yaw: float = 0.0
    look_pitch: float = 0.0
    fire: bool = False
    alt: bool = False
    jump: bool = False
    crouch: bool = False
    walk: bool = False
    reload: bool = False
    switch: int = 0
    lean: int = 0
    use: bool = False
    gadget: bool = False


IDLE = Action()


def _crouch_only(act: Action) -> Action:
    return Action(crouch=act.crouch) if act.crouch else IDLE


@dataclass
class MatchConfig:
    team_size: int = 4
    map_name: str = "transport_ship"
    mode: str = "tdm"  # tdm | dom | bomb | r6
    tick_rate: int = C.TICK_RATE
    kill_limit: int = 60
    time_limit: float = 420.0
    friendly_fire: bool = False
    respawn: bool = True
    respawn_time: float = C.RESPAWN_TIME
    primaries: list | None = None  # per-agent primary weapon, default all "ak47"
    seed: int | None = None
    mode_params: dict = field(default_factory=dict)  # overrides of the mode's tunables (see fpsrl.modes)


def dir_from_angles(yaw: float, pitch: float) -> np.ndarray:
    y, p = math.radians(yaw), math.radians(pitch)
    cp = math.cos(p)
    return np.array([cp * math.cos(y), cp * math.sin(y), math.sin(p)])


def angles_to(src, dst) -> tuple[float, float]:
    d = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return yaw, pitch


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class WeaponState:
    __slots__ = ("name", "mag", "reserve")

    def __init__(self, name: str):
        spec = WEAPONS[name]
        self.name = name
        self.mag = spec.mag
        self.reserve = spec.reserve

    @property
    def spec(self):
        return WEAPONS[self.name]


_BARE_HANDS = WeaponState(MELEE)  # placeholder while an agent holds nothing (dead between rounds)

class Agent:
    def __init__(self, aid: int, team: int, name: str, primary: str = "ak47"):
        self.id = aid
        self.team = team
        self.name = name
        self.primary = primary
        self.pos = np.zeros(3)
        self.prev_pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.pitch = 0.0
        self.height = C.STAND_HEIGHT
        self.crouching = False
        self.walking = False
        self.on_ground = False
        self.hp = float(C.MAX_HP)
        self.alive = False
        self.respawn_at = 0.0
        self.spawn_time = 0.0
        self.protect_until = 0.0
        self.death_time = -100.0
        self.killer = -1
        self.inv: dict[int, WeaponState] = {}
        self.slot = SLOT_PRIMARY
        self.last_slot = SLOT_PISTOL
        self.ready_at = 0.0
        self.next_fire = 0.0
        self.next_jump = 0.0
        self.reload_until = 0.0
        self.fire_queue_until = -1.0
        self.ads = False
        self.ads_since = 0.0
        self.bloom = 0.0
        self.punch = np.zeros(2)  # (pitch, yaw) recoil offset in degrees
        self.prev_punch = np.zeros(2)
        self.shots_in_burst = 0
        self.last_shot = -100.0
        self.throw_at = -1.0  # earliest release time while the pin is out; -1 = not cooking
        self.cook_until = -1.0  # fuse expiry while cooking
        self.throw_under = False
        self.switch_back_at = -1.0
        self.prev_fire = self.prev_alt = self.prev_reload = False
        self.step_dist = 0.0
        self.kills = 0
        self.deaths = 0
        self.headshots = 0
        self.damage_dealt = 0.0
        self.last_damage_time = -100.0
        self.last_attacker = -1
        self.last_damage_dir = np.zeros(3)
        # lean / objective / R6 state
        self.lean = 0.0  # smoothed lean input, -1..1
        self.lean_off = 0.0  # actual sideways head offset in metres (clamped by walls), + = right
        self.loadout = {"primary": primary, "secondary": SECONDARY}
        self.money = 0
        self.armor = 0.0  # fraction of damage absorbed
        self.helmet = False
        self.frozen = False  # freeze time / R6 attacker prep: can look, cannot move or shoot
        self.busy: tuple | None = None  # (kind, start, duration, data) of a hold action (plant, defuse ...)
        self.flash_until = 0.0
        self.flash_t0 = 0.0
        self.view_mode = 0  # 0 body, 1 drone, 2 camera
        self.view_ref = -1  # drone / camera id while view_mode != 0
        self.operator = ""
        self.spotted_until = 0.0
        self.prev_use = self.prev_gadget = False

    # convenient accessors -------------------------------------------------------
    def right(self) -> np.ndarray:
        yr = math.radians(self.yaw)
        return np.array([math.sin(yr), -math.cos(yr)])

    def lean_vec(self) -> np.ndarray:
        return self.right() * self.lean_off

    def eye(self) -> np.ndarray:
        off = self.lean_vec()
        return np.array([self.pos[0] + off[0], self.pos[1] + off[1],
                         self.pos[2] + self.height - C.EYE_FROM_TOP - abs(self.lean_off) * 0.12])

    def view_angles(self) -> tuple[float, float]:
        return self.yaw + self.punch[1], self.pitch + self.punch[0]

    @property
    def weapon(self) -> WeaponState:
        ws = self.inv.get(self.slot)
        return ws if ws is not None else _BARE_HANDS

    @property
    def spec(self):
        return self.weapon.spec

    @property
    def reloading(self) -> bool:
        return self.reload_until > 0.0

    def hitboxes(self):
        """(3,3) min / max arrays: head, torso, legs."""
        return hitboxes_at(self.pos, self.height, self.lean_vec())


def hitboxes_at(pos, h, off=None):
    x, y, z = pos
    hx, hy = (off[0], off[1]) if off is not None else (0.0, 0.0)
    tx, ty = hx * 0.5, hy * 0.5
    mn = np.array([[x + hx - 0.14, y + hy - 0.14, z + h - 0.28],
                   [x + tx - 0.25, y + ty - 0.25, z + h * 0.43],
                   [x - 0.2, y - 0.2, z]])
    mx = np.array([[x + hx + 0.14, y + hy + 0.14, z + h],
                   [x + tx + 0.25, y + ty + 0.25, z + h - 0.28],
                   [x + 0.2, y + 0.2, z + h * 0.43]])
    return mn, mx


@dataclass
class Projectile:
    id: int
    kind: str
    owner: int
    team: int
    pos: np.ndarray
    vel: np.ndarray
    explode_at: float
    spawn_time: float
    last_bounce: float = -1.0


@dataclass
class Smoke:
    id: int
    pos: np.ndarray
    t0: float
    t1: float
    max_radius: float

    def radius(self, t: float) -> float:
        grow = min(1.0, (t - self.t0) / 1.6)
        fade = min(1.0, max(0.0, (self.t1 - t) / 2.0))
        return self.max_radius * grow * (0.35 + 0.65 * fade) if t < self.t1 else 0.0


@dataclass
class Fire:
    """Ground flame left by a molotov. `acc` counts toward the next damage tick."""
    id: int
    pos: np.ndarray
    t0: float
    t1: float
    radius: float
    owner: int
    weapon: str
    dps: float
    acc: float = 0.2


class World:
    def __init__(self, cfg: MatchConfig | None = None):
        from .destruction import Destruction
        from .gadgets import Gadgets
        from .modes import make_mode
        self.cfg = cfg or MatchConfig()
        self.map = load_map(self.cfg.map_name)
        self.dt = 1.0 / self.cfg.tick_rate
        self.destruct = Destruction(self)
        self.gadgets = Gadgets(self)
        self.static_boxes = [b for b in self.map.boxes if b.solid and not b.params.get("destruct")]
        self.geo_version = 0
        self._rebuild_geometry()

        n = self.cfg.team_size
        prim = list(self.cfg.primaries or [])
        self.agents: list[Agent] = []
        for team in (0, 1):
            for i in range(n):
                aid = team * n + i
                primary = prim[aid] if aid < len(prim) else "ak47"
                self.agents.append(Agent(aid, team, f"{TEAM_NAMES[team]}-{i}", primary))
        self.event_log: deque = deque(maxlen=4096)
        self.noises: deque = deque(maxlen=128)
        self.muted = False  # drop events (client-side prediction replays)
        self.lag_comp = False  # record hitbox history so shots can be rewound (network host)
        self.rewind: dict[int, float] = {}  # shooter id -> world time the shooter was seeing
        self._hist: deque = deque(maxlen=40)
        self.mode = make_mode(self)
        self.reset(self.cfg.seed)

    # ------------------------------------------------------------------ geometry
    def _rebuild_geometry(self):
        boxes = self.static_boxes + self.destruct.active_boxes() + self.gadgets.boxes()
        solid = [b for b in boxes if b.solid]
        self.solid_boxes = solid
        self.smin = np.array([b.mn for b in solid], dtype=np.float64).reshape(-1, 3)
        self.smax = np.array([b.mx for b in solid], dtype=np.float64).reshape(-1, 3)
        bullet = [b for b in solid if b.blocks_bullets]
        self.bullet_boxes = bullet
        self.bmin = np.array([b.mn for b in bullet], dtype=np.float64).reshape(-1, 3)
        self.bmax = np.array([b.mx for b in bullet], dtype=np.float64).reshape(-1, 3)
        big = np.argsort(-np.prod(self.bmax - self.bmin, axis=1), kind="stable")
        self.los_min, self.los_max = self.bmin[big], self.bmax[big]
        self.geo_version += 1
        self._vis_tick = -1

    def geometry_changed(self):
        """Called by destruction / gadgets after solid boxes appeared or disappeared."""
        self._rebuild_geometry()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.tick = 0
        self.seq = 0
        self.projectiles: list[Projectile] = []
        self.smokes: list[Smoke] = []
        self.fires: list[Fire] = []
        self.pending_melee: list[tuple] = []
        self.team_kills = [0, 0]
        self.done = False
        self.winner = -1
        self._next_id = 1
        self._vis_tick = -1
        self._vis = None
        self.tick_events: list[dict] = []
        self.event_log.clear()
        self.noises.clear()
        self._hist.clear()
        self.destruct.reset()
        self.gadgets.reset()
        self._rebuild_geometry()
        for a in self.agents:
            a.kills = a.deaths = a.headshots = 0
            a.damage_dealt = 0.0
            a.last_damage_time = a.last_shot = a.death_time = -100.0
            a.last_attacker = -1
            a.last_damage_dir[:] = 0.0
            a.next_jump = a.respawn_at = a.step_dist = 0.0
            a.prev_fire = a.prev_alt = a.prev_reload = a.prev_use = a.prev_gadget = False
            a.money = 0
            a.inv = {}
            a.alive = False
        self.mode.reset()
        return self

    def clear_round(self):
        """Remove transient objects between rounds (projectiles, smoke, fire, gadgets, damage to walls)."""
        self.projectiles = []
        self.smokes = []
        self.fires = []
        self.pending_melee = []
        self.noises.clear()
        self._hist.clear()
        self.rewind.clear()
        self.destruct.reset()
        self.gadgets.reset()
        self._rebuild_geometry()

    def spawn_side(self, side_spawns: list, members: list, keep_inventory=False):
        """Place `members` on distinct points of `side_spawns` (random order)."""
        order = self.rng.permutation(len(side_spawns))
        for k, a in enumerate(members):
            self._spawn_at(a, side_spawns[order[k % len(side_spawns)]], keep_inventory)

    @property
    def score(self) -> list:
        return self.mode.score()

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _emit(self, etype: str, **kw) -> dict:
        if self.muted:
            return {"type": etype, **kw}
        self.seq += 1
        e = {"type": etype, "t": self.time, "seq": self.seq, **kw}
        self.tick_events.append(e)
        self.event_log.append(e)
        return e

    def events_since(self, seq: int):
        return [e for e in self.event_log if e["seq"] > seq]

    # ------------------------------------------------------------------ spawning
    def default_loadout(self, a: Agent):
        a.inv = {SLOT_PRIMARY: WeaponState(a.loadout.get("primary") or a.primary),
                 SLOT_PISTOL: WeaponState(a.loadout.get("secondary") or SECONDARY),
                 SLOT_MELEE: WeaponState(MELEE), SLOT_FRAG: WeaponState("frag"),
                 SLOT_SMOKE: WeaponState("smoke"), SLOT_FLASH: WeaponState("flash"),
                 SLOT_FIRE: WeaponState("molotov")}

    def give(self, a: Agent, name: str, select: bool = False):
        """Put weapon `name` into its slot, replacing what was there."""
        spec = WEAPONS[name]
        a.inv[spec.slot] = WeaponState(name)
        if spec.slot == SLOT_PRIMARY:
            a.primary = name
        if a.slot == spec.slot:
            a.ready_at = self.time + spec.draw_time
            a.reload_until = 0.0
            a.ads = False
            self._emit("switch", agent=a.id, slot=spec.slot, weapon=name)
        elif select:
            self.switch_weapon(a, spec.slot)

    def buy(self, a: Agent, name: str) -> bool:
        if not a.alive:
            return False
        if name not in WEAPONS or name in getattr(self.mode, "SPECIAL_ITEMS", ()):
            buy_special = getattr(self.mode, "buy_special", None)
            return bool(buy_special is not None and buy_special(a, name))
        price = self.mode.buy_price(a, name)
        if price is None or a.money < price:
            return False
        a.money -= price
        spec = WEAPONS[name]
        current = a.inv.get(spec.slot)
        if spec.kind == "grenade" and current is not None and current.name == name:
            current.mag += 1
            self._emit("buy", agent=a.id, weapon=name, price=price)
            return True
        if spec.slot == SLOT_PRIMARY:
            a.loadout["primary"] = name
        elif spec.slot == SLOT_PISTOL:
            a.loadout["secondary"] = name
        self.give(a, name, select=spec.kind == "gun")
        self._emit("buy", agent=a.id, weapon=name, price=price)
        return True

    def _spawn_at(self, a: Agent, spawn, keep_inventory: bool = False):
        x, y, z, yaw = spawn
        a.pos[:] = (x, y, z + 0.01)
        a.prev_pos[:] = a.pos
        a.vel[:] = 0.0
        a.yaw = float(yaw)
        a.pitch = 0.0
        a.height = C.STAND_HEIGHT
        a.crouching = False
        a.on_ground = True
        a.hp = float(C.MAX_HP)
        a.alive = True
        a.spawn_time = self.time
        a.protect_until = self.time + self.mode.spawn_protect
        a.lean = a.lean_off = 0.0
        a.busy = None
        a.flash_until = 0.0
        a.view_mode, a.view_ref = 0, -1
        if not keep_inventory:
            a.armor = 0.0
            a.helmet = False
        if not keep_inventory or not a.inv:
            self.mode.give_loadout(a)
        a.slot = SLOT_PRIMARY if SLOT_PRIMARY in a.inv else SLOT_PISTOL
        a.last_slot = SLOT_PISTOL if a.slot == SLOT_PRIMARY else SLOT_MELEE
        a.ready_at = self.time + a.spec.draw_time * 0.5
        a.next_fire = 0.0
        a.reload_until = 0.0
        a.fire_queue_until = -1.0
        a.ads = False
        a.bloom = 0.0
        a.punch[:] = 0.0
        a.prev_punch[:] = 0.0
        a.shots_in_burst = 0
        a.throw_at = -1.0
        a.cook_until = -1.0
        a.switch_back_at = -1.0
        a.killer = -1
        self._emit("spawn", agent=a.id, pos=a.pos.tolist())

    def _respawn(self, a: Agent):
        spawns = self.mode.spawn_points(a)
        enemies = [e.pos for e in self.agents if e.alive and e.team != a.team]
        others = [o.pos for o in self.agents if o.alive and o.id != a.id]
        best, best_score = spawns[0], -1e9
        for s in spawns:
            p = np.array(s[:3])
            if any(np.linalg.norm(p[:2] - o[:2]) < 0.9 for o in others):
                continue
            d_enemy = min((np.linalg.norm(p - e) for e in enemies), default=100.0)
            score = min(d_enemy, 40.0) + self.rng.uniform(0.0, 6.0)
            if score > best_score:
                best, best_score = s, score
        self._spawn_at(a, best)
        extra = getattr(self.mode, "respawn_protect", 0.0)
        if extra:
            a.protect_until = self.time + extra

    # ------------------------------------------------------------------ main step
    def step(self, actions: dict | None = None) -> list[dict]:
        """Advance one tick. `actions` maps agent id -> Action (missing = idle)."""
        actions = actions or {}
        self.tick_events = []
        if self.done:
            return self.tick_events
        dt = self.dt
        for a in self.agents:
            a.prev_pos[:] = a.pos
            a.prev_punch[:] = a.punch
        self.mode.pre_step(dt)
        for a in self.agents:
            self._update_agent(a, actions.get(a.id, IDLE), dt)
        self._separate_agents()
        self._update_melee()
        self._update_projectiles(dt)
        self._update_fires(dt)
        self.gadgets.update(dt)
        self.smokes = [s for s in self.smokes if self.time < s.t1]
        self.time += dt
        self.tick += 1
        if self.lag_comp:
            self._hist.append((self.time, np.stack([a.pos for a in self.agents]),
                               np.array([a.height for a in self.agents]),
                               np.stack([a.lean_vec() for a in self.agents])))
        self.mode.post_step(dt)
        if self.done:
            self._emit("match_end", winner=self.winner, score=list(self.score))
        return self.tick_events

    def _rewound(self, t: float):
        """Agent positions / heights / lean offsets at world time `t` (clamped to the history)."""
        h = self._hist
        if not h or t >= h[-1][0]:
            return None
        if t <= h[0][0]:
            return h[0][1], h[0][2], h[0][3]
        for i in range(len(h) - 1, 0, -1):
            t0, p0, z0, o0 = h[i - 1]
            t1, p1, z1, o1 = h[i]
            if t0 <= t <= t1:
                k = (t - t0) / max(t1 - t0, 1e-9)
                return p0 + (p1 - p0) * k, z0 + (z1 - z0) * k, o0 + (o1 - o0) * k
        return None

    def tdm_check_end(self):
        lim = self.cfg.kill_limit
        if lim and max(self.team_kills) >= lim:
            self.done = True
            self.winner = int(np.argmax(self.team_kills))
        elif self.cfg.time_limit and self.time >= self.cfg.time_limit:
            self.done = True
            tk = self.team_kills
            self.winner = -1 if tk[0] == tk[1] else int(np.argmax(tk))

    # ------------------------------------------------------------------ agents
    def _update_agent(self, a: Agent, act: Action, dt: float):
        if not a.alive:
            if self.cfg.respawn and self.mode.can_respawn(a) and self.time >= a.respawn_at:
                self._respawn(a)
            else:
                if act.gadget and not a.prev_gadget:
                    self.gadgets.toggle_view(a)
                if a.view_mode:
                    self.gadgets.control(a, act, dt)
            a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload
            a.prev_use, a.prev_gadget = act.use, act.gadget
            return
        gadget_edge = act.gadget and not a.prev_gadget
        use_edge = act.use and not a.prev_use
        a.prev_use, a.prev_gadget = act.use, act.gadget
        if gadget_edge:
            self.gadgets.toggle_view(a)
        if a.view_mode:
            # body stands still while the player looks through a drone / camera
            self.gadgets.control(a, act, dt)
            self._move(a, IDLE, dt)
            self._update_lean(a, 0, dt)
            self._recover(a, dt)
            a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload
            return
        a.yaw = (a.yaw + float(act.look_yaw)) % 360.0
        a.pitch = float(np.clip(a.pitch + float(act.look_pitch), -C.MAX_PITCH, C.MAX_PITCH))
        if a.busy is not None:
            self._update_busy(a, act)
        elif use_edge:
            self.mode.use(a)
        if act.switch and a.busy is None:
            self.switch_weapon(a, int(act.switch))
        self._update_lean(a, int(act.lean) if not a.frozen else 0, dt)
        self._move(a, act if not a.frozen and a.busy is None else _crouch_only(act), dt)
        if a.busy is None and not a.frozen:
            self._weapon(a, act)
        elif a.frozen:
            a.ads = False
            a.fire_queue_until = -1.0
        self._recover(a, dt)
        a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload

    def start_busy(self, a: Agent, kind: str, dur: float, data=None):
        a.busy = (kind, self.time, dur, data)
        a.ads = False
        a.fire_queue_until = -1.0
        self._emit("busy", agent=a.id, kind=kind, dur=dur)

    def _update_busy(self, a: Agent, act: Action):
        kind, t0, dur, data = a.busy
        if not act.use and not (act.fire and kind in ("plant", "place")):
            a.busy = None
            self._emit("busy_cancel", agent=a.id, kind=kind)
            return
        if self.time - t0 >= dur:
            a.busy = None
            self.mode.finish_busy(a, kind, data)

    def _update_lean(self, a: Agent, want: int, dt: float):
        want = float(np.clip(want, -1, 1))
        d = want - a.lean
        step = LEAN_SPEED * dt
        a.lean += float(np.clip(d, -step, step))
        if abs(a.lean) < 1e-4:
            a.lean = a.lean_off = 0.0
            return
        target = a.lean * LEAN_DIST
        # keep the head out of walls: cast sideways from the un-leaned head position
        r = a.right()
        head = np.array([a.pos[0], a.pos[1], a.pos[2] + a.height - 0.15])
        side = np.array([r[0], r[1], 0.0]) * (1.0 if target > 0 else -1.0)
        if len(self.bmin):
            t, _, _ = raycast_boxes(head, side, abs(target) + 0.2, self.bmin, self.bmax)
            lim = max(0.0, float(t[0]) - 0.2)
            if abs(target) > lim:
                target = math.copysign(lim, target)
        a.lean_off = target

    def switch_weapon(self, a: Agent, slot: int):
        if slot == a.slot or slot not in a.inv:
            return False
        ws = a.inv[slot]
        if ws.spec.kind in ("grenade", "gadget") and ws.mag <= 0:
            return False
        if a.throw_at >= 0:
            return False
        a.last_slot = a.slot
        a.slot = slot
        a.ready_at = self.time + ws.spec.draw_time
        a.reload_until = 0.0
        a.ads = False
        a.fire_queue_until = -1.0
        a.switch_back_at = -1.0
        a.shots_in_burst = 0
        self._emit("switch", agent=a.id, slot=slot, weapon=ws.name)
        return True

    # movement -------------------------------------------------------------------
    def _overlaps(self, p, h):
        r = C.PLAYER_RADIUS
        return aabb_overlap_mask((p[0] - r, p[1] - r, p[2]), (p[0] + r, p[1] + r, p[2] + h),
                                 self.smin, self.smax)

    def _drop(self, p, h, max_drop):
        q = p.copy()
        q[2] -= max_drop
        hit = self._overlaps(q, h)
        if hit.any():
            q[2] = self.smax[hit, 2].max() + 1e-4
            return q, True
        return q, False

    def _move(self, a: Agent, act: Action, dt: float):
        spec = a.spec
        want_crouch = bool(act.crouch)
        target_h = C.CROUCH_HEIGHT if want_crouch else C.STAND_HEIGHT
        if a.height != target_h:
            step = C.CROUCH_SPEED * dt
            if target_h < a.height:
                new_h = max(target_h, a.height - step)
                if not a.on_ground:  # tuck the legs up when crouching mid-air
                    p = a.pos.copy()
                    p[2] += a.height - new_h
                    if not self._overlaps(p, new_h).any():
                        a.pos[:] = p
                a.height = new_h
            else:
                new_h = min(target_h, a.height + step)
                if a.on_ground:
                    if not self._overlaps(a.pos, new_h).any():
                        a.height = new_h
                else:
                    p = a.pos.copy()
                    p[2] -= new_h - a.height
                    if not self._overlaps(p, new_h).any():
                        a.pos[:] = p
                        a.height = new_h
        a.crouching = a.height < C.STAND_HEIGHT - 0.2
        a.walking = bool(act.walk)

        yr = math.radians(a.yaw)
        fwd = np.array([math.cos(yr), math.sin(yr)])
        right = np.array([math.sin(yr), -math.cos(yr)])
        wish = fwd * float(act.move_forward) + right * float(act.move_right)
        wl = float(np.hypot(*wish))
        if wl > 1.0:
            wish /= wl
            wl = 1.0
        maxspeed = C.RUN_SPEED * spec.move_mult * self.mode.move_scale
        if a.ads:
            maxspeed *= spec.ads_move_mult
        if a.crouching:
            maxspeed *= C.CROUCH_MULT
        elif act.walk:
            maxspeed *= C.WALK_MULT
        if abs(a.lean) > 0.3:
            maxspeed *= 0.85
        if self.gadgets.items:
            maxspeed *= self.gadgets.speed_mult(a)
        wishspeed = maxspeed * wl
        wishdir = wish / wl if wl > 1e-6 else np.zeros(2)

        vxy = a.vel[:2].copy()
        jumped = False
        if a.on_ground:
            speed = float(np.hypot(*vxy))
            if speed > 1e-6:
                drop = max(speed, C.STOP_SPEED) * C.FRICTION * dt
                vxy *= max(speed - drop, 0.0) / speed
            vxy = self._accelerate(vxy, wishdir, wishspeed, C.GROUND_ACCEL, dt)
            if act.jump and self.time >= a.next_jump and not self._overlaps(a.pos + [0, 0, 0.1], a.height).any():
                a.vel[2] = C.JUMP_SPEED
                a.on_ground = False
                a.next_jump = self.time + C.JUMP_COOLDOWN
                jumped = True
                self._emit("jump", agent=a.id)
        else:
            vxy = self._accelerate(vxy, wishdir, wishspeed, C.AIR_ACCEL, dt)
        a.vel[0], a.vel[1] = vxy
        a.vel[2] -= C.GRAVITY * dt

        was_ground = a.on_ground and not jumped
        fall_speed = a.vel[2]
        before = a.pos.copy()
        self._collide_move(a, dt, was_ground)
        if a.on_ground and not was_ground and not jumped and fall_speed < -4.0:
            self._emit("land", agent=a.id, speed=float(-fall_speed))

        if a.on_ground:
            moved = float(np.hypot(*(a.pos[:2] - before[:2])))
            speed = float(np.hypot(*a.vel[:2]))
            if speed > C.RUN_SPEED * 0.6 and not act.walk and not a.crouching:
                a.step_dist += moved
                if a.step_dist >= C.FOOTSTEP_STRIDE:
                    a.step_dist = 0.0
                    self._emit("footstep", agent=a.id, pos=a.pos.tolist())
                    self.noises.append((self.time, a.pos.copy(), a.team, "step"))

    @staticmethod
    def _accelerate(vxy, wishdir, wishspeed, accel, dt):
        if wishspeed <= 0.0:
            return vxy
        current = float(vxy @ wishdir)
        add = wishspeed - current
        if add <= 0.0:
            return vxy
        return vxy + wishdir * min(accel * dt * wishspeed, add)

    def _collide_move(self, a: Agent, dt: float, was_ground: bool):
        r, h = C.PLAYER_RADIUS, a.height
        pos = a.pos.copy()
        vel = a.vel
        for axis in (0, 1):
            d = vel[axis] * dt
            if d == 0.0:
                continue
            newp = pos.copy()
            newp[axis] += d
            hit = self._overlaps(newp, h)
            if hit.any():
                stepped = False
                if was_ground:
                    up = newp.copy()
                    up[2] += C.STEP_HEIGHT
                    head = pos.copy()
                    head[2] += C.STEP_HEIGHT
                    if not self._overlaps(up, h).any() and not self._overlaps(head, h).any():
                        newp, _ = self._drop(up, h, C.STEP_HEIGHT + 0.01)
                        stepped = True
                if not stepped:
                    if d > 0:
                        newp[axis] = self.smin[hit, axis].min() - r - 1e-4
                    else:
                        newp[axis] = self.smax[hit, axis].max() + r + 1e-4
                    vel[axis] = 0.0
            pos = newp
        newp = pos.copy()
        newp[2] += vel[2] * dt
        hit = self._overlaps(newp, h)
        on_ground = False
        if hit.any():
            if vel[2] <= 0.0:
                newp[2] = self.smax[hit, 2].max() + 1e-4
                on_ground = True
            else:
                newp[2] = self.smin[hit, 2].min() - h - 1e-4
            vel[2] = 0.0
        elif was_ground and vel[2] <= 0.0:
            snapped, found = self._drop(newp, h, C.STEP_HEIGHT)
            if found:
                newp = snapped
                on_ground = True
                vel[2] = 0.0
        a.pos[:] = newp
        a.on_ground = on_ground
        if a.pos[2] < -20.0:  # safety net: should never fall off the ship
            self._kill(a, None, "world", False)

    def _separate_agents(self):
        """Alive agents are cylinders: they push each other aside instead of occupying one body."""
        alive = [a for a in self.agents if a.alive]
        need = C.PLAYER_RADIUS * 2.0
        for _ in range(3):
            pushed = False
            for i, a in enumerate(alive):
                for b in alive[i + 1:]:
                    if a.pos[2] + a.height <= b.pos[2] + 0.02 or b.pos[2] + b.height <= a.pos[2] + 0.02:
                        continue
                    d = b.pos[:2] - a.pos[:2]
                    dist = float(math.hypot(d[0], d[1]))
                    if dist >= need - 1e-4:
                        continue
                    n = np.array([1.0, 0.0]) if dist < 1e-5 else d / dist
                    overlap = need - max(dist, 1e-5)
                    half = overlap * 0.5
                    zero = np.zeros(2)
                    for da, db in ((-n * half, n * half), (-n * overlap, zero), (zero, n * overlap)):
                        pa, pb = a.pos.copy(), b.pos.copy()
                        pa[0] += da[0]
                        pa[1] += da[1]
                        pb[0] += db[0]
                        pb[1] += db[1]
                        if self._overlaps(pa, a.height).any() or self._overlaps(pb, b.height).any():
                            continue
                        a.pos[:] = pa
                        b.pos[:] = pb
                        rel = float((b.vel[:2] - a.vel[:2]) @ n)
                        if rel < 0.0:  # cancel the part of the velocity that is still closing the gap
                            a.vel[:2] += n * (rel * 0.5)
                            b.vel[:2] -= n * (rel * 0.5)
                        pushed = True
                        break
            if not pushed:
                break

    # weapons ------------------------------------------------------------------
    def _weapon(self, a: Agent, act: Action):
        now = self.time
        if a.throw_at >= 0:
            held = bool(act.fire or act.alt)
            if a.weapon.name == "frag" and now >= a.cook_until:
                self._cook_pop(a)
            elif now >= a.throw_at and not held:
                self._release_grenade(a)
        if a.switch_back_at >= 0 and now >= a.switch_back_at:
            a.switch_back_at = -1.0
            self.switch_weapon(a, self.best_slot(a))
        ws = a.weapon
        spec = ws.spec
        fire_edge = act.fire and not a.prev_fire
        alt_edge = act.alt and not a.prev_alt
        ready = now >= a.ready_at

        if spec.kind == "gun":
            if a.reload_until > 0.0 and now >= a.reload_until:
                take = min(spec.mag - ws.mag, ws.reserve)
                ws.mag += take
                ws.reserve -= take
                a.reload_until = 0.0
                self._emit("reload_end", agent=a.id, weapon=ws.name)
            reloading = a.reload_until > 0.0
            was_ads = a.ads
            a.ads = bool(act.alt and spec.ads_fov is not None and ready and not reloading
                         and (not spec.unscope_after_shot or now >= a.next_fire - 1e-9))
            if a.ads and not was_ads:
                a.ads_since = now
            if act.reload and not a.prev_reload and not reloading and ready \
                    and ws.mag < spec.mag and ws.reserve > 0:
                self._start_reload(a, ws)
                return
            if fire_edge:
                a.fire_queue_until = now + 0.15
            want = act.fire if spec.auto else (now <= a.fire_queue_until)
            if want and ready and not reloading:
                if ws.mag <= 0:
                    if ws.reserve > 0:
                        self._start_reload(a, ws)
                    elif fire_edge:
                        self._emit("dry_fire", agent=a.id, weapon=ws.name)
                elif now >= a.next_fire:
                    self._fire_gun(a, ws, spec)
                    a.fire_queue_until = -1.0
        elif spec.kind == "melee":
            a.ads = False
            if ready and now >= a.next_fire:
                if act.alt:
                    self._melee(a, spec, heavy=True)
                elif act.fire:
                    self._melee(a, spec, heavy=False)
        elif spec.kind == "grenade":
            a.ads = False
            if ready and a.throw_at < 0 and a.switch_back_at < 0 and ws.mag > 0 and (fire_edge or alt_edge):
                a.throw_at = now + spec.windup
                a.cook_until = now + spec.fuse
                a.throw_under = bool(alt_edge and not fire_edge)
                self._emit("throw_start", agent=a.id, weapon=ws.name, under=a.throw_under)
        elif spec.kind == "shield":
            a.ads = False
            if ready and fire_edge and ws.mag > 0 and now >= a.next_fire:
                ws.mag -= 1
                a.next_fire = now + spec.interval
                a.protect_until = 0.0
                self.gadgets.shield_flash(a, spec)
        else:  # bomb / placeable gadget
            a.ads = False
            held = spec.kind == "bomb" and act.fire and a.busy is None
            if ready and (fire_edge or held) and ws.mag > 0:
                if spec.kind == "bomb":
                    self.mode.item_fire(a, ws)
                else:
                    self.gadgets.item_fire(a, ws)

    def set_primary(self, a: Agent, name: str):
        """Change the primary weapon; applied immediately during the first seconds after spawning
        when the mode hands out free weapons (TDM / domination)."""
        if not self.mode.free_weapons:
            return False
        a.primary = name
        a.loadout["primary"] = name
        if a.alive and self.time - a.spawn_time < 6.0:
            self.give(a, name, select=True)
            return True
        return False

    def set_loadout(self, a: Agent, key: str) -> bool:
        """Pick a `LOADOUTS` preset (B menu in TDM / domination). Same timing rule as `set_primary`;
        returns True when the weapons were swapped right away."""
        if not self.mode.free_weapons or key not in LOADOUT_BY_KEY:
            return False
        _key, _zh, _en, primary, secondary = LOADOUT_BY_KEY[key]
        a.primary = primary
        a.loadout = {"primary": primary, "secondary": secondary}
        if a.alive and self.time - a.spawn_time < 6.0:
            self.give(a, secondary)
            self.give(a, primary, select=True)
            return True
        return False

    def set_secondary(self, a: Agent, name: str):
        a.loadout["secondary"] = name
        if a.alive and self.mode.free_weapons and self.time - a.spawn_time < 6.0:
            self.give(a, name, select=True)
            return True
        return False

    def best_slot(self, a: Agent) -> int:
        for slot in (SLOT_PRIMARY, SLOT_PISTOL):
            w = a.inv.get(slot)
            if w is not None and w.mag + w.reserve > 0:
                return slot
        return SLOT_MELEE

    def _start_reload(self, a: Agent, ws: WeaponState):
        a.reload_until = self.time + ws.spec.reload_time
        a.ads = False
        a.shots_in_burst = 0
        self._emit("reload", agent=a.id, weapon=ws.name, duration=ws.spec.reload_time)

    def current_spread(self, a: Agent) -> float:
        spec = a.spec
        if spec.kind != "gun":
            return 0.0
        base = spec.spread_ads if a.ads else spec.spread_base
        frac = float(np.hypot(a.vel[0], a.vel[1])) / C.RUN_SPEED
        s = base + a.bloom * (0.6 if a.ads else 1.0)
        s += max(0.0, (frac - 0.3) / 0.7) * spec.spread_move
        if not a.on_ground:
            s += spec.spread_air
        if a.crouching and a.on_ground:
            s *= spec.crouch_mult
        return s

    def _fire_gun(self, a: Agent, ws: WeaponState, spec):
        now = self.time
        ws.mag -= 1
        a.protect_until = 0.0
        if now - a.last_shot > spec.interval * 1.8:
            a.shots_in_burst = 0
        spread = self.current_spread(a)
        eye = a.eye()
        yaw, pitch = a.view_angles()
        hits = []
        dealt: dict[int, list] = {}  # victim -> [damage, group, direction]; pellets land as one hit
        for _ in range(spec.pellets):
            rr = spread * math.sqrt(self.rng.random())
            th = 2.0 * math.pi * self.rng.random()
            d = dir_from_angles(yaw + rr * math.cos(th), pitch + rr * math.sin(th))
            hit = self.trace_bullet(eye, d, spec.range, a.id, spec.penetration)
            hits.append(hit)
            victim = hit["agent"]
            if victim is not None:
                dist = hit["t"]
                dmg = spec.damage * (spec.falloff ** (dist / 10.0)) * hit["mult"]
                group = hit["group"]
                if group == 0:
                    dmg *= spec.head_mult
                elif group == 2:
                    dmg *= spec.leg_mult
                acc = dealt.setdefault(victim, [0.0, group, d])
                acc[0] += dmg
                acc[1] = min(acc[1], group)
            for box, _p in hit["walls"]:
                self.damage_box(box, spec.wall_damage, "bullet")
            if hit["stop_box"] is not None:
                self.damage_box(hit["stop_box"], spec.wall_damage, "bullet")
            if hit["gadget"] is not None:
                self.gadgets.shoot(hit["gadget"], spec.damage)
        for victim, (dmg, group, d) in dealt.items():
            self.apply_damage(self.agents[victim], dmg, a, ws.name, group, d)
        last = hits[-1]
        extra = {}
        if spec.pellets > 1:
            extra["pellets"] = [[[round(float(v), 2) for v in h["point"]], [round(float(v), 2) for v in h["normal"]],
                                 h["mat"], h["agent"]] for h in hits[:-1]]
        self._emit("shot", agent=a.id, weapon=ws.name, origin=eye.tolist(), end=last["point"].tolist(),
                   normal=last["normal"].tolist(), hit_agent=last["agent"],
                   group=last["group"], mat=last["mat"],
                   pierce=[p.tolist() for _b, p in last["walls"]][:3], **extra)
        if not spec.silenced or self.rng.random() < 0.15:
            self.noises.append((now, eye.copy(), a.team, "shot"))
        i = min(a.shots_in_burst, len(spec.recoil) - 1)
        rp, ry = spec.recoil[i]
        j = spec.recoil_jitter
        a.punch[0] += rp * (1.0 + self.rng.uniform(-j, j))
        a.punch[1] += ry + self.rng.uniform(-j, j) * 0.35
        a.shots_in_burst += 1
        a.last_shot = now
        a.bloom = min(spec.spread_max, a.bloom + spec.spread_per_shot)
        if now - a.next_fire < self.dt:
            a.next_fire = a.next_fire + spec.interval
        else:
            a.next_fire = now + spec.interval
        if spec.unscope_after_shot:
            a.ads = False

    def _recover(self, a: Agent, dt: float):
        spec = a.spec
        if self.time - a.last_shot > max(0.09, spec.interval * 1.1):
            a.punch *= math.exp(-spec.punch_recover * dt)
        a.bloom = max(0.0, a.bloom - spec.spread_recover * dt)

    def soft_box(self, box) -> bool:
        """True if bullets pass through `box` (unreinforced destructible wall / barricade)."""
        p = box.params
        if "panel" in p:
            return self.destruct.panel_soft(p["panel"])
        if "dyn" in p:
            return self.gadgets.box_soft(p["dyn"])
        return False

    def damage_box(self, box, amount: float, kind: str = "bullet"):
        p = box.params
        if "panel" in p:
            self.destruct.damage_panel(p["panel"], amount, kind)
        elif "dyn" in p:
            self.gadgets.damage_box(p["dyn"], amount, kind)

    def trace_bullet(self, origin, d, max_dist, shooter_id, penetration: float = 0.0):
        """Hitscan. Soft walls are pierced (damage multiplied by `penetration` per wall, max 3)."""
        walls = []
        o = np.asarray(origin, dtype=np.float64)
        travelled = 0.0
        while True:
            t_w, idx_w, n_w = raycast_boxes(o[None], d[None], max_dist - travelled, self.bmin, self.bmax)
            tw = travelled + float(t_w[0])
            box = self.bullet_boxes[idx_w[0]] if idx_w[0] >= 0 else None
            if box is not None and penetration > 0.0 and len(walls) < 3 and self.soft_box(box):
                hp = origin + d * tw
                walls.append((tw, box, hp))
                # exit point of the box along the ray
                safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
                far = np.where(d > 0, box.mx, box.mn)
                t_exit = float(np.min((np.asarray(far) - hp) / safe))
                travelled = tw + max(t_exit, 0.0) + 1e-3
                o = origin + d * travelled
                if travelled >= max_dist:
                    tw, box = max_dist, None
                    break
                continue
            break
        result = {"t": tw, "agent": None, "group": -1, "normal": n_w[0],
                  "mat": box.mat if box is not None else None, "walls": [], "mult": 1.0, "stop_box": box}
        cands = [g for g in self.agents if g.alive and g.id != shooter_id]
        if cands:
            past = None
            rt = self.rewind.get(shooter_id)
            if rt is not None:
                past = self._rewound(max(rt, self.time - 0.25))
            if past is None:
                mins, maxs = zip(*(g.hitboxes() for g in cands))
            else:
                mins, maxs = zip(*(hitboxes_at(past[0][g.id], past[1][g.id], past[2][g.id]) for g in cands))
            t_a, idx_a, n_a = raycast_boxes(origin[None], d[None], min(tw, max_dist),
                                            np.concatenate(mins), np.concatenate(maxs))
            if idx_a[0] >= 0 and t_a[0] < tw:
                v = cands[idx_a[0] // 3]
                group = int(idx_a[0] % 3)
                if self.gadgets.shield_blocks(v, d, group):
                    result.update(t=float(t_a[0]), normal=-d, mat="metal", stop_box=None)
                else:
                    result.update(t=float(t_a[0]), agent=v.id, group=group, normal=n_a[0], mat="flesh",
                                  stop_box=None)
        t = result["t"] if np.isfinite(result["t"]) else max_dist
        result["gadget"] = None
        if self.gadgets.items:
            g = self.gadgets.ray_hit(np.asarray(origin, dtype=np.float64), d, t)
            if g is not None and g[0] < t:
                t = g[0]
                result.update(agent=None, group=-1, normal=-d, mat="metal", stop_box=None, gadget=g[1])
        result["t"] = t
        result["point"] = origin + d * t
        passed = [(b, p) for tw_, b, p in walls if tw_ < t]
        result["walls"] = passed
        result["mult"] = penetration ** len(passed) if passed else 1.0
        return result

    def segment_clear(self, p, q, smoke=False) -> bool:
        """True if the segment hits no bullet-blocking box (and optionally no smoke).

        Uses the size-sorted LOS box list with chunked early-out, same path as `visibility`,
        instead of a full R×N raycast every call.
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if float(np.linalg.norm(q - p)) < 1e-6:
            return True
        if len(self.los_min) and segments_blocked(p[None], q[None], self.los_min, self.los_max,
                                                   chunk=256)[0]:
            return False
        if smoke and self.smokes:
            return not self._smoke_blocks(p[None], q[None])[0]
        return True

    def _smoke_blocks(self, p, q):
        cs = [s.pos for s in self.smokes]
        rs = [s.radius(self.time) for s in self.smokes]
        return segments_hit_spheres(p, q, cs, rs)

    def apply_damage(self, victim: Agent, amount: float, attacker: Agent | None, weapon: str,
                     group: int = 1, direction=None, kind: str = "bullet") -> float:
        if not victim.alive or self.time < victim.protect_until:
            return 0.0
        if getattr(self.mode, "phase", None) in ("freeze", "prep"):
            return 0.0
        if attacker is not None and attacker.team == victim.team and attacker.id != victim.id \
                and not self.cfg.friendly_fire and weapon not in ("frag", "molotov"):
            return 0.0
        armored_hit = kind == "explosion" or (kind == "bullet" and (group == 1 or (group == 0 and victim.helmet)))
        if victim.armor > 0.0 and armored_hit:
            amount *= 1.0 - victim.armor
        dealt = float(min(amount, victim.hp))
        victim.hp -= amount
        victim.last_damage_time = self.time
        if attacker is not None:
            victim.last_attacker = attacker.id
            victim.last_damage_dir = attacker.pos - victim.pos
            if attacker.id != victim.id:
                attacker.damage_dealt += dealt
        self._emit("damage", attacker=attacker.id if attacker else -1, victim=victim.id,
                   amount=dealt, group=group, weapon=weapon, kind=kind)
        if victim.hp <= 0.0:
            self._kill(victim, attacker, weapon, group == 0)
        return dealt

    def _kill(self, victim: Agent, attacker: Agent | None, weapon: str, headshot: bool):
        if not victim.alive:
            return
        victim.alive = False
        victim.hp = 0.0
        victim.deaths += 1
        victim.death_time = self.time
        victim.respawn_at = self.time + self.cfg.respawn_time
        victim.killer = attacker.id if attacker else -1
        victim.throw_at = -1.0
        victim.cook_until = -1.0
        victim.reload_until = 0.0
        victim.ads = False
        victim.busy = None
        victim.lean = victim.lean_off = 0.0
        if victim.view_mode:
            self.gadgets.exit_view(victim)
        if attacker is not None and attacker.team != victim.team:
            attacker.kills += 1
            if headshot:
                attacker.headshots += 1
            self.team_kills[attacker.team] += 1
        self._emit("kill", attacker=attacker.id if attacker else -1, victim=victim.id,
                   weapon=weapon, headshot=bool(headshot), pos=victim.pos.tolist())
        self.mode.on_kill(victim, attacker, weapon, headshot)

    # melee --------------------------------------------------------------------
    def _melee(self, a: Agent, spec, heavy: bool):
        now = self.time
        a.protect_until = 0.0
        a.next_fire = now + (spec.heavy_interval if heavy else spec.interval)
        delay = spec.heavy_hit_delay if heavy else spec.hit_delay
        self.pending_melee.append((now + delay, a.id, heavy))
        self._emit("melee", agent=a.id, heavy=heavy)

    def _update_melee(self):
        keep = []
        for t_hit, aid, heavy in self.pending_melee:
            if self.time < t_hit:
                keep.append((t_hit, aid, heavy))
                continue
            a = self.agents[aid]
            if not a.alive or a.spec.kind != "melee":
                continue
            spec = a.spec
            rng = spec.heavy_range if heavy else spec.melee_range
            eye = a.eye()
            best = None
            wall = None
            for dy, dp in ((0, 0), (7, 0), (-7, 0), (0, 6), (0, -8), (14, -4), (-14, -4)):
                d = dir_from_angles(a.yaw + dy, a.pitch + dp)
                hit = self.trace_bullet(eye, d, rng, a.id)
                if hit["agent"] is not None:
                    if best is None or hit["t"] < best["t"]:
                        best = hit
                elif wall is None and dy == 0 and dp == 0 and hit["t"] < rng - 1e-3:
                    wall = hit
            if best is None and wall is not None:
                self._emit("melee_hit", agent=a.id, victim=-1, point=wall["point"].tolist(),
                           normal=wall["normal"].tolist(), heavy=heavy, backstab=False)
                if wall["stop_box"] is not None:
                    br = spec.extra.get("breach_radius")
                    if br:
                        self.destruct.explode(wall["point"] - dir_from_angles(a.yaw, a.pitch) * 0.05, br, "soft",
                                              floors=bool(spec.extra.get("floors")), source=a.id)
                    else:
                        self.damage_box(wall["stop_box"], spec.wall_damage * (1.6 if heavy else 1.0), "melee")
            if best is not None:
                v = self.agents[best["agent"]]
                vf = dir_from_angles(v.yaw, 0.0)[:2]
                to_v = v.pos[:2] - a.pos[:2]
                n = float(np.linalg.norm(to_v))
                backstab = n > 1e-6 and float(vf @ (to_v / n)) > 0.45
                dmg = spec.heavy_damage if heavy else spec.damage
                if backstab:
                    dmg *= 2.5 if heavy else 2.0
                self._emit("melee_hit", agent=a.id, victim=v.id, point=best["point"].tolist(),
                           normal=best["normal"].tolist(), heavy=heavy, backstab=backstab)
                self.apply_damage(v, dmg, a, spec.name, 1, None, "melee")
        self.pending_melee = keep

    # grenades -----------------------------------------------------------------
    def _release_grenade(self, a: Agent):
        ws = a.weapon
        spec = ws.spec
        left = a.cook_until - self.time if a.cook_until >= 0 else spec.fuse
        a.throw_at = -1.0
        a.cook_until = -1.0
        if spec.kind != "grenade" or ws.mag <= 0:
            return
        ws.mag -= 1
        a.protect_until = 0.0
        pitch = a.pitch + (6.0 if not a.throw_under else 2.0)
        d = dir_from_angles(a.yaw, pitch)
        speed = spec.throw_speed * (0.42 if a.throw_under else 1.0)
        vel = d * speed + a.vel * 0.6
        vel[2] += 1.2 if not a.throw_under else 0.8
        yr = math.radians(a.yaw)
        right = np.array([math.sin(yr), -math.cos(yr), 0.0])
        eye = a.eye()
        pos = eye + d * 0.3 + right * 0.12 - np.array([0, 0, 0.1])
        if not self.segment_clear(eye, pos):
            pos = eye.copy()
        p = Projectile(self._new_id(), spec.name, a.id, a.team, pos, vel,
                       self.time + max(0.05, left), self.time)
        self.projectiles.append(p)
        self._emit("throw", agent=a.id, weapon=spec.name, proj=p.id, pos=pos.tolist())
        a.switch_back_at = self.time + 0.35

    def _cook_pop(self, a: Agent):
        """HE grenade held past its fuse: it goes off in the hand."""
        ws = a.weapon
        spec = ws.spec
        a.throw_at = -1.0
        a.cook_until = -1.0
        if spec.name != "frag" or ws.mag <= 0:
            return
        ws.mag -= 1
        a.protect_until = 0.0
        pos = a.pos + np.array([0.0, 0.0, a.height * 0.45])
        pr = Projectile(self._new_id(), spec.name, a.id, a.team, pos, np.zeros(3), self.time, self.time)
        self._detonate(pr)
        if a.alive:
            a.switch_back_at = self.time + 0.35

    def _proj_overlaps(self, p):
        r = C.GRENADE_RADIUS
        return aabb_overlap_mask(p - r, p + r, self.smin, self.smax)

    def _update_projectiles(self, dt: float):
        keep = []
        r = C.GRENADE_RADIUS
        for pr in self.projectiles:
            spec = WEAPONS[pr.kind]
            speed = float(np.linalg.norm(pr.vel))
            n = max(1, int(math.ceil(speed * dt / 0.08)))
            h = dt / n
            burst = False
            ignite = False
            for _ in range(n):
                pr.vel[2] -= C.GRAVITY * h
                for axis in (0, 1, 2):
                    step = pr.vel[axis] * h
                    if step == 0.0:
                        continue
                    pr.pos[axis] += step
                    hit = self._proj_overlaps(pr.pos)
                    if hit.any():
                        if pr.kind in self.gadgets.PROJECTILES and pr.explode_at > self.time:
                            pr.explode_at = self.time
                            pr.vel[:] = 0.0
                            pr.pos[axis] -= step * 0.5
                            burst = True
                            break
                        if step > 0:
                            pr.pos[axis] = self.smin[hit, axis].min() - r - 1e-4
                        else:
                            pr.pos[axis] = self.smax[hit, axis].max() + r + 1e-4
                        vin = abs(pr.vel[axis])
                        pr.vel[axis] = -pr.vel[axis] * C.GRENADE_RESTITUTION
                        for o in (0, 1, 2):
                            if o != axis:
                                pr.vel[o] *= C.GRENADE_FRICTION
                        if axis == 2 and step < 0 and abs(pr.vel[2]) < 1.0:
                            pr.vel[2] = 0.0
                        if vin > 2.0 and self.time - pr.last_bounce > 0.08:
                            pr.last_bounce = self.time
                            self._emit("bounce", proj=pr.id, pos=pr.pos.tolist(), speed=vin)
                        if spec.fire_radius > 0.0 and axis == 2 and step < 0:
                            ignite = True
                            burst = True
                            break
                if burst:
                    break
            if ignite or (spec.fire_radius <= 0.0 and self.time >= pr.explode_at):
                self._detonate(pr)
            elif spec.fire_radius > 0.0 and pr.pos[2] < -30.0:
                pass
            else:
                keep.append(pr)
        self.projectiles = keep

    def _detonate(self, pr: Projectile):
        spec = WEAPONS[pr.kind]
        if pr.kind == "smoke":
            s = Smoke(pr.id, pr.pos.copy(), self.time, self.time + spec.smoke_duration, spec.smoke_radius)
            self.smokes.append(s)
            self._emit("smoke", proj=pr.id, pos=pr.pos.tolist(), duration=spec.smoke_duration,
                       radius=spec.smoke_radius)
            return
        if spec.fire_radius > 0.0:
            self.fires.append(Fire(pr.id, pr.pos.copy(), self.time, self.time + spec.fire_duration,
                                   spec.fire_radius, pr.owner, pr.kind, spec.fire_dps))
            self._emit("fire", proj=pr.id, pos=pr.pos.tolist(), duration=spec.fire_duration,
                       radius=spec.fire_radius, owner=pr.owner, weapon=pr.kind)
            self.noises.append((self.time, pr.pos.copy(), pr.team, "explode"))
            return
        if spec.flash_radius > 0.0:
            self._emit("flash", proj=pr.id, pos=pr.pos.tolist(), owner=pr.owner)
            self.flash_at(pr.pos + np.array([0.0, 0.0, 0.1]), spec.flash_radius)
            return
        if pr.kind in self.gadgets.PROJECTILES:
            self.gadgets.projectile_hit(pr)
            return
        self._emit("explode", proj=pr.id, pos=pr.pos.tolist(), weapon=pr.kind, owner=pr.owner)
        self.noises.append((self.time, pr.pos.copy(), pr.team, "explode"))
        owner = self.agents[pr.owner]
        origin = pr.pos + np.array([0.0, 0.0, 0.12])
        R = spec.blast_radius
        self.explosion(origin, R, spec.damage, owner, pr.kind, spec.wall_damage)

    def flash_at(self, origin, radius: float, cone_dir=None):
        """Blind agents that can see `origin` (scaled by distance and facing)."""
        for v in self.agents:
            if not v.alive:
                continue
            eye = v.eye()
            dvec = origin - eye
            dist = float(np.linalg.norm(dvec))
            if dist > radius or not self.segment_clear(eye, origin, smoke=True):
                continue
            if cone_dir is not None:
                if float(np.dot(-dvec / max(dist, 1e-6), cone_dir)) < 0.5:
                    continue
            facing = float(np.dot(dir_from_angles(v.yaw, v.pitch), dvec / max(dist, 1e-6)))
            strength = (1.0 - dist / radius) * (0.35 + 0.65 * max(0.0, facing))
            dur = 1.2 + 6.4 * strength
            if dur > 0.5:
                v.flash_t0 = self.time
                v.flash_until = max(v.flash_until, self.time + dur)
                self._emit("flashed", agent=v.id, dur=dur)

    def _update_fires(self, dt: float):
        alive = []
        for f in self.fires:
            if self.time >= f.t1:
                continue
            alive.append(f)
            f.acc += dt
            if f.acc < 0.2:
                continue
            f.acc -= 0.2
            self._burn(f)
        self.fires = alive

    def _burn(self, f: Fire):
        owner = self.agents[f.owner] if 0 <= f.owner < len(self.agents) else None
        origin = f.pos + np.array([0.0, 0.0, 0.25])
        for v in self.agents:
            if not v.alive:
                continue
            dx, dy = v.pos[0] - f.pos[0], v.pos[1] - f.pos[1]
            dist = math.hypot(dx, dy)
            if dist > f.radius or abs(v.pos[2] - f.pos[2]) > 1.5:
                continue
            body = v.pos + np.array([0.0, 0.0, v.height * 0.45])
            if not self.segment_clear(origin, body):
                continue
            fall = 1.0 - 0.4 * (dist / max(f.radius, 1e-3))
            self.apply_damage(v, f.dps * 0.2 * fall, owner, f.weapon, 1, None, "fire")

    def explosion(self, origin, R: float, damage: float, owner: Agent | None, weapon: str,
                  wall_damage: float = 0.0):
        if wall_damage > 0.0:
            self.destruct.blast(origin, R * 0.45, wall_damage)
            self.gadgets.blast(origin, R * 0.6, wall_damage)
        for v in self.agents:
            if not v.alive:
                continue
            center = v.pos + np.array([0.0, 0.0, v.height * 0.55])
            dist = float(np.linalg.norm(center - origin))
            if dist > R:
                continue
            head = v.eye()
            if not (self.segment_clear(origin, center) or self.segment_clear(origin, head)):
                continue
            dmg = damage * (1.0 - dist / R) ** 0.9
            if owner is not None and v.id == owner.id and weapon != "frag":
                dmg *= 0.6
            self.apply_damage(v, dmg, owner, weapon, 1, None, "explosion")

    # perception ---------------------------------------------------------------
    def visibility(self) -> np.ndarray:
        """Symmetric bool matrix [i, j]: line of sight between enemy agents (walls + smoke)."""
        if self._vis_tick == self.tick and self._vis is not None:
            return self._vis
        n = len(self.agents)
        vis = np.zeros((n, n), dtype=bool)
        pairs, p0, p1 = [], [], []
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            ea = a.eye()
            for j in range(i + 1, n):
                b = self.agents[j]
                if not b.alive or b.team == a.team:
                    continue
                for target in (b.eye(), b.pos + np.array([0.0, 0.0, b.height * 0.6])):
                    pairs.append((i, j))
                    p0.append(ea)
                    p1.append(target)
        if pairs:
            p0 = np.array(p0)
            p1 = np.array(p1)
            clear = ~segments_blocked(p0, p1, self.los_min, self.los_max, chunk=256)
            if self.smokes:
                clear &= ~self._smoke_blocks(p0, p1)
            for (i, j), c in zip(pairs, clear):
                if c:
                    vis[i, j] = vis[j, i] = True
        self._vis = vis
        self._vis_tick = self.tick
        return vis

    def lidar(self, a: Agent, angles_deg, pitch_deg: float = 0.0, max_dist: float = 50.0) -> np.ndarray:
        """Distances of horizontal rays from the eye at yaw offsets `angles_deg`."""
        ang = np.radians(a.yaw + np.asarray(angles_deg, dtype=np.float64))
        p = math.radians(pitch_deg)
        dirs = np.stack([np.cos(ang) * math.cos(p), np.sin(ang) * math.cos(p),
                         np.full_like(ang, math.sin(p))], axis=1)
        o = np.repeat(a.eye()[None], len(ang), axis=0)
        t, _, _ = raycast_boxes(o, dirs, max_dist, self.bmin, self.bmax)
        return np.minimum(t, max_dist)

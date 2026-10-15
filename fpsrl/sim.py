"""Headless, deterministic game simulation.

This is the single source of truth for gameplay. The Panda3D client, scripted
bots and the RL environments all drive the same `World.step(actions)`.
The only game mode is elimination (`fpsrl.mode.Elimination`).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .geometry import aabb_overlap_mask, raycast_boxes, segments_blocked, segments_hit_spheres
from .maps import load_map
from .weapons import (GRENADES, LOADOUT_BY_KEY, LOADOUT_KEYS, MELEE, SLOT_MELEE, SLOT_PISTOL, SLOT_PRIMARY,
                      WEAPONS, random_loadout)

TEAM_NAMES = ("BL", "GR")
LEAN_DIST = 0.38
LEAN_ROLL = 10.0
LEAN_SPEED = 6.0


@dataclass
class Action:
    """Per-tick input of one agent.

    move_*: [-1, 1]; look_*: degrees to add this tick (yaw positive = turn left,
    pitch positive = look up); buttons are "held" states; switch: 0 keeps the
    current weapon, 1..7 selects a slot; lean: -1 left, 0, +1 right.
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


IDLE = Action()


def _crouch_only(act: Action) -> Action:
    return Action(crouch=act.crouch) if act.crouch else IDLE


@dataclass
class MatchConfig:
    team_size: int = 4
    map_name: str = "garage"
    tick_rate: int = C.TICK_RATE
    friendly_fire: bool = False
    # per-agent loadout key ("rifle" | "sniper" | "shotgun"); missing / None / "random" = random
    # draw at match reset
    loadouts: list | None = None
    seed: int | None = None
    mode_params: dict = field(default_factory=dict)  # overrides of Elimination.DEFAULTS


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


_BARE_HANDS = WeaponState(MELEE)  # placeholder while an agent holds nothing


class Agent:
    def __init__(self, aid: int, team: int, name: str, loadout: str | None = None):
        self.id = aid
        self.team = team
        self.name = name
        self.loadout_pref = loadout  # configured key, None / "random" = draw at match reset
        self.loadout = loadout if loadout in LOADOUT_BY_KEY else LOADOUT_KEYS[0]
        self.next_loadout = None  # requested switch, applied at the next spawn
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
        self.spawn_time = 0.0
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
        self.lean = 0.0  # smoothed lean input, -1..1
        self.lean_off = 0.0  # actual sideways head offset in metres (clamped by walls), + = right
        self.third_person = False  # True when controlled from the over-the-shoulder camera
        self.frozen = False  # freeze time: can look, cannot move or shoot
        self.flash_until = 0.0
        self.flash_t0 = 0.0

    @property
    def primary(self) -> str:
        return LOADOUT_BY_KEY[self.loadout][3]

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
        from .mode import Elimination
        self.cfg = cfg or MatchConfig()
        self.map = load_map(self.cfg.map_name)
        self.dt = 1.0 / self.cfg.tick_rate
        self.static_boxes = [b for b in self.map.boxes if b.solid]
        solid = self.static_boxes
        self.smin = np.array([b.mn for b in solid], dtype=np.float64).reshape(-1, 3)
        self.smax = np.array([b.mx for b in solid], dtype=np.float64).reshape(-1, 3)
        bullet = [b for b in solid if b.blocks_bullets]
        self.bullet_boxes = bullet
        self.bmin = np.array([b.mn for b in bullet], dtype=np.float64).reshape(-1, 3)
        self.bmax = np.array([b.mx for b in bullet], dtype=np.float64).reshape(-1, 3)
        big = np.argsort(-np.prod(self.bmax - self.bmin, axis=1), kind="stable")
        self.los_min, self.los_max = self.bmin[big], self.bmax[big]

        n = self.cfg.team_size
        lo = list(self.cfg.loadouts or [])
        self.agents: list[Agent] = []
        for team in (0, 1):
            for i in range(n):
                aid = team * n + i
                self.agents.append(Agent(aid, team, f"{TEAM_NAMES[team]}-{i}", lo[aid] if aid < len(lo) else None))
        self.event_log: deque = deque(maxlen=4096)
        self.noises: deque = deque(maxlen=128)
        self.zone = (self.map.objectives or {}).get("zone")
        self._zone_acc = 0.0
        self.mode = Elimination(self)
        self.reset(self.cfg.seed)

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
        for a in self.agents:
            a.kills = a.deaths = a.headshots = 0
            a.damage_dealt = 0.0
            a.last_damage_time = a.last_shot = a.death_time = -100.0
            a.last_attacker = -1
            a.last_damage_dir[:] = 0.0
            a.next_jump = a.step_dist = 0.0
            a.prev_fire = a.prev_alt = a.prev_reload = False
            a.inv = {}
            a.alive = False
            a.next_loadout = None
            a.loadout = a.loadout_pref if a.loadout_pref in LOADOUT_BY_KEY else random_loadout(self.rng)
        self.mode.reset()
        return self

    def clear_round(self):
        """Remove transient objects between rounds (projectiles, smoke, fire)."""
        self.projectiles = []
        self.smokes = []
        self.fires = []
        self.pending_melee = []
        self._zone_acc = 0.0
        self.noises.clear()

    def zone_dist(self, pos) -> float:
        """Signed distance outside the safe zone (> 0 means in the poison zone)."""
        if not self.zone:
            return -1e9
        cx, cy = self.zone["center"]
        return math.hypot(pos[0] - cx, pos[1] - cy) - self.zone["radius"]

    def outside_zone(self, pos) -> bool:
        return self.zone_dist(pos) > 0.0

    @property
    def score(self) -> list:
        return self.mode.score()

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _emit(self, etype: str, **kw) -> dict:
        self.seq += 1
        e = {"type": etype, "t": self.time, "seq": self.seq, **kw}
        self.tick_events.append(e)
        self.event_log.append(e)
        return e

    def events_since(self, seq: int):
        return [e for e in self.event_log if e["seq"] > seq]

    # ------------------------------------------------------------------ loadouts
    def give_loadout(self, a: Agent):
        _key, _zh, _en, primary, secondary = LOADOUT_BY_KEY[a.loadout]
        a.inv = {SLOT_PRIMARY: WeaponState(primary), SLOT_PISTOL: WeaponState(secondary),
                 SLOT_MELEE: WeaponState(MELEE)}
        for g in GRENADES:
            a.inv[WEAPONS[g].slot] = WeaponState(g)

    def set_loadout(self, a: Agent, key: str) -> bool:
        """Pick one of the fixed `LOADOUTS`. Applied right away during freeze time, otherwise at the
        next round start. Returns True when the weapons were swapped right away."""
        if key not in LOADOUT_BY_KEY:
            return False
        a.loadout_pref = key
        if a.alive and self.mode.phase == "freeze":
            a.loadout = key
            a.next_loadout = None
            self.give_loadout(a)
            self.switch_weapon(a, SLOT_PRIMARY, force=True)
            return True
        a.next_loadout = key
        return False

    def _spawn_at(self, a: Agent, spawn):
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
        a.lean = a.lean_off = 0.0
        a.flash_until = 0.0
        if a.next_loadout is not None:
            a.loadout, a.next_loadout = a.next_loadout, None
        self.give_loadout(a)
        a.slot = SLOT_PRIMARY
        a.last_slot = SLOT_PISTOL
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
        self._update_zone(dt)
        self.smokes = [s for s in self.smokes if self.time < s.t1]
        self.time += dt
        self.tick += 1
        self.mode.post_step(dt)
        if self.done:
            self._emit("match_end", winner=self.winner, score=list(self.score))
        return self.tick_events

    # ------------------------------------------------------------------ agents
    def _update_agent(self, a: Agent, act: Action, dt: float):
        if not a.alive:
            a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload
            return
        a.yaw = (a.yaw + float(act.look_yaw)) % 360.0
        a.pitch = float(np.clip(a.pitch + float(act.look_pitch), -C.MAX_PITCH, C.MAX_PITCH))
        if act.switch:
            self.switch_weapon(a, int(act.switch))
        self._update_lean(a, int(act.lean) if not a.frozen else 0, dt)
        self._move(a, act if not a.frozen else _crouch_only(act), dt)
        if not a.frozen:
            self._weapon(a, act)
        else:
            a.ads = False
            a.fire_queue_until = -1.0
        self._recover(a, dt)
        a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload

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

    def switch_weapon(self, a: Agent, slot: int, force: bool = False):
        if (slot == a.slot and not force) or slot not in a.inv:
            return False
        ws = a.inv[slot]
        if ws.spec.kind == "grenade" and ws.mag <= 0:
            return False
        if a.throw_at >= 0:
            return False
        if slot != a.slot:
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
        maxspeed = C.RUN_SPEED * spec.move_mult
        if a.ads:
            maxspeed *= spec.ads_move_mult
        if a.crouching:
            maxspeed *= C.CROUCH_MULT
        elif act.walk:
            maxspeed *= C.WALK_MULT
        if abs(a.lean) > 0.3:
            maxspeed *= 0.85
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
        if a.pos[2] < -20.0:  # safety net: fell out of the map
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

    def third_person_camera(self, a: Agent, pos=None, punch=None,
                            ads_blend: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Over-the-shoulder camera `(cam_pos, cam_dir)` framing the upper body (PUBG-style)."""
        pos = a.pos if pos is None else np.asarray(pos, dtype=np.float64)
        punch = a.punch if punch is None else np.asarray(punch, dtype=np.float64)
        if ads_blend is None:
            spec = a.spec if a.slot in a.inv else None
            k = 1.0 if (a.alive and a.ads and spec is not None and spec.ads_fov) else 0.0
        else:
            k = float(ads_blend)
        yr = math.radians(a.yaw)
        fwd = np.array([math.cos(yr), math.sin(yr), 0.0])
        right = np.array([math.sin(yr), -math.cos(yr), 0.0])
        side = -1.0 if a.lean_off < -0.05 else 1.0
        anchor = np.array([pos[0], pos[1], pos[2] + a.height - 0.16])
        back = 1.65 + (0.85 - 1.65) * k
        lat = (0.36 + (0.42 - 0.36) * k) * side
        up = 0.08 + (0.02 - 0.08) * k
        pitch = math.radians(a.pitch + punch[0])
        want = anchor - fwd * (back * math.cos(pitch)) + right * lat + np.array([0.0, 0.0, up - back * math.sin(pitch)])
        dv = want - anchor
        dist = float(np.linalg.norm(dv))
        dv_unit = dv / max(dist, 1e-8)
        tt, _, _ = raycast_boxes(anchor[None], dv_unit[None], dist, self.bmin, self.bmax)
        cam_pos = anchor + dv_unit * max(0.15, min(dist, float(tt[0]) - 0.2))
        cam_dir = dir_from_angles(a.yaw + punch[1], a.pitch + punch[0])
        return cam_pos, cam_dir

    def aim_angles(self, a: Agent, max_dist: float = 200.0) -> tuple[float, float]:
        """Firing `(yaw, pitch)` from `a.eye()`. In third-person, rays from `a.eye()` toward the
        3D point under the camera's screen-centre crosshair."""
        spec = a.spec if a.slot in a.inv else None
        scoped_fp = bool(a.ads and spec is not None and spec.scope_overlay)
        if not a.third_person or scoped_fp:
            return a.view_angles()
        eye = a.eye()
        cam_pos, cam_dir = self.third_person_camera(a)
        t_eye = max(0.0, float(np.dot(eye - cam_pos, cam_dir)))
        ray_origin = cam_pos + cam_dir * t_eye
        hit = self.trace_bullet(ray_origin, cam_dir, max_dist, a.id)
        target = ray_origin + cam_dir * max(0.35, float(hit["t"]))
        return angles_to(eye, target)

    def _fire_gun(self, a: Agent, ws: WeaponState, spec):
        now = self.time
        ws.mag -= 1
        if now - a.last_shot > spec.interval * 1.8:
            a.shots_in_burst = 0
        spread = self.current_spread(a)
        eye = a.eye()
        yaw, pitch = self.aim_angles(a, spec.range)
        hits = []
        dealt: dict[int, list] = {}  # victim -> [damage, group, direction]; pellets land as one hit
        for _ in range(spec.pellets):
            rr = spread * math.sqrt(self.rng.random())
            th = 2.0 * math.pi * self.rng.random()
            d = dir_from_angles(yaw + rr * math.cos(th), pitch + rr * math.sin(th))
            hit = self.trace_bullet(eye, d, spec.range, a.id)
            hits.append(hit)
            victim = hit["agent"]
            if victim is not None:
                dmg = spec.damage * (spec.falloff ** (hit["t"] / 10.0))
                group = hit["group"]
                if group == 0:
                    dmg *= spec.head_mult
                elif group == 2:
                    dmg *= spec.leg_mult
                acc = dealt.setdefault(victim, [0.0, group, d])
                acc[0] += dmg
                acc[1] = min(acc[1], group)
        for victim, (dmg, group, d) in dealt.items():
            self.apply_damage(self.agents[victim], dmg, a, ws.name, group, d)
        last = hits[-1]
        extra = {}
        if spec.pellets > 1:
            extra["pellets"] = [[[round(float(v), 2) for v in h["point"]], [round(float(v), 2) for v in h["normal"]],
                                 h["mat"], h["agent"]] for h in hits[:-1]]
        self._emit("shot", agent=a.id, weapon=ws.name, origin=eye.tolist(), end=last["point"].tolist(),
                   normal=last["normal"].tolist(), hit_agent=last["agent"],
                   group=last["group"], mat=last["mat"], **extra)
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

    def trace_bullet(self, origin, d, max_dist, shooter_id):
        """Hitscan against the map and the hit boxes of every other living agent."""
        origin = np.asarray(origin, dtype=np.float64)
        t_w, idx_w, n_w = raycast_boxes(origin[None], d[None], max_dist, self.bmin, self.bmax)
        tw = float(t_w[0])
        box = self.bullet_boxes[idx_w[0]] if idx_w[0] >= 0 else None
        result = {"t": tw, "agent": None, "group": -1, "normal": n_w[0], "mat": box.mat if box is not None else None}
        cands = [g for g in self.agents if g.alive and g.id != shooter_id]
        if cands:
            mins, maxs = zip(*(g.hitboxes() for g in cands))
            t_a, idx_a, n_a = raycast_boxes(origin[None], d[None], min(tw, max_dist),
                                            np.concatenate(mins), np.concatenate(maxs))
            if idx_a[0] >= 0 and t_a[0] < tw:
                v = cands[idx_a[0] // 3]
                result.update(t=float(t_a[0]), agent=v.id, group=int(idx_a[0] % 3), normal=n_a[0], mat="flesh")
        t = result["t"] if np.isfinite(result["t"]) else max_dist
        result["t"] = t
        result["point"] = origin + d * t
        return result

    def segment_clear(self, p, q, smoke=False) -> bool:
        """True if the segment hits no bullet-blocking box (and optionally no smoke)."""
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
        if not victim.alive or self.mode.phase == "freeze":
            return 0.0
        if attacker is not None and attacker.team == victim.team and attacker.id != victim.id \
                and not self.cfg.friendly_fire and weapon not in ("frag", "molotov"):
            return 0.0
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
        victim.killer = attacker.id if attacker else -1
        victim.throw_at = -1.0
        victim.cook_until = -1.0
        victim.reload_until = 0.0
        victim.ads = False
        victim.lean = victim.lean_off = 0.0
        if attacker is not None and attacker.team != victim.team:
            attacker.kills += 1
            if headshot:
                attacker.headshots += 1
            self.team_kills[attacker.team] += 1
        self._emit("kill", attacker=attacker.id if attacker else -1, victim=victim.id,
                   weapon=weapon, headshot=bool(headshot), pos=victim.pos.tolist())

    # melee --------------------------------------------------------------------
    def _melee(self, a: Agent, spec, heavy: bool):
        now = self.time
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
            yaw, pitch = self.aim_angles(a, rng)
            best = None
            wall = None
            for dy, dp in ((0, 0), (7, 0), (-7, 0), (0, 6), (0, -8), (14, -4), (-14, -4)):
                d = dir_from_angles(yaw + dy, pitch + dp)
                hit = self.trace_bullet(eye, d, rng, a.id)
                if hit["agent"] is not None:
                    if best is None or hit["t"] < best["t"]:
                        best = hit
                elif wall is None and dy == 0 and dp == 0 and hit["t"] < rng - 1e-3:
                    wall = hit
            if best is None and wall is not None:
                self._emit("melee_hit", agent=a.id, victim=-1, point=wall["point"].tolist(),
                           normal=wall["normal"].tolist(), heavy=heavy, backstab=False)
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
        yaw, base_pitch = self.aim_angles(a, 60.0)
        pitch = base_pitch + (6.0 if not a.throw_under else 2.0)
        d = dir_from_angles(yaw, pitch)
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
                            break
                if ignite:
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
        self._emit("explode", proj=pr.id, pos=pr.pos.tolist(), weapon=pr.kind, owner=pr.owner)
        self.noises.append((self.time, pr.pos.copy(), pr.team, "explode"))
        owner = self.agents[pr.owner]
        origin = pr.pos + np.array([0.0, 0.0, 0.12])
        self.explosion(origin, spec.blast_radius, spec.damage, owner, pr.kind)

    def flash_at(self, origin, radius: float):
        """Blind agents that can see `origin` (scaled by distance and facing)."""
        for v in self.agents:
            if not v.alive:
                continue
            eye = v.eye()
            dvec = origin - eye
            dist = float(np.linalg.norm(dvec))
            if dist > radius or not self.segment_clear(eye, origin, smoke=True):
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

    def _update_zone(self, dt: float):
        """Poison zone outside the OBJ edge: slowly drains HP while in the live phase."""
        if not self.zone or self.mode.phase != "live":
            return
        self._zone_acc += dt
        if self._zone_acc < 0.5:
            return
        self._zone_acc -= 0.5
        cx, cy = self.zone["center"]
        dmg = float(self.zone.get("dps", 8.0)) * 0.5
        for v in self.agents:
            if v.alive and self.outside_zone(v.pos):
                v.last_damage_dir = np.array([v.pos[0] - cx, v.pos[1] - cy, 0.0])
                self.apply_damage(v, dmg, None, "zone", 1, None, "zone")

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

    def explosion(self, origin, R: float, damage: float, owner: Agent | None, weapon: str):
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

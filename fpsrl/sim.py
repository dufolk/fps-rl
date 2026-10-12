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
from .geometry import aabb_overlap_mask, raycast_boxes, segments_hit_spheres
from .maps import load_map
from .weapons import (MELEE, SECONDARY, SLOT_FRAG, SLOT_MELEE, SLOT_PISTOL, SLOT_PRIMARY,
                      SLOT_SMOKE, WEAPONS)

TEAM_NAMES = ("BL", "GR")


@dataclass
class Action:
    """Per-tick input of one agent.

    move_*: [-1, 1]; look_*: degrees to add this tick (yaw positive = turn left,
    pitch positive = look up); buttons are "held" states; switch: 0 keeps the
    current weapon, 1..5 selects a slot.
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


IDLE = Action()


@dataclass
class MatchConfig:
    team_size: int = 4
    map_name: str = "transport_ship"
    tick_rate: int = C.TICK_RATE
    kill_limit: int = 60
    time_limit: float = 420.0
    friendly_fire: bool = False
    respawn: bool = True
    respawn_time: float = C.RESPAWN_TIME
    primaries: list | None = None  # per-agent primary weapon, default all "ak47"
    seed: int | None = None


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
        self.throw_at = -1.0
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

    # convenient accessors -------------------------------------------------------
    def eye(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], self.pos[2] + self.height - C.EYE_FROM_TOP])

    def view_angles(self) -> tuple[float, float]:
        return self.yaw + self.punch[1], self.pitch + self.punch[0]

    @property
    def weapon(self) -> WeaponState:
        return self.inv[self.slot]

    @property
    def spec(self):
        return WEAPONS[self.inv[self.slot].name]

    @property
    def reloading(self) -> bool:
        return self.reload_until > 0.0

    def hitboxes(self):
        """(3,3) min / max arrays: head, torso, legs."""
        return hitboxes_at(self.pos, self.height)


def hitboxes_at(pos, h):
    x, y, z = pos
    mn = np.array([[x - 0.14, y - 0.14, z + h - 0.28],
                   [x - 0.25, y - 0.25, z + h * 0.43],
                   [x - 0.2, y - 0.2, z]])
    mx = np.array([[x + 0.14, y + 0.14, z + h],
                   [x + 0.25, y + 0.25, z + h - 0.28],
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


class World:
    def __init__(self, cfg: MatchConfig | None = None):
        self.cfg = cfg or MatchConfig()
        self.map = load_map(self.cfg.map_name)
        self.dt = 1.0 / self.cfg.tick_rate
        solid = [b for b in self.map.boxes if b.solid]
        self.solid_boxes = solid
        self.smin = np.array([b.mn for b in solid], dtype=np.float64)
        self.smax = np.array([b.mx for b in solid], dtype=np.float64)
        bullet = [b for b in self.map.boxes if b.solid and b.blocks_bullets]
        self.bullet_boxes = bullet
        self.bmin = np.array([b.mn for b in bullet], dtype=np.float64)
        self.bmax = np.array([b.mx for b in bullet], dtype=np.float64)

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
        self.reset(self.cfg.seed)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.tick = 0
        self.seq = 0
        self.projectiles: list[Projectile] = []
        self.smokes: list[Smoke] = []
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
        for team in (0, 1):
            spawns = list(self.map.spawns[team])
            order = self.rng.permutation(len(spawns))
            members = [a for a in self.agents if a.team == team]
            for k, a in enumerate(members):
                a.kills = a.deaths = a.headshots = 0
                a.damage_dealt = 0.0
                a.last_damage_time = a.last_shot = a.death_time = -100.0
                a.last_attacker = -1
                a.last_damage_dir[:] = 0.0
                a.next_jump = a.respawn_at = a.step_dist = 0.0
                a.prev_fire = a.prev_alt = a.prev_reload = False
                self._spawn_at(a, spawns[order[k % len(spawns)]])
        return self

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
    def _give_loadout(self, a: Agent):
        a.inv = {SLOT_PRIMARY: WeaponState(a.primary), SLOT_PISTOL: WeaponState(SECONDARY),
                 SLOT_MELEE: WeaponState(MELEE), SLOT_FRAG: WeaponState("frag"),
                 SLOT_SMOKE: WeaponState("smoke")}

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
        a.protect_until = self.time + C.SPAWN_PROTECT
        self._give_loadout(a)
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
        a.switch_back_at = -1.0
        a.killer = -1
        self._emit("spawn", agent=a.id, pos=a.pos.tolist())

    def _respawn(self, a: Agent):
        spawns = self.map.spawns[a.team]
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
        for a in self.agents:
            self._update_agent(a, actions.get(a.id, IDLE), dt)
        self._update_melee()
        self._update_projectiles(dt)
        self.smokes = [s for s in self.smokes if self.time < s.t1]
        self.time += dt
        self.tick += 1
        if self.lag_comp:
            self._hist.append((self.time, np.stack([a.pos for a in self.agents]),
                               np.array([a.height for a in self.agents])))
        self._check_end()
        return self.tick_events

    def _rewound(self, t: float):
        """Agent positions / heights at world time `t` (clamped to the recorded history)."""
        h = self._hist
        if not h or t >= h[-1][0]:
            return None
        if t <= h[0][0]:
            return h[0][1], h[0][2]
        for i in range(len(h) - 1, 0, -1):
            t0, p0, z0 = h[i - 1]
            t1, p1, z1 = h[i]
            if t0 <= t <= t1:
                k = (t - t0) / max(t1 - t0, 1e-9)
                return p0 + (p1 - p0) * k, z0 + (z1 - z0) * k
        return None

    def _check_end(self):
        lim = self.cfg.kill_limit
        if lim and max(self.team_kills) >= lim:
            self.done = True
            self.winner = int(np.argmax(self.team_kills))
        elif self.cfg.time_limit and self.time >= self.cfg.time_limit:
            self.done = True
            tk = self.team_kills
            self.winner = -1 if tk[0] == tk[1] else int(np.argmax(tk))
        if self.done:
            self._emit("match_end", winner=self.winner, score=list(self.team_kills))

    # ------------------------------------------------------------------ agents
    def _update_agent(self, a: Agent, act: Action, dt: float):
        if not a.alive:
            if self.cfg.respawn and self.time >= a.respawn_at:
                self._respawn(a)
            a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload
            return
        a.yaw = (a.yaw + float(act.look_yaw)) % 360.0
        a.pitch = float(np.clip(a.pitch + float(act.look_pitch), -C.MAX_PITCH, C.MAX_PITCH))
        if act.switch:
            self.switch_weapon(a, int(act.switch))
        self._move(a, act, dt)
        self._weapon(a, act)
        self._recover(a, dt)
        a.prev_fire, a.prev_alt, a.prev_reload = act.fire, act.alt, act.reload

    def switch_weapon(self, a: Agent, slot: int):
        if slot == a.slot or slot not in a.inv:
            return False
        ws = a.inv[slot]
        if ws.spec.kind == "grenade" and ws.mag <= 0:
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
        maxspeed = C.RUN_SPEED * spec.move_mult
        if a.ads:
            maxspeed *= spec.ads_move_mult
        if a.crouching:
            maxspeed *= C.CROUCH_MULT
        elif act.walk:
            maxspeed *= C.WALK_MULT
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

    # weapons ------------------------------------------------------------------
    def _weapon(self, a: Agent, act: Action):
        now = self.time
        if a.throw_at >= 0 and now >= a.throw_at:
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
        else:  # grenade
            a.ads = False
            if ready and a.throw_at < 0 and a.switch_back_at < 0 and ws.mag > 0 and (fire_edge or alt_edge):
                a.throw_at = now + spec.windup
                a.throw_under = bool(alt_edge and not fire_edge)
                self._emit("throw_start", agent=a.id, weapon=ws.name, under=a.throw_under)

    def set_primary(self, a: Agent, name: str):
        """Change the primary weapon; applied immediately during the first seconds after spawning."""
        a.primary = name
        if a.alive and self.time - a.spawn_time < 6.0:
            a.inv[SLOT_PRIMARY] = WeaponState(name)
            if a.slot == SLOT_PRIMARY:
                a.ready_at = self.time + WEAPONS[name].draw_time
                a.reload_until = 0.0
                a.ads = False
                self._emit("switch", agent=a.id, slot=SLOT_PRIMARY, weapon=name)
            else:
                self.switch_weapon(a, SLOT_PRIMARY)
            return True
        return False

    def best_slot(self, a: Agent) -> int:
        p = a.inv[SLOT_PRIMARY]
        if p.mag + p.reserve > 0:
            return SLOT_PRIMARY
        s = a.inv[SLOT_PISTOL]
        if s.mag + s.reserve > 0:
            return SLOT_PISTOL
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
        for _ in range(spec.pellets):
            rr = spread * math.sqrt(self.rng.random())
            th = 2.0 * math.pi * self.rng.random()
            d = dir_from_angles(yaw + rr * math.cos(th), pitch + rr * math.sin(th))
            hit = self.trace_bullet(eye, d, spec.range, a.id)
            hits.append(hit)
            victim = hit["agent"]
            if victim is not None:
                dist = hit["t"]
                dmg = spec.damage * (spec.falloff ** (dist / 10.0))
                group = hit["group"]
                if group == 0:
                    dmg *= spec.head_mult
                elif group == 2:
                    dmg *= spec.leg_mult
                self.apply_damage(self.agents[victim], dmg, a, ws.name, group, d)
        last = hits[-1]
        self._emit("shot", agent=a.id, weapon=ws.name, origin=eye.tolist(), end=last["point"].tolist(),
                   normal=last["normal"].tolist(), hit_agent=last["agent"],
                   group=last["group"], mat=last["mat"])
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
        t_w, idx_w, n_w = raycast_boxes(origin[None], d[None], max_dist, self.bmin, self.bmax)
        tw = float(t_w[0])
        result = {"t": tw, "agent": None, "group": -1,
                  "normal": n_w[0], "mat": self.bullet_boxes[idx_w[0]].mat if idx_w[0] >= 0 else None}
        cands = [g for g in self.agents if g.alive and g.id != shooter_id]
        if cands:
            past = None
            rt = self.rewind.get(shooter_id)
            if rt is not None:
                past = self._rewound(max(rt, self.time - 0.25))
            if past is None:
                mins, maxs = zip(*(g.hitboxes() for g in cands))
            else:
                mins, maxs = zip(*(hitboxes_at(past[0][g.id], past[1][g.id]) for g in cands))
            t_a, idx_a, n_a = raycast_boxes(origin[None], d[None], min(tw, max_dist),
                                            np.concatenate(mins), np.concatenate(maxs))
            if idx_a[0] >= 0 and t_a[0] < tw:
                result.update(t=float(t_a[0]), agent=cands[idx_a[0] // 3].id, group=int(idx_a[0] % 3),
                              normal=n_a[0], mat="flesh")
        t = result["t"] if np.isfinite(result["t"]) else max_dist
        result["t"] = t
        result["point"] = origin + d * t
        return result

    def segment_clear(self, p, q, smoke=False) -> bool:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        d = q - p
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return True
        t, _, _ = raycast_boxes(p[None], (d / dist)[None], dist, self.bmin, self.bmax)
        if t[0] < dist - 1e-3:
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
        if attacker is not None and attacker.team == victim.team and attacker.id != victim.id \
                and not self.cfg.friendly_fire:
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
        victim.respawn_at = self.time + self.cfg.respawn_time
        victim.killer = attacker.id if attacker else -1
        victim.throw_at = -1.0
        victim.reload_until = 0.0
        victim.ads = False
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
                self.apply_damage(v, dmg, a, "knife", 1, None, "melee")
        self.pending_melee = keep

    # grenades -----------------------------------------------------------------
    def _release_grenade(self, a: Agent):
        a.throw_at = -1.0
        ws = a.weapon
        spec = ws.spec
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
                       self.time + spec.fuse, self.time)
        self.projectiles.append(p)
        self._emit("throw", agent=a.id, weapon=spec.name, proj=p.id, pos=pos.tolist())
        a.switch_back_at = self.time + 0.35

    def _proj_overlaps(self, p):
        r = C.GRENADE_RADIUS
        return aabb_overlap_mask(p - r, p + r, self.smin, self.smax)

    def _update_projectiles(self, dt: float):
        keep = []
        r = C.GRENADE_RADIUS
        for pr in self.projectiles:
            speed = float(np.linalg.norm(pr.vel))
            n = max(1, int(math.ceil(speed * dt / 0.08)))
            h = dt / n
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
            if self.time >= pr.explode_at:
                self._detonate(pr)
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
        self._emit("explode", proj=pr.id, pos=pr.pos.tolist(), weapon=pr.kind, owner=pr.owner)
        self.noises.append((self.time, pr.pos.copy(), pr.team, "explode"))
        owner = self.agents[pr.owner]
        origin = pr.pos + np.array([0.0, 0.0, 0.12])
        R = spec.blast_radius
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
            dmg = spec.damage * (1.0 - dist / R) ** 0.9
            if v.id == owner.id:
                dmg *= 0.6
            self.apply_damage(v, dmg, owner, pr.kind, 1, None, "explosion")

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
            d = p1 - p0
            dist = np.linalg.norm(d, axis=1)
            t, _, _ = raycast_boxes(p0, d / np.maximum(dist, 1e-9)[:, None], dist, self.bmin, self.bmax)
            clear = t >= dist - 1e-3
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

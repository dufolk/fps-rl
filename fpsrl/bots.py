"""Scripted opponents: grid navigation + perception / aiming model."""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .geometry import point_in_poly, poly_center
from .mode import ATT
from .nav import NavGrid
from .sim import IDLE, Action, Agent, World, angles_to, wrap180
from .weapons import SLOT_FIRE, SLOT_FLASH, SLOT_FRAG

DIFFICULTY_LEVELS = {"easy": 0.2, "normal": 0.5, "hard": 0.85, "insane": 1.0}


class Bot:
    """Rule-based player controller. Call `act()` once per simulation tick."""

    # per-round team decisions keyed by (Elimination.round_uid, team): the site attackers push
    _plans: dict = {}

    def __init__(self, world: World, agent_id: int, difficulty: float | str = "normal", seed: int | None = None):
        self.world = world
        self.aid = agent_id
        self.nav = NavGrid.for_world(world)
        self.rng = np.random.default_rng(seed)
        self.set_difficulty(difficulty)
        self.reset()

    def set_difficulty(self, difficulty):
        d = DIFFICULTY_LEVELS.get(difficulty, difficulty) if isinstance(difficulty, str) else float(difficulty)
        d = float(np.clip(d, 0.0, 1.0))
        self.skill = d
        self.reaction = (0.62 - 0.42 * d, 0.85 - 0.5 * d)
        self.aim_err = 3.6 - 2.9 * d
        self.turn_speed = 260.0 + 620.0 * d
        self.gain = 5.0 + 11.0 * d
        self.head_ratio = 0.08 + 0.42 * d
        self.recoil_comp = 0.25 + 0.65 * d
        self.fov = 70.0 + 20.0 * d

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.target = -1
        self.react_at = 0.0
        self.aim_head = False
        self.noise = np.zeros(2)
        self.path: list[np.ndarray] = []
        self.path_i = 0
        self.goal = None
        self.replan_at = 0.0
        self.planned_at = 0.0
        self.last_seen: dict[int, tuple[float, np.ndarray]] = {}
        self.strafe = 0.0
        self.strafe_until = 0.0
        self.burst_left = 0
        self.burst_pause_until = 0.0
        self.crouch_until = 0.0
        self.stuck_pos = None
        self.stuck_check = 0.0
        self.jump_until = 0.0
        self.investigate = None
        self.grenade_state = None
        self.grenade_slot = SLOT_FRAG
        self.grenade_until = 0.0
        self.was_alive = False
        self.task = None

    @property
    def agent(self) -> Agent:
        return self.world.agents[self.aid]

    # ------------------------------------------------------------------ helpers
    def _turn_towards(self, a: Agent, yaw: float, pitch: float, dt: float, gain: float):
        ey = wrap180(yaw - a.yaw)
        ep = pitch - a.pitch
        k = min(1.0, gain * dt)
        lim = self.turn_speed * dt
        return float(np.clip(ey * k, -lim, lim)), float(np.clip(ep * k, -lim, lim)), ey, ep

    def _move_axes(self, a: Agent, world_dir) -> tuple[float, float]:
        n = float(np.hypot(world_dir[0], world_dir[1]))
        if n < 1e-6:
            return 0.0, 0.0
        dx, dy = world_dir[0] / n, world_dir[1] / n
        yr = math.radians(a.yaw)
        fwd = dx * math.cos(yr) + dy * math.sin(yr)
        right = dx * math.sin(yr) - dy * math.cos(yr)
        return fwd, right

    def _plan(self, goal):
        a = self.agent
        self.goal = np.asarray(goal, dtype=np.float64)
        self.path = self.nav.find_path(a.pos, self.goal)
        self.path_i = 1 if len(self.path) > 1 else 0
        self.planned_at = self.world.time
        self.replan_at = self.world.time + self.rng.uniform(2.5, 4.0)

    def _pick_goal(self):
        w = self.world
        a = self.agent
        goal, task, bias = self._objective()
        self.task = task if goal is not None else None
        if self.investigate is not None:
            g, self.investigate = self.investigate, None
            self.task = None
            return g
        recent = [(t, p) for eid, (t, p) in self.last_seen.items() if w.time - t < 8.0]
        if recent and self.rng.random() < 0.7 * (1.0 - bias):
            self.task = None
            return max(recent, key=lambda x: x[0])[1]
        if goal is not None and self.rng.random() < bias:
            return goal
        self.task = None
        pts = w.map.strategic_points
        enemy = np.mean([s[:2] for s in w.map.spawns[1 - w.mode.side(a.team)]], axis=0)
        own = np.mean([s[:2] for s in w.map.spawns[w.mode.side(a.team)]], axis=0)
        span = max(float(np.linalg.norm(enemy - own)), 1.0)
        weights = np.array([1.0 + 0.6 * max(0.0, 1.0 - float(np.linalg.norm(np.asarray(p[:2]) - enemy)) / span)
                            for p in pts])
        k = self.rng.choice(len(pts), p=weights / weights.sum())
        x, y = pts[k][:2]
        x, y = x + self.rng.uniform(-1.5, 1.5), y + self.rng.uniform(-1.0, 1.0)
        n = self.nav.nearest_cell((x, y, a.pos[2]))
        return np.array([x, y, self.nav.nodes[n, 2] if n >= 0 else a.pos[2]])

    # ------------------------------------------------------------------ objectives
    def _zone_point(self, zone, spread: float = 0.35) -> np.ndarray:
        poly = np.asarray(zone["poly"], dtype=np.float64)
        lo, hi = poly.min(0), poly.max(0)
        for _ in range(6):
            x, y = (lo + hi) * 0.5 + (hi - lo) * spread * self.rng.uniform(-1.0, 1.0, 2)
            if point_in_poly(x, y, poly):
                break
        else:
            x, y = poly_center(poly)
        return np.array([x, y, zone.get("z", zone.get("zmin", -0.5) + 0.5)])

    def _team_site(self, m, team) -> int:
        key = (m.round_uid, team)
        if key not in Bot._plans:
            if len(Bot._plans) > 256:
                Bot._plans.clear()
            Bot._plans[key] = int(self.rng.integers(len(m.sites))) if m.sites else -1
        return Bot._plans[key]

    def _objective(self):
        """(goal, task, bias): attackers push one site per round, defenders spread over the sites;
        `bias` is how strongly the bot prefers that over hunting."""
        m = self.world.mode
        a = self.agent
        if m.phase != "live" or not m.sites:
            return None, None, 0.0
        if m.side(a.team) == ATT:
            site = self._team_site(m, a.team)
            return self._zone_point(m.sites[site], 0.45), "push", 0.8
        mates = sorted(o.id for o in m.members(a.team))
        site = mates.index(a.id) % len(m.sites)
        return self._zone_point(m.sites[site], 0.4), "hold", 0.65

    # ------------------------------------------------------------------ main
    def act(self) -> Action:
        w = self.world
        a = self.agent
        if not a.alive:
            if self.was_alive:
                self.reset()
            return IDLE
        if not self.was_alive:
            self.was_alive = True
            self._plan(self._pick_goal())
        if a.frozen:
            if w.time >= self.replan_at:
                self._plan(self._pick_goal())
            return IDLE
        now, dt = w.time, w.dt
        vis = w.visibility()

        seen = []
        for e in w.agents:
            if e.team == a.team or not e.alive or not vis[a.id, e.id]:
                continue
            dvec = e.pos - a.pos
            dist = float(np.linalg.norm(dvec))
            yaw_to, _ = angles_to(a.eye(), e.eye())
            in_fov = abs(wrap180(yaw_to - a.yaw)) < self.fov
            recently_hit = (now - a.last_damage_time < 1.2 and a.last_attacker == e.id)
            if in_fov or dist < 4.0 or recently_hit or e.id == self.target:
                seen.append((dist, e))
                self.last_seen[e.id] = (now, e.pos.copy())

        if not seen:
            for t, pos, team, kind in w.noises:
                if team != a.team and now - t < 0.6:
                    dist = float(np.linalg.norm(pos - a.pos))
                    if dist < (40.0 if kind != "step" else 14.0):
                        if self.investigate is None or self.rng.random() < 0.05:
                            self.investigate = pos.copy()
                            if self.rng.random() < 0.35:
                                self.replan_at = now
            if now - a.last_damage_time < 0.3 and a.last_attacker >= 0:
                att = w.agents[a.last_attacker]
                self.investigate = att.pos.copy()
                self.replan_at = min(self.replan_at, now + 0.2)

        act = Action()
        if seen:
            seen.sort(key=lambda x: x[0])
            cur = [e for _, e in seen if e.id == self.target]
            e = cur[0] if cur else seen[0][1]
            if e.id != self.target:
                self.target = e.id
                self.react_at = now + self.rng.uniform(*self.reaction)
                self.aim_head = self.rng.random() < self.head_ratio
                self.noise = self.rng.normal(0.0, self.aim_err * 2.2, 2)
            self._combat(a, e, act, dt)
            self.grenade_state = None
        else:
            self.target = -1
            if self.grenade_state is not None:
                self._grenade(a, act, dt)
            else:
                self._navigate(a, act, dt)
                self._maybe_grenade(a)
        return act

    # ------------------------------------------------------------------ combat
    def _combat(self, a: Agent, e: Agent, act: Action, dt: float):
        w = self.world
        now = w.time
        dist = float(np.linalg.norm(e.pos - a.pos))
        aim_pt = e.eye() if self.aim_head else e.pos + np.array([0.0, 0.0, e.height * 0.62])
        aim_pt = aim_pt + e.vel * 0.05
        yaw, pitch = angles_to(a.eye(), aim_pt)
        yaw -= a.punch[1] * self.recoil_comp
        pitch -= a.punch[0] * self.recoil_comp
        # Ornstein-Uhlenbeck aim noise that settles while tracking
        self.noise += -self.noise * (3.0 * dt) + self.rng.normal(0.0, self.aim_err * math.sqrt(dt) * 1.4, 2)
        yaw += self.noise[0]
        pitch += self.noise[1] * 0.6
        dyaw, dpitch, ey, ep = self._turn_towards(a, yaw, pitch, dt, self.gain)
        act.look_yaw, act.look_pitch = dyaw, dpitch

        ws = a.weapon
        spec = ws.spec
        if spec.kind != "gun" or (ws.mag + ws.reserve == 0):
            best = w.best_slot(a)
            if best != a.slot:
                act.switch = best
        reacted = now >= self.react_at
        tol = max(0.8, math.degrees(math.atan2(0.32, max(dist, 0.5)))) * (1.6 - 0.5 * self.skill)
        on_target = abs(wrap180(ey)) < tol and abs(ep) < tol * 1.3

        if spec.kind == "gun" and ws.mag == 0 and ws.reserve > 0:
            act.reload = True
        elif spec.kind == "gun":
            if spec.scope_overlay:
                act.alt = now >= self.react_at - 0.25
                if reacted and on_target and a.ads and now - a.ads_since > 0.18:
                    act.fire = True
            elif reacted and on_target:
                if dist > 16.0:
                    if now >= self.burst_pause_until:
                        if self.burst_left <= 0:
                            self.burst_left = int(self.rng.integers(2, 5 + int(3 * self.skill)))
                        act.fire = True
                        if a.prev_fire and now >= a.next_fire - w.dt:
                            self.burst_left -= 1
                            if self.burst_left <= 0:
                                self.burst_pause_until = now + self.rng.uniform(0.18, 0.4)
                else:
                    act.fire = True
                if not spec.auto:
                    act.fire = act.fire and not a.prev_fire
                if spec.ads_fov and dist > 22.0 and self.skill > 0.6:
                    act.alt = True
        elif spec.kind == "melee":
            act.fire = dist < 2.2 and on_target

        # strafing / spacing
        if now >= self.strafe_until:
            self.strafe = float(self.rng.choice([-1.0, 0.0, 1.0], p=[0.42, 0.16, 0.42]))
            self.strafe_until = now + self.rng.uniform(0.25, 0.75)
            if self.skill > 0.5 and dist > 12.0 and self.rng.random() < 0.18:
                self.crouch_until = now + self.rng.uniform(0.4, 1.0)
        fwd = 0.0
        if spec.kind == "melee":
            fwd = 1.0
        elif spec.pellets > 1 and dist > 8.0:
            fwd = 0.8  # shotgun: close the distance
        elif dist > 28.0:
            fwd = 0.6
        elif dist < 5.0:
            fwd = -0.5
        act.move_forward = fwd
        act.move_right = self.strafe if not (spec.scope_overlay and a.ads) else 0.0
        act.crouch = now < self.crouch_until
        if now - a.last_damage_time < 0.2 and self.rng.random() < 0.02 * self.skill:
            act.jump = True

    # ------------------------------------------------------------------ navigation
    def _navigate(self, a: Agent, act: Action, dt: float):
        w = self.world
        now = w.time
        if not self.path or self.path_i >= len(self.path) or now >= self.replan_at:
            self._plan(self._pick_goal())
        look_yaw, look_pitch = a.yaw, 0.0
        if self.path and self.path_i < len(self.path):
            wp = self.path[self.path_i]
            d = wp - a.pos
            if float(np.hypot(d[0], d[1])) < 0.6:
                self.path_i += 1
                if self.path_i >= len(self.path):
                    self.replan_at = now + self.rng.uniform(0.3, 1.2)
            else:
                act.move_forward, act.move_right = self._move_axes(a, d)
                look_yaw = math.degrees(math.atan2(d[1], d[0]))
        # glance towards the most recent sighting
        recent = [(t, p) for t, p in self.last_seen.values() if now - t < 3.0]
        if recent:
            p = max(recent, key=lambda x: x[0])[1]
            look_yaw, look_pitch = angles_to(a.eye(), p + np.array([0, 0, 1.4]))
        dyaw, dpitch, _, _ = self._turn_towards(a, look_yaw, look_pitch, dt, 6.0)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        # re-derive move axes after the turn is applied this tick
        if self.path and self.path_i < len(self.path):
            d = self.path[self.path_i] - a.pos
            yaw_saved = a.yaw
            a.yaw = (a.yaw + dyaw) % 360.0
            act.move_forward, act.move_right = self._move_axes(a, d)
            a.yaw = yaw_saved
        ws = a.weapon
        if ws.spec.kind == "gun" and ws.mag < ws.spec.mag * 0.4 and ws.reserve > 0:
            act.reload = not a.reloading and not a.prev_reload
        if a.slot != w.best_slot(a) and ws.spec.kind != "grenade":
            act.switch = w.best_slot(a)
        # stuck detection
        if now >= self.stuck_check:
            if self.stuck_pos is not None and float(np.linalg.norm(a.pos - self.stuck_pos)) < 0.4 \
                    and (abs(act.move_forward) + abs(act.move_right)) > 0.1:
                self.jump_until = now + 0.2
                self.replan_at = now + 0.3
                self.path_i = min(self.path_i + 1, max(0, len(self.path) - 1))
            self.stuck_pos = a.pos.copy()
            self.stuck_check = now + 1.0
        act.jump = now < self.jump_until

    # ------------------------------------------------------------------ grenades
    def _nades(self, a: Agent) -> dict:
        owned = {}
        for slot in (SLOT_FRAG, SLOT_FLASH, SLOT_FIRE):
            ws = a.inv.get(slot)
            if ws is not None and ws.spec.kind == "grenade" and ws.mag > 0:
                owned[ws.name] = slot
        return owned

    def _maybe_grenade(self, a: Agent):
        w = self.world
        owned = self._nades(a)
        if not owned or self.rng.random() > 0.01 + 0.02 * self.skill:
            return
        recent = [(t, p) for t, p in self.last_seen.values() if 0.8 < w.time - t < 4.0]
        if not recent:
            return
        _, p = max(recent, key=lambda x: x[0])
        dist = float(np.hypot(*(p[:2] - a.pos[:2])))
        if not (8.0 < dist < 26.0):
            return
        if dist > 16.0 and "flash" in owned:
            name = "flash"
        elif dist < 15.0 and "molotov" in owned:
            name = "molotov"
        elif "frag" in owned:
            name = "frag"
        else:
            name = next(iter(owned))
        self.grenade_slot = owned[name]
        self.grenade_state = p.copy()
        self.grenade_until = w.time + 1.6

    def _grenade(self, a: Agent, act: Action, dt: float):
        w = self.world
        slot = self.grenade_slot
        ws = a.inv.get(slot)
        if w.time > self.grenade_until or ws is None or ws.spec.kind != "grenade" or ws.mag <= 0:
            self.grenade_state = None
            return
        if a.slot != slot:
            act.switch = slot
            return
        p = self.grenade_state
        dist = float(np.hypot(*(p[:2] - a.pos[:2])))
        v = ws.spec.throw_speed
        s = min(1.0, C.GRAVITY * dist / (v * v))
        pitch = math.degrees(0.5 * math.asin(s)) - 2.0
        yaw = math.degrees(math.atan2(p[1] - a.pos[1], p[0] - a.pos[0]))
        dyaw, dpitch, ey, ep = self._turn_towards(a, yaw, pitch, dt, 12.0)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        if abs(ey) < 3.0 and abs(ep) < 3.0 and w.time >= a.ready_at and a.throw_at < 0:
            act.fire = not a.prev_fire

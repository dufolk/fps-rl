"""Scripted opponents: grid navigation + perception / aiming model."""
from __future__ import annotations

import heapq
import math

import numpy as np

from . import config as C
from .sim import IDLE, Action, Agent, World, angles_to, wrap180
from .weapons import SLOT_FRAG, SLOT_MELEE, SLOT_PISTOL, SLOT_PRIMARY

DIFFICULTY_LEVELS = {"easy": 0.2, "normal": 0.5, "hard": 0.85, "insane": 1.0}


class NavGrid:
    """Layered walkable grid derived from the map boxes.

    Every 2D cell can hold several walkable heights (one node per storey); nodes in neighbouring
    cells are linked when their floor heights differ by at most a step. `map.nav_max_floor` caps
    which box tops count as floors (e.g. to keep bots off the ship's bridge roof).
    """

    _cache: dict = {}
    MAX_LAYERS = 6

    def __init__(self, world: World, cell: float = 0.5, max_floor: float | None = None,
                 clearance: float = C.PLAYER_RADIUS + 0.02):
        (x0, y0), (x1, y1) = world.map.bounds
        if max_floor is None:
            max_floor = world.map.nav_max_floor
        self.cell = cell
        self.x0, self.y0 = x0, y0
        self.xs = np.arange(x0 + cell / 2, x1, cell)
        self.ys = np.arange(y0 + cell / 2, y1, cell)
        self.W, self.H = len(self.xs), len(self.ys)
        X, Y = np.meshgrid(self.xs, self.ys)
        P = np.stack([X.ravel(), Y.ravel()], axis=1)
        bmin, bmax = world.smin, world.smax
        r = clearance
        nx, ny, nz, ncell = [], [], [], []
        chunk = max(1, 4_000_000 // max(1, len(bmin)))
        for s in range(0, len(P), chunk):
            Q = P[s:s + chunk]
            inside = ((Q[:, None, 0] >= bmin[None, :, 0]) & (Q[:, None, 0] <= bmax[None, :, 0])
                      & (Q[:, None, 1] >= bmin[None, :, 1]) & (Q[:, None, 1] <= bmax[None, :, 1]))
            near = ((Q[:, None, 0] > bmin[None, :, 0] - r) & (Q[:, None, 0] < bmax[None, :, 0] + r)
                    & (Q[:, None, 1] > bmin[None, :, 1] - r) & (Q[:, None, 1] < bmax[None, :, 1] + r))
            for k in range(len(Q)):
                tops = np.unique(bmax[inside[k], 2])
                tops = tops[tops <= max_floor]
                if not len(tops):
                    continue
                nb = np.nonzero(near[k])[0]
                lo, hi = bmin[nb, 2], bmax[nb, 2]
                found = 0
                for t in tops[::-1]:
                    if ((lo < t + 1.7) & (hi > t + 0.46)).any():
                        continue
                    nx.append(Q[k, 0])
                    ny.append(Q[k, 1])
                    nz.append(t)
                    ncell.append(s + k)
                    found += 1
                    if found >= self.MAX_LAYERS:
                        break
        self.nodes = np.stack([np.array(nx), np.array(ny), np.array(nz)], axis=1).reshape(-1, 3)
        self.node_cell = np.array(ncell, dtype=np.int64)
        self._index_cells()
        self._build_adjacency()
        # keep only the component reachable from the team-0 spawn
        sx, sy, sz, _ = world.map.spawns[0][0]
        reach = self._flood(self.nearest_cell((sx, sy, sz)))
        keep = np.nonzero(reach)[0]
        self.nodes = self.nodes[keep]
        self.node_cell = self.node_cell[keep]
        self._index_cells()
        self._build_adjacency()
        self.walkable_cells = np.arange(len(self.nodes))
        # 2D summary (highest walkable floor per cell) for code that only needs the ground plan
        self.walk = np.zeros(self.H * self.W, dtype=bool)
        self.walk[self.node_cell] = True
        self.walk = self.walk.reshape(self.H, self.W)

    @classmethod
    def for_world(cls, world: World) -> "NavGrid":
        key = world.cfg.map_name
        if key not in cls._cache:
            cls._cache[key] = cls(world)
        return cls._cache[key]

    def _index_cells(self):
        self.cell_nodes: dict[int, list[int]] = {}
        for i, c in enumerate(self.node_cell.tolist()):
            self.cell_nodes.setdefault(c, []).append(i)

    def _build_adjacency(self):
        H, W = self.H, self.W
        self.adj: list[list[tuple[int, float]]] = [[] for _ in range(len(self.nodes))]
        dirs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        z = self.nodes[:, 2] if len(self.nodes) else np.zeros(0)
        cn = self.cell_nodes
        step = C.STEP_HEIGHT

        def match(cell, zi):
            for j in cn.get(cell, ()):
                if abs(z[j] - zi) <= step:
                    return j
            return -1

        for i in range(len(self.nodes)):
            cell = int(self.node_cell[i])
            r, c = divmod(cell, W)
            zi = z[i]
            lst = self.adj[i]
            for dc, dr, cost in dirs:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                j = match(rr * W + cc, zi)
                if j < 0:
                    continue
                if dr and dc and (match(r * W + cc, zi) < 0 or match(rr * W + c, zi) < 0):
                    continue
                lst.append((j, cost + 0.5 * abs(z[j] - zi)))

    def _flood(self, start: int) -> np.ndarray:
        seen = np.zeros(len(self.nodes), dtype=bool)
        if start < 0:
            return seen
        stack = [start]
        seen[start] = True
        while stack:
            n = stack.pop()
            for m, _ in self.adj[n]:
                if not seen[m]:
                    seen[m] = True
                    stack.append(m)
        return seen

    def cell_rc(self, p):
        c = int((p[0] - self.x0) / self.cell)
        r = int((p[1] - self.y0) / self.cell)
        return min(max(r, 0), self.H - 1), min(max(c, 0), self.W - 1)

    def nearest_cell(self, p, max_ring: int = 8) -> int:
        """Nearest walkable node, preferring the storey closest to the query height."""
        r0, c0 = self.cell_rc(p)
        z = p[2] if len(p) > 2 else 0.0
        best, best_d = -1, 1e9
        for ring in range(max_ring + 1):
            for r in range(r0 - ring, r0 + ring + 1):
                for c in range(c0 - ring, c0 + ring + 1):
                    if not (0 <= r < self.H and 0 <= c < self.W) or max(abs(r - r0), abs(c - c0)) != ring:
                        continue
                    for j in self.cell_nodes.get(r * self.W + c, ()):
                        dz = self.nodes[j, 2] - z
                        d = (r - r0) ** 2 + (c - c0) ** 2 + 4.0 * dz * dz + (40.0 if dz > 1.2 else 0.0)
                        if d < best_d:
                            best, best_d = j, d
            if best >= 0 and ring >= 1:
                return best
        return best

    def point(self, idx: int) -> np.ndarray:
        return self.nodes[idx].copy()

    def random_point(self, rng) -> np.ndarray:
        return self.nodes[rng.integers(len(self.nodes))].copy()

    def find_path(self, start, goal) -> list[np.ndarray]:
        s = self.nearest_cell(start)
        g = self.nearest_cell(goal)
        if s < 0 or g < 0:
            return []
        nodes = self.nodes
        gp = nodes[g]
        inv = 1.0 / self.cell

        def h(n):
            dx = abs(nodes[n, 0] - gp[0]) * inv
            dy = abs(nodes[n, 1] - gp[1]) * inv
            return (dx + dy) + (1.414 - 2.0) * min(dx, dy) + abs(nodes[n, 2] - gp[2]) * inv

        openq = [(h(s), 0.0, s)]
        came = {s: -1}
        cost = {s: 0.0}
        while openq:
            _, gcost, n = heapq.heappop(openq)
            if n == g:
                break
            if gcost > cost.get(n, 1e18):
                continue
            for m, w in self.adj[n]:
                nc = gcost + w
                if nc < cost.get(m, 1e18):
                    cost[m] = nc
                    came[m] = n
                    heapq.heappush(openq, (nc + h(m), nc, m))
        if g not in came:
            return []
        path = []
        n = g
        while n != -1:
            path.append(n)
            n = came[n]
        path.reverse()
        return self._smooth([self.point(i) for i in path])

    def _line_walkable(self, a, b) -> bool:
        d = b[:2] - a[:2]
        dist = float(np.hypot(*d))
        steps = max(2, int(dist / (self.cell * 0.5)))
        prev_f = a[2]
        for k in range(1, steps + 1):
            p = a[:2] + d * (k / steps)
            r, c = self.cell_rc(p)
            f = None
            for j in self.cell_nodes.get(r * self.W + c, ()):
                if abs(self.nodes[j, 2] - prev_f) <= C.STEP_HEIGHT:
                    f = self.nodes[j, 2]
                    break
            if f is None:
                return False
            prev_f = f
        return abs(prev_f - b[2]) <= C.STEP_HEIGHT

    def _smooth(self, pts):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = min(len(pts) - 1, i + 24)
            while j > i + 1 and not self._line_walkable(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out


class Bot:
    """Rule-based player controller. Call `act()` once per simulation tick."""

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
        self.grenade_until = 0.0
        self.was_alive = False

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
        self.replan_at = self.world.time + self.rng.uniform(2.5, 4.0)

    def _pick_goal(self):
        w = self.world
        a = self.agent
        if self.investigate is not None:
            g, self.investigate = self.investigate, None
            return g
        recent = [(t, p) for eid, (t, p) in self.last_seen.items() if w.time - t < 8.0]
        if recent and self.rng.random() < 0.7:
            return max(recent, key=lambda x: x[0])[1]
        pts = w.map.strategic_points
        enemy_side = 1.0 if a.team == 0 else -1.0
        weights = np.array([1.0 + 0.6 * max(0.0, enemy_side * x / 20.0) for x, _ in pts])
        k = self.rng.choice(len(pts), p=weights / weights.sum())
        x, y = pts[k]
        return np.array([x + self.rng.uniform(-1.5, 1.5), y + self.rng.uniform(-1.0, 1.0), 0.0])

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
    def _maybe_grenade(self, a: Agent):
        w = self.world
        if a.inv[SLOT_FRAG].mag <= 0 or self.rng.random() > 0.01 + 0.02 * self.skill:
            return
        recent = [(t, p) for t, p in self.last_seen.values() if 0.8 < w.time - t < 4.0]
        if not recent:
            return
        _, p = max(recent, key=lambda x: x[0])
        dist = float(np.hypot(*(p[:2] - a.pos[:2])))
        if 9.0 < dist < 24.0:
            self.grenade_state = p.copy()
            self.grenade_until = w.time + 1.6

    def _grenade(self, a: Agent, act: Action, dt: float):
        w = self.world
        if w.time > self.grenade_until or a.inv[SLOT_FRAG].mag <= 0:
            self.grenade_state = None
            return
        if a.slot != SLOT_FRAG:
            act.switch = SLOT_FRAG
            return
        p = self.grenade_state
        dist = float(np.hypot(*(p[:2] - a.pos[:2])))
        v = 17.0
        s = min(1.0, C.GRAVITY * dist / (v * v))
        pitch = math.degrees(0.5 * math.asin(s)) - 2.0
        yaw = math.degrees(math.atan2(p[1] - a.pos[1], p[0] - a.pos[0]))
        dyaw, dpitch, ey, ep = self._turn_towards(a, yaw, pitch, dt, 12.0)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        if abs(ey) < 3.0 and abs(ep) < 3.0 and w.time >= a.ready_at and a.throw_at < 0:
            act.fire = not a.prev_fire

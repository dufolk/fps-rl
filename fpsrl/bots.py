"""Scripted opponents: grid navigation + perception / aiming model."""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .geometry import point_in_poly, poly_center
from .modes.rounds import ATT, DEF, RoundMode
from .nav import NavGrid
from .sim import IDLE, Action, Agent, World, angles_to, wrap180
from .weapons import SLOT_BOMB, SLOT_FRAG, SLOT_GADGET, SLOT_MELEE, SLOT_PISTOL, SLOT_PRIMARY, SLOT_SMOKE

DIFFICULTY_LEVELS = {"easy": 0.2, "normal": 0.5, "hard": 0.85, "insane": 1.0}
# tasks the bot keeps doing even when it hears / remembers enemies
FORCED_TASKS = ("plant", "defuse", "fetch", "reinforce", "breach")
BUSY_TASKS = ("plant", "defuse", "reinforce", "barricade", "place")


class Bot:
    """Rule-based player controller. Call `act()` once per simulation tick."""

    # per-round team decisions keyed by RoundMode.round_uid: attack site, claimed reinforcement walls
    _plans: dict = {}

    def __init__(self, world: World, agent_id: int, difficulty: float | str = "normal", seed: int | None = None):
        self.world = world
        self.aid = agent_id
        self.nav = NavGrid.for_world(world)
        self.rng = np.random.default_rng(seed)
        self.bought = None
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
        self.path_geo = -1
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
        self.grenade_until = 0.0
        self.was_alive = False
        self.task = None
        self.task_data = None
        self.stuck_n = 0
        self.bash_until = 0.0
        self.breached = None  # round uid in which this bot is done breaching
        self.drone_path: list[np.ndarray] = []
        self.drone_i = 0
        self.drone_replan = 0.0
        self.drone_last = (0.0, np.zeros(3))

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
        self.path_geo = self.world.geo_version
        self.planned_at = self.world.time
        self.replan_at = self.world.time + self.rng.uniform(2.5, 4.0)

    def _pick_goal(self):
        w = self.world
        a = self.agent
        goal, task, bias = self._objective()
        self.task = task if goal is not None else None
        if goal is not None and task in FORCED_TASKS:
            return goal
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

    def _objective(self):
        """(goal, task, bias): where the game mode wants this bot, what to do there and how strongly
        it prefers that over hunting."""
        m = self.world.mode
        a = self.agent
        if m.name == "dom":
            return self._dom_objective(m, a)
        if isinstance(m, RoundMode):
            return self._round_objective(m, a)
        return None, None, 0.0

    def _dom_objective(self, m, a):
        if not m.points:
            return None, None, 0.0
        cur = self.task_data if self.task == "capture" else None
        if cur is not None and (m.owner[cur] != a.team or m.present[cur][1 - a.team]) \
                and self.rng.random() < 0.8:
            k = cur
        else:
            scores = []
            for i, c in enumerate(m.centers):
                d = math.hypot(c[0] - a.pos[0], c[1] - a.pos[1])
                s = 30.0 / (8.0 + d)
                if m.owner[i] != a.team:
                    s *= 3.0
                if m.present[i][1 - a.team]:
                    s *= 1.5
                scores.append(s)
            p = np.array(scores) ** 2
            k = int(self.rng.choice(len(p), p=p / p.sum()))
        self.task_data = k
        hunter = self.aid % 4 == 3
        bias = 0.45 if hunter else (0.85 if m.owner[k] != a.team else 0.5)
        return self._zone_point(m.points[k]), "capture", bias

    def _team_site(self, m, team) -> int:
        key = (m.round_uid, team, "site")
        if key not in Bot._plans:
            if len(Bot._plans) > 256:
                Bot._plans.clear()
            Bot._plans[key] = int(self.rng.choice(m.active_sites)) if m.active_sites else -1
        return Bot._plans[key]

    def _round_objective(self, m, a):
        side = m.side(a.team)
        b = m.bomb
        if m.phase == "over" or not m.sites:
            return None, None, 0.0
        if side == ATT:
            if m.phase in ("freeze", "prep"):
                return None, None, 0.0
            if b is not None and b["state"] == "carried" and b["carrier"] == a.id:
                site = self._team_site(m, a.team)
                return self._zone_point(m.sites[site], 0.25), "plant", 1.0
            if b is not None and b["state"] == "dropped":
                bp = np.asarray(b["pos"])
                closest = min(m.members(a.team, alive_only=True), key=lambda o: float(np.linalg.norm(o.pos - bp)))
                if closest.id == a.id:
                    return bp.copy(), "fetch", 1.0
            if m.phase == "planted" and b is not None:
                bp = np.asarray(b["pos"])
                ang = self.rng.uniform(0, 2 * math.pi)
                r = self.rng.uniform(3.0, 7.0)
                g = bp + np.array([math.cos(ang) * r, math.sin(ang) * r, 0.0])
                return g, "guard", 0.9
            site = self._team_site(m, a.team)
            if site < 0:
                return None, None, 0.0
            slot = self._breach_tool(a)
            if m.name == "r6" and slot is not None and self.breached != m.round_uid:
                t = self.task_data if self.task == "breach" else None
                if t is None or not self.world.destruct.panels[t[0]].alive:
                    t = self._wall_target(m, a, [site], outside=True, hard=a.inv[slot].name == "thermite")
                if t is not None:
                    self.task_data = t
                    return t[1], "breach", 1.0
                self.breached = m.round_uid
            return self._zone_point(m.sites[site], 0.45), "push", 0.8
        # defenders
        if m.phase == "planted" and b is not None:
            return np.asarray(b["pos"], dtype=np.float64).copy(), "defuse", 1.0
        if m.phase == "prep" and getattr(m, "reinforce_left", 0) > 0:
            t = self.task_data if self.task == "reinforce" else None
            if t is None or not self.world.destruct.panel_soft(t[0]):
                t = self._wall_target(m, a, m.active_sites, outside=False)
            if t is not None:
                self.task_data = t
                return t[1], "reinforce", 1.0
        if not m.active_sites:
            return None, None, 0.0
        mates = sorted(o.id for o in m.members(a.team))
        site = m.active_sites[mates.index(a.id) % len(m.active_sites)]
        bias = 0.9 if m.name == "r6" else 0.65
        return self._zone_point(m.sites[site], 0.4), "hold", bias

    def _wall_target(self, m, a, sites, outside: bool, hard: bool = False):
        """(panel id, stand point, look point) of a soft wall next to one of `sites`, approached from
        the site side (defenders reinforcing) or from outside (attackers breaching). Each wall is
        claimed by one bot per round and team."""
        w = self.world
        claimed = Bot._plans.setdefault((m.round_uid, a.team, "walls"), set())
        best, best_d = None, 1e9
        for si in sites:
            zone = m.sites[si]
            c = m.site_centers[si]
            zf = zone.get("z", zone.get("zmin", -0.5) + 0.5)
            for p in w.destruct.panels:
                if p.axis == 2 or not p.alive or (p.reinforced and not hard) or p.wall in claimed:
                    continue
                if not (zf + 0.8 < p.center[2] < zf + 2.2):
                    continue
                d = math.hypot(p.center[0] - c[0], p.center[1] - c[1])
                if d > 10.0:
                    continue
                n = np.zeros(3)
                n[p.axis] = 1.0 if (c[p.axis] > p.center[p.axis]) != outside else -1.0
                stand = p.center + n * 0.85
                stand[2] = zf
                node = self.nav.nearest_cell(stand)
                if node < 0 or np.linalg.norm(self.nav.nodes[node] - stand) > 0.45:
                    continue
                far = math.hypot(stand[0] - a.pos[0], stand[1] - a.pos[1])
                dd = 0.5 * d + far if outside else d + 0.3 * far
                if dd < best_d:
                    best, best_d = (p, stand), dd
        if best is None:
            return None
        p, stand = best
        claimed.add(p.wall)
        look = p.center.copy()
        look[2] = stand[2] + (C.STAND_HEIGHT - C.EYE_FROM_TOP if not outside else 1.0)
        return p.id, stand, look

    def _breach_tool(self, a):
        """Slot of the bot's best wall breaching item with charges left, else None."""
        for slot in (SLOT_GADGET, SLOT_SMOKE):
            ws = a.inv.get(slot)
            if ws is not None and ws.name in ("thermite", "hammer", "breach") and ws.mag > 0:
                return slot
        return None

    def _work_at(self, a: Agent, act: Action, dt: float, stand, look):
        """Walk onto `stand` and face `look`. None while still out of reach (let the path finder
        steer), else whether the bot stands in place and is aimed."""
        off = stand - a.pos
        dist = math.hypot(off[0], off[1])
        if dist > 1.5:
            return None
        yaw, pitch = angles_to(a.eye(), look)
        dyaw, dpitch, ey, ep = self._turn_towards(a, yaw, pitch, dt, 12.0)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        if dist > 0.3:
            yaw_saved = a.yaw
            a.yaw = (a.yaw + dyaw) % 360.0
            fwd, right = self._move_axes(a, off)
            a.yaw = yaw_saved
            act.move_forward, act.move_right = fwd * 0.5, right * 0.5
            act.walk = True
            return False
        act.move_forward = act.move_right = 0.0
        return abs(ey) < 4.0 and abs(ep) < 4.0

    def _task_act(self, a: Agent, act: Action, dt: float) -> bool:
        """Carry out the current task once in place. Returns True when it took over the controls."""
        w = self.world
        m = w.mode
        task = self.task
        if task == "plant":
            if SLOT_BOMB not in a.inv or m.site_at(a.pos) < 0 or not a.on_ground:
                return False
            act.move_forward = act.move_right = 0.0
            if a.slot != SLOT_BOMB:
                act.switch = SLOT_BOMB
            else:
                act.fire = True
            return True
        if task == "defuse":
            b = m.bomb
            if b is None or b["state"] != "planted":
                return False
            bp = np.asarray(b["pos"])
            if math.hypot(*(bp[:2] - a.pos[:2])) > 1.1 or abs(bp[2] - a.pos[2]) > 1.3:
                return False
            act.move_forward = act.move_right = 0.0
            act.crouch = True
            act.use = not a.prev_use
            return True
        if task == "reinforce":
            pid, stand, look = self.task_data
            p = w.destruct.panels[pid]
            if p.reinforced or not p.alive or getattr(m, "reinforce_left", 0) <= 0:
                self.task = None
                self.replan_at = w.time
                return False
            ready = self._work_at(a, act, dt, stand, look)
            if ready is None:
                return False
            act.use = ready and not a.prev_use
            return True
        if task == "breach":
            pid, stand, look = self.task_data
            p = w.destruct.panels[pid]
            slot = self._breach_tool(a)
            placed = any(it["kind"] == "charge" and it["owner"] == a.id for it in w.gadgets.items.values())
            if not p.alive or slot is None or placed:
                # done: back off from the charge, then push through the hole
                self.breached = m.round_uid
                self.task = None
                away = stand + (stand - p.center) * np.array([1.0, 1.0, 0.0]) * 3.0
                self._plan(away)
                self.replan_at = w.time + (2.5 if placed else 0.3)
                return False
            ready = self._work_at(a, act, dt, stand, look)
            if ready is None:
                return False
            if a.slot != slot:
                act.switch = slot
            elif ready and w.time >= a.ready_at:
                act.fire = not a.prev_fire
            return True
        return False

    def _drone_act(self, a: Agent) -> Action:
        """Siege attackers in the prep phase: drive a drone towards the team's site and spot
        defenders on the way. Leaves the drone view once the round goes live."""
        w = self.world
        m = w.mode
        g = w.gadgets
        if m.phase != "prep" or m.side(a.team) != ATT:
            return Action(gadget=bool(a.view_mode) and not a.prev_gadget)
        if w.time >= self.replan_at:
            self._plan(self._pick_goal())
        if not a.view_mode:
            start = m.phase_end - m.p["prep_time"] + 0.5 + (a.id % 3)
            if g.drones_left.get(a.id, 0) <= 0 or w.time < start:
                return IDLE
            self.drone_path = []
            return Action(gadget=not a.prev_gadget)
        it = g.items.get(a.view_ref)
        if it is None:
            return IDLE
        pos = it["pos"]
        act = Action()
        if not self.drone_path or self.drone_i >= len(self.drone_path) or w.time >= self.drone_replan:
            goal = self._zone_point(m.sites[self._team_site(m, a.team)], 0.4)
            self.drone_path = self.nav.find_path(pos, goal)
            self.drone_i = 1 if len(self.drone_path) > 1 else 0
            self.drone_replan = w.time + 6.0
        if self.drone_i < len(self.drone_path):
            d = self.drone_path[self.drone_i] - pos
            if math.hypot(d[0], d[1]) < 0.5:
                self.drone_i += 1
            ey = wrap180(math.degrees(math.atan2(d[1], d[0])) - it["yaw"])
            act.look_yaw = float(np.clip(ey * 0.3, -8.0, 8.0))
            act.move_forward = 1.0 if abs(ey) < 35.0 else 0.2
        t0, p0 = self.drone_last
        if w.time - t0 > 0.6:
            act.jump = float(np.linalg.norm(pos[:2] - p0[:2])) < 0.1 and act.move_forward > 0.5
            self.drone_last = (w.time, pos.copy())
        eye = pos + np.array([0.0, 0.0, 0.13])
        fwd = np.array([math.cos(math.radians(it["yaw"])), math.sin(math.radians(it["yaw"]))])
        for e in w.agents:
            if e.team == a.team or not e.alive or e.spotted_until > w.time + 0.5:
                continue
            v = e.pos[:2] - eye[:2]
            dist = float(np.hypot(*v))
            if dist < 30.0 and float(fwd @ v) > 0.85 * dist \
                    and w.segment_clear(eye, e.pos + np.array([0.0, 0.0, e.height * 0.6]), smoke=True):
                act.fire = not a.prev_fire
                break
        return act

    def _continue_busy(self, a: Agent) -> Action:
        kind = a.busy[0]
        return Action(use=True, fire=kind in ("plant", "place"), crouch=kind == "defuse")

    # ------------------------------------------------------------------ economy
    def maybe_buy(self, a: Agent):
        w = self.world
        m = w.mode
        if self.bought == m.round_uid or not m.can_buy_now(a):
            return
        self.bought = m.round_uid
        side = m.side(a.team)
        money = a.money
        pistol_round = m.round_no == 1 or (m.swapped and m.round_no == m.p["half_rounds"] + 1)
        prim = a.inv.get(SLOT_PRIMARY)
        if prim is None:
            want = None
            if money >= 5700 and self.skill > 0.55 and self.rng.random() < 0.18:
                want = "awm"
            elif money >= 3700:
                want = "ak47" if side == ATT else str(self.rng.choice(["m4a4", "m4a1"]))
            elif money >= 2650:
                want = "famas" if side == DEF or self.rng.random() < 0.5 else "ak47"
            elif money >= 1900 and (self.rng.random() < 0.6 or money >= 2400):
                want = str(self.rng.choice(["mp9", "uzi", "mp5"]))
            if want is not None:
                w.buy(a, want)
        if a.inv.get(SLOT_PRIMARY) is None and pistol_round and a.money >= 700 and self.rng.random() < 0.4:
            w.buy(a, "deagle")
        if a.money >= 300 and SLOT_FRAG not in a.inv and (not pistol_round or self.rng.random() < 0.5):
            w.buy(a, "frag")
        if a.money >= 700 and SLOT_SMOKE not in a.inv and self.rng.random() < 0.5:
            w.buy(a, "smoke")
        if side == DEF and a.money >= 400 and SLOT_BOMB not in a.inv and self.rng.random() < 0.6:
            w.buy(a, "defuser")
        if a.slot != w.best_slot(a):
            w.switch_weapon(a, w.best_slot(a))

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
        if w.mode.name == "bomb":
            self.maybe_buy(a)
        if a.busy is not None and a.busy[0] in BUSY_TASKS:
            return self._continue_busy(a)
        if w.mode.name == "r6" and (a.frozen or a.view_mode):
            return self._drone_act(a)
        if a.frozen:
            if w.time >= self.replan_at:
                self._plan(self._pick_goal())
            return IDLE
        now, dt = w.time, w.dt
        vis = w.visibility()

        seen = []
        for e in w.agents:
            if e.team != a.team and e.alive and e.spotted_until > now:
                self.last_seen[e.id] = (now, e.pos.copy())  # pinged by a team drone / camera
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
        elif w.geo_version != self.path_geo and now - self.planned_at > 1.0:
            self._plan(self.goal)  # walls broken / barricades placed: same goal, new route
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
        bashing = now < self.bash_until
        if bashing:
            look_pitch = 0.0
        dyaw, dpitch, _, _ = self._turn_towards(a, look_yaw, look_pitch, dt, 6.0)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        # re-derive move axes after the turn is applied this tick
        if self.path and self.path_i < len(self.path):
            d = self.path[self.path_i] - a.pos
            yaw_saved = a.yaw
            a.yaw = (a.yaw + dyaw) % 360.0
            act.move_forward, act.move_right = self._move_axes(a, d)
            a.yaw = yaw_saved
        if self._task_act(a, act, dt):
            return
        ws = a.weapon
        if bashing:
            # knife through barricades on the route
            if a.slot != SLOT_MELEE:
                act.switch = SLOT_MELEE
            act.fire = True
            return
        if ws.spec.kind == "gun" and ws.mag < ws.spec.mag * 0.4 and ws.reserve > 0:
            act.reload = not a.reloading and not a.prev_reload
        if a.slot != w.best_slot(a) and ws.spec.kind != "grenade":
            act.switch = w.best_slot(a)
        # stuck detection
        if now >= self.stuck_check:
            if self.stuck_pos is not None and float(np.linalg.norm(a.pos - self.stuck_pos)) < 0.4 \
                    and (abs(act.move_forward) + abs(act.move_right)) > 0.1:
                self.stuck_n += 1
                self.jump_until = now + 0.2
                self.replan_at = now + 0.3
                if self.stuck_n >= 2:
                    self.bash_until = now + 1.5
                    self.replan_at = now + 1.6
                else:
                    self.path_i = min(self.path_i + 1, max(0, len(self.path) - 1))
            else:
                self.stuck_n = 0
            self.stuck_pos = a.pos.copy()
            self.stuck_check = now + 1.0
        act.jump = now < self.jump_until

    # ------------------------------------------------------------------ grenades
    def _maybe_grenade(self, a: Agent):
        w = self.world
        frag = a.inv.get(SLOT_FRAG)
        if frag is None or frag.spec.kind != "grenade" or frag.mag <= 0 \
                or self.rng.random() > 0.01 + 0.02 * self.skill:
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
        frag = a.inv.get(SLOT_FRAG)
        if w.time > self.grenade_until or frag is None or frag.mag <= 0:
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

"""Scripted opponents: grid navigation + perception / aiming model."""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .geometry import point_in_poly, poly_center
from .modes.rounds import ATT, DEF, RoundMode
from .nav import NavGrid
from .sim import IDLE, Action, Agent, World, angles_to, wrap180
from .weapons import (SLOT_BOMB, SLOT_FIRE, SLOT_FLASH, SLOT_FRAG, SLOT_GADGET, SLOT_MELEE, SLOT_PISTOL,
                      SLOT_PRIMARY, SLOT_SMOKE, WEAPONS)

DIFFICULTY_LEVELS = {"easy": 0.2, "normal": 0.5, "hard": 0.85, "insane": 1.0}
# tasks the bot keeps doing even when it hears / remembers enemies
FORCED_TASKS = ("plant", "defuse", "fetch", "reinforce", "breach", "setup", "armor", "ash")
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
        self.chase_retry = 0.0  # zombie: no path to the target, roam until then
        self.direct_check = 0.0  # zombie: next straight-line walkability test
        self.direct_ok = False
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
        self.task_data = None
        self.stuck_n = 0
        self.bash_until = 0.0
        self.breached = None  # round uid in which this bot is done breaching
        self.hold_look = None
        self.entry_flash = None
        self.give_up: set = set()
        self.setup_since = 0.0
        self.setup_t0 = 0.0
        self.setup_for = None
        self.setup_mag = -1
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
        k = m.active
        if k < 0:
            # waiting for the next point: drift toward the middle of the ones still to play
            left = [i for i in range(len(m.points)) if not m.played[i]]
            if not left:
                return None, None, 0.0
            c = np.mean([m.centers[i] for i in left], axis=0)
            return np.array([c[0], c[1], a.pos[2]]), "capture", 0.3
        self.task_data = k
        hunter = self.aid % 4 == 3
        behind = m.cap[a.team] <= m.cap[1 - a.team]
        enemy_on = m.present[1 - a.team] > 0
        bias = 0.45 if hunter else (0.9 if behind or enemy_on else 0.75)
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
            ash = a.inv.get(SLOT_GADGET)
            if m.name == "r6" and ash is not None and ash.name == "ash_round" and ash.mag > 0:
                t = self.task_data if self.task == "ash" else None
                if t is None or not self.world.destruct.panels[t[0]].alive:
                    t = self._wall_target(m, a, [site], outside=True)
                    if t is not None:
                        pid, stand, look = t
                        back = stand - look
                        back[2] = 0.0
                        n = float(np.linalg.norm(back)) or 1.0
                        stand = stand + back / n * 5.0
                        node = self.nav.nearest_cell(stand)
                        if node >= 0:
                            stand = self.nav.nodes[node].copy()
                        t = (pid, stand, look)
                if t is not None:
                    self.task_data = t
                    return t[1], "ash", 1.0
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
        if m.phase == "prep" and m.name == "r6":
            setup = self._setup_goal(m, a)
            if setup is not None:
                return setup
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
        if m.name == "r6" and m.phase == "live":
            fresh = [(t, p) for t, p in self.last_seen.values() if self.world.time - t < 6.0]
            if fresh:
                p = max(fresh, key=lambda x: x[0])[1]
                if math.hypot(p[0] - a.pos[0], p[1] - a.pos[1]) > 5.0:
                    return p.copy(), "rotate", 0.85
            stand, look = self._anchor(m, a, site)
            self.hold_look = look
            return stand, "hold", 0.95
        return self._zone_point(m.sites[site], 0.4), "hold", 0.65

    def _anchor(self, m, a, site):
        """Stand just inside a door of `site` and watch it. Cached per bot per round."""
        key = (m.round_uid, a.id, "anchor")
        cached = Bot._plans.get(key)
        if cached is not None:
            return cached
        zone = m.sites[site]
        c = np.asarray(m.site_centers[site], dtype=np.float64)
        zf = zone.get("z", zone.get("zmin", -0.5) + 0.5)
        claimed = Bot._plans.setdefault((m.round_uid, a.team, "doors"), set())
        best, bd = None, 1e9
        for i, s in enumerate(self.world.map.objectives.get("slots", [])):
            if i in claimed:
                continue
            sc = (np.asarray(s["mn"]) + np.asarray(s["mx"])) * 0.5
            if abs(sc[2] - (zf + 1.0)) > 2.2:
                continue
            d = math.hypot(sc[0] - c[0], sc[1] - c[1])
            if not 0.6 < d < 8.0:
                continue
            if d < bd:
                best, bd = (i, sc), d
        if best is None:
            stand = self._zone_point(zone, 0.2)
            away = stand[:2] - c
            n = float(np.linalg.norm(away)) or 1.0
            look = np.array([stand[0] + away[0] / n * 4.0, stand[1] + away[1] / n * 4.0, stand[2] + 1.3])
        else:
            i, sc = best
            claimed.add(i)
            inward = c - sc[:2]
            n = float(np.linalg.norm(inward)) or 1.0
            stand = np.array([sc[0] + inward[0] / n * 1.4, sc[1] + inward[1] / n * 1.4, zf])
            node = self.nav.nearest_cell(stand)
            if node >= 0 and np.linalg.norm(self.nav.nodes[node] - stand) < 1.2:
                stand = self.nav.nodes[node].copy()
            look = np.array([sc[0], sc[1], stand[2] + 1.2])
        Bot._plans[key] = (stand, look)
        return stand, look

    def _can_reach(self, a, stand) -> bool:
        path = self.nav.find_path(a.pos, stand)
        return bool(path) and float(np.linalg.norm(path[-1][:2] - np.asarray(stand)[:2])) < 1.3

    def _reachable_near(self, a, point):
        """A point on the navmesh near `point` that `a` can actually walk to."""
        path = self.nav.find_path(a.pos, point)
        if not path or float(np.linalg.norm(path[-1][:2] - np.asarray(point)[:2])) > 4.0:
            return None
        return path[-1].copy()

    def _door_target(self, m, a):
        """(stand, look) facing an unclaimed doorway of the defended sites, for castle / kapkan."""
        claimed = Bot._plans.setdefault((m.round_uid, a.team, "doors"), set())
        best, bd = None, 1e9
        for si in m.active_sites:
            c = np.asarray(m.site_centers[si], dtype=np.float64)
            zf = m.sites[si].get("z", m.sites[si].get("zmin", -0.5) + 0.5)
            for i, s in enumerate(self.world.map.objectives.get("slots", [])):
                if i in claimed:
                    continue
                sc = (np.asarray(s["mn"]) + np.asarray(s["mx"])) * 0.5
                if abs(sc[2] - (zf + 1.0)) > 2.2:
                    continue
                d = math.hypot(sc[0] - c[0], sc[1] - c[1])
                far = math.hypot(sc[0] - a.pos[0], sc[1] - a.pos[1])
                if d > 8.0 or d + 0.15 * far >= bd:
                    continue
                inward = c - sc[:2]
                n = float(np.linalg.norm(inward)) or 1.0
                stand = np.array([sc[0] + inward[0] / n * 1.15, sc[1] + inward[1] / n * 1.15, zf])
                node = self.nav.nearest_cell(stand)
                if node < 0 or np.linalg.norm(self.nav.nodes[node] - stand) > 1.2:
                    continue
                stand = self.nav.nodes[node].copy()
                if not self._can_reach(a, stand):
                    continue
                best, bd = (i, stand, sc, zf), d + 0.15 * far
        if best is None:
            return None
        i, stand, sc, zf = best
        claimed.add(i)
        return stand, np.array([sc[0], sc[1], zf + 1.2])

    def _setup_goal(self, m, a):
        """One defender gadget placement: unique gadget first (two charges), then a generic one
        once the reinforcements are down. Armor packs on the ground get picked up immediately."""
        w = self.world
        for it in w.gadgets.items.values():
            if it["kind"] == "armor_bag" and it["team"] == a.team and it.get("charges", 0) > 0 and a.armor <= 0:
                self.task_data = it["pos"].copy()
                return it["pos"].copy(), "armor", 1.0
        op = m.operator_of(a)
        reinforce_pending = getattr(m, "reinforce_left", 0) > 0
        names = [op.gadget]
        if not reinforce_pending:
            names += list(op.generics)
        for name in names:
            if name in self.give_up:
                continue
            spec = WEAPONS[name]
            if spec.kind != "gadget" or spec.place in ("", "launch"):
                continue
            ws = a.inv.get(spec.slot)
            if ws is None or ws.name != name or ws.mag <= 0:
                continue
            used = spec.mag - ws.mag
            if used >= (1 if name in ("armor_bag", "dshield") or name in op.generics else 2):
                continue
            if self.task == "setup" and isinstance(self.task_data, tuple) and self.task_data[0] == name:
                return self.task_data[1], "setup", 1.0
            if spec.place == "slot":
                door = self._door_target(m, a)
                if door is None:
                    continue
                stand, look = door
            else:
                site = m.active_sites[a.id % len(m.active_sites)]
                zone = m.sites[site]
                zf = zone.get("z", zone.get("zmin", -0.5) + 0.5)
                center = np.asarray(m.site_centers[site], dtype=np.float64)
                stand = self._reachable_near(a, np.array([center[0], center[1], zf]))
                if stand is None:
                    continue
                look = np.array([center[0], center[1], stand[2] + 0.9])
            self.task_data = (name, stand, look)
            return stand, "setup", 1.0
        return None

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
        if task in ("reinforce", "setup", "armor") and getattr(m, "phase", "") != "prep":
            self.task = None
            self.replan_at = w.time
            return False
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
        if task == "armor":
            pos = self.task_data
            if a.armor > 0 or not any(it["kind"] == "armor_bag" and it["team"] == a.team and it.get("charges", 0) > 0
                                      for it in w.gadgets.items.values()):
                self.task = None
                self.replan_at = w.time
                return False
            if math.hypot(pos[0] - a.pos[0], pos[1] - a.pos[1]) > 1.2:
                return False
            act.move_forward = act.move_right = 0.0
            act.use = not a.prev_use
            return True
        if task == "setup":
            name, stand, look = self.task_data
            spec = WEAPONS[name]
            ws = a.inv.get(spec.slot)
            if ws is None or ws.name != name or ws.mag <= 0 or name in self.give_up:
                self.task = None
                self.replan_at = w.time
                return False
            if self.setup_for != name:
                self.setup_for = name
                self.setup_t0 = w.time
                self.setup_since = 0.0
                self.setup_mag = -1
            if self.stuck_n >= 2 and math.hypot(stand[0] - a.pos[0], stand[1] - a.pos[1]) > 1.5:
                # the stand is past a barricade; drop the gadget in the corridor instead of stalling
                back = a.pos - stand
                back[2] = 0.0
                n = float(np.linalg.norm(back)) or 1.0
                look = a.pos + back / n * 2.0
                look[2] = a.pos[2] + 0.8
                stand = a.pos.copy()
                self.task_data = (name, stand, look)
                self.stuck_n = 0
                self.setup_t0 = w.time
            ready = self._work_at(a, act, dt, stand, look)
            if ready is None:
                if w.time - self.setup_t0 > 12.0:
                    self.give_up.add(name)
                    self.task = None
                    self.replan_at = w.time
                return False
            if self.setup_mag != ws.mag:
                self.setup_mag = ws.mag
                self.setup_since = w.time
            elif w.time - self.setup_since > 4.0:
                self.give_up.add(name)
                self.task = None
                self.replan_at = w.time
                return False
            if a.slot != spec.slot:
                act.switch = spec.slot
            elif ready and w.time >= a.ready_at:
                act.fire = not a.prev_fire
            return True
        if task == "ash":
            pid, stand, look = self.task_data
            p = w.destruct.panels[pid]
            ws = a.inv.get(SLOT_GADGET)
            if not p.alive or ws is None or ws.name != "ash_round" or ws.mag <= 0:
                self.task = None
                self.replan_at = w.time
                return False
            ready = self._work_at(a, act, dt, stand, look)
            if ready is None:
                return False
            if a.slot != SLOT_GADGET:
                act.switch = SLOT_GADGET
            elif ready and w.time >= a.ready_at:
                act.fire = not a.prev_fire
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
        if a.armor <= 0 and a.money >= (1000 if pistol_round else 1800):
            w.buy(a, "kevlar_helmet")
        elif a.armor > 0 and not a.helmet and a.money >= 350:
            w.buy(a, "kevlar_helmet")
        if a.money >= 300 and SLOT_FRAG not in a.inv and (not pistol_round or self.rng.random() < 0.5):
            w.buy(a, "frag")
        if a.money >= 200 and SLOT_FLASH not in a.inv and self.rng.random() < 0.55:
            w.buy(a, "flash")
        if a.money >= 700 and SLOT_SMOKE not in a.inv and self.rng.random() < 0.5:
            w.buy(a, "smoke")
        if a.money >= 400 and SLOT_FIRE not in a.inv and self.rng.random() < 0.4:
            w.buy(a, "molotov")
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
        if getattr(w.mode, "economy", False):
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
        if getattr(a, "role", "human") != "human":
            return self._zombie_act(a, dt)
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

    def _zombie_act(self, a: Agent, dt: float) -> Action:
        """Run at the nearest human and stab. The knife is the only weapon a zombie keeps."""
        w = self.world
        now = w.time
        act = Action()
        if a.slot != SLOT_MELEE:
            act.switch = SLOT_MELEE
        humans = [e for e in w.agents if e.alive and getattr(e, "role", "human") == "human"]
        if not humans:
            return act
        e = min(humans, key=lambda o: float(np.linalg.norm(o.pos - a.pos)))
        dist = float(np.linalg.norm(e.pos - a.pos))
        yaw, pitch = angles_to(a.eye(), e.pos + np.array([0.0, 0.0, e.height * 0.5]))
        dyaw, dpitch, ey, _ep = self._turn_towards(a, yaw, pitch, dt, self.gain)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        if dist < 1.9 and abs(ey) < 30.0:
            act.fire = not a.prev_fire
            act.move_forward = 1.0 if dist > 1.1 else 0.0
            return act
        # open ground to the target: run straight at it, no path search
        if dist < 14.0 and abs(float(e.pos[2] - a.pos[2])) < 0.5:
            if now >= self.direct_check:
                self.direct_check = now + 0.2
                self.nav.refresh()
                self.direct_ok = self.nav.grid._line_walkable(a.pos, e.pos, self.nav.ok_list)
            if self.direct_ok:
                d = e.pos - a.pos
                yaw_saved = a.yaw
                a.yaw = (a.yaw + dyaw) % 360.0
                act.move_forward, act.move_right = self._move_axes(a, d)
                a.yaw = yaw_saved
                self.path = []
                self.goal = None
                return act
        # re-route at most ~once a second, sooner when the target moved a lot relative to the
        # distance; after a miss (target out of reach) roam for a while
        moved = float(np.linalg.norm(self.goal - e.pos)) if self.goal is not None else 1e9
        stale = self.goal is None or now >= self.replan_at or not self.path or (
            moved > max(1.5, 0.25 * dist) and now - self.planned_at > 0.8)
        if stale and now >= self.chase_retry:
            self.task = "chase"
            self._plan(e.pos)
            if not self.path:
                self.chase_retry = now + 1.5
        self._navigate(a, act, dt)
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

        shield = a.inv.get(SLOT_GADGET)
        if shield is not None and shield.spec.kind == "shield" and dist < 16.0:
            if a.slot != SLOT_GADGET:
                act.switch = SLOT_GADGET
            elif reacted and on_target and dist < 9.0 and shield.mag > 0:
                act.fire = not a.prev_fire
            act.move_forward = 1.0 if dist > 2.2 else 0.0
            act.move_right = self.strafe * 0.4
            return

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
        anchored = (self.task == "hold" and self.hold_look is not None and self.goal is not None
                    and math.hypot(self.goal[0] - a.pos[0], self.goal[1] - a.pos[1]) < 1.4
                    and abs(self.goal[2] - a.pos[2]) < 1.5)
        # glance towards the most recent sighting; an anchor watches its door otherwise
        recent = [(t, p) for t, p in self.last_seen.values() if now - t < 3.0]
        if recent:
            p = max(recent, key=lambda x: x[0])[1]
            look_yaw, look_pitch = angles_to(a.eye(), p + np.array([0, 0, 1.4]))
        elif anchored:
            look_yaw, look_pitch = angles_to(a.eye(), self.hold_look)
            self.replan_at = max(self.replan_at, now + 2.5)
        bashing = now < self.bash_until
        if bashing:
            look_pitch = 0.0
        dyaw, dpitch, _, _ = self._turn_towards(a, look_yaw, look_pitch, dt, 6.0)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        # re-derive move axes after the turn is applied this tick
        if self.path and self.path_i < len(self.path) and not anchored:
            d = self.path[self.path_i] - a.pos
            yaw_saved = a.yaw
            a.yaw = (a.yaw + dyaw) % 360.0
            act.move_forward, act.move_right = self._move_axes(a, d)
            a.yaw = yaw_saved
        if anchored:
            act.move_forward = act.move_right = 0.0
            act.crouch = self.aid % 2 == 0
        if self._task_act(a, act, dt):
            return
        m = w.mode
        if m.name == "r6" and m.side(a.team) == ATT and self.task in ("push", "guard") \
                and self.entry_flash != m.round_uid:
            site = self._team_site(m, a.team)
            if site >= 0:
                c = m.site_centers[site]
                nades = self._nades(a)
                if math.hypot(c[0] - a.pos[0], c[1] - a.pos[1]) < 14.0 and "flash" in nades:
                    self.entry_flash = m.round_uid
                    self.grenade_slot = nades["flash"]
                    self.grenade_state = np.array([c[0], c[1], a.pos[2] + 0.5])
                    self.grenade_until = w.time + 1.6
        if self._shoot_gadget(a, act, dt):
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

    def _shoot_gadget(self, a: Agent, act: Action, dt: float) -> bool:
        """Aim at an enemy drone, camera, trap, jammer or breach charge. Stops to kill one up close."""
        if self.task in FORCED_TASKS:
            return False
        w = self.world
        eye = a.eye()
        best, bd = None, 16.0
        for it in w.gadgets.items.values():
            if it["kind"] not in ("drone", "camera", "jammer", "edd", "charge") or it["team"] == a.team:
                continue
            dist = float(np.linalg.norm(it["pos"] - eye))
            if dist > bd or not w.segment_clear(eye, it["pos"], smoke=True):
                continue
            yaw_to, _ = angles_to(eye, it["pos"])
            if dist > 4.0 and abs(wrap180(yaw_to - a.yaw)) > self.fov:
                continue
            best, bd = it, dist
        if best is None:
            return False
        yaw, pitch = angles_to(eye, best["pos"])
        dyaw, dpitch, ey, ep = self._turn_towards(a, yaw, pitch, dt, self.gain)
        act.look_yaw, act.look_pitch = dyaw, dpitch
        if a.weapon.spec.kind != "gun":
            act.switch = w.best_slot(a)
        elif abs(ey) < 3.5 and abs(ep) < 3.5:
            act.fire = True if a.weapon.spec.auto else not a.prev_fire
        if bd < 9.0:
            act.move_forward = act.move_right = 0.0
        return True

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

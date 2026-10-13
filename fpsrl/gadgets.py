"""Placeable / controllable gadgets (mostly for the Siege mode).

Entities live in `Gadgets.items` (gid -> dict). Static ones that block movement (barricades,
armor panels, deployable shields) contribute collision boxes tagged `params={"dyn": gid}`; small
shootable ones (drones, cameras, jammers) are tested separately by `ray_hit`.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .geometry import aabb_overlap_mask, raycast_boxes
from .maps.base import Box
from .weapons import WEAPONS

DRONE_SPEED = 2.6
DRONE_SIZE = 0.13
SLOW_KINDS = ("barbed",)
BOX_KINDS = ("barricade", "castle", "dshield")
SHOOTABLE = {"drone": 0.13, "camera": 0.12, "jammer": 0.14, "edd": 0.08, "charge": 0.1}
HP = {"barricade": 100.0, "castle": 700.0, "dshield": 500.0, "drone": 1.0, "camera": 1.0,
      "jammer": 1.0, "edd": 1.0, "barbed": 1.0, "armor_bag": 1e9, "charge": 1.0}


def _dir(yaw, pitch=0.0):
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])


class Gadgets:
    PROJECTILES = ("ash_round",)

    def __init__(self, world):
        self.w = world
        self.items: dict[int, dict] = {}
        self._next = 1
        self.version = 0
        self._boxes: list[Box] = []
        self.drones_left: dict[int, int] = {}

    # ------------------------------------------------------------------ bookkeeping
    def reset(self):
        self.items = {}
        self._next = 1
        self.drones_left = {}
        for i, cam in enumerate(self.w.map.cameras):
            x, y, z, yaw, pitch = cam
            self._add("camera", -1, 1, (x, y, z), yaw=yaw, pitch=pitch, base_yaw=yaw)
        self._rebuild()

    def round_start(self, drones: int):
        """Called by the Siege mode after spawning."""
        m = self.w.mode
        for a in self.w.agents:
            self.drones_left[a.id] = drones if m.side(a.team) == 0 else 0
        for it in self.items.values():
            if it["kind"] == "camera":
                it["team"] = m.team_of_side(1)

    def _add(self, kind, owner, team, pos, **kw) -> dict:
        gid = self._next
        self._next += 1
        it = {"id": gid, "kind": kind, "owner": owner, "team": team, "pos": np.array(pos, dtype=np.float64),
              "hp": HP.get(kind, 1.0), "t": self.w.time, **kw}
        self.items[gid] = it
        self.version += 1
        return it

    def _remove(self, gid, reason="destroyed"):
        it = self.items.pop(gid, None)
        if it is None:
            return
        self.version += 1
        self.w._emit("gadget_destroyed", gid=gid, kind=it["kind"], pos=it["pos"].tolist(), reason=reason)
        for a in self.w.agents:
            if a.view_ref == gid:
                self.exit_view(a)
                if self._view_locked(a):
                    self.toggle_view(a)  # re-enter another drone already out; prep does not deploy a spare
        if it["kind"] in BOX_KINDS:
            self._rebuild()
            self.w.geometry_changed()

    def _rebuild(self):
        boxes = []
        for it in self.items.values():
            if it["kind"] in BOX_KINDS:
                mat = {"barricade": "wood", "castle": "metal_dark", "dshield": "metal"}[it["kind"]]
                boxes.append(Box(tuple(it["mn"]), tuple(it["mx"]), mat, tag=it["kind"],
                                 params={"dyn": it["id"]}))
        self._boxes = boxes

    def boxes(self):
        return list(self._boxes)

    def box_soft(self, gid) -> bool:
        it = self.items.get(gid)
        return it is not None and it["kind"] == "barricade"

    def damage_box(self, gid, amount, kind="bullet"):
        it = self.items.get(gid)
        if it is None:
            return
        k = it["kind"]
        if k == "castle" and kind == "bullet":
            return
        if k == "dshield" and kind not in ("explosion", "hard"):
            return
        mult = 0.6 if (k == "barricade" and kind == "bullet") else 1.0
        if k == "barricade" and kind == "melee":
            mult = 2.0
        it["hp"] -= amount * mult
        if it["hp"] <= 0.0:
            self._remove(gid)

    def blast(self, origin, radius, damage):
        for gid, it in list(self.items.items()):
            if it["kind"] in ("armor_bag",):
                continue
            c = (np.asarray(it["mn"]) + np.asarray(it["mx"])) * 0.5 if "mn" in it else it["pos"]
            if np.linalg.norm(c - origin) > radius:
                continue
            if it["kind"] in BOX_KINDS:
                self.damage_box(gid, damage * 3.0, "explosion")
            elif it["kind"] != "camera" or damage > 50:
                self._remove(gid)

    # ------------------------------------------------------------------ shooting
    def ray_hit(self, origin, d, max_t):
        """Nearest small shootable gadget on the ray, as (t, gid) or None."""
        best = None
        for gid, it in self.items.items():
            r = SHOOTABLE.get(it["kind"])
            if r is None:
                continue
            p = it["pos"]
            t, _, _ = raycast_boxes(origin[None], d[None], max_t, (p - r)[None], (p + r)[None])
            if np.isfinite(t[0]) and (best is None or t[0] < best[0]):
                best = (float(t[0]), gid)
        return best

    def shoot(self, gid, amount):
        it = self.items.get(gid)
        if it is None:
            return
        it["hp"] -= amount
        if it["hp"] <= 0.0:
            self._remove(gid, "shot")

    def shield_blocks(self, v, d, group: int) -> bool:
        if v.slot not in v.inv or v.spec.kind != "shield" or group == 2:
            return False
        fwd = _dir(v.yaw)
        return float(fwd @ -d) > 0.35

    def shield_flash(self, a, spec):
        fwd = _dir(a.yaw, a.pitch)
        self.w._emit("flash", proj=-1, pos=(a.eye() + fwd * 0.6).tolist(), owner=a.id, shield=True)
        self.w.flash_at(a.eye() + fwd * 0.7, spec.flash_radius, cone_dir=fwd)

    def speed_mult(self, a) -> float:
        for it in self.items.values():
            if it["kind"] == "barbed" and it["team"] != a.team:
                if abs(a.pos[0] - it["pos"][0]) < 0.8 and abs(a.pos[1] - it["pos"][1]) < 0.8 \
                        and abs(a.pos[2] - it["pos"][2]) < 0.6:
                    if self.w.tick % 20 == 0:
                        self.w.noises.append((self.w.time, a.pos.copy(), a.team, "step"))
                    return 0.4
        return 1.0

    def jammed(self, pos) -> bool:
        for it in self.items.values():
            if it["kind"] == "jammer" and np.linalg.norm(it["pos"] - pos) < WEAPONS["jammer"].extra["radius"]:
                return True
        return False

    # ------------------------------------------------------------------ placement
    def _aim(self, a, dist=1.8):
        eye = a.eye()
        d = _dir(a.yaw, a.pitch)
        t, idx, n = raycast_boxes(eye[None], d[None], dist, self.w.bmin, self.w.bmax)
        if idx[0] < 0:
            return None
        return self.w.bullet_boxes[idx[0]], eye + d * float(t[0]), n[0]

    def _slot_near(self, a, dist=2.2):
        slots = self.w.map.objectives.get("slots", [])
        eye = a.eye()
        fwd = _dir(a.yaw)
        best, bd = -1, dist
        for i, s in enumerate(slots):
            c = (np.asarray(s["mn"]) + np.asarray(s["mx"])) * 0.5
            dv = c - eye
            dv[2] = 0.0
            dd = float(np.linalg.norm(dv))
            if dd < bd and (dd < 0.6 or float(fwd @ (dv / max(dd, 1e-6))) > 0.3) \
                    and s["mn"][2] - 0.5 <= a.pos[2] <= s["mx"][2]:
                best, bd = i, dd
        return best

    def _slot_items(self, si):
        return [it for it in self.items.values() if it.get("slot") == si]

    def _ground_pos(self, a, ahead=1.0):
        p = a.pos + np.append(_dir(a.yaw)[:2], 0.0) * ahead
        p[2] = a.pos[2]
        if not self.w.segment_clear(a.eye(), p + [0, 0, 0.3]):
            return None
        return p

    def use(self, a) -> bool:
        """F key without objective: reinforce wall, barricade slot, take armor."""
        w = self.w
        m = w.mode
        if m.name != "r6" or m.side(a.team) != 1:
            for it in self.items.values():
                if it["kind"] == "armor_bag" and it["team"] == a.team and it["charges"] > 0 \
                        and np.linalg.norm(it["pos"] - a.pos) < 1.4 and a.armor <= 0:
                    return self._take_armor(a, it)
            return False
        for it in self.items.values():
            if it["kind"] == "armor_bag" and it["team"] == a.team and it["charges"] > 0 \
                    and np.linalg.norm(it["pos"] - a.pos) < 1.4 and a.armor <= 0:
                return self._take_armor(a, it)
        si = self._slot_near(a)
        if si >= 0 and not self._slot_items(si):
            w.start_busy(a, "barricade", 1.5, si)
            return True
        hit = self._aim(a, 1.6)
        if hit is not None and getattr(m, "reinforce_left", 0) > 0:
            box, _, _ = hit
            p = w.destruct.panel_at(box)
            if p is not None and not p.reinforced:
                w.start_busy(a, "reinforce", 3.0, p.id)
                return True
        return False

    def _take_armor(self, a, it):
        it["charges"] -= 1
        a.armor = 0.2
        a.helmet = False
        self.w._emit("armor", agent=a.id)
        if it["charges"] <= 0:
            self._remove(it["id"], "empty")
        return True

    def item_fire(self, a, ws):
        spec = ws.spec
        w = self.w
        name = spec.name
        if spec.place == "launch":
            if w.mode.name == "r6" and w.mode.phase not in ("live", "planted"):
                return
            ws.mag -= 1
            d = _dir(a.yaw, a.pitch)
            from .sim import Projectile
            pr = Projectile(w._new_id(), name, a.id, a.team, a.eye() + d * 0.4, d * spec.extra["speed"],
                            w.time + 3.0, w.time)
            w.projectiles.append(pr)
            w._emit("launch", agent=a.id, weapon=name, proj=pr.id)
            self._after_use(a, ws)
            return
        if spec.place == "wall":
            hit = self._aim(a, 1.6)
            if hit is None:
                return
            box, point, normal = hit
            p = w.destruct.panel_at(box)
            if p is None or (p.reinforced and not spec.extra.get("hard")):
                w._emit("hint", agent=a.id, text="need_soft_wall")
                return
            w.start_busy(a, "place", spec.place_time, {"item": name, "pos": (point + normal * 0.04).tolist(),
                                                       "normal": normal.tolist()})
        elif spec.place == "slot":
            si = self._slot_near(a)
            if si < 0:
                w._emit("hint", agent=a.id, text="need_door")
                return
            if any(it["kind"] in ("castle", "edd") and it["kind"] == name for it in self._slot_items(si)):
                return
            w.start_busy(a, "place", spec.place_time, {"item": name, "slot": si})
        else:  # ground
            p = self._ground_pos(a, 0.9 if name != "dshield" else 0.8)
            if p is None:
                return
            w.start_busy(a, "place", spec.place_time, {"item": name, "pos": p.tolist(), "yaw": a.yaw})

    def _after_use(self, a, ws):
        if ws.mag <= 0:
            self.w.switch_weapon(a, self.w.best_slot(a))

    def finish_busy(self, a, kind, data):
        w = self.w
        if kind == "reinforce":
            if getattr(w.mode, "reinforce_left", 0) > 0 and w.destruct.reinforce_around(data):
                w.mode.reinforce_left -= 1
            return
        if kind == "barricade":
            self._barricade(a, data, "barricade")
            return
        if kind != "place":
            return
        name = data["item"]
        ws = a.inv.get(WEAPONS[name].slot)
        if ws is None or ws.name != name or ws.mag <= 0:
            return
        ws.mag -= 1
        team = a.team
        if name in ("breach", "thermite"):
            spec = WEAPONS[name]
            self._add("charge", a.id, team, data["pos"], item=name, fuse_at=w.time + spec.fuse,
                      normal=data["normal"])
        elif name == "castle":
            self._barricade(a, data["slot"], "castle")
        elif name == "edd":
            s = w.map.objectives["slots"][data["slot"]]
            c = (np.asarray(s["mn"]) + np.asarray(s["mx"])) * 0.5
            c[2] = s["mn"][2] + 0.3
            self._add("edd", a.id, team, c, slot=data["slot"], mn=np.asarray(s["mn"]) - 0.05,
                      mx=np.asarray(s["mx"]) + 0.05)
        elif name == "dshield":
            yaw = data["yaw"]
            p = np.array(data["pos"])
            along_x = abs(math.sin(math.radians(yaw))) > 0.707  # facing ±y -> shield spans x
            hw = 0.55
            if along_x:
                mn, mx = p + [-hw, -0.08, 0.0], p + [hw, 0.08, 1.05]
            else:
                mn, mx = p + [-0.08, -hw, 0.0], p + [0.08, hw, 1.05]
            self._add("dshield", a.id, team, p, mn=mn, mx=mx)
            self._rebuild()
            w.geometry_changed()
        elif name == "barbed":
            self._add("barbed", a.id, team, data["pos"])
        elif name == "jammer":
            self._add("jammer", a.id, team, np.array(data["pos"]) + [0, 0, 0.1])
        elif name == "armor_bag":
            n = len([m for m in w.agents if m.team == team])
            self._add("armor_bag", a.id, team, np.array(data["pos"]) + [0, 0, 0.1], charges=n)
        w._emit("gadget_placed", agent=a.id, item=name)
        self._after_use(a, ws)

    def _barricade(self, a, si, kind):
        s = self.w.map.objectives["slots"][si]
        for it in self._slot_items(si):
            if it["kind"] == "barricade" and kind == "castle":
                self.items.pop(it["id"], None)
            elif it["kind"] in ("barricade", "castle"):
                return
        mn = np.array(s["mn"], dtype=np.float64)
        mx = np.array(s["mx"], dtype=np.float64)
        size = mx - mn
        ax = 0 if size[0] < size[1] else 1
        c = (mn[ax] + mx[ax]) * 0.5
        th = 0.05 if kind == "barricade" else 0.08
        mn[ax], mx[ax] = c - th, c + th
        self._add(kind, a.id, a.team, (mn + mx) * 0.5, slot=si, mn=mn, mx=mx)
        self._rebuild()
        self.w.geometry_changed()
        self.w._emit("gadget_placed", agent=a.id, item=kind)

    # ------------------------------------------------------------------ projectiles
    def projectile_hit(self, pr):
        if pr.kind == "ash_round":
            self._add("charge", pr.owner, pr.team, pr.pos, item="ash_round",
                      fuse_at=self.w.time + WEAPONS["ash_round"].fuse, normal=[0, 0, 1])

    # ------------------------------------------------------------------ views
    def _view_locked(self, a) -> bool:
        locked = getattr(self.w.mode, "view_locked", None)
        return locked is not None and locked(a)

    def toggle_view(self, a):
        w = self.w
        if a.view_mode:
            if self._view_locked(a):
                return  # prep drone stays up; the spare waits until the round is live
            self.exit_view(a)
            return
        m = w.mode
        if m.name != "r6":
            return
        side = m.side(a.team)
        if side == 0:
            mine = [it for it in self.items.values() if it["kind"] == "drone" and it["owner"] == a.id]
            if mine:
                it = mine[0]
            elif a.alive and self.drones_left.get(a.id, 0) > 0 and not self._prep_spare_held(a):
                self.drones_left[a.id] -= 1
                d = _dir(a.yaw)
                p = a.pos + np.array([d[0] * 0.5, d[1] * 0.5, 0.05])
                if w._overlaps(p, 0.2).any():
                    p = a.pos + np.array([0.0, 0.0, 0.05])
                it = self._add("drone", a.id, a.team, p, yaw=a.yaw, pitch=0.0, vz=0.0)
                w._emit("drone_deploy", agent=a.id, gid=it["id"])
            else:
                return
            a.view_mode, a.view_ref = 1, it["id"]
        else:
            cams = self.cameras(a.team)
            if not cams:
                return
            a.view_mode, a.view_ref = 2, cams[0]["id"]
        a.ads = False
        a.busy = None
        w._emit("view", agent=a.id, mode=a.view_mode, gid=a.view_ref)

    def cameras(self, team):
        return [it for it in self.items.values() if it["kind"] == "camera" and it["team"] == team]

    def _prep_spare_held(self, a) -> bool:
        """Prep allows the single drone placed at round start. The spare stays pocketed."""
        if not self._view_locked(a):
            return False
        per = self.w.mode.p.get("drones_per_player", 1)
        return self.drones_left.get(a.id, 0) < per

    def exit_view(self, a):
        if a.view_mode:
            a.view_mode, a.view_ref = 0, -1
            self.w._emit("view", agent=a.id, mode=0, gid=-1)

    def view_pose(self, a):
        """(eye position, yaw, pitch) of the gadget the agent is looking through."""
        it = self.items.get(a.view_ref)
        if it is None:
            return None
        if it["kind"] == "drone":
            return it["pos"] + [0, 0, DRONE_SIZE], it["yaw"], it["pitch"]
        return it["pos"], it["yaw"], it["pitch"]

    def control(self, a, act, dt):
        w = self.w
        it = self.items.get(a.view_ref)
        if it is None:
            self.exit_view(a)
            return
        if it["kind"] == "camera":
            it["yaw"] = float(np.clip((it["yaw"] + act.look_yaw - it["base_yaw"] + 180) % 360 - 180, -70, 70)
                              + it["base_yaw"])
            it["pitch"] = float(np.clip(it["pitch"] + act.look_pitch, -60, 20))
            if act.alt and not a.prev_alt:
                cams = self.cameras(a.team)
                ids = [c["id"] for c in cams]
                if ids:
                    k = (ids.index(it["id"]) + 1) % len(ids) if it["id"] in ids else 0
                    a.view_ref = ids[k]
            if act.fire and not a.prev_fire:
                self._spot(a, it["pos"], it["yaw"], it["pitch"])
            return
        # drone
        it["yaw"] = (it["yaw"] + act.look_yaw) % 360.0
        it["pitch"] = float(np.clip(it["pitch"] + act.look_pitch, -30, 60))
        if act.fire and not a.prev_fire:
            self._spot(a, it["pos"] + [0, 0, DRONE_SIZE], it["yaw"], it["pitch"])
        if self.jammed(it["pos"]):
            return
        yr = math.radians(it["yaw"])
        fwd = np.array([math.cos(yr), math.sin(yr)])
        right = np.array([math.sin(yr), -math.cos(yr)])
        wish = fwd * act.move_forward + right * act.move_right
        n = np.linalg.norm(wish)
        if n > 1:
            wish /= n
        self._drone_move(it, wish * DRONE_SPEED * dt, act.jump, dt)

    def _drone_overlap(self, p):
        r = DRONE_SIZE
        return aabb_overlap_mask((p[0] - r, p[1] - r, p[2]), (p[0] + r, p[1] + r, p[2] + 2 * r),
                                 self.w.smin, self.w.smax)

    def _drone_move(self, it, dxy, jump, dt):
        p = it["pos"]
        for axis in (0, 1):
            q = p.copy()
            q[axis] += dxy[axis]
            if not self._drone_overlap(q).any():
                p[:] = q
            else:
                q[2] += 0.2  # climb small steps
                if not self._drone_overlap(q).any():
                    p[:] = q
        on_ground = self._drone_overlap(p - [0, 0, 0.02]).any()
        if jump and on_ground:
            it["vz"] = 3.2
        it["vz"] = it.get("vz", 0.0) - C.GRAVITY * dt
        q = p.copy()
        q[2] += it["vz"] * dt
        hit = self._drone_overlap(q)
        if hit.any():
            if it["vz"] < 0:
                q[2] = self.w.smax[hit, 2].max() + 1e-4
            else:
                q[2] = p[2]
            it["vz"] = 0.0
        p[:] = q
        if p[2] < -20:
            self._remove(it["id"], "fell")

    def _spot(self, a, eye, yaw, pitch):
        w = self.w
        d0 = _dir(yaw, pitch)
        spotted = []
        for e in w.agents:
            if not e.alive or e.team == a.team:
                continue
            tgt = e.pos + [0, 0, e.height * 0.6]
            v = tgt - eye
            dist = float(np.linalg.norm(v))
            if dist < 40 and float(d0 @ (v / max(dist, 1e-6))) > 0.8 and w.segment_clear(eye, tgt, smoke=True):
                e.spotted_until = w.time + 4.0
                spotted.append(e.id)
        w._emit("spot", agent=a.id, targets=spotted)

    # ------------------------------------------------------------------ tick
    def update(self, dt):
        w = self.w
        if not self.items:
            return
        for gid, it in list(self.items.items()):
            k = it["kind"]
            if k == "drone":
                ctrl = any(a.view_ref == gid for a in w.agents)
                if not ctrl:
                    self._drone_move(it, np.zeros(2), False, dt)
            elif k == "charge":
                if w.time >= it["fuse_at"] and (it["item"] == "thermite" or not self.jammed(it["pos"])):
                    self._detonate_charge(it)
                elif self.jammed(it["pos"]) and it["item"] != "thermite":
                    it["fuse_at"] = max(it["fuse_at"], w.time + 0.5)
            elif k == "edd":
                for e in w.agents:
                    if e.alive and e.team != it["team"]:
                        mn, mx = it["mn"], it["mx"]
                        if (mn[0] - 0.3 < e.pos[0] < mx[0] + 0.3 and mn[1] - 0.3 < e.pos[1] < mx[1] + 0.3
                                and mn[2] - 0.5 < e.pos[2] < mx[2]):
                            self._remove(gid, "triggered")
                            owner = w.agents[it["owner"]] if it["owner"] >= 0 else None
                            w._emit("explode", proj=-1, pos=it["pos"].tolist(), weapon="edd",
                                    owner=it["owner"])
                            w.explosion(it["pos"], 2.6, WEAPONS["edd"].damage * 1.25, owner, "edd", 0.0)
                            break

    def _detonate_charge(self, it):
        w = self.w
        self._remove(it["id"], "detonated")
        item = it["item"]
        owner = w.agents[it["owner"]] if it["owner"] >= 0 else None
        pos = np.asarray(it["pos"])
        w._emit("explode", proj=-1, pos=pos.tolist(), weapon=item, owner=it["owner"])
        w.noises.append((w.time, pos.copy(), it["team"], "explode"))
        if item == "thermite":
            w.destruct.explode(pos, 1.15, "hard", floors=True, source=it["owner"])
            w.explosion(pos, 1.5, 40.0, owner, item, 0.0)
        elif item == "ash_round":
            w.destruct.explode(pos, WEAPONS["ash_round"].extra["radius"], "soft", floors=False,
                               source=it["owner"])
            self.blast(pos, 1.2, 200.0)
            w.explosion(pos, 1.8, 50.0, owner, item, 0.0)
        else:
            w.destruct.explode(pos, WEAPONS["breach"].extra["radius"], "soft", floors=True, source=it["owner"])
            self.blast(pos, 1.0, 200.0)
            w.explosion(pos, 2.2, 75.0, owner, item, 0.0)

    # ------------------------------------------------------------------ network
    def state(self) -> dict:
        out = []
        for it in self.items.values():
            e = [it["id"], it["kind"], it["team"], it["owner"],
                 [round(float(v), 3) for v in it["pos"]], round(float(it.get("yaw", 0.0)), 1),
                 round(float(it.get("pitch", 0.0)), 1)]
            if "mn" in it:
                e.append([[round(float(v), 3) for v in it["mn"]], [round(float(v), 3) for v in it["mx"]]])
            else:
                e.append(None)
            e.append(it.get("item", ""))
            out.append(e)
        return {"v": self.version, "items": out, "drones": {str(k): v for k, v in self.drones_left.items()}}

    def apply_state(self, st: dict):
        if not st:
            return
        changed_boxes = False
        new = {}
        for gid, kind, team, owner, pos, yaw, pitch, box, item in st.get("items", []):
            it = self.items.get(gid) or {"id": gid, "kind": kind, "hp": 1.0, "t": self.w.time}
            it.update(kind=kind, team=team, owner=owner, pos=np.array(pos), yaw=yaw, pitch=pitch, item=item)
            if box is not None:
                if "mn" not in it:
                    changed_boxes = True
                it["mn"], it["mx"] = np.array(box[0]), np.array(box[1])
            new[gid] = it
        if set(k for k, v in new.items() if v["kind"] in BOX_KINDS) != \
                set(k for k, v in self.items.items() if v["kind"] in BOX_KINDS):
            changed_boxes = True
        self.items = new
        self.drones_left = {int(k): v for k, v in st.get("drones", {}).items()}
        if st.get("v") != self.version:
            self.version = st.get("v", self.version)
        if changed_boxes:
            self._rebuild()
            self.w.geometry_changed()

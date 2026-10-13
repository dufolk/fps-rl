"""Destructible walls and floors.

Map boxes tagged with `params["destruct"]` ("soft" wall or "floor") are cut into ~0.75 m panels.
Each panel is its own collision box with hit points; bullets pierce intact unreinforced panels,
explosions / melee / breaching gadgets break them. Reinforced panels only yield to "hard" damage
(thermite). Breaking a panel rebuilds the world's collision arrays.
"""
from __future__ import annotations

import math

import numpy as np

from .maps.base import Box

PANEL = 0.75
PANEL_HP = {"soft": 150.0, "floor": 220.0}


class Panel:
    __slots__ = ("id", "wall", "kind", "box", "hp", "alive", "reinforced", "center", "axis")

    def __init__(self, pid, wall, kind, mn, mx, mat, axis):
        self.id = pid
        self.wall = wall
        self.kind = kind
        self.box = Box(tuple(mn), tuple(mx), mat, tag="panel", params={"panel": pid})
        self.hp = PANEL_HP.get(kind, 150.0)
        self.alive = True
        self.reinforced = False
        self.center = (np.asarray(mn) + np.asarray(mx)) * 0.5
        self.axis = axis  # thickness axis (0 / 1 walls, 2 floors)


class Destruction:
    def __init__(self, world):
        self.w = world
        self.panels: list[Panel] = []
        self.walls: list[list[int]] = []
        self._batch = 0
        self._dirty = False
        self.version = 0
        for b in world.map.boxes:
            kind = b.params.get("destruct")
            if not kind or not b.solid:
                continue
            self._split(b, kind)
        self.centers = np.array([p.center for p in self.panels]).reshape(-1, 3)
        self.mins = np.array([p.box.mn for p in self.panels], dtype=np.float64).reshape(-1, 3)
        self.maxs = np.array([p.box.mx for p in self.panels], dtype=np.float64).reshape(-1, 3)

    def _split(self, b: Box, kind: str):
        mn = np.array(b.mn, dtype=np.float64)
        mx = np.array(b.mx, dtype=np.float64)
        size = mx - mn
        axis = 2 if kind == "floor" else (0 if size[0] < size[1] else 1)
        plane = [i for i in range(3) if i != axis]
        n = [max(1, int(round(size[i] / PANEL))) for i in plane]
        wall = len(self.walls)
        ids = []
        for i in range(n[0]):
            for j in range(n[1]):
                pmn = mn.copy()
                pmx = mx.copy()
                a0, a1 = plane
                pmn[a0] = mn[a0] + size[a0] * i / n[0]
                pmx[a0] = mn[a0] + size[a0] * (i + 1) / n[0]
                pmn[a1] = mn[a1] + size[a1] * j / n[1]
                pmx[a1] = mn[a1] + size[a1] * (j + 1) / n[1]
                p = Panel(len(self.panels), wall, kind, pmn, pmx, b.mat, axis)
                self.panels.append(p)
                ids.append(p.id)
        self.walls.append(ids)

    # ------------------------------------------------------------------ state
    def reset(self):
        for p in self.panels:
            p.alive = True
            p.reinforced = False
            p.hp = PANEL_HP.get(p.kind, 150.0)
        self.version += 1

    def active_boxes(self):
        return [p.box for p in self.panels if p.alive]

    def panel_soft(self, pid: int) -> bool:
        p = self.panels[pid]
        return p.alive and not p.reinforced

    def broken(self) -> list:
        return [p.id for p in self.panels if not p.alive]

    def reinforced(self) -> list:
        return [p.id for p in self.panels if p.reinforced]

    def state(self) -> dict:
        return {"v": self.version, "broken": self.broken(), "reinf": self.reinforced()}

    def apply_state(self, st: dict):
        if not st or st.get("v") == self.version:
            return
        broken = set(st.get("broken", ()))
        reinf = set(st.get("reinf", ()))
        for p in self.panels:
            p.alive = p.id not in broken
            p.reinforced = p.id in reinf
        self.version = st["v"]
        self.w.geometry_changed()

    # ------------------------------------------------------------------ damage
    def _flush(self):
        if self._dirty and self._batch == 0:
            self._dirty = False
            self.version += 1
            self.w.geometry_changed()

    def _break(self, p: Panel, source=-1):
        p.alive = False
        p.hp = 0.0
        self._dirty = True
        self.w._emit("wall_break", panel=p.id, pos=p.center.tolist(), mat=p.box.mat, source=source)

    def damage_panel(self, pid: int, amount: float, kind: str = "bullet", source=-1):
        p = self.panels[pid]
        if not p.alive:
            return
        if p.reinforced and kind != "hard":
            return
        if kind == "bullet" and p.kind == "floor":
            amount *= 0.5
        p.hp -= amount
        if p.hp <= 0.0:
            self._break(p, source)
            self._flush()

    def _near(self, center, radius):
        c = np.asarray(center, dtype=np.float64)
        if not len(self.panels):
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        q = np.clip(c, self.mins, self.maxs)
        d = np.linalg.norm(q - c, axis=1)
        idx = np.nonzero(d <= radius)[0]
        return idx, d[idx]

    def blast(self, center, radius: float, damage: float, kind: str = "soft", source=-1):
        """Area damage that falls off with distance (grenades, bombs)."""
        idx, d = self._near(center, radius)
        self._batch += 1
        for i, di in zip(idx, d):
            self.damage_panel(int(i), damage * (1.0 - di / max(radius, 1e-6)) + 1e-3, kind, source)
        self._batch -= 1
        self._flush()

    def explode(self, center, radius: float, kind: str = "soft", floors: bool = True, source=-1):
        """Break every panel within `radius` (breach charges, thermite, hammer)."""
        idx, _ = self._near(center, radius)
        self._batch += 1
        for i in idx:
            p = self.panels[int(i)]
            if not p.alive or (p.kind == "floor" and not floors):
                continue
            if p.reinforced and kind != "hard":
                continue
            self._break(p, source)
        self._batch -= 1
        self._flush()

    def reinforce_around(self, pid: int) -> list:
        """Reinforce the ~1.5 m wide column of panels of the wall containing `pid`."""
        p0 = self.panels[pid]
        done = []
        for i in self.walls[p0.wall]:
            p = self.panels[i]
            if not p.alive or p.reinforced:
                continue
            if p.axis == 2:
                ok = math.dist(p.center[:2], p0.center[:2]) < 1.1
            else:
                lat = 1 - p.axis
                ok = abs(p.center[lat] - p0.center[lat]) < 0.8
            if ok:
                p.reinforced = True
                p.hp = PANEL_HP.get(p.kind, 150.0)
                done.append(i)
        if done:
            self.version += 1
            self.w._emit("reinforce", panels=done, pos=p0.center.tolist())
        return done

    def panel_at(self, box) -> Panel | None:
        pid = box.params.get("panel") if box is not None else None
        return self.panels[pid] if pid is not None else None

"""Biochemical pyramid, from the zm_lostcities BSP (workshop 1516169386).

The render mesh and brush collision come from tools/import_bsp.py. Spawns that sit on a ledge the
nav mesh cannot climb to are left out, so a round starts on the connected floor.
"""
from __future__ import annotations

import numpy as np

from .base import MapData
from .imported import load


def _connected_spawns(spawns):
    heights = [s[2] for pts in spawns.values() for s in pts]
    base = float(np.median(heights))
    out = {}
    for team, pts in spawns.items():
        kept = [s for s in pts if s[2] <= base + 0.5]
        out[team] = kept or list(pts[:1])
    return out


def _walkable_points(points, spawns):
    pts = [s[:2] for side in spawns.values() for s in side]
    if not pts:
        return points
    arr = np.array(pts, dtype=np.float64)
    kept = []
    for x, y in points:
        if float(np.min(np.hypot(arr[:, 0] - x, arr[:, 1] - y))) <= 28.0:
            kept.append((x, y))
    return kept


def build() -> MapData:
    boxes, spawns, bounds, strategic, ztop, _sea = load("pyramid")
    spawns = _connected_spawns(spawns)
    return MapData(
        name="pyramid",
        boxes=boxes,
        props=[],
        spawns=spawns,
        bounds=bounds,
        strategic_points=_walkable_points(strategic, spawns),
        water_z=-20.0,
        sun_dir=(-0.4, 0.25, -0.85),
        theme="town",
        modes=("bio",),
        nav_max_floor=ztop,
        mesh="pyramid",
    )

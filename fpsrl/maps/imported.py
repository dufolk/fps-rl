"""Shared loader for BSP imports (mesh + brush collision + entity spawns)."""
from __future__ import annotations

import os

import numpy as np

from .base import Box


def load(folder: str):
    """-> boxes, spawns, bounds, strategic points, highest box top, sea level (None without water)."""
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", folder)
    col = np.load(os.path.join(data, "collision.npz"))
    mn, mx = col["mn"].astype(np.float64), col["mx"].astype(np.float64)
    boxes = [Box(tuple(a), tuple(b), "concrete", visible=False, tag="wall") for a, b in zip(mn, mx)]
    sp = np.load(os.path.join(data, "spawns.npz"))

    def stand(row):
        x, y, z, yaw = (float(v) for v in row)
        tops = mx[(mn[:, 0] <= x) & (x <= mx[:, 0]) & (mn[:, 1] <= y) & (y <= mx[:, 1]) & (mx[:, 2] <= z + 0.4), 2]
        floor = float(tops.max()) if len(tops) else z
        return (x, y, floor + 0.02, yaw)

    spawns = {0: [stand(r) for r in sp["ct"]], 1: [stand(r) for r in sp["t"]]}
    bounds = ((float(mn[:, 0].min()), float(mn[:, 1].min())), (float(mx[:, 0].max()), float(mx[:, 1].max())))
    strategic = []
    for a, b in zip(mn, mx):
        if b[2] - a[2] < 1.2 and (b[0] - a[0]) > 4 and (b[1] - a[1]) > 4 and b[2] < 40:
            strategic.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    water = sp["water"] if "water" in sp.files else np.zeros(0)
    sea = float(water[0]) if len(water) else None
    return boxes, spawns, bounds, strategic[:24], float(mx[:, 2].max()), sea

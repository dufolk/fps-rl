"""Garage (假车库): a hospital compound around a flat concrete yard, imported from 医院-假车库_75m.obj.

The render mesh and collision boxes come from `tools/import_garage.py`; the scene is rotated so
its walls run along x / y. Buildings (rotated frame, metres, yard at z = 0):

* hospital - two storeys plus a roof house, south of the yard centre
* office   - long single storey block with a roof terrace, in the middle
* garage   - tall vehicle shed with roller-door openings, north-east
* house    - small gabled house, north-west

Side 0 spawns indoors on the ground floor of the hospital (south), side 1 indoors in the garage
(north-east), with no line of sight between spawns. The circular outer edge of the 75 m OBJ disc
is a poison zone (`zone`): stepping beyond it slowly drains HP.
"""
from __future__ import annotations

import os

import numpy as np

from .base import Box, MapData

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "garage")
BOUND_H = 15.9  # the invisible outer safety walls reach 16 m; nothing else in the map is that tall
ZONE_RADIUS = 74.5  # circular edge of 医院-假车库_75m.obj
ZONE_DPS = 8.0


def _collision():
    d = np.load(os.path.join(DATA, "collision.npz"))
    return (d["mn"].astype(np.float64), d["mx"].astype(np.float64), d["mat"], [str(m) for m in d["mats"]])


def build() -> MapData:
    mn, mx, mat, names = _collision()
    boxes = [Box(tuple(a), tuple(b), names[m], visible=False, tag="wall") for a, b, m in zip(mn, mx, mat)]
    inner = mx[:, 2] < BOUND_H

    def ground(x, y, z_lo=-1.0, z_hi=1.5):
        """Top of the walkable floor under (x, y) within [z_lo, z_hi]."""
        inside = (inner & (mn[:, 0] <= x) & (x <= mx[:, 0]) & (mn[:, 1] <= y) & (y <= mx[:, 1])
                  & (mx[:, 2] >= z_lo) & (mx[:, 2] <= z_hi))
        tops = mx[inside, 2]
        return float(tops.max()) if len(tops) else None

    def roofed(x, y, z):
        inside = inner & (mn[:, 0] <= x) & (x <= mx[:, 0]) & (mn[:, 1] <= y) & (y <= mx[:, 1])
        return bool(np.any(inside & (mn[:, 2] > z + 1.8)))

    def free(x, y, z, r=0.55):
        return not np.any((mn[:, 0] < x + r) & (mx[:, 0] > x - r) & (mn[:, 1] < y + r) & (mx[:, 1] > y - r)
                          & (mn[:, 2] < z + 1.9) & (mx[:, 2] > z + 0.05))

    def indoor_spots(cx, cy, z_lo, z_hi, yaw, x_lo, x_hi, y_lo, y_hi, n=10, step=1.15, radius=8.0):
        """`n` clear indoor standing spots (with ceiling overhead) nearest (cx, cy)."""
        out = []
        k = int(radius / step)
        cand = sorted(((cx + i * step, cy + j * step) for i in range(-k, k + 1) for j in range(-k, k + 1)),
                      key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        for x, y in cand:
            if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
                continue
            z = ground(x, y, z_lo, z_hi)
            if z is not None and roofed(x, y, z) and free(x, y, z) and free(x + 4.0, y, z, 0.45):
                out.append((round(x, 2), round(y, 2), round(z + 0.02, 2), yaw))
            if len(out) == n:
                return out
        raise ValueError(f"only {len(out)} indoor spawn spots near {cx, cy}")

    spawns = {
        0: indoor_spots(1.6, -19.8, 0.7, 1.0, 90.0, -0.5, 5.5, -23.0, -17.5),
        1: indoor_spots(14.0, 21.5, -0.05, 0.2, -90.0, 11.5, 16.5, 17.5, 25.5),
    }
    strategic = [(-18.0, -20.0), (-18.0, 2.0), (-11.0, 4.0), (-11.0, 28.0), (-2.0, 16.0), (4.0, 10.0),
                 (6.0, -8.0), (-7.0, -12.0), (-7.0, -34.0), (6.0, -36.0), (21.0, -34.0), (21.0, -14.0),
                 (24.0, -2.0), (24.0, 9.0), (16.0, 31.0), (28.0, 22.0), (16.0, 19.0), (4.0, -21.0)]

    return MapData(
        name="garage",
        boxes=boxes,
        props=[],
        spawns=spawns,
        bounds=((-85.0, -85.0), (85.0, 85.0)),
        nav_bounds=((-36.0, -44.0), (44.0, 40.0)),
        strategic_points=strategic,
        water_z=-20.0,
        sun_dir=(-0.4, 0.5, -0.77),
        theme="garage",
        objectives={"zone": {"center": (0.0, 0.0), "radius": ZONE_RADIUS, "dps": ZONE_DPS}},
        nav_max_floor=12.0,
        mesh="garage",
        ground_z=float(np.load(os.path.join(DATA, "mesh.npz"))["ground_z"]),
    )

"""Desert 2: de_dust2 from Counter-Strike 1.6, imported from the original BSP geometry.

The render mesh and collision boxes come from `tools/import_dust2.py` (model by vrchris on
Sketchfab, CC-BY 4.0; see data/dust2/CREDITS.txt). Hammer units are scaled by 0.025 m, so the
72-unit player hull is 1.8 m and an 18-unit step is 0.45 m, as in the game.

North (+y) is up: A site north-east, B site north-west, T spawn south, CT spawn north between
the sites. Mid runs north-south through the centre; mid doors open onto CT mid, the lower
tunnels (狗洞) join mid just south of them, and A short (A小) climbs from mid to A.
"""
from __future__ import annotations

import os

import numpy as np

from .base import Box, MapData

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dust2")


def _collision():
    d = np.load(os.path.join(DATA, "collision.npz"))
    return d["mn"].astype(np.float64), d["mx"].astype(np.float64)


def build() -> MapData:
    mn, mx = _collision()
    boxes = [Box(tuple(a), tuple(b), "sandstone", visible=False, tag="wall") for a, b in zip(mn, mx)]
    ztop = float(mx[:, 2].max())

    def ground(x, y, below=None):
        """Top of the walkable floor under (x, y), the highest one below `below` if given."""
        inside = (mn[:, 0] <= x) & (x <= mx[:, 0]) & (mn[:, 1] <= y) & (y <= mx[:, 1]) & (mx[:, 2] < ztop - 0.01)
        tops = mx[inside, 2]
        if below is not None:
            tops = tops[tops <= below]
        return float(tops.max()) if len(tops) else 0.0

    def free(x, y, z):
        r = 0.45
        return not np.any((mn[:, 0] < x + r) & (mx[:, 0] > x - r) & (mn[:, 1] < y + r) & (mx[:, 1] > y - r)
                          & (mn[:, 2] < z + 1.9) & (mx[:, 2] > z + 0.01))

    def spots(cx, cy, yaw, n=10, step=1.6, radius=8.0):
        """`n` clear standing spots nearest (cx, cy), at least `step` apart."""
        out = []
        k = int(radius / step)
        cand = sorted(((cx + i * step, cy + j * step) for i in range(-k, k + 1) for j in range(-k, k + 1)),
                      key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        for x, y in cand:
            z = ground(x, y)
            if free(x, y, z):
                out.append((x, y, z + 0.02, yaw))
            if len(out) == n:
                return out
        raise ValueError(f"only {len(out)} spawn spots near {cx, cy}")

    def rect(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def site(name, x0, y0, x1, y1):
        zs = [round(ground(x, y), 2) for x in np.linspace(x0, x1, 9) for y in np.linspace(y0, y1, 9)]
        z = max(set(zs), key=zs.count)
        return {"name": name, "poly": rect(x0, y0, x1, y1), "zmin": min(zs) - 0.5, "zmax": max(zs) + 3.0, "z": z}

    spawns = {0: spots(2.0, -46.0, 90.0), 1: spots(15.0, 30.0, 200.0)}
    sites = [site("A", 28.0, 30.0, 46.0, 47.0), site("B", -46.0, 26.0, -30.0, 46.0)]
    buy_zones = {0: [rect(-14.0, -54.0, 18.0, -40.0)], 1: [rect(9.0, 24.0, 21.0, 36.0)]}
    strategic = [(0.0, -10.0), (-3.0, 15.0), (-10.0, 28.0), (-24.0, 28.0), (-38.0, 0.0), (-38.0, -25.0),
                 (-38.0, -45.0), (0.0, -47.0), (25.0, -12.0), (42.0, 0.0), (44.0, -22.0), (18.0, 15.0),
                 (28.0, 30.0), (-20.0, 6.0), (37.0, 39.0), (-40.0, 39.0)]

    return MapData(
        name="desert2",
        boxes=boxes,
        props=[],
        spawns=spawns,
        bounds=((-56.4, -66.8), (56.4, 66.8)),
        strategic_points=strategic,
        water_z=-20.0,
        sun_dir=(-0.45, 0.35, -0.82),
        theme="desert",
        modes=("bomb", "elim", "tdm"),
        objectives={"sites": sites},
        buy_zones=buy_zones,
        nav_max_floor=ztop - 1.0,
        mesh="dust2",
    )

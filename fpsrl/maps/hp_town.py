"""HP Town (占点小镇) for Domination, rebuilt from the level export in domin/.

`tools/import_hp_city.py` voxelizes HP_City_Main.obj into boxes (data/hp_city.npz): textured
visual boxes, material-agnostic collision boxes, one box per crate / barrel / car, tree trunks,
the three capture zones (Point_A/B/C) and the spawn points from the design sheet.

World frame: x = (UE X - 5950) / 100 east, y = -(UE Y - 5525) / 100 north, z up, metres.
A sits north of the centre between 红房 (west) and 黑房 (east), B south-west, C east on the
raised east yard. Red (team 0) spawns on the south-east terrace, blue (team 1) in the north-west.
"""
from __future__ import annotations

import os

import numpy as np

from .base import Box, MapData, Prop

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hp_city.npz")
CRATE = (1.13, 1.05)  # 补给箱 footprint / height on each 底座


def build() -> MapData:
    d = np.load(DATA)
    mats = [str(m) for m in d["mats"]]
    boxes: list[Box] = []
    for a, b, m in zip(d["mn"], d["mx"], d["mat"]):
        boxes.append(Box(tuple(map(float, a)), tuple(map(float, b)), mats[m], solid=False, blocks_bullets=False,
                         tag="ground" if a[2] <= -0.99 else "wall"))
    cmn, cmx = d["cmn"].astype(np.float64), d["cmx"].astype(np.float64)
    for a, b in zip(cmn, cmx):
        boxes.append(Box(tuple(a), tuple(b), "concrete", visible=False, tag="wall"))
    def off_grid(v, inward, gap=0.01):
        """Keep prop faces >= `gap` off the 0.1 m voxel planes so they don't z-fight with them."""
        g = np.round(v / 0.1) * 0.1
        return np.where(np.abs(v - g) < gap, g + inward * gap, v)

    for k, (a, b, m) in enumerate(zip(d["pmn"], d["pmx"], d["pmat"])):
        if float(a[2]) >= 6.0:  # lamp heads and tarp scraps above the playable skyline
            continue
        inset = 0.0015 * (k % 11)  # props overlapping each other get distinct face planes
        a, b = off_grid(a.astype(np.float64), 1) + inset, off_grid(b.astype(np.float64), -1) - inset
        boxes.append(Box(tuple(map(float, a)), tuple(map(float, b)), mats[m], tag="cover"))
    props: list[Prop] = []
    for x, y, z, h in d["trees"].astype(np.float64):
        r = 0.2
        boxes.append(Box((x - r, y - r, z), (x + r, y + r, z + min(h * 0.45, 4.0)), "wood_planks", visible=False,
                         tag="trunk"))
        props.append(Prop("tree", (x, y, z), {"scale": h / 3.8, "autumn": True, "seed": int(abs(x * 7 + y * 13))}))
    s, hgt = CRATE[0] / 2, CRATE[1]
    for x, y, z in d["pads"].astype(np.float64):
        boxes.append(Box((x - s, y - s, z), (x + s, y + s, z + hgt), "crate", tag="crate"))

    solid_mn = np.concatenate([cmn, d["pmn"]])
    solid_mx = np.concatenate([cmx, d["pmx"]])

    def ground(x, y, below=99.0, r=0.45):
        """Highest floor top under the player's footprint (voxel floors step by 0.1 m)."""
        inside = (solid_mn[:, 0] < x + r) & (x - r < solid_mx[:, 0]) & (solid_mn[:, 1] < y + r) & (y - r < solid_mx[:, 1])
        tops = solid_mx[inside, 2]
        tops = tops[tops <= below]
        return float(tops.max()) if len(tops) else 0.0

    def free(x, y, z):
        r = 0.45
        return not np.any((solid_mn[:, 0] < x + r) & (solid_mx[:, 0] > x - r) & (solid_mn[:, 1] < y + r)
                          & (solid_mx[:, 1] > y - r) & (solid_mn[:, 2] < z + 1.9) & (solid_mx[:, 2] > z + 0.01))

    def spots(pts, yaw, n=10, step=1.4, radius=7.0):
        """The design-sheet spawns first, then clear spots around them on the same floor."""
        out = []
        for x, y, z in pts:
            g = ground(x, y, z + 0.5)
            out.append((float(x), float(y), g + 0.02, yaw))
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        fz = out[0][2]
        k = int(radius / step)
        cand = sorted(((cx + i * step, cy + j * step) for i in range(-k, k + 1) for j in range(-k, k + 1)),
                      key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        for x, y in cand:
            if len(out) >= n:
                break
            z = ground(x, y, fz + 0.5)
            if abs(z - fz) > 0.5 or not free(x, y, z):
                continue
            if all(np.hypot(x - o[0], y - o[1]) >= step * 0.9 for o in out):
                out.append((float(x), float(y), z + 0.02, yaw))
        return out

    spawns = {0: spots(d["spawn0"].astype(np.float64) - [0, 0, 0.9], 120.0),
              1: spots(d["spawn1"].astype(np.float64) - [0, 0, 0.9], 300.0)}
    points = []
    for name, z in zip(d["zone_names"], d["zone_z"]):
        poly = d[f"zone_{name}"].astype(np.float64).round(2).tolist()
        points.append({"name": str(name), "poly": poly, "zmin": float(z) - 1.0, "zmax": float(z) + 3.6,
                       "z": float(z)})
    x0, y0, x1, y1 = d["reach"].astype(np.float64)
    x0, y0, x1, y1 = x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5
    for mn, mx in (((x0 - 1, y0 - 1, -1), (x0, y1 + 1, 40)), ((x1, y0 - 1, -1), (x1 + 1, y1 + 1, 40)),
                   ((x0, y0 - 1, -1), (x1, y0, 40)), ((x0, y1, -1), (x1, y1 + 1, 40))):
        boxes.append(Box(mn, mx, "concrete", visible=False, tag="wall"))
    candidates = [(-8.0, 9.0), (-17.0, -23.0), (28.0, -3.5), (-27.5, 9.5), (11.5, 9.5), (0.0, 28.0),
                  (27.0, 18.0), (7.0, -27.0), (5.0, -7.0), (-28.0, 26.0), (26.0, -32.0), (-31.0, -8.0),
                  (-31.0, -32.0), (31.0, 28.0), (16.0, -16.0), (-14.0, 27.0)]
    strategic = [(x, y) for x, y in candidates if free(x, y, ground(x, y, 3.5))]
    return MapData(
        name="hp_town",
        boxes=boxes,
        props=props,
        spawns=spawns,
        bounds=((x0, y0), (x1, y1)),
        strategic_points=strategic,
        water_z=-20.0,
        sun_dir=(-0.5, 0.42, -0.7),
        theme="town",
        modes=("dom", "tdm"),
        objectives={"points": points},
        nav_max_floor=5.0,
    )

"""Desert 2 (CF 沙漠灰 / Dust2 layout) for Bomb Defusal.

North (+y) is up. Terrorists (side 0) spawn in the south, counter-terrorists (side 1) in the north.
A site is in the north-east (reached via Long A or Short / Catwalk), B site in the north-west
(reached via the tunnels or from CT mid through B doors). Mid runs north-south through the centre.

Authored on a 2 m grid with floor levels in multiples of 0.4 m (one walkable step per cell).
"""
from __future__ import annotations

import math

from .base import Box, MapData, Prop
from .gridmap import GridMap

L = 0.4  # metres per floor level


def build() -> MapData:
    g = GridMap(50, 50, 2.0, -50.0, 50.0)
    f = g.floor
    # --- B site & surroundings
    f(2, 14, 2, 14, 2 * L, "sand")
    f(2, 5, 2, 6, 3 * L, "stone_dark")          # back-site platform
    f(15, 15, 4, 6, 1 * L, "stone_dark")        # B doors
    # --- CT mid, CT spawn
    f(16, 25, 3, 7, 0.0, "sand")
    f(26, 26, 3, 8, 1 * L, "sand")
    f(27, 28, 3, 8, 2 * L, "sand")
    f(29, 33, 1, 9, 3 * L, "sand")
    # --- A site, goose, pit, A ramp
    f(34, 45, 2, 13, 4 * L, "sand")
    f(46, 47, 2, 3, 4 * L, "sand")
    f(46, 48, 6, 13, 0.0, "sand_dark")          # pit
    f(46, 48, 14, 15, 1 * L, "sand_dark")
    f(38, 45, 14, 15, 3 * L, "sand")            # A ramp
    # --- Long A
    f(38, 46, 16, 32, 2 * L, "sand")
    f(36, 39, 33, 36, 2 * L, "stone_dark")      # long doors
    f(28, 36, 37, 42, 3 * L, "sand")            # outside long
    # --- mid
    f(20, 24, 8, 27, 0.0, "sand_dark")
    f(18, 27, 28, 32, 1 * L, "sand")            # top mid
    f(22, 25, 33, 35, 2 * L, "sand")            # suicide
    f(22, 25, 36, 38, 3 * L, "sand")
    # --- short / catwalk
    f(25, 28, 27, 27, 2 * L, "stone_dark")
    f(25, 28, 26, 26, 3 * L, "stone_dark")
    f(25, 30, 14, 25, 4 * L, "sand")
    f(31, 33, 13, 16, 4 * L, "sand")
    # --- tunnels
    f(8, 19, 22, 23, 1 * L, "stone_dark")       # lower tunnels
    f(4, 7, 15, 29, 1 * L, "stone_dark")        # upper tunnels
    f(4, 14, 28, 30, 1 * L, "stone_dark")
    f(12, 16, 31, 38, 2 * L, "sand")            # outside tunnels
    f(15, 17, 39, 44, 3 * L, "sand")
    # --- T spawn
    f(17, 30, 39, 47, 4 * L, "sand")
    # mid doors: narrow the north end of mid
    g.wall(20, 20, 8, 8)
    g.wall(24, 24, 8, 8)
    # ceilings
    g.roof(4, 7, 16, 29, 3.0, "sandstone_dark")
    g.roof(8, 14, 28, 30, 3.0, "sandstone_dark")
    g.roof(8, 18, 22, 23, 3.0, "sandstone_dark")

    def wall_h(c0, c1, r0, r1):
        k = (c0 * 7 + r0 * 13 + c1 * 3 + r1) % 5
        return 5.5 + k * 0.6

    def wall_m(c0, c1, r0, r1):
        k = (c0 * 5 + r0 * 3) % 4
        return ("sandstone", "plaster_tan", "sandstone", "sandstone_dark")[k]

    boxes = g.build(wall_h, wall_m)
    props = []

    def cx(c):
        return g.x(c)

    def cy(r):
        return g.y(r)

    def crate(x, y, z, s=1.2, h=None, mat="crate"):
        h = h or s
        boxes.append(Box((x - s / 2, y - s / 2, z), (x + s / 2, y + s / 2, z + h), mat, tag="crate"))

    def door_pair(xa, ya, xb, yb, z, gap=1.5, thick=0.12, height=3.0):
        """Two wooden door leaves spanning (xa,ya)-(xb,yb) leaving a centred gap."""
        if abs(xb - xa) > abs(yb - ya):
            mid = (xa + xb) / 2
            for a, b in ((xa, mid - gap / 2), (mid + gap / 2, xb)):
                boxes.append(Box((a, ya - thick / 2, z), (b, ya + thick / 2, z + height), "wood_door", tag="door"))
        else:
            mid = (ya + yb) / 2
            for a, b in ((ya, mid - gap / 2), (mid + gap / 2, yb)):
                boxes.append(Box((xa - thick / 2, a, z), (xa + thick / 2, b, z + height), "wood_door", tag="door"))

    # doors
    door_pair(cx(21), cy(8) - 1.0, cx(24), cy(8) - 1.0, 0.0, gap=1.8)                 # mid doors
    # long doors: the 2 m doorway between outside long (row 37) and the long-doors room (col 36)
    yld = cy(37)
    boxes.append(Box((cx(37), yld - 0.06, 2 * L), (cx(37) + 1.1, yld + 0.06, 2 * L + 3.0), "wood_door", tag="door"))
    boxes.append(Box((cx(36), yld - 0.4, 2 * L + 3.0), (cx(37), yld + 0.4, 6.0), "sandstone", tag="wall"))
    door_pair(cx(38), cy(33), cx(40), cy(33), 2 * L, gap=1.4)                         # long doors (north)
    boxes.append(Box((cx(38), cy(33) - 0.4, 2 * L + 3.0), (cx(40), cy(33) + 0.4, 6.0), "sandstone", tag="wall"))
    boxes.append(Box((cx(15) + 0.4, cy(4) - 0.06, 1 * L), (cx(16), cy(4) + 0.06, 1 * L + 3.0), "wood_door",
                     tag="door"))  # B doors (one leaf open)
    # lintels above doorways
    boxes.append(Box((cx(21), cy(9), 3.0), (cx(24), cy(8), 6.0), "sandstone", tag="wall"))
    boxes.append(Box((cx(15), cy(7), 1 * L + 3.0), (cx(16), cy(4), 6.0), "sandstone", tag="wall"))

    # A site
    zA = 4 * L
    crate(cx(40) + 1.0, cy(6), zA, 1.6)                       # default plant box
    crate(cx(40) + 1.0, cy(6), zA + 1.6, 1.2)
    crate(cx(44), cy(3), zA, 1.4)                             # goose boxes
    crate(cx(44) + 1.4, cy(3), zA, 1.4)
    crate(cx(35) + 0.8, cy(11), zA, 1.2)                      # short boxes
    crate(cx(36) + 0.2, cy(11) - 1.2, zA, 1.0)
    boxes.append(Box((cx(38), cy(16), 2 * L), (cx(38) + 0.6, cy(14), 3 * L + 1.0), "sandstone", tag="wall"))
    crate(cx(45) + 0.4, cy(20), 2 * L, 1.6, mat="crate_big")  # long corner box
    crate(cx(44), cy(26), 2 * L, 1.2)                         # long blue box
    crate(cx(42), cy(30), 2 * L, 1.0)
    # pit walls
    boxes.append(Box((cx(46) - 0.4, cy(14), 0.0), (cx(46), cy(6), 4 * L + 0.9), "sandstone_dark", tag="wall"))
    # B site
    zB = 2 * L
    crate(cx(8), cy(8), zB, 1.6, mat="crate_big")             # B box centre
    crate(cx(8) + 1.6, cy(8), zB, 1.2)
    crate(cx(4), cy(11), zB, 1.2)                             # car area
    crate(cx(11), cy(4), zB, 1.2)                             # back B
    crate(cx(11) + 1.2, cy(4), zB, 1.2)
    crate(cx(11), cy(4), zB + 1.2, 1.2)
    boxes.append(Box((cx(9), cy(13), zB), (cx(9) + 0.8, cy(12) - 0.4, zB + 4.0), "sandstone_dark", tag="wall"))
    boxes.append(Box((cx(3) + 0.5, cy(13), zB), (cx(3) + 1.3, cy(12) - 0.4, zB + 4.0), "sandstone_dark", tag="wall"))
    # B car (collision + prop)
    bxc, byc = cx(11), cy(11)
    boxes.append(Box((bxc - 2.1, byc - 0.85, zB), (bxc + 2.1, byc + 0.85, zB + 1.3), "metal_dark", visible=False,
                     tag="car"))
    props.append(Prop("car", (bxc, byc, zB), {"yaw": 0.0, "color": (0.55, 0.5, 0.35)}))
    # mid: xbox, mid crates
    crate(cx(23) + 1.0, cy(27) + 1.0, 0.0, 1.4, mat="crate_big")
    crate(cx(20) + 0.7, cy(16), 0.0, 1.2)
    # CT spawn / CT mid
    crate(cx(30) + 1.0, cy(5), 3 * L, 1.2)
    crate(cx(18), cy(5), 0.0, 1.2)
    crate(cx(18) + 1.2, cy(5), 0.0, 1.0)
    # T spawn
    crate(cx(20), cy(41), 4 * L, 1.2)
    crate(cx(27), cy(44), 4 * L, 1.4, mat="crate_big")
    crate(cx(30), cy(39), 3 * L, 1.2)                         # outside long box
    # top mid / lower tunnels boxes
    crate(cx(19), cy(30), 1 * L, 1.0)
    crate(cx(9), cy(23) + 0.6, 1 * L, 1.0)
    # catwalk railing (low wall along the mid side, with a gap near the stairs)
    boxes.append(Box((cx(25), cy(25), 4 * L), (cx(25) + 0.25, cy(17), 4 * L + 1.0), "sandstone", tag="wall"))

    # palms & lamps
    for c, r, lvl in ((31, 41, 3), (19, 46, 4), (33, 2, 3), (43, 5, 4), (13, 13, 2), (3, 4, 3), (17, 7, 0),
                      (47, 25, 2)):
        x, y = g.center(c, r)
        props.append(Prop("palm", (x, y, lvl * L), {}))

    # spawns
    spawns = {0: [], 1: []}
    for i in range(10):
        c, r = 19 + (i % 5) * 2, 41 + (i // 5) * 3
        x, y = g.center(c, r)
        spawns[0].append((x, y, 4 * L, 90.0))
    for i in range(10):
        c, r = 29 + (i % 5), 2 + (i // 5) * 4
        x, y = g.center(c, r)
        spawns[1].append((x, y, 3 * L, 220.0))

    def poly(c0, c1, r0, r1):
        xa, ya, xb, yb = g.rect_xy(c0, c1, r0, r1)
        return [[xa, ya], [xb, ya], [xb, yb], [xa, yb]]

    sites = [
        {"name": "A", "poly": poly(36, 45, 2, 12), "zmin": zA - 0.5, "zmax": zA + 4.0, "z": zA},
        {"name": "B", "poly": poly(2, 13, 2, 12), "zmin": zB - 0.5, "zmax": zB + 4.0, "z": zB},
    ]
    buy_zones = {0: [poly(17, 30, 39, 47)], 1: [poly(26, 33, 1, 9)]}

    strategic = [g.center(c, r) for c, r in (
        (40, 7), (8, 8), (22, 18), (22, 30), (42, 24), (37, 39), (28, 18), (6, 22), (14, 25), (14, 35),
        (31, 5), (20, 5), (42, 15), (47, 10), (32, 14), (10, 29))]

    return MapData(
        name="desert2",
        boxes=boxes,
        props=props,
        spawns=spawns,
        bounds=((-50.0, -50.0), (50.0, 50.0)),
        strategic_points=strategic,
        water_z=-20.0,
        sun_dir=(-0.45, 0.35, -0.82),
        theme="desert",
        modes=("bomb", "tdm"),
        objectives={"sites": sites},
        buy_zones=buy_zones,
        nav_max_floor=2.2,
    )

"""HP Town (占点小镇) for Domination.

Source data is in centimetres in the editor's X/Z plane (activity area X 1950-9950, Z 1450-9600).
World coordinates: x = (X - 5950) / 100 (east), y = -(Z - 5525) / 100 (north), so the map spans
about 80 m x 81.5 m centred on the origin.

If `fpsrl/maps/data/hp_town.json` exists (generated from the raw point data by
`tools/import_town.py`), its walls / zones / spawns are used. Otherwise the layout below, traced
from the overlay screenshot, is built.
"""
from __future__ import annotations

import json
import os

from .base import Box, MapData, Prop

CX, CZ = 5950.0, 5525.0
DATA = os.path.join(os.path.dirname(__file__), "data", "hp_town.json")


def to_world(X, Z):
    return (X - CX) / 100.0, -(Z - CZ) / 100.0


def _poly(pts):
    return [list(to_world(X, Z)) for X, Z in pts]


# capture zones traced from the overlay (centimetres, editor X/Z)
ZONES_CM = {
    "A": [(4080, 3580), (6280, 3580), (6280, 4250), (6180, 4250), (6180, 5650), (5800, 5880), (4450, 5880),
          (4080, 5700), (4080, 4200), (4180, 4200), (4180, 3800), (4080, 3800)],
    "B": [(2750, 7000), (4080, 7000), (4080, 7200), (4400, 7200), (4400, 7300), (5600, 7300), (5600, 7600),
          (5700, 7600), (5700, 8550), (2750, 8550)],
    "C": [(8050, 5400), (9100, 5400), (9100, 5550), (9600, 5550), (9600, 6300), (9100, 6300), (9100, 6400),
          (8000, 6400), (8000, 5750), (8050, 5750)],
}


def _zones():
    return [{"name": k, "poly": _poly(v), "zmin": -0.5, "zmax": 4.0, "z": 0.1} for k, v in ZONES_CM.items()]


def _from_json(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    boxes = [Box(tuple(b["mn"]), tuple(b["mx"]), b.get("mat", "brick"), tag=b.get("tag", "wall"))
             for b in d["boxes"]]
    props = [Prop(p["kind"], tuple(p["pos"]), p.get("params", {})) for p in d.get("props", [])]
    return boxes, props, d


def build() -> MapData:
    boxes: list[Box] = []
    props: list[Prop] = []
    x0, x1, y0, y1 = -40.0, 40.0, -40.75, 40.75
    # ground: asphalt everywhere, raised tiled plazas
    boxes.append(Box((x0 - 2, y0 - 2, -1.0), (x1 + 2, y1 + 2, 0.0), "asphalt", tag="ground"))
    extra = None
    if os.path.exists(DATA):
        jb, jp, extra = _from_json(DATA)
        boxes += jb
        props += jp
    else:
        _traced(boxes, props)
    # perimeter walls
    H = 9.0
    for mn, mx in (((x0 - 1, y0 - 1, 0), (x0, y1 + 1, H)), ((x1, y0 - 1, 0), (x1 + 1, y1 + 1, H)),
                   ((x0, y0 - 1, 0), (x1, y0, H)), ((x0, y1, 0), (x1, y1 + 1, H))):
        boxes.append(Box(mn, mx, "brick_dark", tag="wall"))

    spawns = {
        0: [(-36.0 + (i % 5) * 2.6, -37.0 + (i // 5) * 2.6, 0.0, 60.0) for i in range(10)],
        1: [(24.0 + (i % 5) * 2.6, 24.0 + (i // 5) * 2.6, 0.0, 240.0) for i in range(10)],
    }
    zones = _zones()
    if extra is not None:
        if extra.get("spawns"):
            spawns = {int(k): [tuple(s) for s in v] for k, v in extra["spawns"].items()}
        if extra.get("points"):
            zones = extra["points"]
    strategic = [(-8.5, 9.0), (-17.0, -22.0), (28.0, -3.5), (-28.0, 12.0), (-28.0, -6.0), (10.0, 14.0),
                 (12.0, -14.0), (25.0, 12.0), (-6.0, -9.5), (2.0, -36.0), (-20.0, 24.0), (8.0, 24.0),
                 (30.0, -26.0), (-36.0, -20.0), (20.0, -36.0), (-8.0, -30.0)]
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
        objectives={"points": zones},
        nav_max_floor=1.0,
    )


def _traced(boxes, props):
    """Hand-traced building blocks / cover from the overlay (world metres)."""
    mats = ("brick", "plaster_white", "plaster_yellow", "brick", "plaster_blue", "plaster_white")

    def bld(xa, ya, xb, yb, h=7.0, k=0, roof=True):
        m = mats[k % len(mats)]
        boxes.append(Box((xa, ya, 0.0), (xb, yb, h), m, tag="building"))
        if roof:
            boxes.append(Box((xa - 0.3, ya - 0.3, h), (xb + 0.3, yb + 0.3, h + 0.5),
                             "roof" if k % 2 == 0 else "roof_grey", tag="roof"))

    def low(xa, ya, xb, yb, h=1.1, m="concrete"):
        boxes.append(Box((xa, ya, 0.0), (xb, yb, h), m, tag="cover"))

    def plaza(xa, ya, xb, yb, m="tile_floor"):
        boxes.append(Box((xa, ya, 0.0), (xb, yb, 0.1), m, tag="ground"))

    def crate(x, y, s=1.2, z=0.0, m="crate"):
        boxes.append(Box((x - s / 2, y - s / 2, z), (x + s / 2, y + s / 2, z + s), m, tag="crate"))

    # --- north district (Z 1450-3460)
    bld(-38.5, 27.0, -27.0, 38.5, 8.0, 0)
    bld(-22.0, 29.5, -11.0, 38.5, 7.0, 1)
    bld(-3.0, 31.0, 7.0, 38.5, 9.0, 2)
    bld(11.0, 26.5, 16.5, 38.5, 7.0, 3)
    bld(18.2, 32.6, 38.5, 38.5, 8.0, 4)
    plaza(-10.0, 22.5, -4.0, 28.5, "grass")
    props.append(Prop("tree", (-7.0, 25.5, 0.1), {}))
    # north row of shopfronts with alleys (the long Z 3460 line)
    for xa, xb in ((-36.5, -31.0), (-27.0, -20.5), (-6.0, 4.0), (9.0, 18.5), (27.0, 36.9)):
        bld(xa, 20.6, xb, 23.0, 4.0, int(xa) % 5, roof=False)
    # --- west block (X 2300-4040)
    bld(-36.5, 8.5, -26.0, 18.0, 7.0, 1)
    bld(-36.5, -1.6, -27.5, 5.0, 6.0, 3)
    bld(-24.0, 12.0, -21.0, 17.0, 3.2, 5, roof=False)
    low(-24.0, 2.0, -22.8, 7.0)
    # --- A plaza (X 4100-6300, Z 3600-5900) with central fountain
    plaza(-18.4, -3.6, 3.4, 19.2)
    boxes.append(Box((-10.0, 7.7, 0.0), (-7.0, 10.7, 0.8), "stone_dark", tag="fountain"))
    boxes.append(Box((-8.9, 8.8, 0.8), (-8.1, 9.6, 2.2), "stone_dark", tag="fountain"))
    for x, y in ((-15.5, 15.5), (-1.0, 15.5), (-15.0, 1.0), (0.0, 1.5)):
        low(x - 1.6, y - 0.4, x + 1.6, y + 0.4, 0.5, "wood_planks")      # benches
    low(-14.5, 4.0, -12.0, 6.6, 1.1, "concrete")                          # planters
    low(-5.0, 11.0, -2.5, 13.6, 1.1, "concrete")
    crate(-16.5, 9.0)
    crate(1.5, 8.0)
    crate(1.5, 9.2)
    props.append(Prop("tree", (-13.25, 5.3, 1.1), {}))
    props.append(Prop("tree", (-3.75, 12.3, 1.1), {}))
    # walls framing A on east / west (with entrances)
    bld(-20.5, 4.0, -18.6, 12.0, 4.0, 0, roof=False)
    bld(3.6, 9.0, 5.4, 18.0, 4.0, 2, roof=False)
    bld(3.6, -2.0, 5.4, 4.0, 4.0, 2, roof=False)
    # --- centre-east block (X 6300-7900)
    bld(8.0, 9.5, 17.0, 18.0, 8.0, 0)
    bld(8.0, -1.0, 14.0, 5.5, 6.0, 4)
    low(15.5, 1.0, 17.0, 4.0)
    props.append(Prop("car", (18.5, 6.5, 0.0), {"yaw": 90.0, "color": (0.15, 0.3, 0.55)}))
    boxes.append(Box((17.65, 4.4, 0.0), (19.35, 8.6, 1.45), "metal_dark", visible=False, tag="car"))
    # --- east (X 7900-9950)
    bld(22.0, 7.5, 34.0, 18.0, 8.0, 3)
    bld(36.0, 4.0, 40.0, 18.0, 6.0, 1)
    bld(22.0, -22.0, 37.0, -12.5, 7.0, 5)
    # C courtyard cover
    low(26.0, -5.0, 28.6, -3.8)
    crate(31.5, -1.0)
    crate(32.7, -1.0)
    crate(32.1, -1.0, 1.0, 1.2)
    low(23.0, -1.5, 24.2, 0.8, 1.1, "wood_planks")
    props.append(Prop("tree", (30.5, 3.2, 0.0), {}))
    props.append(Prop("tree", (31.0, -12.2, 0.0), {}))
    # --- between A and B (X 4100-5900, Z 5900-7000)
    bld(-14.5, -13.0, -3.5, -6.0, 6.0, 1)
    low(-17.5, -10.5, -16.3, -7.5)
    # --- B yard (X 2750-5700, Z 7000-8550)
    plaza(-31.9, -30.2, 0.4, -14.8, "concrete")
    low(-20.0, -26.0, -17.0, -24.8, 1.1, "wood_planks")
    low(-12.5, -21.0, -11.3, -17.5)
    crate(-26.0, -20.0)
    crate(-24.8, -20.0)
    crate(-26.0, -20.0, 1.2, 1.2)
    crate(-6.0, -27.0, 1.4, 0.0, "crate_big")
    props.append(Prop("car", (-28.5, -26.5, 0.1), {"yaw": 0.0, "color": (0.6, 0.12, 0.1)}))
    boxes.append(Box((-30.6, -27.35, 0.0), (-26.4, -25.65, 1.45), "metal_dark", visible=False, tag="car"))
    # west of B and the south-west spawn
    bld(-38.5, -12.0, -33.5, -4.0, 6.0, 2)
    bld(-38.5, -28.0, -34.0, -16.0, 6.0, 0)
    # --- south-centre divider (X 5875 line) and south-east buildings
    bld(0.8, -31.0, 4.5, -19.0, 7.0, 3)
    bld(0.8, -15.0, 4.5, -4.5, 7.0, 3)
    bld(8.0, -30.0, 18.0, -19.0, 8.0, 4)
    bld(8.0, -13.0, 16.0, -6.0, 6.0, 0)
    low(19.0, -28.0, 20.2, -24.0)
    low(-4.0, -36.5, -1.5, -35.3)
    bld(6.0, -40.75, 18.0, -35.0, 5.0, 1)
    props.append(Prop("tree", (21.5, -36.7, 0.0), {}))
    props.append(Prop("tree", (31.0, -36.7, 0.0), {}))
    props.append(Prop("tree", (-32.5, -39.0, 0.0), {}))
    props.append(Prop("tree", (-36.0, 31.2, 0.0), {}))
    # NE spawn plaza cover
    low(26.0, 29.5, 29.0, 30.7)
    for x, y in ((-30.0, 0.0), (6.5, 19.5), (20.0, -3.0), (-2.0, -16.5), (20.5, 22.5), (-20.0, -32.5)):
        props.append(Prop("lamp", (x, y, 0.0), {"yaw": 0.0}))

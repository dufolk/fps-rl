"""Cargo-ship deck map inspired by CrossFire's "Transport Ship".

The deck runs along X (-40..40) and Y (-13..13). Team 0 spawns on the raised
platform at the west end, team 1 at the east end. Everything on the west half
is defined once and point-mirrored ((x, y) -> (-x, -y)) so both sides are fair.
"""
from __future__ import annotations

from .base import Box, MapData, Prop

CONT_L, CONT_W, CONT_H = 6.06, 2.44, 2.59
CRATE = 0.85
BIG_CRATE = 1.2


def _mirror_box(b: Box) -> Box:
    params = dict(b.params)
    if "door" in params:
        params["door"] = -params["door"]
    return Box((-b.mx[0], -b.mx[1], b.mn[2]), (-b.mn[0], -b.mn[1], b.mx[2]), b.mat,
               b.solid, b.blocks_bullets, b.visible, b.tag, params)


def _mirror_prop(p: Prop) -> Prop:
    params = dict(p.params)
    if "yaw" in params:
        params["yaw"] = params["yaw"] + 180.0
    return Prop(p.kind, (-p.pos[0], -p.pos[1], p.pos[2]), params)


def _container(cx, cy, along, colors, z0=0.0, door=1):
    hx, hy = (CONT_L / 2, CONT_W / 2) if along == "x" else (CONT_W / 2, CONT_L / 2)
    out = []
    for level, color in enumerate(colors):
        z = z0 + level * CONT_H
        out.append(Box((cx - hx, cy - hy, z), (cx + hx, cy + hy, z + CONT_H),
                       f"container_{color}", tag="container",
                       params={"along": along, "door": door}))
    return out


def _crate(x0, y0, levels=1, size=CRATE, z0=0.0):
    return [Box((x0, y0, z0 + i * size), (x0 + size, y0 + size, z0 + (i + 1) * size),
                "crate" if size <= CRATE else "crate_big", tag="crate")
            for i in range(levels)]


def _barrel(x, y, color):
    r = 0.3
    return (Box((x - r, y - r, 0.0), (x + r, y + r, 0.9), "barrel", visible=False, tag="barrel"),
            Prop("barrel", (x, y, 0.0), {"color": color}))


def build() -> MapData:
    half: list[Box] = []
    props: list[Prop] = []

    # --- West superstructure (team 0 base) -------------------------------------------
    half.append(Box((-46.0, -13.6, 0.0), (-40.0, 13.6, 7.5), "superstructure", tag="house"))
    half.append(Box((-40.0, -9.0, 0.0), (-35.0, 9.0, 1.2), "steel_plate", tag="platform"))
    # centre stairs
    half.append(Box((-35.0, -2.5, 0.0), (-34.0, 2.5, 0.8), "steel_plate", tag="stairs"))
    half.append(Box((-34.0, -2.5, 0.0), (-33.0, 2.5, 0.4), "steel_plate", tag="stairs"))
    # side stairs (both sides of the platform)
    for s in (1, -1):
        y_a, y_b = sorted((s * 9.0, s * 10.0))
        y_c, y_d = sorted((s * 10.0, s * 11.0))
        half.append(Box((-39.0, y_a, 0.0), (-36.0, y_b, 0.8), "steel_plate", tag="stairs"))
        half.append(Box((-39.0, y_c, 0.0), (-36.0, y_d, 0.4), "steel_plate", tag="stairs"))
        # platform parapet beside the centre stairs
        y0, y1 = sorted((s * 2.5, s * 9.0))
        half.append(Box((-35.25, y0, 1.2), (-35.0, y1, 2.15), "parapet", tag="parapet"))
    # cargo hatch in front of the base
    half.append(Box((-31.0, -3.0, 0.0), (-27.0, 3.0, 0.4), "hatch", tag="hatch"))

    # --- Containers ------------------------------------------------------------------
    half += _container(-26.0, 7.5, "x", ["red"])
    half += _container(-15.0, 7.8, "x", ["blue", "orange"], door=-1)
    half += _container(-21.0, -5.5, "x", ["green"])
    half += _container(-9.0, -9.8, "x", ["yellow"])
    half += _container(-6.0, 2.5, "y", ["white"])

    # --- Crates ----------------------------------------------------------------------
    half += _crate(-31.0, 7.1, 1)            # climb route onto the red container
    half += _crate(-30.15, 7.1, 2)
    half += _crate(-24.5, 7.2, 1, z0=CONT_H)  # cover on top of the red container
    half += _crate(-25.75, -6.0, 1)          # climb route onto the green container
    half += _crate(-24.9, -6.0, 2)
    half += _crate(-26.1, 1.9, 1, BIG_CRATE)
    half += _crate(-10.6, -2.6, 1, BIG_CRATE)
    half += _crate(-19.4, 0.8, 1)
    half += _crate(-19.4, 1.65, 2)
    half += _crate(-20.6, 10.9, 1, BIG_CRATE)
    half += _crate(-3.6, 7.0, 1, BIG_CRATE)
    half += _crate(-14.5, -11.5, 1)

    # --- Barrels ---------------------------------------------------------------------
    for x, y, c in ((-33.0, -11.6, "blue"), (-32.3, -12.25, "red"), (-33.65, -12.3, "blue"),
                    (-12.6, 12.2, "red"), (-13.3, 12.3, "blue")):
        b, p = _barrel(x, y, c)
        half.append(b)
        props.append(p)

    # --- Crane pillars ---------------------------------------------------------------
    half.append(Box((-22.4, 12.2, 0.0), (-21.6, 13.0, 11.0), "mast", tag="mast"))
    props.append(Prop("crane_boom", (-22.0, 12.6, 11.0), {"yaw": -65.0}))

    # --- Deck lamps (visual) ---------------------------------------------------------
    for x in (-30.0, -18.0, -6.0):
        props.append(Prop("lamp", (x, 13.05, 1.1), {"yaw": -90.0}))
        props.append(Prop("lamp", (x + 6.0, -13.05, 1.1), {"yaw": 90.0}))

    boxes = list(half) + [_mirror_box(b) for b in half]
    props = list(props) + [_mirror_prop(p) for p in props]

    # --- Self-symmetric centre pieces --------------------------------------------------
    boxes += _container(0.0, 0.0, "y", ["teal"])

    # --- Deck, bulwarks and invisible clip walls --------------------------------------
    boxes.append(Box((-46.0, -13.6, -1.0), (46.0, 13.6, 0.0), "deck", tag="deck"))
    for s in (1, -1):
        y0, y1 = sorted((s * 13.0, s * 13.6))
        boxes.append(Box((-40.0, y0, 0.0), (40.0, y1, 1.1), "bulwark", tag="bulwark"))
        y0, y1 = sorted((s * 13.0, s * 14.5))
        boxes.append(Box((-46.0, y0, 1.1), (46.0, y1, 16.0), "clip",
                         blocks_bullets=False, visible=False, tag="clip"))

    # --- Visual-only dressing ----------------------------------------------------------
    deco: list[Box] = []
    for s in (1, -1):
        y0, y1 = sorted((s * 13.6, s * 14.0))
        deco.append(Box((-47.0, y0, -9.0), (47.0, y1, 1.1), "hull", solid=False, blocks_bullets=False))
    for sx in (1, -1):
        x0, x1 = sorted((sx * 46.0, sx * 47.0))
        deco.append(Box((x0, -13.6, -9.0), (x1, 13.6, 1.1), "hull", solid=False, blocks_bullets=False))
        # upper superstructure tiers + funnel
        x0, x1 = sorted((sx * 46.0, sx * 41.5))
        deco.append(Box((x0, -10.0, 7.5), (x1, 10.0, 10.5), "superstructure", solid=False, blocks_bullets=False))
        x0, x1 = sorted((sx * 46.0, sx * 43.0))
        deco.append(Box((x0, -6.0, 10.5), (x1, 6.0, 12.8), "superstructure", solid=False, blocks_bullets=False))
        x0, x1 = sorted((sx * 45.5, sx * 43.8))
        deco.append(Box((x0, -1.4, 12.8), (x1, 1.4, 16.5), "funnel", solid=False, blocks_bullets=False))
        # bridge wing
        x0, x1 = sorted((sx * 43.0, sx * 41.5))
        deco.append(Box((x0, -13.4, 10.2), (x1, 13.4, 10.5), "steel_plate", solid=False, blocks_bullets=False))
    boxes += deco

    # Antennas / radar on top of the superstructure.
    for sx in (1, -1):
        props.append(Prop("antenna", (sx * 44.5, 3.5, 12.8), {}))
        props.append(Prop("radar", (sx * 44.5, -3.0, 12.8), {}))
        props.append(Prop("door", (sx * 40.0, 0.0, 1.2), {"yaw": 0.0 if sx < 0 else 180.0}))

    spawns = {0: [], 1: []}
    for y in (-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, -6.25, 6.25, -1.25, 1.25):
        spawns[0].append((-37.8 + (0.0 if abs(y) % 2.5 < 1e-6 else 1.4), y, 1.2, 0.0))
        spawns[1].append((37.8 - (0.0 if abs(y) % 2.5 < 1e-6 else 1.4), -y, 1.2, 180.0))

    pts_half = [(-20.0, 11.5), (-20.0, -2.0), (-12.0, -12.0), (-3.0, 5.0), (-28.0, -8.0),
                (-9.0, 0.5), (-14.0, 4.0), (-27.0, 11.5), (-16.0, -9.0), (-3.0, -6.0)]
    strategic = pts_half + [(-x, -y) for x, y in pts_half] + [(0.0, 7.0), (0.0, -7.0)]

    for sx in (1, -1):
        props.append(Prop("ship_bow_lights", (sx * 47.0, 0.0, 1.1), {}))

    return MapData(
        name="transport_ship",
        boxes=boxes,
        props=props,
        spawns=spawns,
        bounds=((-40.0, -13.0), (40.0, 13.0)),
        strategic_points=strategic,
        water_z=-5.0,
        sun_dir=(-0.62, 0.42, -0.42),
    )

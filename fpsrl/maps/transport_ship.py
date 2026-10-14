"""Transport Ship, from the CrossFire remake BSP (workshop 970165514).

The render mesh and brush collision come from tools/import_bsp.py. CT is team 0, T is team 1.

The BSP's own cargo (CS:GO static props) is not packed in the map, so the open main deck and the
hull corridors were bare. On top of the import this module adds:
  * the container / crate cover of the old box-built ship, point-mirrored and fitted onto the main deck;
  * crate steps up the deck-side container stacks and out of the corridor ladder pits (ladders are
    not simulated);
  * cover crates along both hull corridors.
Every added box is set on the highest imported surface under it and dropped if it would cut into
the imported geometry, so a re-import cannot bury or float them.
"""
from __future__ import annotations

import numpy as np

from .base import Box, MapData
from .imported import load

CONT_L, CONT_W, CONT_H = 6.06, 2.44, 2.59
CRATE = 0.85
BIG_CRATE = 1.2

# Main deck measured from collision.npz: deck top z = 2.85 over x -24..22, y -11.9..7.6
# (the cargo hold below it is sealed). The old layout spanned x -40..40, y -13..13 around (0, 0).
DECK_Z = 2.85
DECK_CX, DECK_CY = -1.0, -2.17
X_SCALE, Y_SCALE = 0.72, 0.72

# (x0, y0, levels, size, z hint) crate stacks; z hint = roughly the surface they stand on.
STEPS = [
    # north side: deck -> single container (5.25) -> double container (7.65)
    # (the stacks start at y 7.65; the upper containers at x -20.38 and -2.62)
    (-23.5, 6.45, 1, BIG_CRATE, DECK_Z),
    (-21.7, 8.6, 1, BIG_CRATE, 5.25),
    (-5.6, 6.45, 1, BIG_CRATE, DECK_Z),
    (-3.92, 8.6, 1, BIG_CRATE, 5.25),
    (19.4, 6.45, 1, BIG_CRATE, DECK_Z),
    # south side: single container (5.25) -> double container (7.68, ends at x 1.56)
    (1.68, -14.2, 1, BIG_CRATE, 5.25),
    # corridor ladder pits (floor 0.65, deck 2.85): two-step stacks against the ladder wall
    (9.45, -14.3, 2, CRATE, 0.65),
    (8.6, -14.3, 1, CRATE, 0.65),
    (-11.35, 9.6, 2, CRATE, 0.65),
    (-10.5, 9.6, 1, CRATE, 0.65),
]
# cover in the hull corridors (floor 0.65, 3 m wide): alternate walls, leave ~1.9 m to pass
CORRIDOR = [
    # south corridor, y -14.92 .. -11.92, x -26 .. 10
    (-23.0, -14.85, 1, CRATE), (-18.0, -12.8, 2, CRATE), (-13.0, -14.85, 1, BIG_CRATE),
    (-7.5, -12.8, 1, CRATE), (-2.5, -14.85, 2, CRATE), (3.0, -13.15, 1, BIG_CRATE),
    # north corridor, y 7.58 .. 10.58, x -10 .. 24
    (-6.0, 9.7, 1, CRATE), (-1.0, 7.65, 2, CRATE), (4.5, 9.35, 1, BIG_CRATE),
    (10.0, 7.65, 1, CRATE), (15.0, 9.7, 2, CRATE), (20.0, 7.65, 1, BIG_CRATE),
]
SPAWN_CLEAR = 2.5


def _container(cx, cy, along, colors, z0=0.0, door=1):
    hx, hy = (CONT_L / 2, CONT_W / 2) if along == "x" else (CONT_W / 2, CONT_L / 2)
    out = []
    for level, color in enumerate(colors):
        z = z0 + level * CONT_H
        out.append(Box((cx - hx, cy - hy, z), (cx + hx, cy + hy, z + CONT_H),
                       f"container_{color}", tag="container", params={"along": along, "door": door}))
    return out


def _crate(x0, y0, levels=1, size=CRATE, z0=0.0):
    return [Box((x0, y0, z0 + i * size), (x0 + size, y0 + size, z0 + (i + 1) * size),
                "crate" if size <= CRATE else "crate_big", tag="crate")
            for i in range(levels)]


def _mirror(b: Box) -> Box:
    params = dict(b.params)
    if "door" in params:
        params["door"] = -params["door"]
    return Box((-b.mx[0], -b.mx[1], b.mn[2]), (-b.mn[0], -b.mn[1], b.mx[2]), b.mat,
               b.solid, b.blocks_bullets, b.visible, b.tag, params)


def _legacy_groups() -> list[list[Box]]:
    """Old box-ship cover (deck at z = 0, centred on the origin) as stacks that move together."""
    half = [
        _container(-26.0, 7.5, "x", ["red"]) + _crate(-24.5, 7.2, 1, z0=CONT_H),
        _container(-15.0, 7.8, "x", ["blue", "orange"], door=-1),
        _container(-21.0, -5.5, "x", ["green"]),
        _container(-9.0, -9.8, "x", ["yellow"]),
        _container(-6.0, 2.5, "y", ["white"]),
        _crate(-31.0, 7.1, 1), _crate(-30.15, 7.1, 2),
        _crate(-25.75, -6.0, 1), _crate(-24.9, -6.0, 2),
        _crate(-26.1, 1.9, 1, BIG_CRATE), _crate(-10.6, -2.6, 1, BIG_CRATE),
        _crate(-19.4, 0.8, 1), _crate(-19.4, 1.65, 2),
        _crate(-20.6, 10.9, 1, BIG_CRATE), _crate(-3.6, 7.0, 1, BIG_CRATE),
        _crate(-14.5, -11.5, 1),
    ]
    return half + [[_mirror(b) for b in g] for g in half] + [_container(0.0, 0.0, "y", ["teal"])]


def _shift(group: list[Box], dx: float, dy: float, dz: float) -> list[Box]:
    return [Box((b.mn[0] + dx, b.mn[1] + dy, b.mn[2] + dz), (b.mx[0] + dx, b.mx[1] + dy, b.mx[2] + dz),
                b.mat, b.solid, b.blocks_bullets, b.visible, b.tag, dict(b.params)) for b in group]


def _bounds(group):
    return (np.min([b.mn for b in group], axis=0), np.max([b.mx for b in group], axis=0))


class _Fitter:
    """Sets added stacks on the imported floor and rejects ones that would intersect it."""

    def __init__(self, boxes: list[Box], spawns):
        self.mn = np.array([b.mn for b in boxes], np.float64)
        self.mx = np.array([b.mx for b in boxes], np.float64)
        self.spawns = np.array([s[:2] for team in spawns.values() for s in team], np.float64)
        self.placed: list[tuple[np.ndarray, np.ndarray]] = []
        self.dropped = 0

    def _xy_hits(self, lo, hi, pad=0.0):
        return ((self.mn[:, 0] < hi[0] - pad) & (self.mx[:, 0] > lo[0] + pad)
                & (self.mn[:, 1] < hi[1] - pad) & (self.mx[:, 1] > lo[1] + pad))

    def floor(self, lo, hi, z_hint: float) -> float | None:
        """Highest imported top under the footprint within 0.4 m of z_hint."""
        hit = self._xy_hits(lo, hi, 0.05) & (np.abs(self.mx[:, 2] - z_hint) < 0.4)
        return float(self.mx[hit, 2].max()) if hit.any() else None

    def clear(self, lo, hi) -> bool:
        hit = self._xy_hits(lo, hi, 0.02) & (self.mn[:, 2] < hi[2] - 0.02) & (self.mx[:, 2] > lo[2] + 0.02)
        if hit.any():
            return False
        for a, b in self.placed:
            if np.all(a[:2] < hi[:2] - 0.02) and np.all(b[:2] > lo[:2] + 0.02) and a[2] < hi[2] and b[2] > lo[2]:
                return False
        if len(self.spawns):
            cx, cy = np.clip(self.spawns[:, 0], lo[0], hi[0]), np.clip(self.spawns[:, 1], lo[1], hi[1])
            if np.any(np.hypot(self.spawns[:, 0] - cx, self.spawns[:, 1] - cy) < SPAWN_CLEAR):
                return False
        return True

    def put(self, group: list[Box], z_hint: float, out: list[Box]) -> bool:
        """`group` has its base at z = 0; set it on the floor near z_hint if it fits."""
        lo, hi = _bounds(group)
        z = self.floor(lo, hi, z_hint)
        if z is None:
            self.dropped += 1
            return False
        moved = _shift(group, 0.0, 0.0, z)
        lo, hi = _bounds(moved)
        if not self.clear(lo, hi):
            self.dropped += 1
            return False
        self.placed.append((lo, hi))
        out += moved
        return True


def _deck_cover(fit: _Fitter) -> list[Box]:
    out: list[Box] = []
    containers, crates = [], []
    for g in _legacy_groups():
        lo, hi = _bounds(g)
        c = (lo + hi) / 2
        moved = _shift(g, c[0] * (X_SCALE - 1) + DECK_CX, c[1] * (Y_SCALE - 1) + DECK_CY, 0.0)
        (containers if g[0].tag == "container" else crates).append(moved)
    for g in containers:
        fit.put(g, DECK_Z, out)
    for g in crates:
        # scaling the layout moves a crate stack into the container it leaned on: push it out
        lo, hi = _bounds(g)
        for a, b in fit.placed:
            if lo[2] > 0.1 or not (np.all(a[:2] < hi[:2]) and np.all(b[:2] > lo[:2])):
                continue
            pen = [b[0] - lo[0], hi[0] - a[0], b[1] - lo[1], hi[1] - a[1]]
            k = int(np.argmin(pen))
            d = pen[k] + 0.02
            g = _shift(g, (d, -d, 0, 0)[k], (0, 0, d, -d)[k], 0.0)
            lo, hi = _bounds(g)
        fit.put(g, DECK_Z, out)
    return out


def build() -> MapData:
    boxes, spawns, bounds, strategic, ztop, sea = load("transport_ship")
    fit = _Fitter(boxes, spawns)
    added = _deck_cover(fit)
    for x0, y0, levels, size, z_hint in STEPS:
        fit.put(_crate(x0, y0, levels, size), z_hint, added)
    for x0, y0, levels, size in CORRIDOR:
        fit.put(_crate(x0, y0, levels, size), 0.65, added)
    return MapData(
        name="transport_ship",
        boxes=boxes + added,
        props=[],
        spawns=spawns,
        bounds=bounds,
        strategic_points=strategic,
        water_z=sea if sea is not None else -4.0,
        sun_dir=(-0.62, 0.42, -0.42),
        theme="ship",
        modes=("tdm", "bio"),
        nav_max_floor=ztop,
        mesh="transport_ship",
    )

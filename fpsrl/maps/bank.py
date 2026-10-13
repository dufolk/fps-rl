"""Bank (Rainbow Six Siege style) for the Siege mode.

A 48 x 32 m building in a city block: basement (z -4), first floor (z 0), second floor (z 4) and an
accessible roof (z 8). Main stairs connect 1F-2F (lobby) and B-1F, back stairs in the north-east
run from the basement to the roof hut, and an outdoor fire escape on the east face reaches 2F and
the roof. A garage ramp on the east leads into the basement.

Interior partitions are soft (destructible, reinforceable) drywall; stairwells, the vault and the
outer shell are hard. Every door / window opening is a barricade slot. Several 1.5 m floor hatches
are destructible. Sites come in pairs (site_sets): CEO + Executive Lounge (2F), Open Area + Staff
Room (1F), Tellers + Archives (1F), Lockers + CCTV (B).
"""
from __future__ import annotations

from .base import Box, MapData, Prop
from .gridmap import rect_minus

FX0, FY0, FX1, FY1 = -24.0, -16.0, 24.0, 16.0  # building footprint
T_EXT = 0.4
T_INT = 0.2
ZB, Z1, Z2, ZR = -4.0, 0.0, 4.0, 8.0
SLAB = 0.3
STOREY = 4.0
DOOR_H = 2.2


class Builder:
    def __init__(self):
        self.boxes: list[Box] = []
        self.slots: list[dict] = []
        self.props: list[Prop] = []

    def box(self, mn, mx, mat, soft=False, floor=False, tag="", **kw):
        params = {}
        if soft:
            params["destruct"] = "soft"
        if floor:
            params["destruct"] = "floor"
        self.boxes.append(Box(tuple(map(float, mn)), tuple(map(float, mx)), mat, tag=tag, params=params, **kw))

    def wall(self, x0, y0, x1, y1, z0, z1, mat="drywall", soft=True, openings=(), tag="wall"):
        """Axis-aligned wall slab with openings: (centre along the wall, width, bottom, top, slot?)
        where bottom / top are offsets from z0."""
        along_x = (x1 - x0) >= (y1 - y0)
        a0, a1 = (x0, x1) if along_x else (y0, y1)
        ops = sorted(openings, key=lambda o: o[0])
        cur = a0

        def seg(s0, s1, za, zb):
            if s1 - s0 < 1e-3 or zb - za < 1e-3:
                return
            if along_x:
                self.box((s0, y0, za), (s1, y1, zb), mat, soft=soft, tag=tag)
            else:
                self.box((x0, s0, za), (x1, s1, zb), mat, soft=soft, tag=tag)

        for o in ops:
            c, w, bot, top = o[:4]
            slot = o[4] if len(o) > 4 else True
            s0, s1 = max(a0, c - w / 2), min(a1, c + w / 2)
            seg(cur, s0, z0, z1)
            seg(s0, s1, z0, z0 + bot)
            seg(s0, s1, z0 + top, z1)
            if slot:
                if along_x:
                    mn, mx = (s0, y0 - 0.05, z0 + bot), (s1, y1 + 0.05, z0 + top)
                else:
                    mn, mx = (x0 - 0.05, s0, z0 + bot), (x1 + 0.05, s1, z0 + top)
                self.slots.append({"mn": list(mn), "mx": list(mx)})
            cur = s1
        seg(cur, a1, z0, z1)

    def wx(self, xa, xb, y, z, doors=(), windows=(), soft=True, mat=None, h=STOREY - SLAB, t=T_INT):
        """Interior wall along x at y."""
        ops = [(c, w, 0.0, DOOR_H) for c, w in doors] + [(c, w, 1.0, 2.3) for c, w in windows]
        self.wall(xa, y - t / 2, xb, y + t / 2, z, z + h, mat or ("drywall" if soft else "concrete"), soft, ops)

    def wy(self, ya, yb, x, z, doors=(), windows=(), soft=True, mat=None, h=STOREY - SLAB, t=T_INT):
        ops = [(c, w, 0.0, DOOR_H) for c, w in doors] + [(c, w, 1.0, 2.3) for c, w in windows]
        self.wall(x - t / 2, ya, x + t / 2, yb, z, z + h, mat or ("drywall" if soft else "concrete"), soft, ops)

    def slab(self, z, holes=(), hatches=(), mat="concrete", outer=(FX0 + T_EXT, FY0 + T_EXT, FX1 - T_EXT, FY1 - T_EXT)):
        """Floor slab (top at z) minus holes; hatches are destructible floor patches."""
        for x0, y0, x1, y1 in rect_minus(outer, list(holes) + list(hatches)):
            self.box((x0, y0, z - SLAB), (x1, y1, z), mat, tag="slab")
            if z > ZB:
                self.box((x0, y0, z - SLAB - 0.02), (x1, y1, z - SLAB), "tile_white", tag="ceiling",
                         solid=False, blocks_bullets=False)
        for x0, y0, x1, y1 in hatches:
            self.box((x0, y0, z - SLAB), (x1, y1, z), "wood_floor", floor=True, tag="hatch")

    def straight_stairs(self, x0, x1, y_start, z_start, direction, steps=10, run=0.5, rise=0.4):
        """Straight flight along +y (direction=1) or -y (-1); rising (rise>0) or descending (rise<0)."""
        for k in range(steps):
            ya = y_start + direction * run * k
            yb = ya + direction * run
            top = z_start + rise * (k + 1)
            self.box((x0, min(ya, yb), top - 0.45), (x1, max(ya, yb), top), "concrete", tag="stairs")

    def switchback(self, xa, xb, ys, z0, levels, landing_slabs=False):
        """Two side-by-side half flights per storey: flight A climbs north on the west half, a landing,
        flight B climbs south on the east half to the arrival landing at ys-2..ys."""
        w = (xb - xa) / 2 - 0.05
        half = STOREY / 2
        n = int(round(half / 0.4))
        run = 0.5
        L = n * run
        for lv in range(levels):
            z = z0 + lv * STOREY
            self.straight_stairs(xa, xa + w, ys, z, 1, n, run, 0.4)
            self.box((xa, ys + L, z + half - 0.3), (xb, ys + L + 2.0, z + half), "concrete", tag="landing")
            self.straight_stairs(xb - w, xb, ys + L, z + half, -1, n, run, 0.4)
            if landing_slabs:
                self.box((xa, ys - 2.0, z + STOREY - 0.3), (xb, ys, z + STOREY), "metal", tag="landing")
        return L


def build() -> MapData:
    b = Builder()
    X0, Y0, X1, Y1 = -40.0, -30.0, 40.0, 30.0
    # ------------------------------------------------------------------ outside ground, ramp
    ramp = (24.4, 0.0, 37.0, 5.0)
    for x0, y0, x1, y1 in rect_minus((X0 - 2, Y0 - 2, X1 + 2, Y1 + 2), [(FX0, FY0, FX1, FY1), ramp]):
        b.box((x0, y0, -1.0), (x1, y1, 0.0), "asphalt", tag="ground")
    b.box((FX0 - 3, FY0 - 4, 0.0), (FX1 + 0.4, FY0, 0.12), "concrete", tag="ground")  # front sidewalk
    rx0, ry0, rx1, ry1 = ramp
    nst = 10
    dx = (rx1 - rx0) / nst
    for k in range(nst):
        top = -0.4 * (k + 1)
        xa = rx1 - dx * (k + 1)
        b.box((xa, ry0, -5.0), (xa + dx, ry1, top), "concrete", tag="ramp")
    b.box((rx0, ry0 - 0.3, -5.0), (rx1, ry0, 1.0), "concrete", tag="wall")
    b.box((rx0, ry1, -5.0), (rx1, ry1 + 0.3, 1.0), "concrete", tag="wall")
    # ------------------------------------------------------------------ outer shell
    ext = "brick"
    # basement (solid, garage opening east)
    b.wall(FX0, FY0, FX1, FY0 + T_EXT, -5.0, ZB + STOREY, ext, False, tag="ext")
    b.wall(FX0, FY1 - T_EXT, FX1, FY1, -5.0, ZB + STOREY, ext, False, tag="ext")
    b.wall(FX0, FY0 + T_EXT, FX0 + T_EXT, FY1 - T_EXT, -5.0, ZB + STOREY, ext, False, tag="ext")
    b.wall(FX1 - T_EXT, FY0 + T_EXT, FX1, FY1 - T_EXT, -5.0, ZB + STOREY, ext, False,
           openings=[(2.5, 4.4, 1.0, 3.4, False)], tag="ext")  # garage (bottom offset = basement floor)
    # 1F
    win = lambda c, w=1.6: (c, w, 1.0, 2.4)  # noqa: E731
    door = lambda c, w=1.2: (c, w, 0.0, DOOR_H)  # noqa: E731
    b.wall(FX0, FY0, FX1, FY0 + T_EXT, Z1, Z2, ext, False,
           openings=[door(0.0, 3.0), win(-18), win(-12), win(12), win(18)], tag="ext")
    b.wall(FX0, FY1 - T_EXT, FX1, FY1, Z1, Z2, ext, False,
           openings=[door(-15.0), win(-6), win(6), win(13)], tag="ext")
    b.wall(FX0, FY0 + T_EXT, FX0 + T_EXT, FY1 - T_EXT, Z1, Z2, ext, False,
           openings=[win(-10), win(0), door(4.0), win(11)], tag="ext")
    b.wall(FX1 - T_EXT, FY0 + T_EXT, FX1, FY1 - T_EXT, Z1, Z2, ext, False,
           openings=[win(-10), win(-1)], tag="ext")
    # 2F
    b.wall(FX0, FY0, FX1, FY0 + T_EXT, Z2, ZR, "concrete", False,
           openings=[win(c) for c in (-18, -12, -4, 4, 12, 18)], tag="ext")
    b.wall(FX0, FY1 - T_EXT, FX1, FY1, Z2, ZR, "concrete", False,
           openings=[win(c) for c in (-18, -6, 6, 13)], tag="ext")
    b.wall(FX0, FY0 + T_EXT, FX0 + T_EXT, FY1 - T_EXT, Z2, ZR, "concrete", False,
           openings=[win(-8), win(2), win(12)], tag="ext")
    b.wall(FX1 - T_EXT, FY0 + T_EXT, FX1, FY1 - T_EXT, Z2, ZR, "concrete", False,
           openings=[door(-14.9, 1.2), win(-1)], tag="ext")
    # roof parapet (gap for the fire escape)
    b.wall(FX0, FY0, FX1, FY0 + T_EXT, ZR, ZR + 0.9, "concrete", False, tag="ext")
    b.wall(FX0, FY1 - T_EXT, FX1, FY1, ZR, ZR + 0.9, "concrete", False, tag="ext")
    b.wall(FX0, FY0 + T_EXT, FX0 + T_EXT, FY1 - T_EXT, ZR, ZR + 0.9, "concrete", False, tag="ext")
    b.wall(FX1 - T_EXT, FY0 + T_EXT, FX1, FY1 - T_EXT, ZR, ZR + 0.9, "concrete", False,
           openings=[(-14.9, 1.4, 0.0, 0.9, False)], tag="ext")
    # cornices between storeys (visual banding, slightly proud of the wall)
    for z in (Z2 - 0.15, ZR - 0.15):
        b.box((FX0 - 0.15, FY0 - 0.15, z), (FX1 + 0.15, FY0, z + 0.3), "concrete_dark", tag="trim")
        b.box((FX0 - 0.15, FY1, z), (FX1 + 0.15, FY1 + 0.15, z + 0.3), "concrete_dark", tag="trim")
        b.box((FX0 - 0.15, FY0, z), (FX0, FY1, z + 0.3), "concrete_dark", tag="trim")

    # ------------------------------------------------------------------ slabs
    back_stair = (19.2, 10.0, 23.6, 14.5 + 1.0)
    s1 = (-2.0, 0.0, 2.0, 5.0)     # lobby stairs 1F->2F
    s2 = (-7.0, 0.0, -4.0, 5.0)    # B->1F stairs
    b.slab(ZB, mat="concrete_dark")
    b.slab(Z1, holes=[back_stair, s2], hatches=[(-15.75, -10.75, -14.25, -9.25), (-0.75, -11.75, 0.75, -10.25)],
           mat="tile_floor")
    b.slab(Z2, holes=[back_stair, s1], hatches=[(14.25, -10.75, 15.75, -9.25), (-16.75, 2.25, -15.25, 3.75)],
           mat="carpet")
    b.slab(ZR, holes=[back_stair], hatches=[(14.25, -9.75, 15.75, -8.25), (-16.75, -9.75, -15.25, -8.25)],
           mat="concrete")
    # ------------------------------------------------------------------ stairs
    b.straight_stairs(s1[0], s1[2], 0.0, Z1, 1, 10, 0.5, 0.4)      # up north to 2F
    b.straight_stairs(s2[0], s2[2], 0.0, Z1, 1, 10, 0.5, -0.4)     # down north to B
    b.switchback(19.2, 23.6, 10.0, ZB, 3)                           # back stairs B->roof
    b.switchback(24.0, 28.6, -14.0, Z1, 2, landing_slabs=True)     # outdoor fire escape
    # stair railings / enclosures
    for x in (s2[0] - 0.15, s2[2] + 0.05):
        b.box((x, 0.4, Z1), (x + 0.1, 5.0, Z1 + 1.0), "metal", tag="rail")
    for x in (s1[0] - 0.15, s1[2] + 0.05):
        b.box((x, 0.0, Z2), (x + 0.1, 5.0, Z2 + 1.0), "metal", tag="rail")
    b.box((s1[0] - 0.15, -0.15, Z2), (s1[2] + 0.15, -0.05, Z2 + 1.0), "metal", tag="rail")
    b.wy(0.0, 5.0, s2[0] - 0.2, ZB, soft=False, h=STOREY - SLAB)   # wall off the space under B stairs
    b.wy(0.0, 5.0, s2[2] + 0.2, ZB, soft=False, h=STOREY - SLAB)
    b.wx(s2[0] - 0.3, s2[2] + 0.3, -0.1, ZB, soft=False)
    # back stairwell shell (hard), door on the west side at each level + roof hut
    for z in (ZB, Z1, Z2):
        b.wy(6.0, FY1 - T_EXT, 19.0, z, doors=[(7.0, 1.4)], soft=False)
    b.wy(6.0, FY1 - T_EXT, 19.0, ZR, doors=[(7.0, 1.4)], soft=False, h=3.0)
    b.wx(19.0, FX1 - T_EXT, 6.0, ZR, soft=False, h=3.0)
    b.box((18.8, 5.8, ZR + 3.0), (FX1, FY1, ZR + 3.3), "concrete", tag="roof")
    b.box((19.1, FY1 - T_EXT, ZR), (FX1, FY1, ZR + 3.0), "concrete", tag="wall")
    b.box((FX1 - T_EXT, 6.0, ZR), (FX1, FY1, ZR + 3.0), "concrete", tag="wall")

    # ------------------------------------------------------------------ basement rooms (z -4)
    z = ZB
    b.wy(FY0 + T_EXT, -6.0, -6.0, z, doors=[(-10.0, 1.2)])                       # lockers | CCTV
    b.wx(FX0 + T_EXT, -6.0, -2.0, z, doors=[(-12.0, 1.2), (-19.0, 1.2)])        # lockers north wall
    b.wy(-6.0, -2.0, -6.0, z, doors=[(-4.0, 1.2)])
    b.wx(-6.0, 6.0, -6.0, z, doors=[(0.0, 1.4)])                                # CCTV north wall
    b.wy(FY0 + T_EXT, -2.0, 6.0, z, doors=[(-4.0, 1.6)], soft=False, mat="vault")  # vault west wall
    b.wx(6.0, FX1 - T_EXT, -2.0, z, doors=[(14.0, 1.6)], soft=False, mat="vault")  # vault north wall
    b.wy(2.0, FY1 - T_EXT, -9.0, z, doors=[(10.0, 1.2)])                         # server room
    b.wx(FX0 + T_EXT, -9.0, 2.0, z, doors=[(-16.0, 1.2)])
    b.wy(6.0, FY1 - T_EXT, 6.0, z, doors=[(9.0, 1.4)])                           # garage | corridor
    # ------------------------------------------------------------------ 1F rooms (z 0)
    z = Z1
    b.wy(FY0 + T_EXT, -2.0, -8.0, z, doors=[(-6.0, 1.2)], windows=[(-11.0, 2.0)])  # security | lobby
    b.wx(FX0 + T_EXT, -8.0, -2.0, z, doors=[(-14.0, 1.2)])                      # security | open area
    b.wy(FY0 + T_EXT, -4.0, 8.0, z, doors=[(-4.8, 1.2)], windows=[(-13.0, 2.0), (-10.0, 2.0), (-7.0, 2.0)])
    b.wx(8.0, FX1 - T_EXT, -4.0, z, doors=[(16.0, 1.2)])                        # tellers | archives
    b.wy(-4.0, 6.0, 8.0, z, doors=[(2.0, 1.2)])                                 # archives | lobby
    b.wx(8.0, FX1 - T_EXT, 6.0, z, doors=[(12.0, 1.2)])                         # archives north wall
    b.wy(-2.0, 6.0, -8.0, z, doors=[(3.0, 1.4)], windows=[(-0.5, 2.0)])         # open area | lobby
    b.wx(FX0 + T_EXT, -12.0, 8.0, z, doors=[(-18.0, 1.2)])                      # open area | staff room
    b.wy(8.0, FY1 - T_EXT, -12.0, z, doors=[(12.0, 1.2)])                       # staff room east wall
    b.wx(-8.0, 6.0, 8.0, z, doors=[(-4.0, 1.4), (2.0, 1.4)], soft=True)         # lobby | north offices
    b.wy(8.0, FY1 - T_EXT, 5.0, z, doors=[(11.0, 1.2)])
    # lobby pillars
    for x, y in ((-5.0, -9.0), (5.0, -9.0), (-5.0, -3.0), (5.0, -3.0)):
        b.box((x - 0.35, y - 0.35, Z1), (x + 0.35, y + 0.35, Z2 - SLAB), "marble", tag="pillar")
    # ------------------------------------------------------------------ 2F rooms (z 4)
    z = Z2
    b.wy(FY0 + T_EXT, -4.0, 8.0, z, doors=[(-6.0, 1.2)], windows=[(-11.0, 2.0)])   # CEO west wall
    b.wx(8.0, FX1 - T_EXT, -4.0, z, doors=[(12.0, 1.2)])                        # CEO | lounge
    b.wy(-4.0, 6.0, 8.0, z, doors=[(0.0, 1.2)])
    b.wx(8.0, 19.0, 6.0, z, doors=[(14.0, 1.2)])                                # lounge north
    b.wy(FY0 + T_EXT, -2.0, -8.0, z, doors=[(-8.0, 1.2)])                       # admin office
    b.wx(FX0 + T_EXT, -8.0, -2.0, z, doors=[(-16.0, 1.2)])
    b.wy(-2.0, FY1 - T_EXT, -8.0, z, doors=[(9.0, 1.2)])                        # west offices
    b.wx(FX0 + T_EXT, -8.0, 6.0, z, doors=[(-15.0, 1.2)])
    b.wx(-8.0, 19.0, 8.0, z, doors=[(-4.0, 1.4), (12.0, 1.2)])                  # north meeting rooms
    b.wy(8.0, FY1 - T_EXT, 5.0, z, doors=[(12.0, 1.2)])

    # ------------------------------------------------------------------ furniture / cover
    def crate(x, y, zf, s=1.0, h=None, mat="crate"):
        b.box((x - s / 2, y - s / 2, zf), (x + s / 2, y + s / 2, zf + (h or s)), mat, tag="crate")

    def desk(x, y, zf, yaw=0.0):
        b.box((x - 0.8, y - 0.4, zf), (x + 0.8, y + 0.4, zf + 0.77), "wood_floor", tag="desk", visible=False)
        b.props.append(Prop("desk", (x, y, zf), {"yaw": yaw}))

    # basement: lockers rows, CCTV desks, vault boxes, servers, cars in the garage
    for x in (-21.5, -18.0):
        b.box((x, -14.0, ZB), (x + 0.6, -6.0, ZB + 2.0), "metal_door", tag="lockers")
    desk(-2.0, -12.0, ZB)
    desk(2.0, -12.0, ZB)
    b.box((-5.6, -15.4, ZB), (5.6, -14.8, ZB + 2.2), "metal_dark", tag="monitors")
    for x, y in ((10.0, -12.0), (12.0, -12.0), (16.0, -8.0), (20.0, -13.0), (20.0, -5.0)):
        crate(x, y, ZB, 1.2, mat="vault")
    for y in (5.0, 9.0, 13.0):
        b.box((-21.0, y, ZB), (-12.0, y + 0.8, ZB + 2.0), "metal_dark", tag="servers")
    for x, y, col in ((10.0, 10.0, (0.15, 0.15, 0.18)), (15.0, 4.0, (0.55, 0.55, 0.58))):
        b.box((x - 2.1, y - 0.85, ZB), (x + 2.1, y + 0.85, ZB + 1.4), "metal_dark", visible=False, tag="car")
        b.props.append(Prop("car", (x, y, ZB), {"yaw": 0.0, "color": col}))
    # 1F: teller counter, open area desks, staff tables
    b.box((10.0, -12.0, Z1), (11.0, -6.0, Z1 + 1.1), "marble", tag="counter")
    for x, y in ((-20.0, 0.0), (-16.0, 0.0), (-20.0, 4.0), (-16.0, 4.0), (-12.0, 2.0)):
        desk(x, y, Z1)
    crate(-20.0, 12.0, Z1, 1.4, 0.8, "wood_planks")
    crate(-16.0, 13.0, Z1, 1.4, 0.8, "wood_planks")
    for x, y in ((12.0, 0.0), (16.0, 3.0), (20.0, -2.0)):
        b.box((x - 1.4, y - 0.3, Z1), (x + 1.4, y + 0.3, Z1 + 2.0), "metal_dark", tag="shelf")
    b.box((-3.0, -14.0, Z1), (3.0, -13.2, Z1 + 1.0), "marble", tag="desk")      # reception
    b.box((-21.0, -14.0, Z1), (-17.0, -12.0, Z1 + 0.9), "wood_floor", tag="desk")
    # 2F: CEO desk, lounge sofas, meeting tables
    desk(16.0, -12.0, Z2)
    b.box((12.0, -8.0, Z2), (14.0, -7.2, Z2 + 0.8), "carpet_blue", tag="sofa")
    for x, y in ((12.0, 2.0), (17.0, 2.0)):
        b.box((x - 1.0, y - 0.4, Z2), (x + 1.0, y + 0.4, Z2 + 0.8), "carpet_blue", tag="sofa")
    for x, y in ((-20.0, -10.0), (-14.0, -6.0), (-20.0, 2.0), (-14.0, 12.0)):
        desk(x, y, Z2)
    b.box((-4.0, 11.0, Z2), (2.0, 13.0, Z2 + 0.77), "wood_floor", tag="table")
    # roof: AC units / vents
    for x, y, sx, sy in ((-18.0, 8.0, 3.0, 2.0), (-6.0, -8.0, 2.0, 2.0), (6.0, 6.0, 3.0, 1.5), (10.0, -12.0, 2.0, 2.0)):
        b.box((x, y, ZR), (x + sx, y + sy, ZR + 1.4), "metal", tag="ac")
    # street: parked cars, lamps, trees
    for x, y, yaw, col in ((-30.0, -22.0, 0.0, (0.6, 0.12, 0.1)), (-10.0, -22.5, 0.0, (0.12, 0.14, 0.2)),
                           (18.0, -22.0, 180.0, (0.75, 0.75, 0.72)), (32.0, -14.0, 90.0, (0.15, 0.3, 0.55)),
                           (-32.0, 10.0, 90.0, (0.3, 0.32, 0.3)), (10.0, 23.0, 0.0, (0.55, 0.42, 0.12))):
        sx, sy = (2.1, 0.85) if yaw in (0.0, 180.0) else (0.85, 2.1)
        b.box((x - sx, y - sy, 0.0), (x + sx, y + sy, 1.45), "metal_dark", visible=False, tag="car")
        b.props.append(Prop("car", (x, y, 0.0), {"yaw": yaw, "color": col}))
    for x, y in ((-34.0, -19.0), (-20.0, -19.5), (0.0, -19.5), (20.0, -19.5), (34.0, -19.0), (-34.0, 20.0),
                 (0.0, 20.0), (30.0, 20.0)):
        b.props.append(Prop("lamp", (x, y, 0.0), {"yaw": 90.0 if y < 0 else -90.0}))
    for x, y in ((-28.0, -18.5), (-8.0, -18.5), (8.0, -18.5), (28.0, -18.5), (-28.0, 19.0), (20.0, 19.0)):
        b.props.append(Prop("tree", (x, y, 0.0), {}))
    for x, y in ((-26.0, 22.0), (-22.0, 25.0), (26.0, 24.0), (34.0, 4.0)):
        crate(x, y, 0.0, 1.2)
    # ceiling lights
    for zf in (ZB, Z1, Z2):
        for x in (-16.0, -4.0, 4.0, 14.0):
            for y in (-10.0, 2.0, 11.0):
                b.props.append(Prop("ceiling_light", (x, y, zf + STOREY - SLAB - 0.07), {"yaw": 0.0}))

    # ------------------------------------------------------------------ objectives
    def rect(x0, y0, x1, y1):
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def site(name, r, zf):
        return {"name": name, "poly": rect(*r), "zmin": zf - 0.5, "zmax": zf + 3.0, "z": zf}

    sites = [
        site("CEO", (8.2, -15.6, 23.6, -4.2), Z2),
        site("Lounge", (8.2, -3.8, 19.0, 5.8), Z2),
        site("Open Area", (-23.6, -1.8, -8.2, 7.8), Z1),
        site("Staff", (-23.6, 8.2, -12.2, 15.6), Z1),
        site("Tellers", (8.2, -15.6, 23.6, -4.2), Z1),
        site("Archives", (8.2, -3.8, 23.6, 5.8), Z1),
        site("Lockers", (-23.6, -15.6, -6.2, -2.2), ZB),
        site("CCTV", (-5.8, -15.6, 5.8, -6.2), ZB),
    ]
    site_sets = [[0, 1], [2, 3], [4, 5], [6, 7]]

    def pts(cx, cy, zf, n=5, dx=1.6):
        return [(cx + (i % 3) * dx - dx, cy + (i // 3) * dx, zf, 0.0) for i in range(n)]

    defender_spawns = {
        0: pts(15.0, -1.0, Z2),
        1: pts(-18.0, 5.6, Z1),
        2: pts(15.0, 0.5, Z1),
        3: pts(-11.0, -11.0, ZB),
    }
    attackers = [(-14.0 + i * 3.0, -26.0, 0.0, 90.0) for i in range(5)] + \
                [(33.0, -24.0 + i * 3.0, 0.0, 180.0) for i in range(3)] + \
                [(-20.0, 26.0, 0.0, 270.0), (0.0, 26.0, 0.0, 270.0)]
    defenders = [p for v in defender_spawns.values() for p in v]
    cameras = [
        (0.0, -15.0, Z1 + 3.2, 90.0, -22.0),        # lobby entrance
        (-22.8, -1.0, Z1 + 3.2, 20.0, -25.0),       # open area
        (22.8, -15.0, Z2 + 3.2, 135.0, -25.0),      # CEO
        (-22.8, -15.0, ZB + 3.2, 45.0, -25.0),      # lockers
        (5.0, -2.5, ZB + 3.2, 200.0, -25.0),        # basement corridor
        (-23.0, -15.0, 6.0, 45.0, -20.0),           # exterior south-west corner (on the wall)
    ]
    strategic = [(0.0, -9.0), (15.0, -10.0), (-16.0, 3.0), (-18.0, 12.0), (15.0, 1.0), (-14.0, -9.0),
                 (0.0, -26.0), (32.0, -10.0), (-32.0, 0.0), (0.0, 24.0), (30.0, 2.5), (-14.0, -8.0)]
    return MapData(
        name="bank",
        boxes=b.boxes,
        props=b.props,
        spawns={0: attackers, 1: defenders},
        bounds=((X0, Y0), (X1, Y1)),
        strategic_points=strategic,
        water_z=-20.0,
        sun_dir=(-0.4, 0.55, -0.73),
        theme="bank",
        modes=("r6", "tdm"),
        objectives={"sites": sites, "site_sets": site_sets, "defender_spawns": defender_spawns,
                    "slots": b.slots},
        buy_zones={},
        cameras=cameras,
        nav_max_floor=ZR + 0.2,
        nav_levels=(ZB, Z1, Z2, ZR),
    )

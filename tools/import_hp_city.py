"""Rebuild the domination map from the Unreal export in domin/ as textured boxes.

    py tools/import_hp_city.py [domin]

Reads HP_City_Main.obj (every placed static mesh of the level, grouped per mesh type), voxelizes the
playable block, makes enclosed air solid and greedily merges the solid columns into boxes. Writes
fpsrl/maps/data/hp_city.npz:
  mn, mx, mat     visual boxes (metres, world frame) and their material index into `mats`
  cmn, cmx        collision boxes: the same solid space merged without regard to material
  pmn, pmx, pmat  compact props (crates, barrels, car ...), one box per instance, solid and visible
  trees           (x, y, base z, height) per tree
  reach           (xmin, ymin, xmax, ymax) of the walkable area
  pads            (x, y, z) top centre of each supply crate pedestal (底座)
  zone_<name>     capture polygon outlines from Point_A/B/C (x, y), plus zone_z (floor height)

Unreal centimetres, Y-up OBJ (x = UE X, y = UE Z, z = UE Y) -> world metres:
  x = (X - 5950) / 100 east, y = -(Y - 5525) / 100 north, z = height / 100.
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fpsrl", "maps", "data", "hp_city.npz")

CX, CY = 5950.0, 5525.0
CELL = 0.2                                     # horizontal voxel (m)
X0, X1, Y0, Y1 = -42.6, 43.4, -42.8, 42.8      # voxelized block (world m)
ZEDGES = np.round(np.concatenate([np.arange(-1.0, 5.0, 0.1), np.arange(5.0, 10.0, 0.25),
                                  np.arange(10.0, 48.01, 0.5)]), 3)
FINE_TOP = 5.0                                 # below this the vertical resolution is 0.1 m
CLIMB = 1.1        # highest ledge a player gets onto (jump)
HEAD = 1.2         # crouched height: thinner air gaps are not walkable
STEP_CLEAR = 0.5   # the body column above this height must be 3 cells (0.6 m) wide
HIGH = 5.0         # out of reach: coarser heights, one material per column
SKY = 6.0          # floating boxes whose underside is at least this high are dropped
SPECK = 2          # floating boxes of at most this many cells above SPECK_Z are dropped
SPECK_Z = 3.0

ROOF = "roof_grey"
BUILDINGS = {
    "HP_City_Building_01": "brick", "HP_City_Building_02": "plaster_white", "HP_City_Building_02a": "plaster_yellow",
    "HP_City_Building_03a": "brick", "HP_City_Building_04": "plaster_red", "HP_City_Building_04a": "plaster_dark",
    "HP_City_Building_05": "plaster_white", "HP_City_Building_05a": "plaster_blue",
    "HP_City_Building_06": "plaster_yellow", "HP_City_Building_08": "brick_dark", "HP_City_Building_08a": "brick",
    "HP_City_Building_10": "concrete", "HP_City_Building_10a": "concrete",
}
# painted later = wins a shared voxel; ground first so props and walls stay visible
GROUPS = [
    ("HP_City_Ground005", "asphalt"), ("HP_City_Ground006", "asphalt"), ("HP_City_Ground001", "asphalt"),
    ("HP_City_Ground002", "concrete"), ("HP_City_Ground07", "tile_floor"), ("HP_City_Parterre", "grass"),
    ("HP_City_Stairs001_a", "concrete"), ("HP_City_Stairs001_b", "concrete"), ("HP_City_Stairs001_c", "concrete"),
    ("HP_City_Tower", "concrete_dark"),
    ("HP_City_Wall01", "brick"), ("HP_City_Wall02", "brick"), ("HP_City_Wall03", "concrete"),
    ("HP_City_Wall04", "brick_dark"), ("HP_City_Wall05", "concrete"), ("HP_City_Wall06", "brick"),
    *BUILDINGS.items(),
    ("HP_City_Awning", ROOF), ("HP_City_Shed", ROOF), ("HP_City_Shed_02", ROOF),
    ("HP_City_billboard01", "metal"), ("HP_City_billboard02", "metal"),
    ("HP_City_Pole", "wood_planks"), ("HP_City_WoodBoard001", "wood_planks"),
    ("HP_City_metalFence_01", "metal_dark"), ("HP_City_Door", "wood_door"),
    ("HP_City_Sculpture_001", "stone_dark"), ("HP_City_oil", "metal"),
]
# compact props: one box per instance (overlapping parts merged)
PROPS = [
    ("ST_Prop_Cover", "stone_dark"), ("HP_City_Metalbox", "metal"), ("HP_City_RustyCar", "metal_door"),
    ("HP_City_Clusters_01", "concrete_dark"), ("HP_City_Barrel", "paint_blue"), ("HP_City_FireHydrant", "paint_red"),
    ("HP_City_Telephone", "metal"), ("HP_City_Box03", "wood_planks"), ("HP_City_Woodbox", "wood_planks"),
    ("HP_City_Woodbox01", "wood_planks"), ("HP_City_Woodbox02", "wood_planks"), ("HP_City_Woodbox04", "wood_planks"),
    ("HP_City_Woodbox05", "wood_planks"), ("HP_City_Box_cloth01", "tarp"), ("HP_City_Box_cloth02", "tarp"),
    ("HP_City_Box_cloth03", "tarp"), ("HP_City_Tarp01", "tarp"), ("HP_City_Tarp02", "tarp"),
]
GROUND = {"asphalt", "concrete", "tile_floor", "grass"}


def parse(path):
    V = []
    faces = collections.defaultdict(list)
    cur = None
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("g "):
                cur = line[2:].strip()
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces[cur].append((idx[0], idx[k], idx[k + 1]))
    V = np.array(V, dtype=np.float64)
    W = np.stack([(V[:, 0] - CX) / 100.0, -(V[:, 2] - CY) / 100.0, V[:, 1] / 100.0], axis=1)
    return W, {k: np.array(v, dtype=np.int64) for k, v in faces.items()}


def sample(A, B, C, step):
    """Points on a barycentric grid over each triangle, no two neighbours further than `step` apart."""
    L = np.maximum(np.maximum(np.linalg.norm(B - A, axis=1), np.linalg.norm(C - A, axis=1)),
                   np.linalg.norm(C - B, axis=1))
    k = np.maximum(1, np.ceil(L / step).astype(np.int64))
    for kk in np.unique(k):
        sel = np.nonzero(k == kk)[0]
        i, j = np.meshgrid(np.arange(kk + 1), np.arange(kk + 1), indexing="ij")
        m = i + j <= kk
        u = (i[m] / kk)[None, :, None]
        v = (j[m] / kk)[None, :, None]
        per = max(1, 4_000_000 // m.sum())
        for s in range(0, len(sel), per):
            t = sel[s:s + per]
            a, b, c = A[t][:, None], B[t][:, None], C[t][:, None]
            yield (a + u * (b - a) + v * (c - a)).reshape(-1, 3), t, m.sum()


def voxelize(W, faces, mats):
    nx = int(round((X1 - X0) / CELL))
    ny = int(round((Y1 - Y0) / CELL))
    nz = len(ZEDGES) - 1
    lab = np.zeros((nx, ny, nz), dtype=np.uint8)
    for group, mat in GROUPS:
        if group not in faces:
            print("missing group", group)
            continue
        F = faces[group]
        A, B, C = W[F[:, 0]], W[F[:, 1]], W[F[:, 2]]
        lo = np.minimum(np.minimum(A, B), C)
        hi = np.maximum(np.maximum(A, B), C)
        keep = (hi[:, 0] >= X0) & (lo[:, 0] <= X1) & (hi[:, 1] >= Y0) & (lo[:, 1] <= Y1) & (lo[:, 2] < ZEDGES[-1])
        A, B, C, lo = A[keep], B[keep], C[keep], lo[keep]
        n = np.cross(B - A, C - A)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        ids = np.full(len(A), mats.index(mat), dtype=np.uint8)
        if group in BUILDINGS:
            ids[n[:, 2] > 0.7] = mats.index(ROOF)
        for fine in (True, False):
            part = (lo[:, 2] < FINE_TOP) == fine
            for P, t, per in sample(A[part], B[part], C[part], 0.08 if fine else 0.3):
                ix = np.floor((P[:, 0] - X0) / CELL).astype(np.int64)
                iy = np.floor((P[:, 1] - Y0) / CELL).astype(np.int64)
                iz = np.searchsorted(ZEDGES, P[:, 2], side="right") - 1
                ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
                lab[ix[ok], iy[ok], iz[ok]] = np.repeat(ids[part][t], per)[ok]
        print(f"  {group:28s} {len(A):6d} tris")
    return lab


def cell_range(lo, hi, origin, n):
    """Cells whose centre lies inside [lo, hi]."""
    a = int(np.ceil((lo - origin) / CELL - 0.5))
    b = int(np.floor((hi - origin) / CELL - 0.5)) + 1
    return max(a, 0), min(b, n)


def prop_mask(shape, pmn, pmx):
    m = np.zeros(shape, dtype=bool)
    for a, b in zip(pmn, pmx):
        i0, i1 = cell_range(a[0], b[0], X0, shape[0])
        j0, j1 = cell_range(a[1], b[1], Y0, shape[1])
        zc = (ZEDGES[:-1] + ZEDGES[1:]) / 2
        k = np.nonzero((zc >= a[2]) & (zc <= b[2]))[0]
        if i1 > i0 and j1 > j0 and len(k):
            m[i0:i1, j0:j1, k[0]:k[-1] + 1] = True
    return m


def air_gaps(solid):
    """Per column: air runs (a, b) in voxel layers, each with solid right below it."""
    nx, ny, _ = solid.shape
    gaps = {}
    for i in range(nx):
        for j in range(ny):
            d = np.diff(np.concatenate([[1], solid[i, j].astype(np.int8), [1]]))
            gaps[(i, j)] = list(zip(np.nonzero(d == -1)[0], np.nonzero(d == 1)[0]))
    return gaps


def reachable(solid, seeds):
    """Air gaps a player can walk into from the spawn points (jump-height climbs, any drop)."""
    nx, ny, nz = solid.shape
    wide = ndimage.binary_erosion(~solid, structure=np.ones((3, 3, 1), dtype=bool), border_value=0)
    cw = np.concatenate([np.zeros((nx, ny, 1), dtype=np.int32), np.cumsum(wide, axis=2, dtype=np.int32)], axis=2)
    gaps = air_gaps(solid)
    top = np.append(ZEDGES[1:], 1e9)

    def stand(i, j, a, b):
        f, c = ZEDGES[a], (ZEDGES[b] if b < nz else 1e9)
        if c - f < HEAD:
            return False
        k0 = min(int(np.searchsorted(ZEDGES, f + STEP_CLEAR)), nz)
        k1 = min(int(np.searchsorted(ZEDGES, f + HEAD)), nz)
        return cw[i, j, k1] - cw[i, j, k0] == k1 - k0

    ok = {}
    for key, gs in gaps.items():
        ok[key] = [stand(*key, a, b) for a, b in gs]
    seen = set()
    stack = []
    for x, y, z in seeds:
        i, j = int((x - X0) / CELL), int((y - Y0) / CELL)
        for g, (a, b) in enumerate(gaps[(i, j)]):
            if ZEDGES[a] - 0.6 <= z <= top[b - 1] and ok[(i, j)][g]:
                seen.add((i, j, g))
                stack.append((i, j, g))
    if not stack:
        raise ValueError("no spawn point stands in a walkable gap")
    while stack:
        i, j, g = stack.pop()
        a, b = gaps[(i, j)][g]
        f, c = ZEDGES[a], (ZEDGES[b] if b < nz else 1e9)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if n not in gaps:
                continue
            for h, (a2, b2) in enumerate(gaps[n]):
                if not ok[n][h] or (n[0], n[1], h) in seen:
                    continue
                f2, c2 = ZEDGES[a2], (ZEDGES[b2] if b2 < nz else 1e9)
                if f2 - f > CLIMB or min(c, c2) - max(f, f2) < HEAD:
                    continue
                seen.add((n[0], n[1], h))
                stack.append((n[0], n[1], h))
    return gaps, seen


def solidify(lab, props, seeds):
    """Solid = painted voxels + covered air no player can reach (building shells, crawl spaces, under ground)."""
    solid = lab > 0
    solid[:, :, ZEDGES[1:] <= 0.0] = True
    solid |= props
    gaps, seen = reachable(solid, seeds)
    nz = solid.shape[2]
    filled = 0
    for (i, j), gs in gaps.items():
        for g, (a, b) in enumerate(gs):
            if b < nz and (i, j, g) not in seen:
                solid[i, j, a:b] = True
                filled += b - a
    print(f"walkable gaps {len(seen)}, filled {filled} voxels")
    return solid & ~props, seen


def _snap_up(k, step):
    """Layer edge index rounded up to a multiple of `step` metres (only above HIGH)."""
    z = ZEDGES[k]
    if z <= HIGH:
        return k
    return int(np.searchsorted(ZEDGES, min(np.ceil(z / step - 1e-6) * step, ZEDGES[-1]) - 1e-6))


def runs(solid, lab, mats, step=1.0):
    """-> bottom (top index, material) per column, and floating runs (lo, hi, material).

    Above HIGH nobody walks: heights snap outwards to `step` metres and a column takes its wall
    material (not the roof's), so tall buildings merge into few boxes."""
    nx, ny, nz = solid.shape
    ground = np.array([m in GROUND for m in mats])
    roof = mats.index(ROOF)
    bottom_top = np.zeros((nx, ny), dtype=np.int64)
    bottom_mat = np.zeros((nx, ny), dtype=np.int64)
    upper = collections.defaultdict(list)

    def pick(painted, fallback):
        if not len(painted):
            return fallback
        obj = painted[~ground[painted]]
        wall = obj[obj != roof]
        p = wall if len(wall) else obj if len(obj) else painted
        return int(np.bincount(p).argmax())

    for i in range(nx):
        for j in range(ny):
            s = solid[i, j]
            L = lab[i, j]
            d = np.diff(np.concatenate([[0], s.astype(np.int8), [0]]))
            starts = np.nonzero(d == 1)[0]
            ends = np.nonzero(d == -1)[0]
            below = 0  # top of the previous run: snapped runs must not overlap it (coplanar faces flicker)
            for r, (a, b) in enumerate(zip(starts, ends)):
                seg = L[a:b]
                painted = seg[seg > 0]
                if r == 0:
                    bottom_top[i, j] = below = _snap_up(b, step)
                    if ZEDGES[b] > HIGH:
                        bottom_mat[i, j] = pick(painted, roof)
                    else:
                        obj = painted[~ground[painted]]
                        p = obj if len(obj) else painted
                        bottom_mat[i, j] = int(np.bincount(p).argmax()) if len(p) else mats.index("asphalt")
                else:
                    hi = _snap_up(b, step)
                    if ZEDGES[a] > HIGH:
                        lo = max(np.floor(ZEDGES[a] / step) * step, HIGH)
                        a = int(np.searchsorted(ZEDGES, lo - 1e-6))
                        m = pick(painted, roof)
                    else:
                        m = int(np.bincount(painted).argmax()) if len(painted) else mats.index("concrete")
                    a = max(a, below)
                    if hi <= a:
                        continue
                    below = hi
                    upper[(a, hi, m)].append((i, j))
    return bottom_top, bottom_mat, upper


def coarsen_far(bottom_top, bottom_mat, upper, seen, block=5):
    """Columns further than a block from any walkable cell become `block`-cell solid towers of the
    block's median height."""
    nx, ny = bottom_top.shape
    reach = np.zeros((nx, ny), dtype=bool)
    for i, j, _ in seen:
        reach[i, j] = True
    near = ndimage.binary_dilation(reach, structure=np.ones((2 * block + 1, 2 * block + 1), dtype=bool))
    far = np.zeros((nx, ny), dtype=bool)
    for i in range(0, nx, block):
        for j in range(0, ny, block):
            s = (slice(i, i + block), slice(j, j + block))
            if near[s].any():
                continue
            far[s] = True
            t = np.sort(bottom_top[s].ravel())
            pick = t[len(t) // 2]
            k = int(np.argmin(np.abs(bottom_top[s] - pick).ravel()))
            bottom_mat[s] = bottom_mat[s].ravel()[k]
            bottom_top[s] = pick
    for key in list(upper):
        cells = [c for c in upper[key] if not far[c]]
        if cells:
            upper[key] = cells
        else:
            del upper[key]
    print(f"coarsened {int(far.sum())} of {nx * ny} columns")


def covering(target, allowed=None):
    """Rectangles covering every `target` cell, each lying inside `allowed` (a superset); without
    `allowed` the rectangles are disjoint."""
    todo = target.copy()
    if allowed is None:
        allowed = todo
    nx, ny = target.shape
    out = []
    for i, j in zip(*np.nonzero(target)):
        if not todo[i, j]:
            continue
        j1 = j
        while j1 + 1 < ny and allowed[i, j1 + 1]:
            j1 += 1
        j0 = j
        while j0 - 1 >= 0 and allowed[i, j0 - 1]:
            j0 -= 1
        i1 = i
        while i1 + 1 < nx and allowed[i1 + 1, j0:j1 + 1].all():
            i1 += 1
        todo[i:i1 + 1, j0:j1 + 1] = False
        out.append((i, i1, j0, j1))
    return out


def boxes(bottom_top, bottom_mat, upper, by_mat=True):
    """Merged boxes; with by_mat=False materials are ignored (fewer boxes, for collision).

    Visual boxes (by_mat) never overlap: overlapping boxes share coplanar faces that z-fight.
    Collision boxes may extend under taller neighbours to merge into fewer boxes."""
    if not by_mat:
        bottom_mat = np.zeros_like(bottom_mat)
        merged = collections.defaultdict(list)
        for (a, b, _m), cells in upper.items():
            merged[(a, b, 0)] += cells
        upper = merged
    mn, mx, mt = [], [], []
    for t in np.unique(bottom_top):
        for m in np.unique(bottom_mat[bottom_top == t]):
            target = (bottom_top == t) & (bottom_mat == m)
            for i, i1, j, j1 in covering(target, None if by_mat else bottom_top >= t):
                mn.append((X0 + i * CELL, Y0 + j * CELL, ZEDGES[0]))
                mx.append((X0 + (i1 + 1) * CELL, Y0 + (j1 + 1) * CELL, ZEDGES[t]))
                mt.append(m)
    for (a, b, m), cells in upper.items():
        if ZEDGES[a] >= SKY:  # nothing walkable up here; the solid columns already carry the skyline
            continue
        grid = np.zeros(bottom_top.shape, dtype=bool)
        ii, jj = zip(*cells)
        grid[list(ii), list(jj)] = True
        for i, i1, j, j1 in covering(grid):
            if (i1 - i + 1) * (j1 - j + 1) <= SPECK and ZEDGES[a] > SPECK_Z:
                continue
            mn.append((X0 + i * CELL, Y0 + j * CELL, ZEDGES[a]))
            mx.append((X0 + (i1 + 1) * CELL, Y0 + (j1 + 1) * CELL, ZEDGES[b]))
            mt.append(m)
    return np.array(mn), np.array(mx), np.array(mt)


def components(W, F):
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in F:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    groups = collections.defaultdict(set)
    for f in F:
        for v in f:
            groups[find(v)].add(v)
    return [W[sorted(s)] for s in groups.values()]


def instances(W, F):
    """AABBs of the connected parts of a mesh group, overlapping ones merged into one."""
    bx = [(P.min(0), P.max(0)) for P in components(W, F)]
    bx = [b for b in bx if b[1][0] > X0 and b[0][0] < X1 and b[1][1] > Y0 and b[0][1] < Y1]
    mn = np.array([b[0] for b in bx]).reshape(-1, 3)
    mx = np.array([b[1] for b in bx]).reshape(-1, 3)
    i = 0
    while i < len(mn):
        ov = np.all(mn <= mx[i] + 0.02, axis=1) & np.all(mx >= mn[i] - 0.02, axis=1)
        ov[i] = False
        if ov.any():
            mn[i] = np.minimum(mn[i], mn[ov].min(0))
            mx[i] = np.maximum(mx[i], mx[ov].max(0))
            keep = ~ov
            i = int(keep[:i].sum())
            mn, mx = mn[keep], mx[keep]
            continue
        i += 1
    return mn, mx


def trees(W, faces):
    """(x, y, base z, height) per tree: clusters of Tree01 parts within 3 m of each other."""
    parts = components(W, faces["HP_City_Tree01"])
    out = []
    for P in sorted(parts, key=lambda P: -len(P)):
        c = np.median(P[:, :2], axis=0)
        if not (X0 < c[0] < X1 and Y0 < c[1] < Y1):
            continue
        near = [o for o in out if np.hypot(c[0] - o[0], c[1] - o[1]) < 3.0]
        if near:
            o = near[0]
            o[2] = min(o[2], P[:, 2].min())
            o[3] = max(o[3], P[:, 2].max())
            continue
        out.append([c[0], c[1], P[:, 2].min(), P[:, 2].max()])
    return np.array([[x, y, z0, z1 - z0] for x, y, z0, z1 in out])


def trunks(tr, r=0.2):
    """Collision boxes of the tree trunks (fpsrl/maps/hp_town.py builds the same ones)."""
    mn = np.c_[tr[:, 0] - r, tr[:, 1] - r, tr[:, 2]]
    mx = np.c_[tr[:, 0] + r, tr[:, 1] + r, tr[:, 2] + np.minimum(tr[:, 3] * 0.45, 4.0)]
    return mn, mx


def pads(W, faces):
    out = []
    for P in components(W, faces["ST_Prop_Cover"]):
        c = np.r_[(P[:, :2].min(0) + P[:, :2].max(0)) / 2, P[:, 2].max()]
        if not any(np.hypot(c[0] - o[0], c[1] - o[1]) < 2.0 for o in out):
            out.append(c)
    return np.array(out)


def _snap(vals, tol=0.3):
    """Map each value to the mean of its cluster (values closer than `tol` chain together)."""
    order = np.argsort(vals)
    out = np.empty_like(vals)
    k = 0
    while k < len(order):
        e = k
        while e + 1 < len(order) and vals[order[e + 1]] - vals[order[e]] < tol:
            e += 1
        out[order[k:e + 1]] = vals[order[k:e + 1]].mean()
        k = e + 1
    return out


def zone(W, F):
    """Capture polygon from a Point_X_01 marker: a bevelled ribbon standing on the zone outline."""
    verts = np.unique(F)
    sx = dict(zip(verts, np.round(_snap(W[verts, 0]), 2)))
    sy = dict(zip(verts, np.round(_snap(W[verts, 1]), 2)))
    adj = collections.defaultdict(set)
    for f in F:
        for p, q in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            a, b = (sx[p], sy[p]), (sx[q], sy[q])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    start = min(adj)
    loop, prev = [start], None
    while True:
        cur = loop[-1]
        nxt = sorted(adj[cur] - {prev})[0] if prev is not None else sorted(adj[cur])[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev = cur
    if len(loop) != len(adj):
        raise ValueError(f"zone outline is not a simple loop ({len(loop)} of {len(adj)} corners)")
    poly = np.array(loop)
    if np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
        poly = poly[::-1]
    return poly, float(W[verts, 2].min())


def ue(X, Y, Z):
    return (X - CX) / 100.0, -(Y - CY) / 100.0, Z / 100.0


# spawn points from the level design sheet (UE units, capsule centre ~0.9 m above the feet)
SPAWNS_UE = {
    0: [(8946.535156, 8997.428711, 351.494873), (8681.863281, 8979.089844, 351.494690),
        (8470.192383, 8992.833984, 351.494568), (8317.511719, 9004.556641, 351.494568)],
    1: [(2726.252441, 2622.192871, 105.057068), (3006.368896, 2622.184082, 105.057068),
        (3313.184814, 2622.177246, 105.057068), (3537.385986, 2622.173340, 105.057190)],
}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "domin")
    W, faces = parse(os.path.join(src, "HP_City_Main.obj"))
    print(f"{len(W)} verts, {sum(len(f) for f in faces.values())} tris, {len(faces)} groups")
    mats = ["", *sorted({m for _, m in GROUPS + PROPS} | {ROOF, "wood_planks"})]
    lab = voxelize(W, faces, mats)
    pmn, pmx, pmt = [], [], []
    for group, mat in PROPS:
        a, b = instances(W, faces[group])
        pmn.append(a)
        pmx.append(b)
        pmt += [mats.index(mat)] * len(a)
        print(f"  {group:28s} {len(a):4d} boxes")
    tr = trees(W, faces)
    pmn, pmx = np.concatenate(pmn), np.concatenate(pmx)
    tmn, tmx = trunks(tr)
    seeds = [ue(*p) for sp in SPAWNS_UE.values() for p in sp]
    seeds = [(x, y, z - 0.9) for x, y, z in seeds]
    solid, seen = solidify(lab, prop_mask(lab.shape, np.r_[pmn, tmn], np.r_[pmx, tmx]), seeds)
    print("solid voxels", int(solid.sum()))
    bt, bm, up = runs(solid, lab, mats)
    coarsen_far(bt, bm, up, seen)
    mn, mx, mt = boxes(bt, bm, up)
    cmn, cmx, _ = boxes(bt, bm, up, by_mat=False)
    print(f"boxes: {len(mn)} visual / {len(cmn)} collision + {len(pmn)} props")
    reach = np.array(sorted({(i, j) for i, j, _ in seen}))
    lo = np.array([X0, Y0]) + reach.min(0) * CELL
    hi = np.array([X0, Y0]) + (reach.max(0) + 1) * CELL
    print(f"walkable area x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}")
    out = {"mn": mn.astype(np.float32), "mx": mx.astype(np.float32), "mat": mt.astype(np.int16),
           "cmn": cmn.astype(np.float32), "cmx": cmx.astype(np.float32),
           "pmn": pmn.astype(np.float32), "pmx": pmx.astype(np.float32), "pmat": np.array(pmt, dtype=np.int16),
           "mats": np.array(mats), "trees": tr.astype(np.float32), "reach": np.r_[lo, hi].astype(np.float32),
           "pads": pads(W, faces).astype(np.float32)}
    for team, sp in SPAWNS_UE.items():
        out[f"spawn{team}"] = np.array([ue(*p) for p in sp], dtype=np.float32)
    names, zs = [], []
    for name in "ABC":
        poly, z = zone(W, faces[f"Point_{name}_01"])
        out[f"zone_{name}"] = poly.astype(np.float32)
        names.append(name)
        zs.append(z)
        print(f"zone {name}: {len(poly)} verts, z {z:.2f}")
    out["zone_names"] = np.array(names)
    out["zone_z"] = np.array(zs, dtype=np.float32)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT, f"{len(out['trees'])} trees, {len(out['pads'])} pads")


if __name__ == "__main__":
    main()

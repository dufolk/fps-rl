"""Import the fake-garage scene (医院-假车库_75m.obj) as the `garage` map.

    py tools/import_garage.py [path/to/医院-假车库_75m.obj]

The OBJ is one untextured mesh in centimetres, Y up, its walls about 50 degrees off the axes and
its UVs only tile indices. This script

* converts it to metres, Z up, rotates it so the walls line up with x / y (the sim's collision is
  axis-aligned boxes) and puts the yard at z = 0,
* sorts every triangle into a material (yard, grass, each building's walls / floors / roofs /
  ceilings, vehicles, trees, small props) and gives it world-space UVs, so the procedural textures
  in `fpsrl/render/textures.py` tile at a real-world scale,
* adds a skirt around the terrain disc so its edge meets the far ground plane,
* voxelises everything inside the playable bounds (0.1 m x 0.1 m x 0.05 m) and merges the voxels
  into collision boxes, plus invisible walls around the bounds.

Writes fpsrl/maps/data/garage/mesh.npz and collision.npz.
"""
from __future__ import annotations

import collections
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fpsrl", "maps", "data", "garage")
SRC = os.path.join(ROOT, "医院-假车库_75m.obj")

# world collision bounds (metres) and inner compound bounds where 0.5m terrain resolution is kept
BOUNDS = ((-85.0, -85.0), (85.0, 85.0))
YARD_BOUNDS = ((-36.0, -44.0), (44.0, 40.0))
ZONE_RADIUS = 74.5  # circular edge of the 75m OBJ disc (poison zone outside)
CELL, DZ = 0.1, 0.05
WALL_H = 16.0

# building footprints in the rotated frame: name -> (x0, y0, x1, y1), materials
BUILDINGS = {
    "hospital": ((-3.5, -31.5, 16.8, -11.3), dict(wall="stucco_white", floor="tile_white", roof="roof_flat",
                                                  ceil="drywall", slope="roof_flat")),
    "office": ((-6.5, -5.2, 18.6, 7.2), dict(wall="stucco_tan", floor="concrete", roof="roof_flat",
                                             ceil="drywall", slope="roof_flat")),
    "garage": ((9.4, 11.0, 22.6, 28.2), dict(wall="cinder", floor="garage_floor", roof="roof_metal",
                                             ceil="roof_metal", slope="roof_metal")),
    "house": ((-15.6, 9.4, -6.4, 23.2), dict(wall="brick", floor="wood_floor", roof="roof", ceil="drywall",
                                            slope="roof")),
}
VEHICLE_PAINT = ["car_white", "car_red", "car_blue", "car_green"]
# collision material per render material (bullet impact effects only care about metal / wood / other)
HIT_MAT = {"bark": "wood_planks", "wood_floor": "wood_planks", "roof_metal": "metal", "metal_dark": "metal",
           "garage_door": "metal", **{p: "metal" for p in VEHICLE_PAINT}}


# ----------------------------------------------------------------------------- load
def load_obj(path):
    v, f = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) for p in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # fan any polygon into triangles
                    f.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int64) - 1
    P = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1) / 100.0  # cm, Y up -> m, Z up
    return P, f


def face_normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    return n / np.maximum(ln, 1e-12)[:, None], ln / 2


def align(P, F):
    """Rotate about the scene centre so the dominant wall direction is the x axis; yard at z = 0."""
    n, area = face_normals(P, F)
    wall = np.abs(n[:, 2]) < 0.05
    ang = np.degrees(np.arctan2(n[wall, 1], n[wall, 0])) % 90.0
    hist, edges = np.histogram(ang, bins=360, range=(0, 90), weights=area[wall])
    k = int(np.argmax(hist))
    sel = (ang >= edges[max(k - 4, 0)]) & (ang <= edges[min(k + 5, 360)])
    a = float(np.average(ang[sel], weights=area[wall][sel]))
    th = np.radians(-a)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = (P[:, :2].min(0) + P[:, :2].max(0)) / 2
    Q = P.copy()
    Q[:, :2] = (P[:, :2] - c) @ R.T
    # yard height: the most common height of up-facing flat area
    tri = Q[F]
    up = n[:, 2] > 0.99
    z = tri[up, :, 2].mean(1)
    h, e = np.histogram(z, bins=4000, weights=area[up])
    z0 = float(np.average(z[(z >= e[np.argmax(h)]) & (z <= e[np.argmax(h) + 1])]))
    Q[:, 2] -= z0
    print(f"rotated by {-a:.2f} deg, yard at {z0:.3f} m")
    return Q


def components(P, F):
    key = np.round(P / 0.005).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    WF = weld.ravel()[F]
    nv = int(WF.max()) + 1
    r = np.concatenate([WF[:, 0], WF[:, 1], WF[:, 2]])
    c = np.concatenate([WF[:, 1], WF[:, 2], WF[:, 0]])
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(nv, nv)), directed=False)
    return lab[WF[:, 0]], WF


# ----------------------------------------------------------------------------- materials
def covered_above(P, F, idx, up=True, eps=0.05):
    """For triangles `idx`: is there any geometry straight above (or below) their centroid?"""
    tri = P[F]
    A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]
    lo = np.minimum(np.minimum(A, B), C)
    hi = np.maximum(np.maximum(A, B), C)
    out = np.zeros(len(idx), dtype=bool)
    cen = tri[idx].mean(1)
    for s in range(0, len(idx), 256):
        q = cen[s:s + 256]
        cand = ((lo[None, :, 0] <= q[:, None, 0]) & (hi[None, :, 0] >= q[:, None, 0])
                & (lo[None, :, 1] <= q[:, None, 1]) & (hi[None, :, 1] >= q[:, None, 1]))
        cand &= (hi[None, :, 2] > q[:, None, 2] + eps) if up else (lo[None, :, 2] < q[:, None, 2] - eps)
        for i, j in zip(*np.nonzero(cand)):
            a, b, c = A[j, :2], B[j, :2], C[j, :2]
            p = q[i, :2]
            d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(d) < 1e-12:
                continue
            l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d
            l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d
            if l1 < -1e-6 or l2 < -1e-6 or l1 + l2 > 1 + 1e-6:
                continue
            z = l1 * A[j, 2] + l2 * B[j, 2] + (1 - l1 - l2) * C[j, 2]
            if (z > q[i, 2] + eps) if up else (z < q[i, 2] - eps):
                out[s + i] = True
        # (python loop only over candidate pairs; few thousand faces)
    return out


def classify(P, F, comp):
    n, area = face_normals(P, F)
    tri = P[F]
    cen = tri.mean(1)
    mats = np.empty(len(F), dtype=object)
    sizes = {}
    for k in np.unique(comp):
        p = tri[comp == k].reshape(-1, 3)
        sizes[k] = (p.min(0), p.max(0), int((comp == k).sum()))
    terrain = int(np.argmax(np.bincount(comp)))
    # --- terrain: yard where flat at z ~ 0, grass on the slopes, gravel where they meet
    ter = comp == terrain
    flat_v = np.abs(tri[:, :, 2]) < 0.02
    nflat = flat_v.sum(1)
    mats[ter & (nflat == 3)] = "yard"
    mats[ter & (nflat == 0)] = "grass"
    mats[ter & (nflat > 0) & (nflat < 3)] = "gravel"
    # --- buildings
    in_bld = np.zeros(len(F), dtype=object)
    for name, ((x0, y0, x1, y1), _) in BUILDINGS.items():
        m = (~ter) & (cen[:, 0] >= x0) & (cen[:, 0] <= x1) & (cen[:, 1] >= y0) & (cen[:, 1] <= y1)
        in_bld[m] = name
    bld = np.array([b != 0 for b in in_bld])
    nz = n[:, 2]
    up_flat = bld & (nz > 0.95)
    dn_flat = bld & (nz < -0.95)
    covered_up = np.zeros(len(F), dtype=bool)
    i_up = np.nonzero(up_flat)[0]
    covered_up[i_up] = covered_above(P, F, i_up, up=True)
    for name, (_, m) in BUILDINGS.items():
        sel = in_bld == name
        mats[sel & (np.abs(nz) <= 0.3)] = m["wall"]
        mats[sel & (nz > 0.3) & (nz <= 0.95)] = m["slope"]
        mats[sel & (nz < -0.3) & (nz >= -0.95)] = m["ceil"]
        mats[sel & up_flat & covered_up] = m["floor"]
        mats[sel & up_flat & ~covered_up] = m["roof"]
        mats[sel & dn_flat] = m["ceil"]
    # the garage's roller doors: walls of the garage facing west / east with a door-sized opening
    # are plain walls in the mesh; leave them as cinder block.
    # --- props outside the buildings
    paint = 0
    for k, (lo, hi, nt) in sorted(sizes.items(), key=lambda kv: -kv[1][2]):
        if k == terrain:
            continue
        m = (comp == k) & ~bld
        if not m.any():
            continue
        ext = hi - lo
        if ext[2] > 8.0 and max(ext[0], ext[1]) < 14.0:
            mats[m] = "bark"
        elif 1.0 < ext[2] < 4.5 and 3.0 < max(ext[0], ext[1]) < 10.0 and nt > 80:
            mats[m] = VEHICLE_PAINT[paint % len(VEHICLE_PAINT)]
            paint += 1
        else:
            mats[m] = "metal_dark" if max(ext) < 3.0 else "concrete"
    mats[mats == None] = "concrete"  # noqa: E711
    return mats.astype(str), terrain


UV_SCALE = {"yard": 0.25, "grass": 0.12, "gravel": 0.3, "stucco_white": 0.5, "stucco_tan": 0.5,
            "tile_white": 0.5, "drywall": 0.4, "roof_flat": 0.25, "concrete": 0.25, "cinder": 0.5,
            "garage_floor": 0.25, "roof_metal": 0.5, "brick": 0.5, "wood_floor": 0.5, "roof": 0.5, "bark": 0.6,
            "metal_dark": 0.5}


def triplanar_uv(tri, n, scale):
    """Per-corner UVs from world position, projected on the plane the face is most aligned with."""
    ax = np.argmax(np.abs(n), axis=1)
    u = np.where(ax[:, None] == 2, tri[:, :, 0], np.where(ax[:, None] == 0, tri[:, :, 1], tri[:, :, 0]))
    v = np.where(ax[:, None] == 2, tri[:, :, 1], tri[:, :, 2])
    flip = ((ax == 0) & (n[:, 0] < 0)) | ((ax == 1) & (n[:, 1] > 0))  # read left-to-right on every wall
    u = np.where(flip[:, None], -u, u)
    return np.stack([u, v], axis=2) * scale


def skirt(P, F, terrain_mask, z_far, reach=90.0):
    """Quads from the terrain disc's open edge out and down to the far ground plane."""
    tf = F[terrain_mask]
    key = np.round(P / 0.005).astype(np.int64)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    W = weld.ravel()[tf]
    e = np.concatenate([W[:, [0, 1]], W[:, [1, 2]], W[:, [2, 0]]])
    orig = np.concatenate([tf[:, [0, 1]], tf[:, [1, 2]], tf[:, [2, 0]]])
    s = np.sort(e, axis=1)
    _, inv, cnt = np.unique(s, axis=0, return_inverse=True, return_counts=True)
    border = orig[cnt[inv.ravel()] == 1]
    a, b = P[border[:, 0]], P[border[:, 1]]

    def out(p):
        d = p[:, :2] / np.maximum(np.linalg.norm(p[:, :2], axis=1, keepdims=True), 1e-9)
        q = p.copy()
        q[:, :2] += d * reach
        q[:, 2] = z_far
        return q

    a2, b2 = out(a), out(b)
    # orientation: the terrain is wound counter-clockwise seen from above, so (b, a, a2) faces up
    tris = np.concatenate([np.stack([b, a, a2], 1), np.stack([b, a2, b2], 1)])
    n, _ = face_normals(tris.reshape(-1, 3), np.arange(len(tris) * 3).reshape(-1, 3))
    flip = n[:, 2] < 0
    tris[flip] = tris[flip][:, ::-1]
    return tris


# ----------------------------------------------------------------------------- collision
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
        per = max(1, 3_000_000 // int(m.sum()))
        for s in range(0, len(sel), per):
            t = sel[s:s + per]
            a, b, c = A[t][:, None], B[t][:, None], C[t][:, None]
            yield (a + u * (b - a) + v * (c - a)).reshape(-1, 3), t, int(m.sum())


def covering(target, allowed=None):
    """Rectangles covering every `target` cell, each inside `allowed` (a superset of target)."""
    todo = target.copy()
    if allowed is None:
        allowed = target
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


def terrain_height(P, F, nx, ny, x0, y0, cell=CELL):
    """Highest terrain surface at every cell centre."""
    top = np.full((nx, ny), -np.inf)
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    for t in P[F]:
        a, b, c = t
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:
            continue
        i0 = max(0, int((t[:, 0].min() - x0) / cell) - 1)
        i1 = min(nx - 1, int((t[:, 0].max() - x0) / cell) + 1)
        j0 = max(0, int((t[:, 1].min() - y0) / cell) - 1)
        j1 = min(ny - 1, int((t[:, 1].max() - y0) / cell) + 1)
        if i1 < i0 or j1 < j0:
            continue
        gx, gy = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        z = l1 * a[2] + l2 * b[2] + l3 * c[2]
        sub = top[i0:i1 + 1, j0:j1 + 1]
        np.copyto(sub, np.maximum(sub, np.where(inside, z, -np.inf)))
    return top


ZB = -4.5  # z of voxel layer 0 (shared lattice for every object grid; below lowest OBJ terrain at -3.83 m)
TCELL = 0.5  # terrain collision cell
TSTEP = 0.15  # terrain collision height step on the slopes (well under the 0.45 m step height)
FILL_Z = 12  # seal slabs: a down face with an up face at most this many voxels above is solid between
TRUNK_H = 3.0  # trees only collide up to this height (trunk and the lowest branches)
GAP_FILL = 0.9  # vertical gaps smaller than this inside a building are filled


def solid_grid(tri, nrm, lo_c, hi_c, z_cap):
    """Dense voxels of one object: surfaces, slabs sealed along z, wall cavities closed in x / y.
    `lo_c` / `hi_c` are integer lattice bounds (ix, iy, iz). Returns (solid, exact top z per voxel)."""
    from scipy import ndimage
    pad = 3
    lo_c = np.asarray(lo_c) - pad
    shape = tuple(int(v) for v in np.asarray(hi_c) + pad - lo_c)
    S = np.zeros(shape, dtype=bool)
    UP = np.zeros(shape, dtype=bool)
    DN = np.zeros(shape, dtype=bool)
    TOP = np.full(shape, -np.inf, dtype=np.float32)
    (x0, y0), _ = BOUNDS
    for pts, t, per in sample(tri[:, 0], tri[:, 1], tri[:, 2], 0.035):
        keep = pts[:, 2] <= z_cap
        ix = np.floor((pts[:, 0] - x0) / CELL).astype(np.int64) - lo_c[0]
        iy = np.floor((pts[:, 1] - y0) / CELL).astype(np.int64) - lo_c[1]
        iz = np.floor((pts[:, 2] - ZB) / DZ).astype(np.int64) - lo_c[2]
        ok = keep & (ix >= 0) & (ix < shape[0]) & (iy >= 0) & (iy < shape[1]) & (iz >= 0) & (iz < shape[2])
        ix, iy, iz = ix[ok], iy[ok], iz[ok]
        S[ix, iy, iz] = True
        fz = np.repeat(nrm[t, 2], per)[ok]
        UP[ix[fz > 0.3], iy[fz > 0.3], iz[fz > 0.3]] = True
        DN[ix[fz < -0.3], iy[fz < -0.3], iz[fz < -0.3]] = True
        flat = fz > 0.9995  # level floors / stair treads keep their exact height
        np.maximum.at(TOP, (ix[flat], iy[flat], iz[flat]), pts[ok][flat, 2].astype(np.float32))
    nz = shape[2]
    for g in range(1, min(FILL_Z, nz - 1) + 1):
        cand = DN[:, :, :nz - g] & UP[:, :, g:]
        if cand.any():
            for m in range(g + 1):
                S[:, :, m:nz - g + m] |= cand
    S = ndimage.binary_closing(S, structure=np.ones((5, 5, 1), dtype=bool))
    # gaps no player fits through (crouched a player is 1.2 m) are filled along z: roof trusses,
    # cornices and stacked trim become one slab instead of dozens of thin boxes
    L = int(round(GAP_FILL / DZ))
    Sp = np.pad(S, ((0, 0), (0, 0), (L, L)))
    S = ndimage.binary_closing(Sp, structure=np.ones((1, 1, L), dtype=bool))[:, :, L:-L]
    lab, n = ndimage.label(S)
    if n:
        size = np.bincount(lab.ravel())
        S &= size[lab] >= 12  # drop specks (a few voxels of trim) that would each be a box
    return S, TOP, lo_c


def grid_boxes(S, TOP, lo_c, q, mat):
    """Merge a voxel grid into boxes: z runs per column, columns with equal runs into rectangles.
    Voxels at or below the terrain surface (q, on the CELL lattice of BOUNDS) are dropped."""
    (x0, y0), _ = BOUNDS
    nx, ny, nz = S.shape
    zs = ZB + (lo_c[2] + np.arange(nz)) * DZ  # voxel bottoms
    gi = np.clip(lo_c[0] + np.arange(nx), 0, q.shape[0] - 1)
    gj = np.clip(lo_c[1] + np.arange(ny), 0, q.shape[1] - 1)
    qq = q[gi][:, gj]
    S = S & (zs[None, None, :] + DZ > qq[:, :, None] + 1e-6)
    d = np.diff(np.concatenate([np.zeros((nx, ny, 1), bool), S, np.zeros((nx, ny, 1), bool)], axis=2).astype(np.int8),
                axis=2)
    runs = collections.defaultdict(list)
    for i, j, k in zip(*np.nonzero(d == 1)):
        k1 = k
        while k1 + 1 < nz and S[i, j, k1 + 1]:
            k1 += 1
        zlo = max(float(zs[k]), float(qq[i, j]))
        ztp = float(zs[k1] + DZ)
        if np.isfinite(TOP[i, j, k1]):
            ztp = round(float(TOP[i, j, k1]), 2)
        if ztp - zlo < 0.02:  # a floor lying on a voxel boundary: keep the slab one voxel thick
            zlo = ztp - DZ
        runs[(round(zlo, 3), round(ztp, 3))].append((i, j))
    mn, mx = [], []
    for (zlo, ztp), cells in runs.items():
        grid = np.zeros((nx, ny), dtype=bool)
        ii, jj = np.array(cells).T
        grid[ii, jj] = True
        for i, i1, j, j1 in covering(grid):
            mn.append((x0 + (lo_c[0] + i) * CELL, y0 + (lo_c[1] + j) * CELL, zlo))
            mx.append((x0 + (lo_c[0] + i1 + 1) * CELL, y0 + (lo_c[1] + j1 + 1) * CELL, ztp))
    return mn, mx, [mat] * len(mn)


PCELL = 0.2  # prop height-field cell


def prop_boxes(tri, z_cap, q, mat):
    """A free-standing prop (vehicle, tree trunk, bin, pole) as a height field: solid from the ground
    up to its highest surface in each cell, 0.1 m height steps, merged like the terrain."""
    (x0, y0), _ = BOUNDS
    pts = np.concatenate([p for p, _, _ in sample(tri[:, 0], tri[:, 1], tri[:, 2], PCELL * 0.4)])
    pts = pts[pts[:, 2] <= z_cap]
    if not len(pts):
        return [], [], []
    k = int(round(PCELL / CELL))
    ix = np.floor((pts[:, 0] - x0) / PCELL).astype(np.int64)
    iy = np.floor((pts[:, 1] - y0) / PCELL).astype(np.int64)
    i0, j0 = ix.min(), iy.min()
    H = np.full((ix.max() - i0 + 1, iy.max() - j0 + 1), -np.inf)
    np.maximum.at(H, (ix - i0, iy - j0), pts[:, 2])
    base = np.full(H.shape, np.inf)
    np.minimum.at(base, (ix - i0, iy - j0), pts[:, 2])
    gi = np.clip((i0 + np.arange(H.shape[0])) * k, 0, q.shape[0] - 1)
    gj = np.clip((j0 + np.arange(H.shape[1])) * k, 0, q.shape[1] - 1)
    ground = q[gi][:, gj]
    H = np.where(np.isfinite(H), np.ceil(H / 0.1) * 0.1, -np.inf)
    H[H <= ground + 0.05] = -np.inf
    zlo = float(np.min(np.where(np.isfinite(base), ground, np.inf)))
    mn, mx = [], []
    for t in np.unique(H[np.isfinite(H)]):
        for i, i1, j, j1 in covering(H == t, H >= t):
            mn.append((x0 + (i0 + i) * PCELL, y0 + (j0 + j) * PCELL, zlo))
            mx.append((x0 + (i0 + i1 + 1) * PCELL, y0 + (j0 + j1 + 1) * PCELL, float(t)))
    return mn, mx, [mat] * len(mn)


def objects(P, F, mats, comp, terrain):
    """Collision groups: each building footprint, and every other component inside the bounds."""
    tri = P[F]
    cen = tri.mean(1)
    (bx0, by0), (bx1, by1) = BOUNDS
    taken = comp == terrain
    groups = []
    for name, ((x0, y0, x1, y1), _) in BUILDINGS.items():
        m = (~taken) & (cen[:, 0] >= x0) & (cen[:, 0] <= x1) & (cen[:, 1] >= y0) & (cen[:, 1] <= y1)
        # whole components that mostly sit in the footprint come along (eaves, steps, pipes)
        ks = np.unique(comp[m])
        m = np.isin(comp, ks) & ~taken
        groups.append((name, m, "concrete", WALL_H))
        taken = taken | m
    for k in np.unique(comp[~taken]):
        m = (comp == k) & ~taken
        p = tri[m].reshape(-1, 3)
        lo, hi = p.min(0), p.max(0)
        if hi[0] < bx0 or lo[0] > bx1 or hi[1] < by0 or lo[1] > by1:
            continue
        ext = hi - lo
        if np.prod(np.maximum(ext, 0.05)) < 0.02:
            continue  # bits of trim, nothing to bump into
        mat = mats[m][0]
        hit = HIT_MAT.get(mat, "concrete")
        groups.append((f"comp{k}:{mat}", m, hit, TRUNK_H if mat == "bark" else WALL_H))
    return groups


def collision(P, F, mats, comp, terrain, sk):
    (x0, y0), (x1, y1) = BOUNDS
    nx, ny = int(round((x1 - x0) / CELL)), int(round((y1 - y0) / CELL))
    terrain_mask = comp == terrain
    t0 = time.time()
    P_ter = np.concatenate([P, sk.reshape(-1, 3)], axis=0)
    F_ter = np.concatenate([F[terrain_mask], np.arange(len(P), len(P) + len(sk) * 3).reshape(-1, 3)], axis=0)
    ht = terrain_height(P_ter, F_ter, nx, ny, x0, y0)
    assert np.isfinite(ht).all(), "terrain does not cover the bounds"
    zb = float(np.floor(ht.min()) - 1.0)  # floor of every terrain box
    # terrain collision on a coarser lattice: 0.5 m / 0.15 m in the compound, 1.0 m / 0.30 m on outer hills
    k = int(round(TCELL / CELL))
    tn = (nx // k, ny // k)
    coarse = ht[:tn[0] * k, :tn[1] * k].reshape(tn[0], k, tn[1], k).mean(axis=(1, 3))
    (yx0, yy0), (yx1, yy1) = YARD_BOUNDS
    xs = x0 + (np.arange(tn[0]) + 0.5) * TCELL
    ys = y0 + (np.arange(tn[1]) + 0.5) * TCELL
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    in_yard = (gx >= yx0) & (gx <= yx1) & (gy >= yy0) & (gy <= yy1)
    c2 = coarse.reshape(tn[0] // 2, 2, tn[1] // 2, 2).mean(axis=(1, 3))
    outer_q = np.repeat(np.repeat(np.round(c2 / 0.30) * 0.30, 2, 0), 2, 1)
    tq = np.where(np.abs(coarse) < 0.02, 0.0, np.where(in_yard, np.round(coarse / TSTEP) * TSTEP, outer_q))
    q = np.repeat(np.repeat(tq, k, 0), k, 1)
    q = np.pad(q, ((0, nx - q.shape[0]), (0, ny - q.shape[1])), mode="edge")
    print(f"  terrain raster {time.time() - t0:.1f}s, z {ht.min():.2f}..{ht.max():.2f}")
    hit_names = sorted({HIT_MAT.get(m, "concrete") for m in mats} | {"concrete"})
    n, _ = face_normals(P, F)
    mn, mx, mt = [], [], []
    t0 = time.time()
    per_group = []
    for name, m, hit, z_cap in objects(P, F, mats, comp, terrain):
        tri, nrm = P[F[m]], n[m]
        lo = tri.reshape(-1, 3).min(0)
        hi = tri.reshape(-1, 3).max(0)
        lo_c = (max(0, int(np.floor((lo[0] - x0) / CELL))), max(0, int(np.floor((lo[1] - y0) / CELL))),
                max(0, int(np.floor((max(lo[2], ht.min() - 0.1) - ZB) / DZ))))
        hi_c = (min(nx, int(np.ceil((hi[0] - x0) / CELL)) + 1), min(ny, int(np.ceil((hi[1] - y0) / CELL)) + 1),
                int(np.ceil((min(hi[2], z_cap) - ZB) / DZ)) + 1)
        if hi_c[0] <= lo_c[0] or hi_c[1] <= lo_c[1] or hi_c[2] <= lo_c[2]:
            continue
        if name in BUILDINGS:
            S, TOP, lo_p = solid_grid(tri, nrm, lo_c, hi_c, z_cap)
            a, b, c = grid_boxes(S, TOP, lo_p, q, hit_names.index(hit))
        else:
            a, b, c = prop_boxes(tri, z_cap, q, hit_names.index(hit))
        mn += a
        mx += b
        mt += c
        per_group.append((len(a), name))
    nb = len(mn)
    per_group.sort(reverse=True)
    print(f"  {nb} object boxes in {time.time() - t0:.1f}s; most: " +
          ", ".join(f"{nm} {c}" for c, nm in per_group[:8]))
    ground = hit_names.index("concrete")
    for t in np.unique(tq):
        for i, i1, j, j1 in covering(tq == t, tq >= t):
            mn.append((x0 + i * TCELL, y0 + j * TCELL, zb))
            mx.append((x0 + (i1 + 1) * TCELL, y0 + (j1 + 1) * TCELL, float(t)))
            mt.append(ground)
    print(f"  {len(mn) - nb} terrain boxes")
    for a, b in (((x0 - 1, y0 - 1), (x0, y1 + 1)), ((x1, y0 - 1), (x1 + 1, y1 + 1)),
                 ((x0, y0 - 1), (x1, y0)), ((x0, y1), (x1, y1 + 1))):
        mn.append((a[0], a[1], zb))
        mx.append((b[0], b[1], WALL_H))
        mt.append(ground)
    return (np.array(mn, dtype=np.float32), np.array(mx, dtype=np.float32), np.array(mt, dtype=np.uint8),
            np.array(hit_names), np.float32(zb))


# ----------------------------------------------------------------------------- main
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC
    t0 = time.time()
    P, F = load_obj(src)
    print(f"{len(P)} vertices, {len(F)} triangles ({time.time() - t0:.1f}s)")
    P = align(P, F)
    comp, _ = components(P, F)
    mats, terrain = classify(P, F, comp)
    terrain_mask = comp == terrain
    for m, c in collections.Counter(mats).most_common():
        print(f"  {m:14s} {c:6d}")
    os.makedirs(OUT, exist_ok=True)
    # render mesh: one flat-shaded vertex triple per triangle, grouped by material
    n, _ = face_normals(P, F)
    tri = P[F]
    z_far = float(np.floor(tri[terrain_mask, :, 2].min()) - 1.5)
    sk = skirt(P, F, terrain_mask, z_far)
    skn, _ = face_normals(sk.reshape(-1, 3), np.arange(len(sk) * 3).reshape(-1, 3))
    out = {}
    names = sorted(set(mats))
    for k, m in enumerate(names):
        sel = mats == m
        t, nn = tri[sel], n[sel]
        if m == "grass":
            t, nn = np.concatenate([t, sk]), np.concatenate([nn, skn])
        uv = triplanar_uv(t, nn, UV_SCALE.get(m, 0.5))
        out[f"pos{k}"] = t.reshape(-1, 3).astype(np.float32)
        out[f"uv{k}"] = uv.reshape(-1, 2).astype(np.float32)
        out[f"tri{k}"] = np.arange(len(t) * 3, dtype=np.uint32).reshape(-1, 3)
    np.savez_compressed(os.path.join(OUT, "mesh.npz"), materials=np.array(names), ground_z=np.float32(z_far),
                        **out)
    t0 = time.time()
    mn, mx, mt, hit_names, zb = collision(P, F, mats, comp, terrain, sk)
    np.savez_compressed(os.path.join(OUT, "collision.npz"), mn=mn, mx=mx, mat=mt, mats=hit_names, zb=zb)
    print(f"{len(mn)} collision boxes ({time.time() - t0:.1f}s); far ground at {z_far} m")
    with open(os.path.join(OUT, "SOURCE.txt"), "w", encoding="utf-8") as f:
        f.write("Source: 医院-假车库_75m.obj (untextured scene mesh, cm, Y up).\n"
                "Imported with tools/import_garage.py: metres, Z up, walls aligned to the axes, yard at z = 0,\n"
                "procedural materials with world-space UVs, collision voxelised into boxes.\n")


if __name__ == "__main__":
    main()

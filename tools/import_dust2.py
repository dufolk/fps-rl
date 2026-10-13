"""Import de_dust2 (vrchris, Sketchfab, CC-BY 4.0) as the desert2 map.

    py tools/import_dust2.py [de_dust2-cs-map.zip]

Writes fpsrl/maps/data/dust2/:
  mesh.npz        render mesh in metres, grouped by material (positions, uvs, triangle indices)
  collision.npz   AABBs (mn, mx) for the simulation, built from vertical ray casts through the mesh
  *.png           the original textures

Collision: the map is sampled on a CELL-unit grid. Along each column the mesh faces are walked
bottom to top; an upward face ends a solid run (floor top) and a downward face starts one
(ceiling underside). Columns the mesh never covers are solid (inside brushes / outside the map).
Equal runs in neighbouring columns are merged into rectangles.
"""
from __future__ import annotations

import io
import os
import re
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "fpsrl", "maps", "data", "dust2")

SCALE = 0.025      # metres per Hammer unit: 72-unit player hull = 1.8 m, 18-unit step = 0.45 m
CELL = 8.0         # collision sampling grid (units)
ZQ = 4.0           # collision heights are snapped to this (units); ramps become 0.1 m steps
MIN_GAP = 4.0      # air gaps thinner than this are filled
MIN_SOLID = 1.0    # solid runs thinner than this are dropped
CLIMB = 64.0       # step + crouch-jump onto a large crate
HULL = 36.0        # crouched hull: thinner air gaps are not walkable
STEEP = 0.7        # floor normals with a smaller z component cannot be stood on


def load(zip_path):
    outer = zipfile.ZipFile(zip_path)
    inner = zipfile.ZipFile(io.BytesIO(outer.read("source/de_dust2.zip")))
    obj = inner.read("de_dust2.obj").decode("latin-1")
    mtl = inner.read("de_dust2.mtl").decode("latin-1")
    textures = {os.path.basename(n): outer.read(n) for n in outer.namelist()
                if n.startswith("textures/") and n.endswith(".png")}
    return obj, mtl, textures


def parse_obj(text):
    v, vt, polys = [], [], []
    mat = None
    for line in text.splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("usemtl "):
            mat = line.split()[1]
        elif line.startswith("f "):
            idx = [q.split("/") for q in line.split()[1:]]
            polys.append(([int(q[0]) - 1 for q in idx], [int(q[1]) - 1 if len(q) > 1 and q[1] else -1 for q in idx],
                          mat))
    return np.array(v, dtype=np.float64), np.array(vt, dtype=np.float64), polys


def parse_mtl(text):
    names, tex = {}, {}
    comment = None
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            comment = line[2:]
        elif line.startswith("newmtl "):
            cur = line.split()[1]
            names[cur] = comment or cur
        elif line.startswith("map_Kd ") and cur:
            tex[cur] = os.path.splitext(line.split()[1])[0] + ".png"
    return names, tex


def newell(p):
    n = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


# ----------------------------------------------------------------------------- collision
def column_hits(V, polys, x0, y0, nx, ny):
    """-> per-cell list of (z, +1 floor / -1 ceiling)."""
    hits = [[] for _ in range(nx * ny)]
    for vi, _ti, mat in polys:
        p = V[vi]
        n = newell(p)
        if abs(n[2]) < 0.05 or 0 < n[2] < STEEP:
            continue  # walls; slopes steeper than 45 degrees act as walls too (as in GoldSrc)
        sign = 1 if n[2] > 0 else -1
        i0 = max(0, int(np.floor((p[:, 0].min() - x0) / CELL - 0.5)))
        i1 = min(nx - 1, int(np.ceil((p[:, 0].max() - x0) / CELL - 0.5)))
        j0 = max(0, int(np.floor((p[:, 1].min() - y0) / CELL - 0.5)))
        j1 = min(ny - 1, int(np.ceil((p[:, 1].max() - y0) / CELL - 0.5)))
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        cx = x0 + (ii + 0.5) * CELL
        cy = y0 + (jj + 0.5) * CELL
        inside = np.ones(cx.shape, dtype=bool)
        k = len(p)
        for e in range(k):
            a, b = p[e], p[(e + 1) % k]
            cross = (b[0] - a[0]) * (cy - a[1]) - (b[1] - a[1]) * (cx - a[0])
            inside &= cross * sign >= -1e-6
        if not inside.any():
            continue
        # plane: n . (q - p0) = 0  ->  z = p0z - (nx (x - p0x) + ny (y - p0y)) / nz
        z = p[0, 2] - (n[0] * (cx - p[0, 0]) + n[1] * (cy - p[0, 1])) / n[2]
        for i, j, zz in zip(ii[inside], jj[inside], z[inside]):
            hits[i * ny + j].append((float(zz), sign))
    return hits


def runs_of(h, zbase, ztop):
    if not h:
        return ((zbase, ztop),)
    h.sort(key=lambda t: (t[0], -t[1]))  # at equal z the floor (+1) closes the run first
    runs = []
    solid, start = True, zbase
    for z, s in h:
        if s > 0 and solid:
            runs.append([start, z])
            solid = False
        elif s < 0 and not solid:
            start = z
            solid = True
    if solid:
        runs.append([start, ztop])
    merged = []
    for lo, hi in runs:
        lo = round(lo / ZQ) * ZQ
        hi = round(hi / ZQ) * ZQ
        if hi - lo < MIN_SOLID:
            continue
        if merged and lo - merged[-1][1] < MIN_GAP:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return tuple((lo, hi) for lo, hi in merged)


def rectangles(cells):
    """Greedy cover of a set of (i, j) cells with axis-aligned rectangles."""
    cells = set(cells)
    out = []
    for i, j in sorted(cells):
        if (i, j) not in cells:
            continue
        j1 = j
        while (i, j1 + 1) in cells:
            j1 += 1
        i1 = i
        while all((i1 + 1, jj) in cells for jj in range(j, j1 + 1)):
            i1 += 1
        for a in range(i, i1 + 1):
            for b in range(j, j1 + 1):
                cells.discard((a, b))
        out.append((i, i1, j, j1))
    return out


def covering(target, allowed):
    """Rectangles covering every `target` cell, each lying inside `allowed` (a superset)."""
    todo = target.copy()
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


def gaps_of(runs, ztop):
    """Air gaps (floor, ceiling) between consecutive solid runs."""
    out = []
    for k, (_lo, hi) in enumerate(runs):
        ceil = runs[k + 1][0] if k + 1 < len(runs) else ztop
        if ceil > hi:
            out.append((hi, ceil))
    return out


def reachable_gaps(cols, nx, ny, ztop):
    """Air gaps a player can stand in, flooded from the largest walkable region."""
    gaps = [gaps_of(r, ztop) if r[0][1] < ztop else [] for r in cols]

    def links(c, g, drops):
        i, j = divmod(c, ny)
        f, top = gaps[c][g]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if not (0 <= a < nx and 0 <= b < ny):
                continue
            n = a * ny + b
            for h, (f2, t2) in enumerate(gaps[n]):
                if t2 - f2 < HULL:
                    continue
                if f2 - f > CLIMB or (not drops and f - f2 > CLIMB):
                    continue
                if min(top, t2) - max(f, f2) < HULL:
                    continue
                yield n, h

    seen = set()
    best = []
    for c in range(nx * ny):
        for g, (f, t) in enumerate(gaps[c]):
            if (c, g) in seen or t - f < HULL or t >= ztop - 1 and f >= ztop - 1:
                continue
            comp = [(c, g)]
            seen.add((c, g))
            k = 0
            while k < len(comp):
                for nb in links(*comp[k], False):
                    if nb not in seen:
                        seen.add(nb)
                        comp.append(nb)
                k += 1
            if len(comp) > len(best):
                best = comp
    reach = set(best)
    stack = list(best)
    while stack:
        for nb in links(*stack.pop(), True):
            if nb not in reach:
                reach.add(nb)
                stack.append(nb)
    return gaps, reach


def fill_unreachable(cols, gaps, reach, ny, zbase, ztop):
    out = []
    for c, runs in enumerate(cols):
        keep = [(f, t) for g, (f, t) in enumerate(gaps[c]) if (c, g) in reach]
        if not keep:
            out.append(((zbase, ztop),))
            continue
        solid, lo = [], zbase
        for f, t in keep:
            solid.append((lo, f))
            lo = t
        if lo < ztop:
            solid.append((lo, ztop))
        out.append(tuple(solid))
    return out


def collision(V, polys):
    x0 = np.floor(V[:, 0].min() / CELL) * CELL - CELL
    y0 = np.floor(V[:, 1].min() / CELL) * CELL - CELL
    nx = int(np.ceil((V[:, 0].max() - x0) / CELL)) + 1
    ny = int(np.ceil((V[:, 1].max() - y0) / CELL)) + 1
    zbase, ztop = V[:, 2].min() - 32.0, V[:, 2].max() + 256.0
    hits = column_hits(V, polys, x0, y0, nx, ny)
    cols = [runs_of(hits[c], zbase, ztop) for c in range(nx * ny)]
    gaps, reach = reachable_gaps(cols, nx, ny, ztop)
    cols = fill_unreachable(cols, gaps, reach, ny, zbase, ztop)
    print(f"walkable gaps: {len(reach)}")
    # The first run of every column starts at zbase, so a box (zbase, t) may extend under any
    # column whose first run reaches t or higher without changing the solid space.
    floor_top = np.zeros((nx, ny))
    upper: dict[tuple, list] = {}
    for i in range(nx):
        for j in range(ny):
            runs = cols[i * ny + j]
            floor_top[i, j] = runs[0][1]
            for run in runs[1:]:
                upper.setdefault(run, []).append((i, j))
    mn, mx = [], []
    for t in np.unique(floor_top):
        for i, i1, j, j1 in covering(floor_top == t, floor_top >= t):
            mn.append((x0 + i * CELL, y0 + j * CELL, zbase))
            mx.append((x0 + (i1 + 1) * CELL, y0 + (j1 + 1) * CELL, t))
    for (lo, hi), cells in upper.items():
        for i, i1, j, j1 in rectangles(cells):
            mn.append((x0 + i * CELL, y0 + j * CELL, lo))
            mx.append((x0 + (i1 + 1) * CELL, y0 + (j1 + 1) * CELL, hi))
    return np.array(mn), np.array(mx)


# ----------------------------------------------------------------------------- render mesh
def mesh_groups(V, VT, polys):
    groups: dict[str, dict] = {}
    for vi, ti, mat in polys:
        g = groups.setdefault(mat, {"pos": [], "uv": [], "tri": []})
        base = len(g["pos"])
        for a, t in zip(vi, ti):
            g["pos"].append(V[a])
            g["uv"].append(VT[t] if t >= 0 else (0.0, 0.0))
        for k in range(1, len(vi) - 1):
            g["tri"].append((base, base + k, base + k + 1))
    return groups


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "de_dust2-cs-map.zip")
    obj, mtl, textures = load(zip_path)
    V, VT, polys = parse_obj(obj)
    names, tex = parse_mtl(mtl)
    origin = np.array([(V[:, 0].min() + V[:, 0].max()) / 2, (V[:, 1].min() + V[:, 1].max()) / 2, V[:, 2].min()])
    print(f"{len(V)} verts, {len(polys)} faces, origin {origin.tolist()}")

    mn, mx = collision(V, polys)
    print(f"collision: {len(mn)} boxes")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "collision.npz"),
                        mn=((mn - origin) * SCALE).astype(np.float32), mx=((mx - origin) * SCALE).astype(np.float32))

    groups = mesh_groups(V, VT, polys)
    arrays = {}
    mats = []
    for k, (mat, g) in enumerate(sorted(groups.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0))):
        mats.append((mat, names.get(mat, mat), tex.get(mat, "")))
        arrays[f"pos{k}"] = ((np.array(g["pos"]) - origin) * SCALE).astype(np.float32)
        arrays[f"uv{k}"] = np.array(g["uv"], dtype=np.float32)
        arrays[f"tri{k}"] = np.array(g["tri"], dtype=np.int32)
    arrays["materials"] = np.array([m[0] for m in mats])
    arrays["names"] = np.array([m[1] for m in mats])
    arrays["textures"] = np.array([m[2] for m in mats])
    np.savez_compressed(os.path.join(OUT, "mesh.npz"), **arrays)
    for fn, data in textures.items():
        with open(os.path.join(OUT, fn), "wb") as f:
            f.write(data)
    with open(os.path.join(OUT, "CREDITS.txt"), "w", encoding="utf-8") as f:
        f.write("de_dust2 - CS map by pancakesbassoondonut (@vrchris)\n"
                "https://sketchfab.com/3d-models/de-dust2-cs-map-056008d59eb849a29c0ab6884c0c3d87\n"
                "Licensed under CC Attribution 4.0 (http://creativecommons.org/licenses/by/4.0/).\n"
                "Imported with tools/import_dust2.py: scaled to metres, collision rebuilt as boxes.\n")
    print("wrote", OUT, f"origin_units={origin.tolist()}")


if __name__ == "__main__":
    main()

"""Convert the HP Town point data (centimetres, editor X/Z) into fpsrl/maps/data/hp_town.json.

    py tools/import_town.py <points.(npy|csv|json)> [--zones zones.json] [--cell 0.4] [--min-pts 3]
                            [--height 3.2] [--out fpsrl/maps/data/hp_town.json]

Points: an (N, 2) / (N, 3) array. With 3 columns the X and Z columns are columns 0 and 2 (Y-up
editor convention); CSV files with a header use the columns named X and Z.
Zones (optional): {"A": [[X, Z], ...], "B": ..., "C": ...} in the same centimetre units; defaults
to the polygons traced in fpsrl/maps/hp_town.py.

Pipeline: rasterize onto a `cell` grid inside the activity area -> drop cells with fewer than
`min-pts` points -> close 1-cell gaps -> remove tiny blobs -> mark dense round blobs as trees ->
greedily merge occupied cells into wall boxes.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fpsrl.maps.hp_town import CX, CZ, ZONES_CM, to_world  # noqa: E402

AREA = (1950.0, 9950.0, 1450.0, 9600.0)  # X min/max, Z min/max (cm)


def load_points(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".json":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        arr = np.asarray(d["points"] if isinstance(d, dict) else d, dtype=np.float64)
    else:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        head = [h.strip().lower() for h in rows[0]]
        if "x" in head and "z" in head:
            ix, iz = head.index("x"), head.index("z")
            arr = np.array([[float(r[ix]), float(r[iz])] for r in rows[1:] if r], dtype=np.float64)
        else:
            arr = np.array([[float(v) for v in r] for r in rows if r], dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[1] >= 3:
        arr = arr[:, [0, 2]]
    return arr[:, :2]


def rasterize(pts, cell_cm, min_pts):
    x0, x1, z0, z1 = AREA
    W = int(np.ceil((x1 - x0) / cell_cm))
    H = int(np.ceil((z1 - z0) / cell_cm))
    m = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= z0) & (pts[:, 1] < z1)
    p = pts[m]
    c = ((p[:, 0] - x0) / cell_cm).astype(int)
    r = ((p[:, 1] - z0) / cell_cm).astype(int)
    counts = np.zeros((H, W), dtype=np.int32)
    np.add.at(counts, (r, c), 1)
    return counts, counts >= min_pts


def close_gaps(occ):
    """Fill single empty cells sandwiched between occupied cells (dotted wall lines)."""
    o = occ.copy()
    h = occ[:, :-2] & occ[:, 2:]
    v = occ[:-2, :] & occ[2:, :]
    o[:, 1:-1] |= h
    o[1:-1, :] |= v
    return o


def components(occ):
    H, W = occ.shape
    lab = -np.ones((H, W), dtype=np.int32)
    comps = []
    for r in range(H):
        for c in range(W):
            if not occ[r, c] or lab[r, c] >= 0:
                continue
            stack = [(r, c)]
            lab[r, c] = len(comps)
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = rr + dr, cc + dc
                    if 0 <= a < H and 0 <= b < W and occ[a, b] and lab[a, b] < 0:
                        lab[a, b] = len(comps)
                        stack.append((a, b))
            comps.append(cells)
    return comps


def merge_rects(occ):
    H, W = occ.shape
    used = np.zeros_like(occ)
    out = []
    for r in range(H):
        for c in range(W):
            if not occ[r, c] or used[r, c]:
                continue
            c1 = c
            while c1 + 1 < W and occ[r, c1 + 1] and not used[r, c1 + 1]:
                c1 += 1
            r1 = r
            while r1 + 1 < H and occ[r1 + 1, c:c1 + 1].all() and not used[r1 + 1, c:c1 + 1].any():
                r1 += 1
            used[r:r1 + 1, c:c1 + 1] = True
            out.append((r, r1, c, c1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--zones")
    ap.add_argument("--cell", type=float, default=0.4, help="grid cell in metres")
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--min-blob", type=int, default=3, help="drop blobs with fewer cells")
    ap.add_argument("--height", type=float, default=3.2)
    ap.add_argument("--out", default=os.path.join("fpsrl", "maps", "data", "hp_town.json"))
    a = ap.parse_args()

    pts = load_points(a.points)
    cell_cm = a.cell * 100.0
    counts, occ = rasterize(pts, cell_cm, a.min_pts)
    occ = close_gaps(occ)
    x0, _, z0, _ = AREA
    boxes, props = [], []
    keep = np.zeros_like(occ)
    for cells in components(occ):
        if len(cells) < a.min_blob:
            continue
        rs = np.array([c[0] for c in cells])
        cs = np.array([c[1] for c in cells])
        h, w = int(np.ptp(rs)) + 1, int(np.ptp(cs)) + 1
        density = counts[rs, cs].mean()
        fill = len(cells) / float(h * w)
        # compact, roughly round and very dense blobs are trees / fountains
        if 4 <= h <= 14 and 4 <= w <= 14 and abs(h - w) <= 3 and fill > 0.55 and density > 3 * a.min_pts:
            X = x0 + (cs.mean() + 0.5) * cell_cm
            Z = z0 + (rs.mean() + 0.5) * cell_cm
            x, y = to_world(X, Z)
            props.append({"kind": "tree", "pos": [round(x, 2), round(y, 2), 0.0]})
            boxes.append({"mn": [round(x - 0.35, 3), round(y - 0.35, 3), 0.0],
                          "mx": [round(x + 0.35, 3), round(y + 0.35, 3), 3.0], "mat": "wood_planks", "tag": "trunk"})
            continue
        keep[rs, cs] = True
    mats = ("brick", "plaster_white", "plaster_yellow", "concrete")
    for k, (r0, r1, c0, c1) in enumerate(merge_rects(keep)):
        Xa, Xb = x0 + c0 * cell_cm, x0 + (c1 + 1) * cell_cm
        Za, Zb = z0 + r0 * cell_cm, z0 + (r1 + 1) * cell_cm
        xa, ya = to_world(Xa, Zb)
        xb, yb = to_world(Xb, Za)
        boxes.append({"mn": [round(xa, 3), round(ya, 3), 0.0], "mx": [round(xb, 3), round(yb, 3), a.height],
                      "mat": mats[(r0 // 25 + c0 // 25) % len(mats)], "tag": "wall"})
    zones_cm = ZONES_CM
    if a.zones:
        with open(a.zones, encoding="utf-8") as fh:
            zones_cm = json.load(fh)
    points = [{"name": k, "poly": [list(to_world(X, Z)) for X, Z in v], "zmin": -0.5, "zmax": 4.0, "z": 0.0}
              for k, v in zones_cm.items()]
    out = {"source": os.path.basename(a.points), "units": "m", "origin_cm": [CX, CZ], "cell": a.cell,
           "boxes": boxes, "props": props, "points": points}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print(f"{len(pts)} points -> {len(boxes)} boxes, {len(props)} trees -> {a.out}")


if __name__ == "__main__":
    main()

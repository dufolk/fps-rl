"""Vectorised ray / AABB / sphere queries used by the simulation."""
from __future__ import annotations

import numpy as np


def raycast_boxes(origins, dirs, max_t, bmin, bmax):
    """Nearest hit of R rays against N axis-aligned boxes.

    Returns (t[R], box_index[R], normal[R,3]); t is inf and index -1 on a miss.
    `dirs` should be unit vectors for t to be a distance.
    """
    o = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(dirs, dtype=np.float64).reshape(-1, 3)
    r = o.shape[0]
    max_t = np.broadcast_to(np.asarray(max_t, dtype=np.float64), (r,))
    if bmin.shape[0] == 0 or r == 0:
        return np.full(r, np.inf), np.full(r, -1, dtype=np.int64), np.zeros((r, 3))
    safe = np.where(np.abs(d) < 1e-12, 1e-12, d)
    inv = 1.0 / safe
    t1 = (bmin[None, :, :] - o[:, None, :]) * inv[:, None, :]
    t2 = (bmax[None, :, :] - o[:, None, :]) * inv[:, None, :]
    tmin3 = np.minimum(t1, t2)
    tnear = tmin3.max(axis=2)
    tfar = np.maximum(t1, t2).min(axis=2)
    hit = (tnear <= tfar) & (tfar >= 0.0) & (tnear <= max_t[:, None])
    tn = np.where(hit, np.maximum(tnear, 0.0), np.inf)
    rows = np.arange(r)
    idx = tn.argmin(axis=1)
    t = tn[rows, idx]
    miss = ~np.isfinite(t)
    axis = tmin3[rows, idx].argmax(axis=1)
    normal = np.zeros((r, 3))
    normal[rows, axis] = -np.sign(safe[rows, axis])
    normal[miss] = 0.0
    idx = np.where(miss, -1, idx)
    return t, idx, normal


def aabb_overlap_mask(mn, mx, bmin, bmax):
    """Boolean mask of boxes strictly overlapping the box [mn, mx]."""
    return ((mn[0] < bmax[:, 0]) & (mx[0] > bmin[:, 0])
            & (mn[1] < bmax[:, 1]) & (mx[1] > bmin[:, 1])
            & (mn[2] < bmax[:, 2]) & (mx[2] > bmin[:, 2]))


def segments_hit_spheres(p, q, centers, radii):
    """For R segments p->q and S spheres, return bool[R] whether any sphere is crossed."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
    if len(centers) == 0:
        return np.zeros(p.shape[0], dtype=bool)
    c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    rad = np.asarray(radii, dtype=np.float64).reshape(-1)
    d = q - p
    dd = np.maximum((d * d).sum(axis=1), 1e-12)
    t = ((c[None, :, :] - p[:, None, :]) * d[:, None, :]).sum(axis=2) / dd[:, None]
    t = np.clip(t, 0.0, 1.0)
    closest = p[:, None, :] + t[:, :, None] * d[:, None, :]
    dist2 = ((closest - c[None, :, :]) ** 2).sum(axis=2)
    return (dist2 < (rad[None, :] ** 2)).any(axis=1)


def point_in_poly(x: float, y: float, poly) -> bool:
    """Even-odd test of point (x, y) against polygon [[x, y], ...]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def poly_center(poly):
    a = np.asarray(poly, dtype=np.float64)
    return a.mean(axis=0)

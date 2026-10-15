"""Layered navigation graph for the scripted bots.

Every 2D cell of a grid holds one node per standable surface with head room, so storeys, stairs
and crates stack in the same cell. One-way drop edges lead off ledges. The map geometry is static,
so the graph is built once per map and shared by every world.
"""
from __future__ import annotations

import heapq

import numpy as np

from . import config as C

HEAD_LO, HEAD_HI = 0.46, 1.7  # body column above a floor that must be free (above step height)
DIRS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))


class NavGrid:
    """Static per-map graph. `NavGrid.for_world(world)` returns the cached instance of the map.

    `map.nav_max_floor` caps which box tops count as floors.
    """

    _cache: dict = {}
    MAX_LAYERS = 6
    DROP_MAX = 4.5

    def __init__(self, world, cell: float = 0.5, max_floor: float | None = None,
                 clearance: float = C.PLAYER_RADIUS + 0.02):
        (x0, y0), (x1, y1) = world.map.bounds
        if max_floor is None:
            max_floor = world.map.nav_max_floor
        self.cell = cell
        self.x0, self.y0 = x0, y0
        self.xs = np.arange(x0 + cell / 2, x1, cell)
        self.ys = np.arange(y0 + cell / 2, y1, cell)
        self.W, self.H = len(self.xs), len(self.ys)
        static = [b for b in world.static_boxes if b.solid]
        self._bmin = np.array([b.mn for b in static], dtype=np.float64).reshape(-1, 3)
        self._bmax = np.array([b.mx for b in static], dtype=np.float64).reshape(-1, 3)
        bmin, bmax = self._bmin, self._bmax
        r = clearance
        nodes, ncell = [], []
        self._near: dict[int, np.ndarray] = {}
        for row, y in enumerate(self.ys):
            s = row * self.W
            Q = np.stack([self.xs, np.full(self.W, y)], axis=1)
            sel = np.nonzero((y > bmin[:, 1] - r) & (y < bmax[:, 1] + r))[0]
            smin, smax = bmin[sel], bmax[sel]
            inside = ((Q[:, None, 0] >= smin[None, :, 0]) & (Q[:, None, 0] <= smax[None, :, 0])
                      & (y >= smin[None, :, 1]) & (y <= smax[None, :, 1]))
            near = (Q[:, None, 0] > smin[None, :, 0] - r) & (Q[:, None, 0] < smax[None, :, 0] + r)
            for k in range(len(Q)):
                ins = sel[np.nonzero(inside[k])[0]]
                if not len(ins):
                    continue
                tops = np.unique(bmax[ins, 2])
                tops = tops[tops <= max_floor]
                if not len(tops):
                    continue
                nb = sel[np.nonzero(near[k])[0]]
                self._near[s + k] = nb
                lo, hi = bmin[nb, 2], bmax[nb, 2]
                ilo, ihi = bmin[ins, 2], bmax[ins, 2]
                tops = tops[::-1]
                T = tops[:, None]
                ok = ~(((lo < T + HEAD_HI) & (hi > T + HEAD_LO)).any(axis=1)
                       | ((ilo < T - 1e-6) & (ihi > T + 1e-6)).any(axis=1))
                found = []
                for t in tops[ok]:
                    if found and found[-1] - t <= C.STEP_HEIGHT:
                        continue
                    nodes.append((Q[k, 0], Q[k, 1], t))
                    ncell.append(s + k)
                    found.append(t)
                    if len(found) >= self.MAX_LAYERS:
                        break
        self.nodes = np.array(nodes, dtype=np.float64).reshape(-1, 3)
        self.node_cell = np.array(ncell, dtype=np.int64)
        self._build_edges()
        # keep only what can be reached from some spawn point
        reach = np.zeros(len(self.nodes), dtype=bool)
        for sp in world.map.spawns.values():
            for x, y, z, _ in sp:
                n = self._nearest(np.array([x, y, z]))
                if n >= 0 and not reach[n]:
                    reach |= self._flood(n)
        keep = np.nonzero(reach)[0]
        self.nodes = self.nodes[keep]
        self.node_cell = self.node_cell[keep]
        self._build_edges()
        del self._near, self._bmin, self._bmax

    @classmethod
    def for_world(cls, world) -> "NavGrid":
        key = world.cfg.map_name
        if key not in cls._cache:
            cls._cache[key] = cls(world)
        return cls._cache[key]

    # ------------------------------------------------------------------ build
    def _build_edges(self):
        H, W = self.H, self.W
        cn: dict[int, list[int]] = {}
        for i, c in enumerate(self.node_cell.tolist()):
            cn.setdefault(c, []).append(i)
        self.cell_nodes = cn
        zl = self.nodes[:, 2].tolist()
        step = C.STEP_HEIGHT
        bmin, bmax = self._bmin, self._bmax
        adj: list[list[tuple]] = [[] for _ in range(len(self.nodes))]

        def match(cell, zi):
            for j in cn.get(cell, ()):
                if abs(zl[j] - zi) <= step:
                    return j
            return -1

        for i, cell in enumerate(self.node_cell.tolist()):
            r, c = divmod(cell, W)
            zi = zl[i]
            lst = adj[i]
            for dc, dr, cost in DIRS:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                j = match(rr * W + cc, zi)
                if j >= 0:
                    if dr and dc:
                        s1, s2 = match(r * W + cc, zi), match(rr * W + c, zi)
                        if s1 < 0 or s2 < 0:
                            continue
                    lst.append((j, cost + 0.5 * abs(zl[j] - zi)))
                if dr and dc:
                    continue
                # drop (one or two cells out, past the ledge clearance) to the highest lower floor
                # whose column is free of boxes
                for dist in (1, 2):
                    rr, cc = r + dr * dist, c + dc * dist
                    if not (0 <= rr < H and 0 <= cc < W):
                        break
                    ncell = rr * W + cc
                    low = [k for k in cn.get(ncell, ()) if zi - self.DROP_MAX <= zl[k] < zi - step]
                    nb = self._near.get(ncell)
                    if not low or nb is None:
                        continue
                    j = max(low, key=lambda k: zl[k])
                    blocked = ((bmin[nb, 2] < zi + HEAD_HI) & (bmax[nb, 2] > zl[j] + HEAD_LO)).any()
                    if dist == 2 and not blocked:
                        mid = self._near.get((r + dr) * W + c + dc)
                        if mid is not None:
                            blocked = ((bmin[mid, 2] < zi + HEAD_HI) & (bmax[mid, 2] > zi + HEAD_LO)).any()
                    if blocked:
                        continue
                    lst.append((j, cost * dist + 0.3 * (zi - zl[j])))
                    break
        self.adj = adj

    def _flood(self, start: int) -> np.ndarray:
        seen = np.zeros(len(self.nodes), dtype=bool)
        seen[start] = True
        stack = [start]
        while stack:
            n = stack.pop()
            for m, _ in self.adj[n]:
                if not seen[m]:
                    seen[m] = True
                    stack.append(m)
        return seen

    # ------------------------------------------------------------------ queries
    def cell_rc(self, p):
        c = int((p[0] - self.x0) / self.cell)
        r = int((p[1] - self.y0) / self.cell)
        return min(max(r, 0), self.H - 1), min(max(c, 0), self.W - 1)

    def _nearest(self, p, max_ring: int = 8) -> int:
        """Nearest node, preferring the storey closest to the query height."""
        r0, c0 = self.cell_rc(p)
        z = p[2] if len(p) > 2 else 0.0
        best, best_d = -1, 1e9
        for ring in range(max_ring + 1):
            for r in range(r0 - ring, r0 + ring + 1):
                for c in range(c0 - ring, c0 + ring + 1):
                    if not (0 <= r < self.H and 0 <= c < self.W) or max(abs(r - r0), abs(c - c0)) != ring:
                        continue
                    for j in self.cell_nodes.get(r * self.W + c, ()):
                        dz = self.nodes[j, 2] - z
                        d = (r - r0) ** 2 + (c - c0) ** 2 + 4.0 * dz * dz + (40.0 if dz > 1.2 else 0.0)
                        if d < best_d:
                            best, best_d = j, d
            if best >= 0 and ring >= 1:
                return best
        return best

    def nearest_cell(self, p, max_ring: int = 8) -> int:
        return self._nearest(p, max_ring)

    def random_point(self, rng) -> np.ndarray:
        return self.nodes[rng.integers(len(self.nodes))].copy()

    def _astar(self, s: int, g: int):
        nodes = self.nodes
        gp = nodes[g]
        inv = 1.0 / self.cell
        xyz = getattr(self, "_xyz", None)
        if xyz is None:
            xyz = self._xyz = (nodes[:, 0].tolist(), nodes[:, 1].tolist(), nodes[:, 2].tolist())
        xs, ys, zs = xyz
        gx, gy, gz = float(gp[0]), float(gp[1]), float(gp[2])

        def h(n):
            dx = abs(xs[n] - gx) * inv
            dy = abs(ys[n] - gy) * inv
            return dx + dy - 0.586 * min(dx, dy) + abs(zs[n] - gz) * inv

        adj = self.adj
        openq = [(h(s), 0.0, s)]
        came = {s: -1}
        cost = {s: 0.0}
        while openq:
            _, gcost, n = heapq.heappop(openq)
            if n == g:
                break
            if gcost > cost.get(n, 1e18):
                continue
            for m, w in adj[n]:
                nc = gcost + w
                if nc < cost.get(m, 1e18):
                    cost[m] = nc
                    came[m] = n
                    heapq.heappush(openq, (nc + h(m), nc, m))
        if g not in came:
            return None
        path = []
        n = g
        while n != -1:
            path.append(n)
            n = came[n]
        path.reverse()
        return path

    def _line_walkable(self, a, b) -> bool:
        d = b[:2] - a[:2]
        dist = float(np.hypot(*d))
        steps = max(2, int(dist / (self.cell * 0.5)))
        prev_f = a[2]
        for k in range(1, steps + 1):
            p = a[:2] + d * (k / steps)
            r, c = self.cell_rc(p)
            f = None
            for j in self.cell_nodes.get(r * self.W + c, ()):
                if abs(self.nodes[j, 2] - prev_f) <= C.STEP_HEIGHT:
                    f = self.nodes[j, 2]
                    break
            if f is None:
                return False
            prev_f = f
        return abs(prev_f - b[2]) <= C.STEP_HEIGHT

    def _smooth(self, pts):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = min(len(pts) - 1, i + 24)
            while j > i + 1 and not self._line_walkable(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def find_path(self, start, goal) -> list[np.ndarray]:
        s = self._nearest(start)
        e = self._nearest(goal)
        if s < 0 or e < 0:
            return []
        path = self._astar(s, e)
        if path is None:
            return []
        return self._smooth([self.nodes[i].copy() for i in path])

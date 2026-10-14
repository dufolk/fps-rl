"""Layered navigation graph for the scripted bots.

Every 2D cell of a grid holds one node per standable surface with head room, so storeys, stairs
and crates stack in the same cell. The graph is built once per map with destructible panels and
gadget boxes left out, which makes it the superset of every destruction state: each node remembers
the panels that block its body column or carry its floor. `WorldNav` masks that graph for one world
and refreshes the mask whenever the world's collision geometry changes (walls broken, barricades
placed ...). One-way drop edges lead off ledges and down through broken floor hatches.
"""
from __future__ import annotations

import heapq
import weakref

import numpy as np

from . import config as C

HEAD_LO, HEAD_HI = 0.46, 1.7  # body column above a floor that must be free (above step height)
DIRS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))

_views: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class NavGrid:
    """Static per-map graph. Use `NavGrid.for_world(world)` to get the masked per-world view.

    `map.nav_max_floor` caps which box tops count as floors (e.g. to keep bots off the ship's
    bridge roof).
    """

    _cache: dict = {}
    MAX_LAYERS = 6
    DROP_MAX = 4.5
    SOFT_COST = 25.0

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
        d = world.destruct
        self.n_panels = len(d.panels)
        ns = len(static)
        self._ns = ns
        self._bmin = np.concatenate([np.array([b.mn for b in static], dtype=np.float64).reshape(-1, 3), d.mins])
        self._bmax = np.concatenate([np.array([b.mx for b in static], dtype=np.float64).reshape(-1, 3), d.maxs])
        bmin, bmax = self._bmin, self._bmax
        r = clearance
        nodes, ncell, support, blk_node, blk_panel = [], [], [], [], []
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
                is_panel = nb >= ns
                ilo, ihi = bmin[ins, 2], bmax[ins, 2]
                tops = tops[::-1]
                T = tops[:, None]
                ok = ~(((lo[nb < ns] < T + HEAD_HI) & (hi[nb < ns] > T + HEAD_LO)).any(axis=1)
                       | ((ilo[ins < ns] < T - 1e-6) & (ihi[ins < ns] > T + 1e-6)).any(axis=1))
                found = []
                for t in tops[ok]:
                    if found and found[-1] - t <= C.STEP_HEIGHT:
                        continue
                    hit = (lo < t + HEAD_HI) & (hi > t + HEAD_LO)
                    buried = ins[(ilo < t - 1e-6) & (ihi > t + 1e-6)]
                    under = ins[np.abs(ihi - t) < 1e-6]
                    i = len(nodes)
                    nodes.append((Q[k, 0], Q[k, 1], t))
                    ncell.append(s + k)
                    support.append(-1 if (under < ns).any() else int(under[0]) - ns)
                    for b in [*nb[hit].tolist(), *buried.tolist()]:
                        blk_node.append(i)
                        blk_panel.append(b - ns)
                    found.append(t)
                    if len(found) >= self.MAX_LAYERS:
                        break
        self.nodes = np.array(nodes, dtype=np.float64).reshape(-1, 3)
        self.node_cell = np.array(ncell, dtype=np.int64)
        self.support = np.array(support, dtype=np.int64)
        self.blk_node = np.array(blk_node, dtype=np.int64)
        self.blk_panel = np.array(blk_panel, dtype=np.int64)
        self._build_edges()
        # keep only what can be reached from some spawn point
        reach = np.zeros(len(self.nodes), dtype=bool)
        for sp in world.map.spawns.values():
            for x, y, z, _ in sp:
                n = self._nearest(np.array([x, y, z]), None)
                if n >= 0 and not reach[n]:
                    reach |= self._flood(n)
        keep = np.nonzero(reach)[0]
        remap = np.full(len(self.nodes), -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        self.nodes = self.nodes[keep]
        self.node_cell = self.node_cell[keep]
        self.support = self.support[keep]
        m = remap[self.blk_node] >= 0
        self.blk_node = remap[self.blk_node[m]]
        self.blk_panel = self.blk_panel[m]
        self._build_edges()
        del self._near, self._bmin, self._bmax

    @classmethod
    def for_world(cls, world) -> "WorldNav":
        view = _views.get(world)
        if view is None:
            key = world.cfg.map_name
            if key not in cls._cache:
                cls._cache[key] = cls(world)
            view = _views[world] = WorldNav(cls._cache[key], world)
        return view

    # ------------------------------------------------------------------ build
    def _build_edges(self):
        H, W = self.H, self.W
        cn: dict[int, list[int]] = {}
        for i, c in enumerate(self.node_cell.tolist()):
            cn.setdefault(c, []).append(i)
        self.cell_nodes = cn
        z = self.nodes[:, 2]
        zl = z.tolist()
        step = C.STEP_HEIGHT
        bmin, bmax, ns = self._bmin, self._bmax, self._ns
        de_src, de_dst, de_edge, de_panel = [], [], [], []
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
                ncell = rr * W + cc
                j = match(ncell, zi)
                if j >= 0:
                    if dr and dc:
                        s1, s2 = match(r * W + cc, zi), match(rr * W + c, zi)
                        if s1 < 0 or s2 < 0:
                            continue
                    else:
                        s1 = s2 = -1
                    lst.append((j, cost + 0.5 * abs(zl[j] - zi), s1, s2, -1))
                if dr and dc:
                    continue
                # drop (one or two cells out, past the ledge clearance) to the highest lower floor
                # whose column is free of static boxes; panels in the way gate the edge
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
                    col = nb[(bmin[nb, 2] < zi + HEAD_HI) & (bmax[nb, 2] > zl[j] + HEAD_LO)]
                    if dist == 2:
                        mid = self._near.get((r + dr) * W + c + dc)
                        if mid is not None:
                            col = np.concatenate([col, mid[(bmin[mid, 2] < zi + HEAD_HI)
                                                           & (bmax[mid, 2] > zi + HEAD_LO)]])
                    if (col < ns).any():
                        continue
                    e = len(de_src)
                    de_src.append(i)
                    de_dst.append(j)
                    for b in set(col.tolist()):
                        de_edge.append(e)
                        de_panel.append(b - ns)
                    lst.append((j, cost * dist + 0.3 * (zi - zl[j]), -1, -1, e))
                    break
        self.adj = adj
        self.n_drops = len(de_src)
        self.de_edge = np.array(de_edge, dtype=np.int64)
        self.de_panel = np.array(de_panel, dtype=np.int64)

    def _flood(self, start: int) -> np.ndarray:
        seen = np.zeros(len(self.nodes), dtype=bool)
        seen[start] = True
        stack = [start]
        while stack:
            n = stack.pop()
            for m, *_ in self.adj[n]:
                if not seen[m]:
                    seen[m] = True
                    stack.append(m)
        return seen

    # ------------------------------------------------------------------ queries
    def cell_rc(self, p):
        c = int((p[0] - self.x0) / self.cell)
        r = int((p[1] - self.y0) / self.cell)
        return min(max(r, 0), self.H - 1), min(max(c, 0), self.W - 1)

    def _nearest(self, p, ok, max_ring: int = 8) -> int:
        """Nearest usable node, preferring the storey closest to the query height."""
        r0, c0 = self.cell_rc(p)
        z = p[2] if len(p) > 2 else 0.0
        best, best_d = -1, 1e9
        for ring in range(max_ring + 1):
            for r in range(r0 - ring, r0 + ring + 1):
                for c in range(c0 - ring, c0 + ring + 1):
                    if not (0 <= r < self.H and 0 <= c < self.W) or max(abs(r - r0), abs(c - c0)) != ring:
                        continue
                    for j in self.cell_nodes.get(r * self.W + c, ()):
                        if ok is not None and not ok[j]:
                            continue
                        dz = self.nodes[j, 2] - z
                        d = (r - r0) ** 2 + (c - c0) ** 2 + 4.0 * dz * dz + (40.0 if dz > 1.2 else 0.0)
                        if d < best_d:
                            best, best_d = j, d
            if best >= 0 and ring >= 1:
                return best
        return best

    def scc(self) -> np.ndarray:
        """Strongly connected component per node (drops are one-way), computed once."""
        lab = getattr(self, "_scc", None)
        if lab is None:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import connected_components
            src = [n for n, lst in enumerate(self.adj) for _ in lst]
            dst = [m for lst in self.adj for m, *_ in lst]
            g = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(len(self.nodes),) * 2)
            _n, lab = connected_components(g, directed=True, connection="strong")
            self._scc = lab
        return lab

    def _astar(self, s: int, g: int, ok, drop_ok, cost_add):
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
            for m, w, s1, s2, e in adj[n]:
                if not ok[m] or (s1 >= 0 and not (ok[s1] and ok[s2])) or (e >= 0 and not drop_ok[e]):
                    continue
                nc = gcost + w + cost_add[m]
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

    def _line_walkable(self, a, b, ok) -> bool:
        d = b[:2] - a[:2]
        dist = float(np.hypot(*d))
        steps = max(2, int(dist / (self.cell * 0.5)))
        prev_f = a[2]
        for k in range(1, steps + 1):
            p = a[:2] + d * (k / steps)
            r, c = self.cell_rc(p)
            f = None
            for j in self.cell_nodes.get(r * self.W + c, ()):
                if ok[j] and abs(self.nodes[j, 2] - prev_f) <= C.STEP_HEIGHT:
                    f = self.nodes[j, 2]
                    break
            if f is None:
                return False
            prev_f = f
        return abs(prev_f - b[2]) <= C.STEP_HEIGHT

    def _smooth(self, pts, ok):
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = min(len(pts) - 1, i + 24)
            while j > i + 1 and not self._line_walkable(pts[i], pts[j], ok):
                j -= 1
            out.append(pts[j])
            i = j
        return out


class WorldNav:
    """A `NavGrid` masked by one world's current destruction and gadget state.

    Nodes whose body column holds an intact panel, or whose floor panel is broken, are unusable;
    nodes behind a barricade cost `SOFT_COST` extra (bots knife through), other gadget boxes
    (armor panels, shields, reinforced walls) block unless no other route exists.
    """

    def __init__(self, grid: NavGrid, world):
        self.grid = grid
        self.world = weakref.proxy(world)
        self.nodes = grid.nodes
        self.cell = grid.cell
        self.version = -1

    def refresh(self):
        w = self.world
        if self.version == w.geo_version:
            return
        self.version = w.geo_version
        self.unreachable = set()
        g = self.grid
        n = len(g.nodes)
        ok = np.ones(n, dtype=bool)
        drop_ok = np.ones(g.n_drops, dtype=bool)
        if g.n_panels:
            alive = np.fromiter((p.alive for p in w.destruct.panels), dtype=bool, count=g.n_panels)
            ok &= np.bincount(g.blk_node[alive[g.blk_panel]], minlength=n) == 0
            has = g.support >= 0
            ok[has] &= alive[g.support[has]]
            drop_ok &= np.bincount(g.de_edge[alive[g.de_panel]], minlength=g.n_drops) == 0
        cost = np.zeros(n)
        hard = np.zeros(n, dtype=bool)
        nd = g.nodes
        r = C.PLAYER_RADIUS
        for b in w.gadgets.boxes():
            mn, mx = b.mn, b.mx
            m = ((nd[:, 0] > mn[0] - r) & (nd[:, 0] < mx[0] + r) & (nd[:, 1] > mn[1] - r) & (nd[:, 1] < mx[1] + r)
                 & (nd[:, 2] + 0.05 < mx[2]) & (nd[:, 2] + HEAD_HI > mn[2]))
            if w.gadgets.box_soft(b.params["dyn"]):
                cost[m] = NavGrid.SOFT_COST
            else:
                hard |= m
        self.ok = ok
        self.ok_list = (ok & ~hard).tolist()
        self.loose_list = ok.tolist()
        self.drop_list = drop_ok.tolist()
        self.cost_list = cost.tolist()
        self.loose_cost = (cost + hard * NavGrid.SOFT_COST * 4).tolist()

    def point(self, idx: int) -> np.ndarray:
        return self.nodes[idx].copy()

    def random_point(self, rng) -> np.ndarray:
        self.refresh()
        idx = np.nonzero(self.ok)[0]
        return self.nodes[idx[rng.integers(len(idx))]].copy()

    def usable(self, idx: int) -> bool:
        self.refresh()
        return bool(self.ok_list[idx])

    def nearest_cell(self, p, max_ring: int = 8) -> int:
        self.refresh()
        return self.grid._nearest(p, self.loose_list, max_ring)

    def find_path(self, start, goal) -> list[np.ndarray]:
        self.refresh()
        g = self.grid
        s = g._nearest(start, self.loose_list)
        e = g._nearest(goal, self.loose_list)
        if s < 0 or e < 0:
            return []
        # A failed search floods the whole graph twice (~30 ms). Reachability is the same for every
        # node of a strongly connected component, so a miss is remembered per component pair until
        # the geometry changes (zombies chasing a human on an unreachable box retry constantly).
        lab = g.scc()
        key = (int(lab[s]), int(lab[e]))
        if key in self.unreachable:
            return []
        path = g._astar(s, e, self.ok_list, self.drop_list, self.cost_list)
        if path is None:
            path = g._astar(s, e, self.loose_list, self.drop_list, self.loose_cost)
        if path is None:
            self.unreachable.add(key)
            return []
        return g._smooth([g.nodes[i].copy() for i in path], self.ok_list)

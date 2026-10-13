"""Paint-by-rectangles map authoring.

A map is a 2D grid of cells (`cell` metres). Each cell is either solid (a building block) or a
walkable floor at some height with a material. Rectangles are painted in order (later wins), then
identical neighbouring cells are greedily merged into as few AABBs as possible.
"""
from __future__ import annotations

from .base import Box

SOLID = None


def rect_minus(outer, holes):
    """Split rectangle `outer` (x0, y0, x1, y1) minus `holes` into a few disjoint rectangles."""
    x0, y0, x1, y1 = outer
    xs = sorted({x0, x1, *[min(max(h[0], x0), x1) for h in holes], *[min(max(h[2], x0), x1) for h in holes]})
    ys = sorted({y0, y1, *[min(max(h[1], y0), y1) for h in holes], *[min(max(h[3], y0), y1) for h in holes]})
    cols, rows = len(xs) - 1, len(ys) - 1
    grid = []
    for r in range(rows):
        row = []
        cy = (ys[r] + ys[r + 1]) / 2
        for c in range(cols):
            cx = (xs[c] + xs[c + 1]) / 2
            inside = any(h[0] < cx < h[2] and h[1] < cy < h[3] for h in holes)
            row.append("skip" if inside else 1)
        grid.append(row)
    out = []
    for c0, c1, r0, r1, _ in GridMap._merge(grid, rows, cols):
        out.append((xs[c0], ys[r0], xs[c1 + 1], ys[r1 + 1]))
    return out


class GridMap:
    def __init__(self, cols: int, rows: int, cell: float, x0: float, y1: float):
        """Column 0 starts at x0 (west), row 0 starts at y1 (north edge), rows grow southwards."""
        self.cols, self.rows, self.cell = cols, rows, cell
        self.x0, self.y1 = x0, y1
        self.cells = [[SOLID] * cols for _ in range(rows)]
        self.roofs = [[None] * cols for _ in range(rows)]

    # ------------------------------------------------------------------ painting
    def floor(self, c0, c1, r0, r1, z, mat="sand"):
        for r in range(max(0, r0), min(self.rows, r1 + 1)):
            for c in range(max(0, c0), min(self.cols, c1 + 1)):
                self.cells[r][c] = (round(float(z), 3), mat)

    def wall(self, c0, c1, r0, r1):
        for r in range(max(0, r0), min(self.rows, r1 + 1)):
            for c in range(max(0, c0), min(self.cols, c1 + 1)):
                self.cells[r][c] = SOLID

    def roof(self, c0, c1, r0, r1, height, mat="concrete"):
        """Ceiling slab `height` above the floor of each covered cell (tunnels, interiors)."""
        for r in range(max(0, r0), min(self.rows, r1 + 1)):
            for c in range(max(0, c0), min(self.cols, c1 + 1)):
                self.roofs[r][c] = (height, mat)

    # ------------------------------------------------------------------ coordinates
    def x(self, c):
        return self.x0 + c * self.cell

    def y(self, r):
        return self.y1 - r * self.cell

    def rect_xy(self, c0, c1, r0, r1):
        """World AABB (xmin, ymin, xmax, ymax) of an inclusive cell rectangle."""
        return self.x(c0), self.y(r1 + 1), self.x(c1 + 1), self.y(r0)

    def center(self, c, r):
        return self.x(c) + self.cell / 2, self.y(r) - self.cell / 2

    def level(self, c, r):
        v = self.cells[r][c]
        return None if v is SOLID else v[0]

    # ------------------------------------------------------------------ merging
    @staticmethod
    def _merge(grid, rows, cols):
        used = [[False] * cols for _ in range(rows)]
        out = []
        for r in range(rows):
            for c in range(cols):
                if used[r][c]:
                    continue
                k = grid[r][c]
                if k == "skip":
                    continue
                c1 = c
                while c1 + 1 < cols and not used[r][c1 + 1] and grid[r][c1 + 1] == k:
                    c1 += 1
                r1 = r
                while r1 + 1 < rows and all(not used[r1 + 1][cc] and grid[r1 + 1][cc] == k for cc in range(c, c1 + 1)):
                    r1 += 1
                for rr in range(r, r1 + 1):
                    for cc in range(c, c1 + 1):
                        used[rr][cc] = True
                out.append((c, c1, r, r1, k))
        return out

    def build(self, wall_height, wall_mat, base_z=-1.0):
        """-> list of Box. `wall_height(c0, c1, r0, r1)` / `wall_mat(...)` may be callables."""
        boxes = []
        solid_grid = [["wall" if v is SOLID else "skip" for v in row] for row in self.cells]
        for c0, c1, r0, r1, _ in self._merge(solid_grid, self.rows, self.cols):
            xa, ya, xb, yb = self.rect_xy(c0, c1, r0, r1)
            h = wall_height(c0, c1, r0, r1) if callable(wall_height) else wall_height
            m = wall_mat(c0, c1, r0, r1) if callable(wall_mat) else wall_mat
            boxes.append(Box((xa, ya, base_z), (xb, yb, h), m, tag="wall"))
        floor_grid = [["skip" if v is SOLID else v for v in row] for row in self.cells]
        for c0, c1, r0, r1, (z, mat) in self._merge(floor_grid, self.rows, self.cols):
            xa, ya, xb, yb = self.rect_xy(c0, c1, r0, r1)
            boxes.append(Box((xa, ya, base_z), (xb, yb, z), mat, tag="ground"))
        roof_grid = [[("skip" if self.roofs[r][c] is None or self.cells[r][c] is SOLID
                       else (self.cells[r][c][0] + self.roofs[r][c][0], self.roofs[r][c][1]))
                      for c in range(self.cols)] for r in range(self.rows)]
        for c0, c1, r0, r1, (z, mat) in self._merge(roof_grid, self.rows, self.cols):
            xa, ya, xb, yb = self.rect_xy(c0, c1, r0, r1)
            boxes.append(Box((xa, ya, z), (xb, yb, z + 0.4), mat, tag="roof"))
        return boxes

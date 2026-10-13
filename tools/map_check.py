"""Sanity-check a map: box count, nav reachability of spawns / objectives / strategic points.

    py tools/map_check.py <map> [mode]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fpsrl.bots import NavGrid  # noqa: E402
from fpsrl.geometry import poly_center  # noqa: E402
from fpsrl.maps import default_mode  # noqa: E402
from fpsrl.sim import MatchConfig, World  # noqa: E402


def check(map_name, mode=None, verbose=True):
    mode = mode or default_mode(map_name)
    t = time.time()
    w = World(MatchConfig(team_size=5, map_name=map_name, mode=mode))
    t_world = time.time() - t
    t = time.time()
    nav = NavGrid(w)
    t_nav = time.time() - t
    problems = []

    def reach(label, p, tol=1.5):
        idx = nav.nearest_cell(p)
        if idx < 0:
            problems.append(f"{label}: no nav node near {np.round(p, 2).tolist()}")
            return
        q = nav.point(idx)
        if np.hypot(q[0] - p[0], q[1] - p[1]) > tol or abs(q[2] - p[2]) > 1.0:
            problems.append(f"{label}: nearest node {np.round(q, 2).tolist()} far from {np.round(p, 2).tolist()}")

    for side, sp in w.map.spawns.items():
        for i, s in enumerate(sp):
            reach(f"spawn{side}[{i}]", np.array(s[:3]))
    obj = w.map.objectives or {}
    for kind in ("points", "sites"):
        for z in obj.get(kind, []):
            cx, cy = poly_center(z["poly"])
            reach(f"{kind}:{z['name']}", np.array([cx, cy, z.get("z", z.get("zmin", 0.0) + 0.5)]), tol=3.0)
    for i, (x, y) in enumerate(w.map.strategic_points):
        idx = nav.nearest_cell((x, y, 0.0), max_ring=12)
        if idx < 0:
            problems.append(f"strategic[{i}] ({x:.1f},{y:.1f}) unreachable")
    # spawn agents must not start inside geometry
    for a in w.agents:
        if np.any(w._overlaps(a.pos, a.height)):
            problems.append(f"agent {a.id} spawned inside geometry at {a.pos}")
    if verbose:
        print(f"{map_name}/{mode}: boxes={len(w.map.boxes)} panels={len(w.destruct.panels)} "
              f"nav_nodes={len(nav.nodes)} world={t_world:.2f}s nav={t_nav:.2f}s")
        for p in problems:
            print("  !", p)
        if not problems:
            print("  ok")
    return problems


if __name__ == "__main__":
    check(*sys.argv[1:])

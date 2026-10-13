"""Headless bot-only match with an event summary, to check bot objective play on a map / mode.

    py tools/bot_match.py [map] [mode] [minutes] [team_size] [difficulty]
"""
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl import MatchConfig, World  # noqa: E402
from fpsrl.bots import Bot  # noqa: E402

SHOWN = ("bomb_planted", "bomb_defused", "bomb_explode", "defuser_complete", "point_captured", "round_end",
         "side_swap", "match_end")
COUNTED = ("kill", "buy", "reinforce", "wall_break", "gadget_placed", "bomb_pickup", "bomb_drop", "busy",
           "busy_cancel", "melee_hit")


def main():
    map_name = sys.argv[1] if len(sys.argv) > 1 else "desert2"
    mode = sys.argv[2] if len(sys.argv) > 2 else "bomb"
    minutes = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    team_size = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    diff = sys.argv[5] if len(sys.argv) > 5 else "normal"
    w = World(MatchConfig(team_size=team_size, seed=1, map_name=map_name, mode=mode, time_limit=minutes * 60))
    w.reset(1)
    t0 = time.time()
    bots = [Bot(w, a.id, diff, seed=a.id) for a in w.agents]
    print(f"nav {len(bots[0].nav.nodes)} nodes, built in {time.time() - t0:.1f}s")
    counts = collections.Counter()
    busy = collections.Counter()
    t0 = time.time()
    while not w.done and w.time < minutes * 60:
        for e in w.step({b.aid: b.act() for b in bots}):
            t = e["type"]
            if t in COUNTED:
                counts[t] += 1
            if t == "busy":
                busy[e["kind"]] += 1
            if t in SHOWN:
                info = {k: v for k, v in e.items() if k not in ("type", "seq", "t", "pos")}
                print(f"{w.time:7.1f}s  {t:18s} {info}")
    print(f"sim {w.time:.0f}s in {time.time() - t0:.0f}s wall; score {w.score} winner {w.winner}")
    print("counts", dict(counts))
    print("busy", dict(busy))
    if hasattr(w.mode, "wins"):
        print("money", [a.money for a in w.agents])


if __name__ == "__main__":
    main()

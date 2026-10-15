"""Headless bot-only elimination match with an event summary.

    py tools/bot_match.py [minutes] [team_size] [difficulty]
"""
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl import MatchConfig, World  # noqa: E402
from fpsrl.bots import Bot  # noqa: E402

SHOWN = ("round_end", "side_swap", "match_end")
COUNTED = ("kill", "throw", "melee_hit")


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    team_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    diff = sys.argv[3] if len(sys.argv) > 3 else "normal"
    w = World(MatchConfig(team_size=team_size, seed=1))
    t0 = time.time()
    bots = [Bot(w, a.id, diff, seed=a.id) for a in w.agents]
    print(f"nav {len(bots[0].nav.nodes)} nodes, built in {time.time() - t0:.1f}s")
    print("loadouts", {a.name: a.loadout for a in w.agents})
    counts = collections.Counter()
    kills_by = collections.Counter()
    t0 = time.time()
    while not w.done and w.time < minutes * 60:
        for e in w.step({b.aid: b.act() for b in bots}):
            t = e["type"]
            if t in COUNTED:
                counts[t] += 1
            if t == "kill" and e["attacker"] >= 0:
                kills_by[w.agents[e["attacker"]].loadout] += 1
            if t in SHOWN:
                info = {k: v for k, v in e.items() if k not in ("type", "seq", "t", "pos")}
                print(f"{w.time:7.1f}s  {t:12s} {info}")
    print(f"sim {w.time:.0f}s in {time.time() - t0:.0f}s wall; score {w.score} winner {w.winner}")
    print("counts", dict(counts))
    print("kills by loadout", dict(kills_by))


if __name__ == "__main__":
    main()

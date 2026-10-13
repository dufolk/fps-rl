"""Multi-agent / self-play skeleton.

Two externally controlled players (one per team) fight while the remaining
slots are filled by bots. Replace `policy()` with your learner(s).

    py examples/multiagent_selfplay.py --render
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fpsrl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="transport_ship")
    p.add_argument("--mode", default="tdm")
    p.add_argument("--team-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    n = args.team_size
    env = fpsrl.FPSMultiAgentEnv(map_name=args.map, mode=args.mode, team_size=n, controlled=[0, n],
                                 render_mode="human" if args.render else None)
    obs, infos = env.reset(seed=0)

    def policy(agent, ob):
        return env.action_space(agent).sample()

    totals = {a: 0.0 for a in env.possible_agents}
    for _ in range(args.steps):
        actions = {a: policy(a, obs[a]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        for a, r in rewards.items():
            totals[a] += r
        if not env.agents:
            print("episode done", {a: round(v, 2) for a, v in totals.items()},
                  "score", env.world.score)
            totals = {a: 0.0 for a in env.possible_agents}
            obs, infos = env.reset()
    env.close()


if __name__ == "__main__":
    main()

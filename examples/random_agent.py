"""Random policy rollout + throughput benchmark.

    py examples/random_agent.py                 # headless, vector obs
    py examples/random_agent.py --render        # watch it in a window
    py examples/random_agent.py --obs rgb       # offscreen RGB observations
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fpsrl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--obs", default="vector", choices=["vector", "rgb", "both"])
    p.add_argument("--difficulty", default="normal")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    env = fpsrl.FPSEnv(team_size=args.team_size, obs_mode=args.obs, bot_difficulty=args.difficulty,
                       render_mode="human" if args.render else None, seed=0)
    obs, info = env.reset(seed=0)
    ep_ret, ep_len, returns = 0.0, 0, []
    t0 = time.perf_counter()
    for _ in range(args.steps):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        ep_ret += r
        ep_len += 1
        if terminated or truncated:
            returns.append(ep_ret)
            print(f"episode return {ep_ret:+.2f}  len {ep_len}  score {info['score']}")
            obs, info = env.reset()
            ep_ret, ep_len = 0.0, 0
    dt = time.perf_counter() - t0
    repeat = env.cfg.action_repeat
    print(f"{args.steps / dt:.0f} env steps/s  ({args.steps * repeat / dt:.0f} sim ticks/s, "
          f"{args.steps * repeat / 60 / dt:.1f}x realtime)")
    env.close()


if __name__ == "__main__":
    main()

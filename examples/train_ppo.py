"""PPO with the structured multi-head policy (pip install stable-baselines3).

    py examples/train_ppo.py --timesteps 2000000 --envs 8
    py examples/train_ppo.py --team-size 3 --loadout sniper
    py examples/train_ppo.py --eval runs/ppo/final.zip     # watch a trained policy

The policy encodes self, enemies, teammates, thrown items and lidar in separate
towers, then a 512-wide trunk. Each discrete head has its own categorical.
A 1v1 vs "easy" bots curriculum is a reasonable starting point; raise
--team-size / --difficulty once the agent learns to aim.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fpsrl  # noqa: E402


def make_env(rank, args, render_mode=None):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from fpsrl.policy import HeadAction
        env = fpsrl.FPSEnv(team_size=args.team_size, bot_difficulty=args.difficulty,
                           learner_loadout=args.loadout, max_episode_steps=args.episode_steps,
                           render_mode=render_mode, seed=args.seed + rank)
        return Monitor(HeadAction(env))
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--team-size", type=int, default=1)
    p.add_argument("--difficulty", default="easy")
    p.add_argument("--episode-steps", type=int, default=1500)
    p.add_argument("--loadout", default="random", choices=["rifle", "sniper", "shotgun", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--eval", default=None, help="path to a saved model to watch")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from fpsrl.policy import MultiHeadPolicy

    if args.eval:
        env = make_env(0, args, render_mode="human")()
        model = PPO.load(args.eval)
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                print("score", info["score"])
                obs, _ = env.reset()

    try:
        import tensorboard  # noqa: F401
        tb_log = args.out
    except ImportError:
        tb_log = None

    vec = SubprocVecEnv([make_env(i, args) for i in range(args.envs)])
    vec = VecMonitor(vec)
    model = PPO(MultiHeadPolicy, vec, n_steps=512, batch_size=1024, n_epochs=4, gamma=0.995,
                gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2,
                tensorboard_log=tb_log, verbose=1, seed=args.seed)
    ckpt = CheckpointCallback(save_freq=max(1, 200_000 // args.envs), save_path=args.out, name_prefix="ppo")
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save(os.path.join(args.out, "final"))
    vec.close()


if __name__ == "__main__":
    main()

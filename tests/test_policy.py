"""The structured policy must consume the numeric dict and emit every head."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl import FPSEnv  # noqa: E402
from fpsrl.envs import HEADS  # noqa: E402
from fpsrl.policy import HeadAction, MultiHeadPolicy  # noqa: E402


def test_multihead_policy_steps():
    from stable_baselines3 import PPO

    env = HeadAction(FPSEnv(team_size=1, action_repeat=1))
    model = PPO(MultiHeadPolicy, env, n_steps=16, batch_size=16, n_epochs=1, verbose=0)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (len(HEADS),)
    assert env.action_space.contains(action)
    obs, reward, term, trunc, info = env.step(action)
    assert isinstance(obs, dict) and obs["ownplayer.state.numeric"].shape == (110,)
    assert np.isfinite(reward)
    env.close()

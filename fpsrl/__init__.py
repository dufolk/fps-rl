"""FPS-RL experiment environment: elimination on desert2 with fixed rifle / sniper / shotgun loadouts."""
from .sim import Action, MatchConfig, World
from .envs import EnvConfig, FPSEnv, FPSMultiAgentEnv, RewardConfig
from .bots import Bot

__version__ = "0.3.0-exp"

# gym id -> EnvConfig overrides
ENV_IDS = {
    "FPSRL-Elim-v0": {},
    "FPSRL-Elim-5v5-v0": {"team_size": 5, "mode_params": {"win_rounds": 5, "half_rounds": 4},
                          "max_episode_steps": 15000},
}

try:
    import gymnasium as _gym

    for _id, _kw in ENV_IDS.items():
        if _id not in _gym.registry:
            _gym.register(id=_id, entry_point="fpsrl.envs:FPSEnv", kwargs=dict(_kw))
except Exception:  # pragma: no cover
    pass

__all__ = ["Action", "MatchConfig", "World", "EnvConfig", "FPSEnv", "FPSMultiAgentEnv",
           "RewardConfig", "Bot", "ENV_IDS"]

"""FPS-RL: a lightweight 3D first-person shooter with a reinforcement-learning interface."""
from .sim import Action, MatchConfig, World
from .envs import EnvConfig, FPSEnv, FPSMultiAgentEnv, RewardConfig
from .bots import Bot

__version__ = "0.2.0"

# gym id -> EnvConfig overrides
ENV_IDS = {
    "FPSRL-TransportShip-v0": {},
    "FPSRL-TransportShip-Dom-v0": {"map_name": "transport_ship", "mode": "dom", "max_episode_steps": 9000},
    "FPSRL-Pyramid-Bio-v0": {"map_name": "pyramid", "mode": "bio", "team_size": 4, "max_episode_steps": 9000},
    "FPSRL-Desert2-Bomb-v0": {"map_name": "desert2", "mode": "bomb", "team_size": 5,
                              "mode_params": {"win_rounds": 5, "half_rounds": 4}, "max_episode_steps": 15000},
    "FPSRL-HPTown-Dom-v0": {"map_name": "hp_town", "mode": "dom", "team_size": 5, "max_episode_steps": 9000},
    "FPSRL-Bank-R6-v0": {"map_name": "bank", "mode": "r6", "team_size": 5, "max_episode_steps": 27000},
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

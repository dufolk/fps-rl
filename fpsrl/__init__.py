"""FPS-RL: a lightweight 3D first-person shooter with a reinforcement-learning interface."""
from .sim import Action, MatchConfig, World
from .envs import EnvConfig, FPSEnv, FPSMultiAgentEnv, RewardConfig
from .bots import Bot

__version__ = "0.1.0"

try:
    import gymnasium as _gym

    if "FPSRL-TransportShip-v0" not in _gym.registry:
        _gym.register(id="FPSRL-TransportShip-v0", entry_point="fpsrl.envs:FPSEnv")
except Exception:  # pragma: no cover
    pass

__all__ = ["Action", "MatchConfig", "World", "EnvConfig", "FPSEnv", "FPSMultiAgentEnv",
           "RewardConfig", "Bot"]

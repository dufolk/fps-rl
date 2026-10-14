"""Structured actor-critic for the numeric dict observation and discrete heads.

Player, thrown-item and lidar inputs each go through their own MLP. A 512-wide
trunk feeds a policy MLP and a value MLP. Fight is a two-level categorical
(instant versus sustained) that is expanded back to the 15-way head.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .envs import HEADS
from .obs import (
    CAMERA_DIM,
    DICT_ENEMIES,
    DICT_TEAMMATES,
    LIDAR_DIM,
    PLAYER_NUMERIC_DIM,
    RESERVE_DIM,
    THROWN_DIM,
    THROWN_SLOTS,
    WEAPON_NUMERIC_DIM,
    WEAPON_STATS_DIM,
)

PLAYER_FC = (128, 64)
STATS_FC = (16, 8)
THROWN_FC = (64, 64)
NEARBY_FC = (128, 64)
LIDAR_FC = (128, 64)
ENV_FC = (128, 64)
TRUNK_DIM = 512
STYLE_FC = (64, 64)
POLICY_FC = (256,)
VALUE_FC = (256,)

# Sustained fight items: reload, consume, aimed high throw, aimed low throw.
_LONG_FIGHT = (3, 7, 13, 14)
_INSTANT_FIGHT = tuple(i for i in range(HEADS["fight"]) if i not in _LONG_FIGHT)


def mlp(in_dim: int, widths: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for width in widths:
        layers.append(nn.Linear(dim, width))
        layers.append(nn.ReLU())
        dim = width
    return nn.Sequential(*layers)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(th.ones(dim))
        self.eps = eps

    def forward(self, x: th.Tensor) -> th.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


def _pool(seq: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    return seq.max(dim=1).values, seq.mean(dim=1)


class DictStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space):
        super().__init__(observation_space, TRUNK_DIM + STATS_FC[-1])
        token_dim = STATS_FC[-1]
        player_in = PLAYER_NUMERIC_DIM + RESERVE_DIM + WEAPON_NUMERIC_DIM + token_dim
        self.stats_mlp = mlp(WEAPON_STATS_DIM, STATS_FC)
        self.own_mlp = mlp(player_in, PLAYER_FC)
        self.enemy_mlp = mlp(player_in, PLAYER_FC)
        self.mate_mlp = mlp(player_in, PLAYER_FC)
        self.thrown_mlp = mlp(THROWN_DIM, THROWN_FC)
        self.nearby_mlp = mlp(PLAYER_FC[-1] * 4 + THROWN_FC[-1] * 2, NEARBY_FC)
        self.lidar_mlp = mlp(LIDAR_DIM, LIDAR_FC)
        self.env_mlp = mlp(LIDAR_FC[-1] + CAMERA_DIM, ENV_FC)
        trunk_in = PLAYER_FC[-1] + NEARBY_FC[-1] + ENV_FC[-1]
        self.norm = RMSNorm(trunk_in)
        self.trunk = mlp(trunk_in, (TRUNK_DIM,))
        self.style = mlp(token_dim, STYLE_FC)
        self.film_gamma = nn.Linear(STYLE_FC[-1], TRUNK_DIM)
        self.film_beta = nn.Linear(STYLE_FC[-1], TRUNK_DIM)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def _token(self, stats: th.Tensor) -> th.Tensor:
        flat = stats.reshape(-1, WEAPON_STATS_DIM)
        return self.stats_mlp(flat).reshape(*stats.shape[:-1], STATS_FC[-1])

    def _encode_players(self, obs: dict, prefix: str, count: int, encoder: nn.Module) -> th.Tensor:
        state = th.stack([obs[f"{prefix}_{i}.state.numeric"] for i in range(count)], dim=1)
        weapon = th.stack([obs[f"{prefix}_{i}.weapon.numeric"] for i in range(count)], dim=1)
        stats = th.stack([obs[f"{prefix}_{i}.weapon.stats"] for i in range(count)], dim=1)
        token = self._token(stats)
        reserve = th.zeros(*state.shape[:-1], RESERVE_DIM, device=state.device, dtype=state.dtype)
        flat = th.cat([state, reserve, weapon, token], dim=-1)
        batch, slots, width = flat.shape
        return encoder(flat.reshape(batch * slots, width)).reshape(batch, slots, -1)

    def forward(self, obs: dict) -> th.Tensor:
        own_token = self._token(obs["ownplayer.weapon.stats"])
        own = self.own_mlp(th.cat([
            obs["ownplayer.state.numeric"],
            obs["ownplayer.numeric.reservepara"],
            obs["ownplayer.weapon.numeric"],
            own_token,
        ], dim=-1))
        enemies = self._encode_players(obs, "enemy", DICT_ENEMIES, self.enemy_mlp)
        target = self.enemy_mlp(th.cat([
            obs["target_enemy.state.numeric"],
            th.zeros_like(obs["ownplayer.numeric.reservepara"]),
            obs["target_enemy.weapon.numeric"],
            self._token(obs["target_enemy.weapon.stats"]),
        ], dim=-1)).unsqueeze(1)
        enemies = th.cat([enemies, target], dim=1)
        mates = self._encode_players(obs, "teammate", DICT_TEAMMATES, self.mate_mlp)
        thrown = self.thrown_mlp(obs["nearby.thrown.numeric"].reshape(-1, THROWN_DIM))
        thrown = thrown.reshape(own.shape[0], THROWN_SLOTS, THROWN_FC[-1])
        enemy_max, enemy_mean = _pool(enemies)
        mate_max, mate_mean = _pool(mates)
        thrown_max, thrown_mean = _pool(thrown)
        nearby = self.nearby_mlp(th.cat([
            enemy_max, enemy_mean, mate_max, mate_mean, thrown_max, thrown_mean,
        ], dim=-1))
        lidar = self.lidar_mlp(obs["environment.lidar_info.numeric"])
        env = self.env_mlp(th.cat([lidar, obs["environment.camera.numeric"]], dim=-1))
        hidden = self.trunk(self.norm(th.cat([own, nearby, env], dim=-1)))
        style = self.style(own_token)
        hidden = hidden * (1.0 + self.film_gamma(style)) + self.film_beta(style)
        return th.cat([hidden, own_token], dim=-1)


class ActionHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.order = list(HEADS)
        self.heads = nn.ModuleDict({
            name: nn.Linear(in_dim, n)
            for name, n in HEADS.items()
            if name != "fight"
        })
        self.fight_type = nn.Linear(in_dim, 2)
        self.fight_instant = nn.Linear(in_dim, len(_INSTANT_FIGHT))
        self.fight_long = nn.Linear(in_dim, len(_LONG_FIGHT))

    def forward(self, latent: th.Tensor) -> th.Tensor:
        fight = _fight_logits(self.fight_type(latent), self.fight_instant(latent), self.fight_long(latent))
        parts = [fight if name == "fight" else self.heads[name](latent) for name in self.order]
        return th.cat(parts, dim=-1)


def _fight_logits(type_logits: th.Tensor, instant_logits: th.Tensor, long_logits: th.Tensor) -> th.Tensor:
    p_type = th.softmax(type_logits, dim=-1)
    instant = th.softmax(instant_logits, dim=-1) * p_type[:, :1]
    longer = th.softmax(long_logits, dim=-1) * p_type[:, 1:2]
    prob = th.zeros(type_logits.shape[0], HEADS["fight"], device=type_logits.device, dtype=type_logits.dtype)
    prob[:, list(_INSTANT_FIGHT)] = instant
    prob[:, list(_LONG_FIGHT)] = longer
    return th.log(prob.clamp(min=1e-10))


class HeadAction(gym.ActionWrapper):
    """Expose the discrete heads as one MultiDiscrete, in `HEADS` order."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._keys = list(HEADS)
        self.action_space = spaces.MultiDiscrete([HEADS[key] for key in self._keys])

    def action(self, act):
        values = np.asarray(act, dtype=np.int64).reshape(-1)
        return {key: int(values[i]) for i, key in enumerate(self._keys)}


class MultiHeadPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = DictStateExtractor
        kwargs["net_arch"] = dict(pi=list(POLICY_FC), vf=list(VALUE_FC))
        kwargs["activation_fn"] = nn.ReLU
        kwargs["share_features_extractor"] = True
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = ActionHead(self.mlp_extractor.latent_dim_pi)
        self.action_net.apply(lambda module: self.init_weights(module, gain=0.01))
        film = self.features_extractor
        nn.init.zeros_(film.film_gamma.weight)
        nn.init.zeros_(film.film_gamma.bias)
        nn.init.zeros_(film.film_beta.weight)
        nn.init.zeros_(film.film_beta.bias)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **(self.optimizer_kwargs or {})
        )

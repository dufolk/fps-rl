"""Vector observations for RL agents."""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .sim import Agent, World, angles_to, wrap180
from .weapons import NUM_SLOTS, SLOT_FRAG, SLOT_SMOKE

SELF_FEATURES = [
    "alive", "hp", "pos_x", "pos_y", "pos_z", "vel_fwd", "vel_right", "vel_up",
    "yaw_sin", "yaw_cos", "pitch", "on_ground", "crouch",
    *[f"slot_{i}" for i in range(1, NUM_SLOTS + 1)],
    "mag", "reserve", "reloading", "ads", "ready", "frags", "smokes",
    "punch_pitch", "punch_yaw", "spread", "protected",
    "dmg_dir_fwd", "dmg_dir_right", "dmg_recency", "time_left", "score_diff",
]
ENEMY_FEATURES = ["present", "visible", "dx", "dy", "dz", "dist", "yaw_err", "pitch_err", "hp"]
ALLY_FEATURES = ["present", "dx", "dy", "dz", "hp"]
PROJ_FEATURES = ["present", "dx", "dy", "dz", "is_frag", "fuse_left"]


class ObsBuilder:
    """Builds a flat float32 vector per agent.

    enemy_mode "visible" only exposes enemies in line of sight (fair play);
    "all" exposes every living enemy (privileged / debugging).
    """

    def __init__(self, world: World, n_lidar: int = 32, lidar_range: float = 50.0,
                 max_enemies: int = 4, max_allies: int = 3, max_projectiles: int = 2,
                 enemy_mode: str = "visible"):
        self.world = world
        self.n_lidar = n_lidar
        self.lidar_range = lidar_range
        self.lidar_angles = np.linspace(0.0, 360.0, n_lidar, endpoint=False) if n_lidar else np.zeros(0)
        self.max_enemies = max_enemies
        self.max_allies = max_allies
        self.max_proj = max_projectiles
        self.enemy_mode = enemy_mode
        self.size = (len(SELF_FEATURES) + n_lidar + max_enemies * len(ENEMY_FEATURES)
                     + max_allies * len(ALLY_FEATURES) + max_projectiles * len(PROJ_FEATURES))

    def layout(self) -> dict:
        """Index ranges of each observation block (useful for custom networks)."""
        out, k = {}, 0
        for name, n in (("self", len(SELF_FEATURES)), ("lidar", self.n_lidar),
                        ("enemies", self.max_enemies * len(ENEMY_FEATURES)),
                        ("allies", self.max_allies * len(ALLY_FEATURES)),
                        ("projectiles", self.max_proj * len(PROJ_FEATURES))):
            out[name] = (k, k + n)
            k += n
        return out

    def _local(self, a: Agent, vec):
        yr = math.radians(a.yaw)
        c, s = math.cos(yr), math.sin(yr)
        return vec[0] * c + vec[1] * s, vec[0] * s - vec[1] * c

    def build(self, aid: int) -> np.ndarray:
        w = self.world
        a = w.agents[aid]
        out = np.zeros(self.size, dtype=np.float32)
        k = 0
        ws = a.weapon
        spec = ws.spec
        vf, vr = self._local(a, a.vel)
        dmg_age = w.time - a.last_damage_time
        df, dr = self._local(a, a.last_damage_dir)
        dn = math.hypot(df, dr) or 1.0
        lim = w.cfg.kill_limit or 100
        my, their = w.team_kills[a.team], w.team_kills[1 - a.team]
        feats = [
            float(a.alive), a.hp / C.MAX_HP, a.pos[0] / 40.0, a.pos[1] / 13.0, a.pos[2] / 5.0,
            vf / 6.0, vr / 6.0, a.vel[2] / 6.0,
            math.sin(math.radians(a.yaw)), math.cos(math.radians(a.yaw)), a.pitch / 90.0,
            float(a.on_ground), (C.STAND_HEIGHT - a.height) / (C.STAND_HEIGHT - C.CROUCH_HEIGHT),
            *[float(a.slot == i) for i in range(1, NUM_SLOTS + 1)],
            ws.mag / max(spec.mag, 1), ws.reserve / max(spec.reserve, 1), float(a.reloading), float(a.ads),
            float(w.time >= a.ready_at), float(a.inv[SLOT_FRAG].mag), float(a.inv[SLOT_SMOKE].mag),
            a.punch[0] / 10.0, a.punch[1] / 10.0, w.current_spread(a) / 10.0, float(w.time < a.protect_until),
            df / dn if dmg_age < 2.0 else 0.0, dr / dn if dmg_age < 2.0 else 0.0,
            max(0.0, 1.0 - dmg_age / 2.0),
            max(0.0, 1.0 - w.time / w.cfg.time_limit) if w.cfg.time_limit else 1.0,
            (my - their) / lim,
        ]
        out[k:k + len(feats)] = feats
        k += len(SELF_FEATURES)

        if self.n_lidar:
            if a.alive:
                out[k:k + self.n_lidar] = w.lidar(a, self.lidar_angles, 0.0, self.lidar_range) / self.lidar_range
            k += self.n_lidar

        eye = a.eye()
        vis = w.visibility()
        enemies = []
        for e in w.agents:
            if e.team == a.team or not e.alive or not a.alive:
                continue
            visible = bool(vis[a.id, e.id])
            if self.enemy_mode == "visible" and not visible:
                continue
            enemies.append((float(np.linalg.norm(e.pos - a.pos)), e, visible))
        enemies.sort(key=lambda x: x[0])
        for dist, e, visible in enemies[:self.max_enemies]:
            rel = e.pos - a.pos
            lx, ly = self._local(a, rel)
            yaw_to, pitch_to = angles_to(eye, e.pos + np.array([0.0, 0.0, e.height * 0.62]))
            out[k:k + len(ENEMY_FEATURES)] = [
                1.0, float(visible), lx / 50.0, ly / 50.0, rel[2] / 10.0, dist / 50.0,
                wrap180(yaw_to - a.yaw) / 180.0, (pitch_to - a.pitch) / 90.0,
                e.hp / C.MAX_HP if self.enemy_mode == "all" else 0.0,
            ]
            k += len(ENEMY_FEATURES)
        k_end = len(SELF_FEATURES) + self.n_lidar + self.max_enemies * len(ENEMY_FEATURES)
        k = k_end

        allies = sorted(((float(np.linalg.norm(o.pos - a.pos)), o) for o in w.agents
                         if o.team == a.team and o.id != a.id and o.alive),
                        key=lambda x: x[0]) if a.alive else []
        for dist, o in allies[:self.max_allies]:
            rel = o.pos - a.pos
            lx, ly = self._local(a, rel)
            out[k:k + len(ALLY_FEATURES)] = [1.0, lx / 50.0, ly / 50.0, rel[2] / 10.0, o.hp / C.MAX_HP]
            k += len(ALLY_FEATURES)
        k = k_end + self.max_allies * len(ALLY_FEATURES)

        projs = sorted((float(np.linalg.norm(p.pos - a.pos)), p.id, p) for p in w.projectiles)
        for dist, _, p in projs[:self.max_proj]:
            rel = p.pos - a.pos
            lx, ly = self._local(a, rel)
            out[k:k + len(PROJ_FEATURES)] = [1.0, lx / 20.0, ly / 20.0, rel[2] / 5.0,
                                             float(p.kind == "frag"), max(0.0, p.explode_at - w.time) / 3.0]
            k += len(PROJ_FEATURES)
        return np.clip(out, -5.0, 5.0)

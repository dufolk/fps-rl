"""Vector observations for RL agents."""
from __future__ import annotations

import math

import numpy as np

from . import config as C
from .sim import Agent, World, angles_to, wrap180
from .weapons import LOADOUT_KEYS, NUM_SLOTS, SLOT_FIRE, SLOT_FLASH, SLOT_FRAG, SLOT_SMOKE

SELF_FEATURES = [
    "alive", "hp", "pos_x", "pos_y", "pos_z", "vel_fwd", "vel_right", "vel_up",
    "yaw_sin", "yaw_cos", "pitch", "on_ground", "crouch",
    *[f"slot_{i}" for i in range(1, NUM_SLOTS + 1)],
    *[f"has_{i}" for i in range(1, NUM_SLOTS + 1)],
    "mag", "reserve", "reloading", "ads", "ready", "frags", "smokes", "flashes", "molotovs",
    "punch_pitch", "punch_yaw", "spread",
    "dmg_dir_fwd", "dmg_dir_right", "dmg_recency", "time_left", "score_diff",
    "lean", "frozen", *[f"loadout_{k}" for k in LOADOUT_KEYS],
]
ENEMY_FEATURES = ["present", "visible", "dx", "dy", "dz", "dist", "yaw_err", "pitch_err", "hp"]
ALLY_FEATURES = ["present", "dx", "dy", "dz", "hp"]
PROJ_FEATURES = ["present", "dx", "dy", "dz", "is_frag", "fuse_left"]
OBJECTIVE_SIZE = 12  # Elimination.obs(): round state + nearest site


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
                     + max_allies * len(ALLY_FEATURES) + max_projectiles * len(PROJ_FEATURES) + OBJECTIVE_SIZE)
        (x0, y0), (x1, y1) = world.map.bounds
        self.center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
        self.half = np.array([max((x1 - x0) * 0.5, 1.0), max((y1 - y0) * 0.5, 1.0)])

    def layout(self) -> dict:
        """Index ranges of each observation block (useful for custom networks)."""
        out, k = {}, 0
        for name, n in (("self", len(SELF_FEATURES)), ("lidar", self.n_lidar),
                        ("enemies", self.max_enemies * len(ENEMY_FEATURES)),
                        ("allies", self.max_allies * len(ALLY_FEATURES)),
                        ("projectiles", self.max_proj * len(PROJ_FEATURES)),
                        ("objective", OBJECTIVE_SIZE)):
            out[name] = (k, k + n)
            k += n
        return out

    def _score_diff(self, a: Agent) -> float:
        score = self.world.score
        return (score[a.team] - score[1 - a.team]) / max(self.world.mode.p["win_rounds"], 1)

    def _time_left(self) -> float:
        return min(self.world.mode.time_left() / 120.0, 1.0)

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
        frag, smoke = a.inv.get(SLOT_FRAG), a.inv.get(SLOT_SMOKE)
        flash, fire = a.inv.get(SLOT_FLASH), a.inv.get(SLOT_FIRE)
        px, py = (a.pos[:2] - self.center) / self.half
        feats = [
            float(a.alive), a.hp / C.MAX_HP, px, py, a.pos[2] / 5.0,
            vf / 6.0, vr / 6.0, a.vel[2] / 6.0,
            math.sin(math.radians(a.yaw)), math.cos(math.radians(a.yaw)), a.pitch / 90.0,
            float(a.on_ground), (C.STAND_HEIGHT - a.height) / (C.STAND_HEIGHT - C.CROUCH_HEIGHT),
            *[float(a.slot == i) for i in range(1, NUM_SLOTS + 1)],
            *[float(i in a.inv) for i in range(1, NUM_SLOTS + 1)],
            ws.mag / max(spec.mag, 1), ws.reserve / max(spec.reserve, 1), float(a.reloading), float(a.ads),
            float(w.time >= a.ready_at),
            float(frag.mag) if frag is not None and frag.spec.kind == "grenade" else 0.0,
            float(smoke.mag) if smoke is not None and smoke.spec.kind == "grenade" else 0.0,
            float(flash.mag) if flash is not None and flash.spec.kind == "grenade" else 0.0,
            float(fire.mag) if fire is not None and fire.spec.kind == "grenade" else 0.0,
            a.punch[0] / 10.0, a.punch[1] / 10.0, w.current_spread(a) / 10.0,
            df / dn if dmg_age < 2.0 else 0.0, dr / dn if dmg_age < 2.0 else 0.0,
            max(0.0, 1.0 - dmg_age / 2.0), self._time_left(), self._score_diff(a),
            a.lean, float(a.frozen), *[float(a.loadout == k) for k in LOADOUT_KEYS],
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
                                             float(p.kind in ("frag", "molotov")),
                                             max(0.0, p.explode_at - w.time) / 3.0]
            k += len(PROJ_FEATURES)
        k = self.size - OBJECTIVE_SIZE
        out[k:] = w.mode.obs(a)
        return np.clip(out, -5.0, 5.0)


# Numeric dict layout. Lengths are fixed; unused channels stay 0.
PLAYER_NUMERIC_DIM = 110
WEAPON_NUMERIC_DIM = 8
WEAPON_STATS_DIM = 36
LIDAR_DIM = 93
CAMERA_DIM = 8
THROWN_SLOTS = 10
THROWN_DIM = 8
RESERVE_DIM = 10
DICT_ENEMIES = 4
DICT_TEAMMATES = 4

_BODY_AT = 56
_LIVE_AT = 66
_LOCATION_AT = 70
_SLOT_AT = 80
_WEAPON_STATUS_AT = 100
_LIDAR_CAP_CM = 10000.0
_LIDAR_LOG = math.log(_LIDAR_CAP_CM)


def vector_dict_space() -> "spaces.Dict":
    """Observation dict of the numeric blocks this simulator can fill."""
    from gymnasium import spaces

    def box(shape):
        return spaces.Box(-5.0, 5.0, shape, np.float32)

    keys = {}
    prefixes = ["ownplayer", "target_enemy",
                *[f"enemy_{i}" for i in range(DICT_ENEMIES)],
                *[f"teammate_{i}" for i in range(DICT_TEAMMATES)]]
    for name in prefixes:
        keys[f"{name}.state.numeric"] = box((PLAYER_NUMERIC_DIM,))
        keys[f"{name}.weapon.numeric"] = box((WEAPON_NUMERIC_DIM,))
        keys[f"{name}.weapon.stats"] = box((WEAPON_STATS_DIM,))
    keys["ownplayer.numeric.reservepara"] = box((RESERVE_DIM,))
    keys["environment.lidar_info.numeric"] = box((LIDAR_DIM,))
    keys["environment.camera.numeric"] = box((CAMERA_DIM,))
    keys["nearby.thrown.numeric"] = box((THROWN_SLOTS, THROWN_DIM))
    return spaces.Dict(keys)


def _clip(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), -5.0, 5.0)


def _log_axis(delta: float) -> float:
    return math.copysign(math.log1p(abs(delta)), delta) / math.log(51.0)


class VectorDictObs:
    """Per-agent numeric dict. Enemies follow `enemy_mode` (line of sight by default).

    `target_enemy` is chosen by the action head: slots 0-3 are the distance-sorted
    visible enemies from the previous observation, 4 keeps the previous id, 5 clears it.
    The head does not move the camera.
    """

    def __init__(self, world: World, enemy_mode: str = "visible"):
        self.world = world
        self.enemy_mode = enemy_mode
        self.space = vector_dict_space()
        self.targets: dict[int, int | None] = {}
        self.slots: dict[int, list[int]] = {}
        (x0, y0), (x1, y1) = world.map.bounds
        self.center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
        self.half = np.array([max((x1 - x0) * 0.5, 1.0), max((y1 - y0) * 0.5, 1.0)])
        self._foot_yaw = np.linspace(-60.0, 60.0, 6)
        self._back_yaw = np.linspace(120.0, 240.0, 6)
        self._eye_yaw = np.linspace(-48.0, 48.0, 11)
        self._eye_pitch = np.linspace(-30.0, 30.0, 7)

    def reset(self):
        self.targets.clear()
        self.slots.clear()

    def note(self, aid: int, head: int):
        """Apply the target head against the enemy slots published last observation."""
        head = int(head)
        if head < 4:
            slots = self.slots.get(aid, [])
            self.targets[aid] = slots[head] if head < len(slots) else None
        elif head == 5:
            self.targets[aid] = None

    def build(self, aid: int) -> dict:
        w = self.world
        me = w.agents[aid]
        out = {key: np.zeros(space.shape, np.float32) for key, space in self.space.spaces.items()}
        vis = w.visibility()
        enemies = []
        mates = []
        for other in w.agents:
            if other.id == me.id or not other.alive or not me.alive:
                continue
            dist = float(np.linalg.norm(other.pos - me.pos))
            seen = bool(vis[me.id, other.id])
            if other.team == me.team:
                mates.append((dist, other, seen))
            elif self.enemy_mode != "visible" or seen:
                enemies.append((dist, other, seen))
        enemies.sort(key=lambda x: x[0])
        mates.sort(key=lambda x: x[0])
        self.slots[aid] = [other.id for _, other, _ in enemies[:DICT_ENEMIES]]

        self._write_player(out, "ownplayer", me, me, 0, True, own=True, ammo=True)
        marked = self.targets.get(aid)
        for rank, (dist, other, seen) in enumerate(enemies[:DICT_ENEMIES]):
            self._write_player(out, f"enemy_{rank}", me, other, rank, seen, own=False, ammo=False,
                               marked=other.id == marked)
        if marked is not None:
            for rank, (dist, other, seen) in enumerate(enemies):
                if other.id == marked:
                    self._write_player(out, "target_enemy", me, other, rank, seen, own=False, ammo=False,
                                       marked=True)
                    break
        for rank, (dist, other, seen) in enumerate(mates[:DICT_TEAMMATES]):
            self._write_player(out, f"teammate_{rank}", me, other, rank, seen, own=False, ammo=True)

        out["ownplayer.numeric.reservepara"] = self._reserve(me, enemies)
        out["environment.lidar_info.numeric"] = self._lidar(me) if me.alive else out["environment.lidar_info.numeric"]
        eye = me.eye()
        xy = (eye[:2] - self.center) / self.half
        out["environment.camera.numeric"] = _clip([
            xy[0], xy[1], 0.0, 0.0, eye[2] / 5.0,
            wrap180(me.yaw) / 180.0, me.pitch / 180.0, 0.0,
        ])
        out["nearby.thrown.numeric"] = self._thrown(me)
        return out

    def _write_player(self, out, prefix, me, other, rank, seen, *, own, ammo, marked=False):
        out[f"{prefix}.state.numeric"] = self._player_state(me, other, rank, seen, own=own, marked=marked)
        if ammo:
            out[f"{prefix}.weapon.numeric"] = self._weapon_numeric(other)
        out[f"{prefix}.weapon.stats"] = self._weapon_stats(other.weapon.spec)

    def _player_state(self, me: Agent, other: Agent, rank: int, seen: bool, *, own: bool, marked: bool) -> np.ndarray:
        w = self.world
        feat = np.zeros(PLAYER_NUMERIC_DIM, np.float32)
        xy = (other.pos[:2] - self.center) / self.half
        feat[0], feat[1], feat[4] = xy[0], xy[1], other.pos[2] / 5.0
        feat[5] = wrap180(other.yaw) / 180.0
        feat[6] = other.pitch / 180.0
        feat[8:11] = other.vel / 6.0
        feat[11] = other.hp / C.MAX_HP
        if not own:
            rel = other.pos - me.pos
            dist = float(np.linalg.norm(rel))
            feat[16:19] = rel / 50.0
            yaw_to, pitch_to = angles_to(me.eye(), other.pos + np.array([0.0, 0.0, other.height * 0.62]))
            yaw_err = wrap180(yaw_to - me.yaw)
            feat[19] = wrap180(other.yaw - me.yaw) / 180.0
            feat[20] = math.log1p(dist) / math.log(101.0)
            feat[21] = yaw_err / 180.0
            feat[22] = (pitch_to - me.pitch) / 90.0
            feat[23] = math.cos(math.radians(yaw_err))
            feat[24] = math.sin(math.radians(yaw_err))
            back_yaw, back_pitch = angles_to(other.eye(), me.eye())
            feat[25] = wrap180(back_yaw - other.yaw) / 180.0
            feat[26] = (back_pitch - other.pitch) / 90.0
            feat[27] = float(seen)
            feat[51] = rank / float(DICT_ENEMIES)
            feat[52:56] = self._ray_status(me, other)
        speed = float(np.hypot(other.vel[0], other.vel[1]))
        feat[32] = float(speed > 1.0 and not other.walking)
        feat[33] = float(other.ads)
        feat[34] = float(other.lean < -0.3)
        feat[35] = float(other.lean > 0.3)
        feat[36] = float(not other.on_ground)
        feat[40] = float(w.time - other.last_shot < 0.25)
        if own or other.team == me.team:
            feat[41] = float(other.throw_at >= 0)
            feat[42] = float(w.time < other.ready_at and other.reload_until <= 0.0)
        if not own and other.team != me.team:
            feat[47] = float(me.last_attacker == other.id and w.time - me.last_damage_time < 2.0)
            feat[48] = float(marked)
        if not other.on_ground and other.vel[2] > 0.5:
            body = 1
        elif other.height < (C.STAND_HEIGHT + C.CROUCH_HEIGHT) * 0.5:
            body = 2
        else:
            body = 0
        feat[_BODY_AT + (body + 1) % 10] = 1.0
        if not own:
            feat[_LIVE_AT + 1] = 1.0
        feat[_LOCATION_AT + 1] = 1.0
        if own:
            feat[_SLOT_AT + (other.slot + 1) % 20] = 1.0
            mode = 4 if other.weapon.spec.auto else (1 if other.weapon.spec.kind == "gun" else 0)
            feat[_WEAPON_STATUS_AT + (mode + 1) % 10] = 1.0
        return _clip(feat)

    def _ray_status(self, me: Agent, other: Agent) -> np.ndarray:
        w = self.world
        feet_me = me.pos + np.array([0.0, 0.0, 0.15])
        feet_other = other.pos + np.array([0.0, 0.0, 0.15])
        pairs = ((me.eye(), other.eye()), (me.eye(), feet_other),
                 (feet_me, other.eye()), (feet_me, feet_other))
        return np.array([float(w.segment_clear(p, q)) for p, q in pairs], np.float32)

    def _weapon_numeric(self, other: Agent) -> np.ndarray:
        ws = other.weapon
        spec = ws.spec
        buf = np.zeros(WEAPON_NUMERIC_DIM, np.float32)
        buf[0] = ws.mag / spec.mag if spec.mag else 0.0
        if other.reload_until > self.world.time:
            buf[1] = (other.reload_until - self.world.time) / max(spec.reload_time, 0.1)
            buf[2] = 1.0
        return _clip(buf)

    def _weapon_stats(self, spec) -> np.ndarray:
        buf = np.zeros(WEAPON_STATS_DIM, np.float32)
        buf[0] = spec.damage / 100.0
        buf[1] = spec.mag / 100.0
        buf[2] = spec.interval
        buf[3] = 4.0 if spec.auto else (1.0 if spec.kind == "gun" else 0.0)
        buf[4] = spec.falloff
        buf[6] = spec.range / 200.0
        buf[8] = spec.spread_base / 10.0 if spec.kind == "gun" else 0.0
        buf[9] = spec.reload_time
        return _clip(buf)

    def _reserve(self, me: Agent, enemies) -> np.ndarray:
        buf = np.zeros(RESERVE_DIM, np.float32)
        if me.throw_at >= 0:
            buf[3] = 1.0
            buf[4] = max(0.0, me.cook_until - self.world.time) / max(me.weapon.spec.fuse, 0.1)
        nearest = self._nearest_throw(me)
        if nearest is not None:
            rel = nearest[0] - me.pos
            buf[5], buf[6], buf[7] = _log_axis(rel[0]), _log_axis(rel[1]), _log_axis(rel[2])
            buf[9] = math.log1p(float(np.linalg.norm(rel))) / math.log(101.0)
        return _clip(buf)

    def _nearest_throw(self, me: Agent):
        w = self.world
        best = None
        best_d = 1e9
        for pr in w.projectiles:
            d = float(np.linalg.norm(pr.pos - me.pos))
            if d < best_d:
                best, best_d = pr.pos, d
        for cloud in (*w.smokes, *w.fires):
            d = float(np.linalg.norm(cloud.pos - me.pos))
            if d < best_d:
                best, best_d = cloud.pos, d
        return None if best is None else (best, best_d)

    def _thrown(self, me: Agent) -> np.ndarray:
        w = self.world
        rows = []
        for pr in w.projectiles:
            rows.append((float(np.linalg.norm(pr.pos - me.pos)), pr.pos,
                         max(0.0, pr.explode_at - w.time), max(0.0, pr.explode_at - pr.spawn_time), 0.0, 0.0))
        for smoke in w.smokes:
            if w.time < smoke.t1:
                rows.append((float(np.linalg.norm(smoke.pos - me.pos)), smoke.pos,
                             max(0.0, smoke.t1 - w.time), smoke.t1 - smoke.t0, smoke.max_radius, 1.0))
        for fire in w.fires:
            if w.time < fire.t1:
                rows.append((float(np.linalg.norm(fire.pos - me.pos)), fire.pos,
                             max(0.0, fire.t1 - w.time), fire.t1 - fire.t0, fire.radius, 1.0))
        rows.sort(key=lambda x: x[0])
        out = np.zeros((THROWN_SLOTS, THROWN_DIM), np.float32)
        for i, (dist, pos, remain, duration, radius, arrived) in enumerate(rows[:THROWN_SLOTS]):
            rel = pos - me.pos
            out[i] = _clip([
                _log_axis(rel[0]), _log_axis(rel[1]), _log_axis(rel[2]),
                remain / 8.0, duration / 8.0, radius / 10.0,
                math.log1p(dist) / math.log(101.0), arrived,
            ])
        return out

    def _lidar(self, me: Agent) -> np.ndarray:
        w = self.world
        reach = 100.0
        foot = w.lidar(me, self._foot_yaw, 0.0, reach)
        back = w.lidar(me, self._back_yaw, 0.0, reach)
        eye = np.zeros(11 * 7, np.float64)
        for vi, pitch in enumerate(self._eye_pitch):
            eye.reshape(11, 7)[:, vi] = w.lidar(me, self._eye_yaw, float(me.pitch + pitch), reach)
        raw = np.concatenate([foot, back, eye])
        cm = np.where(raw >= reach - 1e-3, _LIDAR_CAP_CM, raw * 100.0)
        h_center = cm[12 + 5 * 7:12 + 6 * 7]
        v_center = cm[12 + 3:12 + 77:7]
        extra = np.array([h_center.min(), h_center.max(), v_center.min(), v_center.max()])
        feat = np.log1p(np.clip(np.concatenate([cm, extra]), 0.0, _LIDAR_CAP_CM)) / _LIDAR_LOG
        return _clip(feat)

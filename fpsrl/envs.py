"""Gymnasium RL interfaces.

* `FPSEnv`             single learning agent; every other player is a scripted bot.
* `FPSMultiAgentEnv`   PettingZoo-style parallel API for self-play / MARL; any
                       subset of players can be externally controlled.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .bots import Bot
from .maps import MAPS
from .obs import ObsBuilder, VectorDictObs, vector_dict_space
from .sim import Action, MatchConfig, World
from .weapons import NUM_SLOTS, PRIMARY_CHOICES, SLOT_FRAG, SLOT_SMOKE, random_primary

YAW_BINS = np.array([-30.0, -15.0, -6.0, -2.0, 0.0, 2.0, 6.0, 15.0, 30.0])
PITCH_BINS = np.array([-10.0, -3.0, 0.0, 3.0, 10.0])
MD_NVEC = [3, 3, len(YAW_BINS), len(PITCH_BINS), 2, 2, 2, 2, 2, NUM_SLOTS + 1, 3, 2, 2]
ACTION_FIELDS = ["move_forward", "move_right", "look_yaw", "look_pitch", "fire", "alt",
                 "jump", "crouch", "reload", "switch", "lean", "use", "gadget"]

# Eight directions around a center, then walk / run / stop.
# The last yaw bin is 0 degrees. It does not steer toward an enemy or a throw point.
_MOVE_DIR = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
             (-1.0, 0.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0))
HEAD_YAW = np.array([-60.0, -45.0, -25.0, -15.0, -10.0, -6.0, -3.0, -1.0,
                     0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 25.0, 45.0, 60.0, 0.0])
HEAD_PITCH = np.array([-10.0, -6.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0, 10.0])
HEADS = {
    "move_direction": 9,
    "move_way": 3,
    "yaw": len(HEAD_YAW),
    "pitch": len(HEAD_PITCH),
    "body": 16,
    "fight": 15,
    "target_enemy": 6,
    "throw_target": 9,
    "predict_interval": 11,
    "path_goal": 9,
    "thrown_move_direction": 9,
}


@dataclass
class RewardConfig:
    kill: float = 1.0
    headshot_bonus: float = 0.25
    death: float = -0.5
    damage_dealt: float = 0.5   # per 100 HP dealt to enemies
    damage_taken: float = -0.25  # per 100 HP received
    team_damage: float = -0.5   # per 100 HP dealt to teammates (friendly fire only)
    win: float = 3.0
    lose: float = -3.0
    step: float = 0.0
    # objective modes
    round_win: float = 1.0
    round_lose: float = -1.0
    capture: float = 0.5        # own team wins a point (x0.3 when the agent was not on it)
    point_lost: float = -0.25   # enemy wins a point
    hold: float = 0.0           # per second while the team alone stands in the open point
    plant: float = 0.8
    defuse: float = 0.8
    in_objective: float = 0.0   # per second inside the zone the agent should be in (shaping)


@dataclass
class EnvConfig:
    map_name: str = "transport_ship"
    mode: str = "tdm"  # tdm | dom | bomb | elim | r6 (must be listed in the map's modes)
    mode_params: dict = field(default_factory=dict)
    learner_auto_buy: bool = True  # bomb mode: controlled agents buy like bots during freeze time
    team_size: int = 4
    learner_team: int = 0
    tick_rate: int = 60
    action_repeat: int = 4
    max_episode_steps: int = 2250
    kill_limit: int = 40
    time_limit: float = 0.0
    respawn: bool = True
    end_on_death: bool = False
    friendly_fire: bool = False
    bot_difficulty: float | str = "normal"
    teammate_bots: bool = True
    learner_primary: str = "ak47"
    bot_primaries: str = "mixed"  # "ak47" | "awm" | "mixed"
    action_type: str = "heads"  # "heads" | "multidiscrete" | "box"
    max_turn_per_step: float = 30.0  # degrees, only for action_type="box"
    obs_mode: str = "vector"  # "vector" (numeric dict) | "flat" (162-d) | "rgb" | "both"
    rgb_size: tuple = (128, 96)
    n_lidar: int = 32
    max_enemies: int = 4
    max_allies: int = 3
    enemy_mode: str = "visible"
    render_realtime: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int | None = None

    @classmethod
    def make(cls, cfg=None, **overrides) -> "EnvConfig":
        if isinstance(cfg, EnvConfig):
            data = asdict(cfg)
        else:
            data = dict(cfg or {})
        data.update(overrides)
        if isinstance(data.get("reward"), dict):
            data["reward"] = RewardConfig(**data["reward"])
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise TypeError(f"unknown EnvConfig fields: {sorted(unknown)}")
        return cls(**data)


def _head_space() -> spaces.Dict:
    return spaces.Dict({name: spaces.Discrete(n) for name, n in HEADS.items()})


def _move_axes(index: int) -> tuple[float, float]:
    return _MOVE_DIR[int(index)]


def _next_slot(agent) -> int:
    owned = [slot for slot, ws in sorted(agent.inv.items())
             if ws.mag > 0 or ws.spec.kind in ("gun", "melee")]
    if not owned:
        return 0
    if agent.slot in owned:
        return owned[(owned.index(agent.slot) + 1) % len(owned)]
    return owned[0]


def _arm_throw(act: Action, agent, now: float, slot: int, under: bool):
    """Pull the pin once the grenade is in hand. A cook that is already going stays held."""
    if agent is None:
        act.switch = slot
        return
    if agent.throw_at >= 0:
        act.fire = not under
        act.alt = under
        return
    if agent.slot != slot:
        act.switch = slot
        return
    if now < agent.ready_at:
        return
    if under:
        act.alt = not agent.prev_alt
    else:
        act.fire = not agent.prev_fire


def decode_heads(action, repeat: int, agent=None, now: float = 0.0) -> Action:
    """Map one multi-head action onto a sim Action. Look deltas are per tick.

    `throw_target`, `predict_interval` and `path_goal` have no simulator effect.
    `target_enemy` is read by the observation, not here. The last yaw bin is 0 degrees.
    """
    way = int(action["move_way"])
    cooking = agent is not None and agent.throw_at >= 0
    direction = int(action["thrown_move_direction"] if cooking else action["move_direction"])
    if way == 2:
        forward, right, walk = 0.0, 0.0, False
    else:
        forward, right = _move_axes(direction)
        walk = way == 0
    act = Action(
        move_forward=forward, move_right=right, walk=walk,
        look_yaw=float(HEAD_YAW[int(action["yaw"])]) / repeat,
        look_pitch=float(HEAD_PITCH[int(action["pitch"])]) / repeat,
    )
    body = int(action["body"])
    if body == 2:
        act.crouch = True
    elif body == 4:
        act.jump = True
    elif body == 6:
        act.lean = -1
    elif body == 7:
        act.lean = 1
    fight = int(action["fight"])
    if fight == 1:
        act.fire = True
    elif fight == 2:
        act.alt = True
    elif fight == 3:
        act.reload = True
    elif fight == 4 and agent is not None:
        act.switch = _next_slot(agent)
    elif fight == 8:
        _arm_throw(act, agent, now, SLOT_FRAG, under=False)
    elif fight == 9:
        _arm_throw(act, agent, now, SLOT_SMOKE, under=False)
    elif fight in (11, 13) and (agent is None or agent.throw_at < 0):
        _arm_throw(act, agent, now, SLOT_FRAG, under=False)
    elif fight in (12, 14) and (agent is None or agent.throw_at < 0):
        _arm_throw(act, agent, now, SLOT_FRAG, under=True)
    return act


def decode_action(action, action_type: str, repeat: int, max_turn: float = 30.0,
                  agent=None, now: float = 0.0) -> Action:
    """Map a policy action to a sim Action whose look deltas are per tick."""
    if action_type == "heads":
        return decode_heads(action, repeat, agent, now)
    a = np.asarray(action)
    if action_type == "multidiscrete":
        a = a.astype(np.int64)
        return Action(
            move_forward=float(a[0] - 1), move_right=float(a[1] - 1),
            look_yaw=float(YAW_BINS[a[2]]) / repeat, look_pitch=float(PITCH_BINS[a[3]]) / repeat,
            fire=bool(a[4]), alt=bool(a[5]), jump=bool(a[6]), crouch=bool(a[7]),
            reload=bool(a[8]), switch=int(a[9]), lean=int(a[10]) - 1, use=bool(a[11]), gadget=bool(a[12]),
        )
    if action_type != "box":
        raise ValueError(action_type)
    a = np.clip(a.astype(np.float64), -1.0, 1.0)
    return Action(
        move_forward=float(a[0]), move_right=float(a[1]),
        look_yaw=float(a[2]) * max_turn / repeat, look_pitch=float(a[3]) * max_turn * 0.5 / repeat,
        fire=a[4] > 0, alt=a[5] > 0, jump=a[6] > 0, crouch=a[7] > 0, reload=a[8] > 0,
        switch=int(np.clip(np.round((a[9] + 1.0) * NUM_SLOTS / 2.0), 0, NUM_SLOTS)),
        lean=int(np.clip(np.round(a[10]), -1, 1)), use=a[11] > 0, gadget=a[12] > 0,
    )


class _FPSBase:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def _setup(self, config, render_mode, controlled):
        self.cfg = EnvConfig.make(config)
        cfg = self.cfg
        if cfg.map_name not in MAPS:
            raise ValueError(f"unknown map {cfg.map_name!r}, choose from {sorted(MAPS)}")
        n = cfg.team_size
        rng = np.random.default_rng(cfg.seed)
        primaries = []
        for aid in range(2 * n):
            if cfg.bot_primaries == "mixed":
                primaries.append(random_primary(rng))
            else:
                primaries.append(cfg.bot_primaries)
        for aid in controlled:
            primaries[aid] = cfg.learner_primary
        assert all(p in PRIMARY_CHOICES for p in primaries)
        self.world = World(MatchConfig(
            team_size=n, map_name=cfg.map_name, mode=cfg.mode, mode_params=dict(cfg.mode_params),
            tick_rate=cfg.tick_rate, kill_limit=cfg.kill_limit,
            time_limit=cfg.time_limit, friendly_fire=cfg.friendly_fire, respawn=cfg.respawn,
            primaries=primaries, seed=cfg.seed,
        ))
        self.controlled = list(controlled)
        # bots that only shop for the learners (bomb mode economy)
        self._buyers = [Bot(self.world, aid, cfg.bot_difficulty, seed=aid) for aid in self.controlled] \
            if cfg.learner_auto_buy and getattr(self.world.mode, "economy", False) else []
        self.bots = [Bot(self.world, a.id, cfg.bot_difficulty, seed=(cfg.seed or 0) * 101 + a.id)
                     for a in self.world.agents if a.id not in self.controlled]
        if not cfg.teammate_bots:
            teams = {self.world.agents[c].team for c in self.controlled}
            self.bots = [b for b in self.bots if self.world.agents[b.aid].team not in teams]
        self.obs_builder = ObsBuilder(self.world, cfg.n_lidar, max_enemies=cfg.max_enemies,
                                      max_allies=cfg.max_allies, enemy_mode=cfg.enemy_mode)
        self.state_obs = VectorDictObs(self.world, enemy_mode=cfg.enemy_mode)
        self.render_mode = render_mode
        self._viewer = None
        self._rgb_viewer = None
        self._steps = 0
        self._last_dt = 0.0  # simulated seconds covered by the last env step
        self._last_frame = time.perf_counter()

        flat = spaces.Box(-5.0, 5.0, (self.obs_builder.size,), np.float32)
        numeric = vector_dict_space()
        if cfg.obs_mode == "flat":
            self._obs_space = flat
        elif cfg.obs_mode == "vector":
            self._obs_space = numeric
        elif cfg.obs_mode in ("rgb", "both"):
            w, h = cfg.rgb_size
            rgb = spaces.Box(0, 255, (h, w, 3), np.uint8)
            self._obs_space = rgb if cfg.obs_mode == "rgb" else spaces.Dict({"vector": numeric, "rgb": rgb})
        else:
            raise ValueError(cfg.obs_mode)
        if cfg.action_type == "heads":
            self._act_space = _head_space()
        elif cfg.action_type == "multidiscrete":
            self._act_space = spaces.MultiDiscrete(MD_NVEC)
        elif cfg.action_type == "box":
            self._act_space = spaces.Box(-1.0, 1.0, (len(MD_NVEC),), np.float32)
        else:
            raise ValueError(cfg.action_type)

    # ------------------------------------------------------------------ helpers
    def _observe(self, aid: int):
        mode = self.cfg.obs_mode
        if mode == "flat":
            return self.obs_builder.build(aid)
        if mode == "vector":
            return self.state_obs.build(aid)
        rgb = self._render_rgb(aid)
        if mode == "rgb":
            return rgb
        return {"vector": self.state_obs.build(aid), "rgb": rgb}

    def _note_target(self, aid: int, action):
        if self.cfg.action_type == "heads":
            self.state_obs.note(aid, action["target_enemy"])

    def _render_rgb(self, aid: int) -> np.ndarray:
        if self._rgb_viewer is None:
            from .render.viewer import Viewer
            w, h = self.cfg.rgb_size
            self._rgb_viewer = Viewer.shared(self.world, pov=aid, onscreen=self.render_mode == "human")
            self._rgb_viewer.enable_obs_buffer(w, h)
        return self._rgb_viewer.capture_obs(aid)

    def _reward(self, aid: int, events) -> tuple[float, dict]:
        r = self.cfg.reward
        me = self.world.agents[aid]
        total = r.step
        stats = {"kills": 0, "deaths": 0, "damage_dealt": 0.0, "damage_taken": 0.0, "headshots": 0}
        for e in events:
            t = e["type"]
            if t == "damage":
                victim = self.world.agents[e["victim"]]
                if e["attacker"] == aid and e["victim"] != aid:
                    if victim.team != me.team:
                        total += r.damage_dealt * e["amount"] / 100.0
                        stats["damage_dealt"] += e["amount"]
                    else:
                        total += r.team_damage * e["amount"] / 100.0
                if e["victim"] == aid:
                    total += r.damage_taken * e["amount"] / 100.0
                    stats["damage_taken"] += e["amount"]
            elif t == "kill":
                if e["attacker"] == aid and e["victim"] != aid and \
                        self.world.agents[e["victim"]].team != me.team:
                    total += r.kill + (r.headshot_bonus if e["headshot"] else 0.0)
                    stats["kills"] += 1
                    stats["headshots"] += int(e["headshot"])
                if e["victim"] == aid:
                    total += r.death
                    stats["deaths"] += 1
            elif t == "match_end":
                if e["winner"] == me.team:
                    total += r.win
                elif e["winner"] >= 0:
                    total += r.lose
            elif t == "round_end":
                total += r.round_win if e["team"] == me.team else r.round_lose
                stats["rounds_won"] = stats.get("rounds_won", 0) + int(e["team"] == me.team)
            elif t == "point_captured":
                if e["team"] == me.team:
                    on = me.alive and self.world.mode.in_zone(me.pos, self.world.mode.points[e["point"]])
                    total += r.capture * (1.0 if on else 0.3)
                    stats["captures"] = stats.get("captures", 0) + int(on)
                elif e["team"] >= 0:
                    total += r.point_lost
            elif t == "bomb_planted" and e["agent"] == aid:
                total += r.plant
                stats["plants"] = stats.get("plants", 0) + 1
            elif t == "bomb_defused" and e["agent"] == aid:
                total += r.defuse
                stats["defuses"] = stats.get("defuses", 0) + 1
        dt = self._last_dt
        m = self.world.mode
        if r.hold and m.name == "dom" and m.active >= 0 and m.present[me.team] and not m.present[1 - me.team]:
            total += r.hold * dt
        if r.in_objective and me.alive and self._in_objective(me):
            total += r.in_objective * dt
        return float(total), stats

    def _in_objective(self, a) -> bool:
        m = self.world.mode
        if m.name == "dom":
            return m.active >= 0 and m.in_zone(a.pos, m.points[m.active])
        if hasattr(m, "site_at"):
            b = m.bomb
            if m.side(a.team) == 1 and b is not None and b["state"] == "planted":
                return float(np.linalg.norm(a.pos - np.asarray(b["pos"]))) < 2.0
            return m.site_at(a.pos) >= 0
        return False

    def _bot_actions(self) -> dict:
        return {b.aid: b.act() for b in self.bots}

    def _tick(self, actions: dict) -> list:
        for b in self._buyers:
            if b.agent.alive:
                b.maybe_buy(b.agent)
        acts = self._bot_actions()
        acts.update(actions)
        events = self.world.step(acts)
        if self.render_mode == "human":
            self._draw_human()
        return events

    def _draw_human(self):
        from .render.viewer import Viewer
        if self._viewer is None:
            self._viewer = Viewer.shared(self.world, pov=self.controlled[0] if self.controlled else 0,
                                         onscreen=True)
        if self.cfg.render_realtime:
            target = self._last_frame + self.world.dt
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            self._last_frame = max(target, time.perf_counter() - 0.1)
        self._viewer.draw(self.world.dt)

    def _reset_world(self, seed):
        if seed is None:
            seed = int(self.world.rng.integers(1 << 31))
        self.world.reset(seed)
        self.state_obs.reset()
        for b in self.bots:
            b.reset(seed * 101 + b.aid)
        for b in self._buyers:
            b.reset(seed * 101 + b.aid)
        self._steps = 0
        self._last_dt = 0.0

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb(self.controlled[0] if self.controlled else 0)
        if self.render_mode == "human":
            self._draw_human()
        return None

    def close(self):
        from .render import viewer as _v
        if self._viewer is not None or self._rgb_viewer is not None:
            _v.Viewer.destroy_shared()
        self._viewer = self._rgb_viewer = None


class FPSEnv(_FPSBase, gym.Env):
    """Single-agent environment: the learner plays with/against scripted bots.

    Default action is a dict of discrete heads (move, yaw/pitch bins, body, fight).
    The last yaw bin is a fixed 0 degree turn. `target_enemy` only picks which
    enemy is written into the observation.
    """

    def __init__(self, config: EnvConfig | dict | None = None, render_mode: str | None = None, **kwargs):
        cfg = EnvConfig.make(config, **kwargs)
        learner = cfg.learner_team * cfg.team_size
        self._setup(cfg, render_mode, [learner])
        self.learner = learner
        self.observation_space = self._obs_space
        self.action_space = self._act_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_world(seed)
        return self._observe(self.learner), self._info({})

    def _info(self, stats) -> dict:
        w = self.world
        me = w.agents[self.learner]
        return {"stats": stats, "team_kills": list(w.team_kills), "score": list(w.score), "alive": me.alive,
                "hp": me.hp, "kills": me.kills, "deaths": me.deaths, "time": w.time, "events": []}

    def step(self, action):
        cfg = self.cfg
        me = self.world.agents[self.learner]
        self._note_target(self.learner, action)
        act = decode_action(action, cfg.action_type, cfg.action_repeat, cfg.max_turn_per_step,
                            agent=me, now=self.world.time)
        events = []
        t0 = self.world.time
        for k in range(cfg.action_repeat):
            events += self._tick({self.learner: act})
            if k == 0:
                act.switch = 0
            if self.world.done or (cfg.end_on_death and not me.alive):
                break
        self._last_dt = self.world.time - t0
        self._steps += 1
        reward, stats = self._reward(self.learner, events)
        terminated = self.world.done or (cfg.end_on_death and not me.alive)
        truncated = (not terminated) and self._steps >= cfg.max_episode_steps
        info = self._info(stats)
        info["events"] = events
        return self._observe(self.learner), reward, terminated, truncated, info


class FPSMultiAgentEnv(_FPSBase):
    """PettingZoo-ParallelEnv-style API (no PettingZoo dependency).

    `controlled`: agent ids driven by `step()`; the rest are bots. Agent names
    are "agent_<id>"; team = id // team_size.
    """

    def __init__(self, config: EnvConfig | dict | None = None, controlled=None,
                 render_mode: str | None = None, **kwargs):
        cfg = EnvConfig.make(config, **kwargs)
        ids = list(range(2 * cfg.team_size)) if controlled is None else list(controlled)
        self._setup(cfg, render_mode, ids)
        self.possible_agents = [f"agent_{i}" for i in ids]
        self.agents = list(self.possible_agents)
        self._name_to_id = {f"agent_{i}": i for i in ids}

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def team_of(self, agent) -> int:
        return self.world.agents[self._name_to_id[agent]].team

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._reset_world(seed)
        self.agents = list(self.possible_agents)
        obs = {n: self._observe(i) for n, i in self._name_to_id.items()}
        return obs, {n: {} for n in self.agents}

    def step(self, actions: dict):
        cfg = self.cfg
        decoded = {}
        for n, a in actions.items():
            aid = self._name_to_id[n]
            self._note_target(aid, a)
            decoded[aid] = decode_action(a, cfg.action_type, cfg.action_repeat, cfg.max_turn_per_step,
                                         agent=self.world.agents[aid], now=self.world.time)
        events = []
        t0 = self.world.time
        for k in range(cfg.action_repeat):
            events += self._tick(decoded)
            if k == 0:
                for a in decoded.values():
                    a.switch = 0
            if self.world.done:
                break
        self._last_dt = self.world.time - t0
        self._steps += 1
        obs, rew, term, trunc, info = {}, {}, {}, {}, {}
        for n, i in self._name_to_id.items():
            obs[n] = self._observe(i)
            rew[n], stats = self._reward(i, events)
            term[n] = self.world.done
            trunc[n] = (not self.world.done) and self._steps >= cfg.max_episode_steps
            info[n] = {"stats": stats, "alive": self.world.agents[i].alive}
        if self.world.done or self._steps >= cfg.max_episode_steps:
            self.agents = []
        return obs, rew, term, trunc, info

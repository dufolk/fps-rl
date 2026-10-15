"""Headless sanity tests. Run with `pytest tests` or `py tests/test_sim.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl import EnvConfig, FPSEnv, FPSMultiAgentEnv  # noqa: E402
from fpsrl.bots import Bot  # noqa: E402
from fpsrl.sim import Action, MatchConfig, World, angles_to  # noqa: E402
from fpsrl.weapons import LOADOUT_BY_KEY, LOADOUT_KEYS  # noqa: E402

NO_FREEZE = {"freeze_time": 0.0}


def _world(team_size=1, seed=0, **kw):
    w = World(MatchConfig(team_size=team_size, seed=seed, mode_params=dict(NO_FREEZE), **kw))
    w.reset(seed)
    w.step({})  # leave the (zero length) freeze time
    return w


def _duel(loadout="rifle"):
    w = _world(1, 0, loadouts=[loadout, "rifle"])
    shooter, target = w.agents
    target.pos[:] = shooter.pos + np.array([4.0, 0.0, 0.0])
    target.prev_pos[:] = target.pos
    return w, shooter, target


def _aim(shooter, point):
    shooter.yaw, shooter.pitch = angles_to(shooter.eye(), point)


def test_agents_block_each_other():
    w, a, b = _duel()
    b.pos[:] = b.prev_pos[:] = a.pos + (0.15, 0.0, 0.0)
    w.step({})
    assert np.hypot(*(a.pos - b.pos)[:2]) >= 0.58  # overlapping bodies get pushed apart
    b.pos[:] = b.prev_pos[:] = a.pos + (1.0, 0.0, 0.0)
    b.vel[:] = 0.0
    a.yaw = 0.0
    for _ in range(90):
        w.step({a.id: Action(move_forward=1)})
    gap = float(np.hypot(*(a.pos - b.pos)[:2]))
    assert gap >= 0.55 and a.pos[0] < b.pos[0]  # walking into the other stops at their body


def test_headshot_kills():
    w, shooter, target = _duel("sniper")
    assert shooter.weapon.name == "awm"
    events = []
    for i in range(120):  # semi-auto: needs fresh trigger presses
        _aim(shooter, target.eye())
        events += w.step({shooter.id: Action(fire=i % 2 == 0)})
        if not target.alive:
            break
    kills = [e for e in events if e["type"] == "kill"]
    assert kills and kills[0]["attacker"] == shooter.id and kills[0]["headshot"]


def test_body_damage_and_ammo():
    w, shooter, target = _duel("rifle")
    mag0 = shooter.weapon.mag
    dmg = []
    for _ in range(40):
        mn, mx = target.hitboxes()
        _aim(shooter, (mn[1] + mx[1]) / 2)
        dmg += [e for e in w.step({shooter.id: Action(fire=True)}) if e["type"] == "damage"]
    assert dmg and all(e["attacker"] == shooter.id for e in dmg)
    assert shooter.weapon.mag < mag0


def test_movement_and_collision():
    w = _world(1, 1)
    a = w.agents[0]
    a.vel[:] = 0.0
    for _ in range(24):
        w.step({a.id: Action(move_forward=1.0)})
    assert np.linalg.norm(a.vel[:2]) > 4.0
    (x0, y0), (x1, y1) = w.map.bounds
    for k in range(900):  # run around; must never leave the map or end up inside a wall
        w.step({a.id: Action(move_forward=1.0, move_right=0.6, look_yaw=0.8 if k % 300 < 150 else -0.5)})
        assert not w._overlaps(a.pos, a.height).any(), a.pos
    assert x0 <= a.pos[0] <= x1 and y0 <= a.pos[1] <= y1


def test_freeze_time_blocks_movement_and_fire():
    w = World(MatchConfig(team_size=1, seed=0, mode_params={"freeze_time": 1.0}))
    w.reset(0)
    a = w.agents[0]
    p0, mag0 = a.pos.copy(), a.weapon.mag
    for _ in range(30):
        w.step({a.id: Action(move_forward=1.0, fire=True)})
    assert w.mode.phase == "freeze" and np.allclose(a.pos[:2], p0[:2]) and a.weapon.mag == mag0
    for _ in range(60):
        w.step({a.id: Action(move_forward=1.0)})
    assert w.mode.phase == "live" and not np.allclose(a.pos[:2], p0[:2])


def test_fixed_loadouts():
    w = World(MatchConfig(team_size=4, seed=3, loadouts=["sniper", "shotgun"]))
    assert w.agents[0].loadout == "sniper" and w.agents[0].weapon.name == "awm"
    assert w.agents[1].loadout == "shotgun" and w.agents[1].inv[1].name == "nova"
    for a in w.agents:
        primary = LOADOUT_BY_KEY[a.loadout][3]
        assert a.inv[1].name == primary and a.inv[2].name == "glock" and a.inv[3].name == "knife"
        assert {a.inv[s].name for s in (4, 5, 6, 7)} == {"frag", "smoke", "flash", "molotov"}
    drawn = set()
    for seed in range(12):  # unset loadouts are drawn at random per match
        w.reset(seed)
        drawn |= {a.loadout for a in w.agents[2:]}
    assert drawn == set(LOADOUT_KEYS)
    a = w.agents[2]
    assert w.mode.phase == "freeze"
    other = next(k for k in LOADOUT_KEYS if k != a.loadout)
    assert w.set_loadout(a, other) and a.inv[1].name == LOADOUT_BY_KEY[other][3]


def test_elimination_round_flow():
    w = _world(1, 0, loadouts=["rifle", "rifle"])
    w.mode.p.update(post_time=0.5, win_rounds=2, half_rounds=5)
    shooter, target = w.agents
    ends = []
    for _ in range(2):
        w.apply_damage(target, 500.0, shooter, "ak47")
        for _ in range(60):
            ends += [e for e in w.step({}) if e["type"] == "round_end"]
    assert [e["reason"] for e in ends] == ["elimination", "elimination"]
    assert w.done and w.winner == shooter.team and w.score == [2, 0]


def test_determinism():
    def run():
        w = World(MatchConfig(team_size=2, seed=7))
        w.reset(7)
        bots = [Bot(w, a.id, "hard", seed=a.id) for a in w.agents]
        for _ in range(900):
            w.step({b.aid: b.act() for b in bots})
        return np.stack([a.pos for a in w.agents]), list(w.team_kills)
    (p1, k1), (p2, k2) = run(), run()
    assert np.allclose(p1, p2) and k1 == k2


def test_bot_match_produces_kills():
    w = World(MatchConfig(team_size=3, seed=3))
    w.reset(3)
    bots = [Bot(w, a.id, "hard", seed=a.id) for a in w.agents]
    for _ in range(60 * 60):
        w.step({b.aid: b.act() for b in bots})
    assert sum(w.team_kills) > 0


def test_cook_frag_hits_self_and_team():
    from fpsrl.sim import Projectile, dir_from_angles
    from fpsrl.weapons import SLOT_FRAG

    w = _world(2, 1, friendly_fire=False)
    thrower, mate = w.agents[0], w.agents[1]
    w.switch_weapon(thrower, SLOT_FRAG)
    for _ in range(40):
        w.step({})
    mate.pos[:] = thrower.pos + np.array([0.9, 0.0, 0.0])
    mate.prev_pos[:] = mate.pos
    mate.vel[:] = 0.0
    ev = w.step({thrower.id: Action(fire=True)})
    assert thrower.throw_at >= 0 and thrower.cook_until > w.time
    assert not any(e["type"] == "throw" for e in ev)
    ev = []
    for _ in range(20):
        ev += w.step({})
    assert any(e["type"] == "throw" for e in ev)
    assert thrower.alive and w.projectiles
    assert w.projectiles[0].explode_at - w.time > 1.5

    w = _world(2, 2, friendly_fire=False)
    thrower, mate = w.agents[0], w.agents[1]
    w.switch_weapon(thrower, SLOT_FRAG)
    for _ in range(40):
        w.step({})
    mate.pos[:] = thrower.pos + np.array([0.9, 0.0, 0.0])
    mate.prev_pos[:] = mate.pos
    mate.vel[:] = 0.0
    hp = mate.hp
    for _ in range(200):
        w.step({thrower.id: Action(fire=True)})
        if not thrower.alive:
            break
    assert not thrower.alive and thrower.killer == thrower.id
    assert mate.hp < hp

    w, thrower, victim = _duel()
    mate_pos = thrower.pos + np.array([1.0, 0.0, 0.0])
    victim.pos[:] = mate_pos
    victim.prev_pos[:] = victim.pos
    victim.team = thrower.team
    for a in (thrower, victim):
        origin = a.eye() + dir_from_angles(a.yaw, 0.0) * 1.2
        a.yaw, a.pitch = angles_to(a.eye(), origin)
    origin = thrower.eye() + dir_from_angles(thrower.yaw, thrower.pitch) * 1.2
    victim.yaw, victim.pitch = angles_to(victim.eye(), origin)
    pr = Projectile(w._new_id(), "flash", thrower.id, thrower.team, origin, np.zeros(3), w.time, w.time)
    w._detonate(pr)
    assert thrower.flash_until > w.time and victim.flash_until > w.time

    thrower.hp = victim.hp = 100.0
    pr = Projectile(w._new_id(), "molotov", thrower.id, thrower.team,
                    thrower.pos + np.array([0.4, 0.0, 0.15]), np.zeros(3), w.time, w.time)
    w._detonate(pr)
    for _ in range(40):
        w.step({})
    assert thrower.hp < 100 and victim.hp < 100


def test_env_api():
    from gymnasium.utils.env_checker import check_env
    env = FPSEnv(team_size=2, seed=0)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert set(obs) == set(env.observation_space.spaces)
    assert obs["ownplayer.state.numeric"].shape == (110,)
    assert obs["environment.lidar_info.numeric"].shape == (93,)
    assert obs["nearby.thrown.numeric"].shape == (10, 8)
    assert obs["environment.camera.numeric"].shape == (8,)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
    env.close()

    flat = FPSEnv(EnvConfig(team_size=1, obs_mode="flat", action_type="multidiscrete"))
    obs, _ = flat.reset(seed=0)
    assert obs.shape == flat.observation_space.shape == (flat.obs_builder.size,)
    flat.step(flat.action_space.sample())
    flat.close()

    box = FPSEnv(EnvConfig(team_size=1, action_type="box", obs_mode="flat"))
    box.reset(seed=1)
    box.step(box.action_space.sample())
    box.close()


def test_first_person_rgb_is_still_available():
    rgb = FPSEnv(team_size=1, obs_mode="rgb", rgb_size=(32, 24))
    assert rgb.observation_space.shape == (24, 32, 3)
    rgb.close()

    both = FPSEnv(team_size=1, obs_mode="both", rgb_size=(32, 24), action_repeat=1)
    obs, _ = both.reset(seed=0)
    assert both.observation_space.contains(obs)
    assert obs["vector"]["ownplayer.state.numeric"].shape == (110,)
    assert obs["rgb"].shape == (24, 32, 3) and obs["rgb"].dtype == np.uint8
    assert int(obs["rgb"].max()) > 0
    both.close()


def test_look_bins_do_not_lock_onto_an_enemy():
    from fpsrl.envs import HEADS, decode_heads
    from fpsrl.sim import dir_from_angles, wrap180
    idle = {name: 0 for name in HEADS}
    held = decode_heads({**idle, "yaw": 17, "pitch": 5}, repeat=1)
    assert held.look_yaw == 0.0 and held.look_pitch == 0.0

    env = FPSEnv(team_size=2, action_repeat=1, teammate_bots=False, seed=1)
    env.reset(seed=1)
    w = env.world
    me = w.agents[env.learner]
    foes = [a for a in w.agents if a.team != me.team]
    fwd = dir_from_angles(me.yaw, 0.0)
    right = dir_from_angles(me.yaw + 90.0, 0.0)
    placed = []
    for dist in (2.5, 3.5, 5.0, 7.0):
        for side in (1.6, -1.8, 2.2, -2.4):
            pos = me.pos + fwd * dist + right * side
            pos = np.array(pos, dtype=np.float64)
            pos[2] = me.pos[2]
            aim = pos + np.array([0.0, 0.0, 1.2])
            if w.segment_clear(me.eye(), aim) and not w._overlaps(pos + (0, 0, 0.02), 1.8).any():
                placed.append(pos)
            if len(placed) == 2:
                break
        if len(placed) == 2:
            break
    assert len(placed) == 2, "no open ground in front of the spawn"
    for foe, pos in zip(foes, placed):
        foe.pos[:] = pos
        foe.prev_pos[:] = pos
        foe.alive = True
        foe.hp = 100.0
    me.pitch = 0.0
    w._vis_tick = -1
    env.state_obs.build(env.learner)
    assert len(env.state_obs.slots[env.learner]) >= 2
    action = dict(idle)
    action["yaw"] = 13
    action["pitch"] = 8
    action["target_enemy"] = 0
    y0, p0 = float(me.yaw), float(me.pitch)
    nearer = min(foes, key=lambda a: float(np.linalg.norm(a.pos - me.pos)))
    yaw_to, _ = angles_to(me.eye(), nearer.pos + np.array([0.0, 0.0, nearer.height * 0.62]))
    assert abs(wrap180(yaw_to - y0) - 15.0) > 5.0
    obs, r, term, trunc, info = env.step(action)
    assert abs(wrap180(me.yaw - y0) - 15.0) < 1e-3
    assert abs(me.pitch - p0 - 3.0) < 1e-3
    assert env.state_obs.targets[env.learner] is not None
    assert np.allclose(obs["target_enemy.state.numeric"][:2], obs["enemy_0.state.numeric"][:2])
    env.close()


def test_multiagent_api():
    env = FPSMultiAgentEnv(team_size=2, controlled=[0, 2])
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"agent_0", "agent_2"}
    assert env.team_of("agent_0") == 0 and env.team_of("agent_2") == 1
    obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
    assert set(rew) == {"agent_0", "agent_2"}


def test_segment_clear_matches_raycast():
    """LOS helper (chunked box path used by visibility()) must agree with the bullet raycast."""
    from fpsrl.geometry import raycast_boxes
    w = World(MatchConfig(team_size=1, seed=0))
    rng = np.random.default_rng(0)
    (x0, y0), (x1, y1) = w.map.bounds
    blocked = 0
    for _ in range(300):
        p0 = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), rng.uniform(0.2, 6.0)])
        p1 = p0 + np.array([*rng.uniform(-15, 15, 2), rng.uniform(-1, 1)])
        dist = float(np.linalg.norm(p1 - p0))
        t, _, _ = raycast_boxes(p0[None], ((p1 - p0) / dist)[None], dist, w.bmin, w.bmax)
        hit = bool(t[0] < dist - 1e-3)
        if abs(float(t[0]) - dist) < 1e-2 or float(t[0]) < 1e-3:
            continue  # grazing / starting inside a box: either answer is fine
        assert w.segment_clear(p0, p1) == (not hit)
        blocked += hit
    assert blocked > 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

"""Headless sanity tests. Run with `pytest tests` or `py tests/test_sim.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl import EnvConfig, FPSEnv, FPSMultiAgentEnv  # noqa: E402
from fpsrl.bots import Bot  # noqa: E402
from fpsrl.sim import Action, MatchConfig, World, angles_to  # noqa: E402


def _duel(primary="ak47"):
    w = World(MatchConfig(team_size=1, primaries=[primary, "ak47"], seed=0))
    w.reset(0)
    shooter, target = w.agents
    target.pos[:] = shooter.pos + np.array([4.0, 0.0, 0.0])
    target.prev_pos[:] = target.pos
    shooter.protect_until = target.protect_until = 0.0
    return w, shooter, target


def _aim(shooter, point):
    shooter.yaw, shooter.pitch = angles_to(shooter.eye(), point)


def test_headshot_kills():
    w, shooter, target = _duel("awm")
    events = []
    for i in range(120):  # semi-auto: needs fresh trigger presses
        _aim(shooter, target.eye())
        events += w.step({shooter.id: Action(fire=i % 2 == 0)})
        if not target.alive:
            break
    kills = [e for e in events if e["type"] == "kill"]
    assert kills and kills[0]["attacker"] == shooter.id and kills[0]["headshot"]


def test_body_damage_and_ammo():
    w, shooter, target = _duel("ak47")
    mag0 = shooter.weapon.mag
    dmg = []
    for _ in range(40):
        mn, mx = target.hitboxes()
        _aim(shooter, (mn[1] + mx[1]) / 2)
        dmg += [e for e in w.step({shooter.id: Action(fire=True)}) if e["type"] == "damage"]
    assert dmg and all(e["attacker"] == shooter.id for e in dmg)
    assert shooter.weapon.mag < mag0


def test_movement_and_collision():
    w = World(MatchConfig(team_size=1, seed=1))
    w.reset(1)
    a = w.agents[0]
    a.pos[:] = a.prev_pos[:] = (-36.4, -6.25, 1.2)  # spawn platform, facing the east parapet
    a.vel[:] = 0.0
    a.yaw = 0.0
    for _ in range(12):
        w.step({a.id: Action(move_forward=1.0)})
    assert np.linalg.norm(a.vel[:2]) > 4.0
    for _ in range(48):
        w.step({a.id: Action(move_forward=1.0)})
    assert abs(a.pos[0] - (-35.25 - 0.3)) < 0.02, a.pos
    for _ in range(600):  # run into the bulwark / containers; must never leave the map
        w.step({a.id: Action(move_forward=1.0, move_right=0.6)})
    (x0, y0), (x1, y1) = w.map.bounds
    assert x0 <= a.pos[0] <= x1 and y0 <= a.pos[1] <= y1


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


def test_env_api():
    from gymnasium.utils.env_checker import check_env
    env = FPSEnv(team_size=2, seed=0)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.isfinite(r)
    env.close()

    box = FPSEnv(EnvConfig(team_size=1, action_type="box"))
    box.reset(seed=1)
    box.step(box.action_space.sample())


def test_multiagent_api():
    env = FPSMultiAgentEnv(team_size=2, controlled=[0, 2])
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"agent_0", "agent_2"}
    assert env.team_of("agent_0") == 0 and env.team_of("agent_2") == 1
    obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
    assert set(rew) == {"agent_0", "agent_2"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

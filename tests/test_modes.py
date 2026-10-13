"""Weapons, lean, game modes, maps and env ids. Run with `pytest tests` or `py tests/test_modes.py`."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from fpsrl import ENV_IDS, EnvConfig, FPSEnv  # noqa: E402
from fpsrl.bots import Bot  # noqa: E402
from fpsrl.geometry import poly_center  # noqa: E402
from fpsrl.maps import MAPS  # noqa: E402
from fpsrl.modes.rounds import ATT, DEF  # noqa: E402
from fpsrl.net.protocol import apply_client_cmd  # noqa: E402
from fpsrl.operators import ATTACKERS, DEFENDERS  # noqa: E402
from fpsrl.sim import LEAN_DIST, Action, MatchConfig, World, angles_to  # noqa: E402
from fpsrl.weapons import PRIMARY_CHOICES, SECONDARY_CHOICES, SLOT_BOMB, SLOT_PRIMARY, WEAPONS  # noqa: E402


def _run(w, seconds, actions=None):
    events = []
    for _ in range(int(seconds * w.cfg.tick_rate)):
        events += w.step(actions() if callable(actions) else (actions or {}))
    return events


def _place(w, a, x, y, z):
    """Teleport onto the nav node nearest (x, y, z) so the agent stands on real floor."""
    nav = Bot(w, a.id).nav
    node = nav.nearest_cell((x, y, z))
    assert node >= 0, (x, y, z)
    a.pos[:] = a.prev_pos[:] = nav.nodes[node] + (0.0, 0.0, 0.02)
    a.vel[:] = 0.0


def _zone_spot(w, a, zone):
    cx, cy = poly_center(zone["poly"])
    _place(w, a, cx, cy, zone.get("z", zone.get("zmin", 0.0) + 0.5))


def _aim(a, point):
    a.yaw, a.pitch = angles_to(a.eye(), point)


def _duel():
    w = World(MatchConfig(team_size=1, seed=0))
    w.reset(0)
    shooter, target = w.agents
    target.pos[:] = target.prev_pos[:] = shooter.pos + np.array([4.0, 0.0, 0.0])
    shooter.protect_until = target.protect_until = 0.0
    return w, shooter, target


def test_every_gun_hits():
    for name in PRIMARY_CHOICES + SECONDARY_CHOICES:
        w, shooter, target = _duel()
        w.give(shooter, name, select=True)
        _run(w, 1.2)  # draw time
        assert shooter.weapon.name == name
        hits = []
        for i in range(90):
            mn, mx = target.hitboxes()
            _aim(shooter, (mn[1] + mx[1]) / 2)
            semi = not WEAPONS[name].auto
            hits += [e for e in w.step({shooter.id: Action(fire=(i % 8 == 0) if semi else True)})
                     if e["type"] == "damage" and e.get("weapon") == name]
            if hits:
                break
        assert hits, f"{name} never hit a target 4 m away"


def test_lean_moves_eye_sideways():
    for lean in (1, -1):
        w, a, _ = _duel()
        eye0 = a.eye()
        _run(w, 0.6, {a.id: Action(lean=lean)})
        side = float((a.eye() - eye0)[:2] @ a.right()[:2])
        assert abs(a.lean_off - lean * LEAN_DIST) < 0.05, a.lean_off
        assert side * lean > 0.3, side
        _run(w, 0.6)
        assert abs(a.lean_off) < 0.05


def test_domination_capture_and_score():
    w = World(MatchConfig(team_size=2, map_name="transport_ship", mode="dom", seed=1,
                          mode_params={"capture_time": 1.0, "score_limit": 3}))
    w.reset(1)
    m = w.mode
    a = w.agents[0]
    _zone_spot(w, a, m.points[0])
    ev = _run(w, 1.5)
    assert any(e["type"] == "point_captured" and e["point"] == 0 and e["team"] == a.team for e in ev)
    assert m.owner[0] == a.team
    _run(w, 3.5)
    assert w.done and w.winner == a.team and w.score[a.team] >= 3


def test_bomb_buy_plant_defuse():
    w = World(MatchConfig(team_size=2, map_name="desert2", mode="bomb", seed=2,
                          mode_params={"freeze_time": 0.2, "defuse_time": 1.0, "plant_time": 1.0}))
    w.reset(2)
    m = w.mode
    carrier = w.agents[m.bomb["carrier"]]
    defender = m.members(m.team_of_side(DEF))[0]
    assert m.phase == "freeze" and carrier.money == m.p["start_money"]

    carrier.money = 5000
    price = WEAPONS["m4a1"].price
    assert apply_client_cmd(w, carrier, {"t": "cmd", "c": "buy", "v": "m4a1"})
    assert carrier.money == 5000 - price and carrier.inv[SLOT_PRIMARY].name == "m4a1"
    assert not w.buy(carrier, "m4a1"), "bought a gun already owned"

    for a in w.agents:
        if a is not carrier and a is not defender:
            a.alive = False
    _run(w, 0.3)
    assert m.phase == "live"
    _zone_spot(w, carrier, m.sites[0])
    _run(w, 0.3)
    assert not w.buy(carrier, "ak47"), "bought outside the buy zone"
    assert m.site_at(carrier.pos) == 0
    ev = _run(w, 3.0, {carrier.id: Action(fire=True, switch=SLOT_BOMB)})
    assert any(e["type"] == "bomb_planted" for e in ev), [e["type"] for e in ev]
    assert m.phase == "planted"

    bx, by, bz = m.bomb["pos"]
    _place(w, defender, bx + 0.6, by, bz)
    ev = _run(w, 1.5, {defender.id: Action(use=True, crouch=True)})
    assert any(e["type"] == "bomb_defused" for e in ev), [e["type"] for e in ev]
    end = [e for e in ev if e["type"] == "round_end"]
    assert end and end[0]["side"] == DEF and end[0]["reason"] == "defuse"


def test_siege_operator_reinforce_breach():
    w = World(MatchConfig(team_size=2, map_name="bank", mode="r6", seed=3))
    w.reset(3)
    m = w.mode
    assert m.phase == "prep"
    att = m.members(m.team_of_side(ATT))[0]
    dfn = m.members(m.team_of_side(DEF))[0]
    assert att.frozen and not dfn.frozen

    assert not m.choose_operator(dfn, ATTACKERS[0]), "defender picked an attacker"
    assert apply_client_cmd(w, dfn, {"t": "cmd", "c": "operator", "v": DEFENDERS[-1]})
    assert dfn.operator == DEFENDERS[-1]

    target = Bot(w, dfn.id)._reinforce_target(m, dfn)
    assert target is not None, "no reinforceable wall near the active sites"
    pid, stand, look = target
    dfn.pos[:] = dfn.prev_pos[:] = stand + (0.0, 0.0, 0.02)
    dfn.vel[:] = 0.0
    left = m.reinforce_left

    def hold(use=True):
        _aim(dfn, look)
        return {dfn.id: Action(use=use)}
    # the stand point may be next to a doorway: the first press can barricade it instead
    ev = _run(w, 2.0, hold) + _run(w, 0.1, lambda: hold(False)) + _run(w, 3.5, hold)
    assert any(e["type"] == "reinforce" and pid in e["panels"] for e in ev), [e["type"] for e in ev]
    assert m.reinforce_left == left - 1

    panel = w.destruct.panels[pid]
    w.destruct.explode(panel.center, 0.3, "soft")
    assert panel.alive, "soft breach broke a reinforced wall"
    w.destruct.explode(panel.center, 0.3, "hard")
    assert not panel.alive


def test_maps_reachable():
    from map_check import check
    for name, (_zh, _en, modes) in MAPS.items():
        for mode in modes:
            problems = check(name, mode, verbose=False)
            assert not problems, f"{name}/{mode}: {problems[:5]}"


def test_env_ids():
    from gymnasium.utils.env_checker import check_env
    for env_id, kw in ENV_IDS.items():
        env = FPSEnv(EnvConfig(**kw))
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=0)
        for _ in range(30):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs) and np.isfinite(r), env_id
        assert "score" in info
        env.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

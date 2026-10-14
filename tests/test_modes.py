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
from fpsrl.maps import MAPS, resolve_map_mode  # noqa: E402
from fpsrl.modes.rounds import ATT, DEF  # noqa: E402
from fpsrl.net.protocol import apply_client_cmd  # noqa: E402
from fpsrl.operators import ATTACKERS, DEFENDERS  # noqa: E402
from fpsrl.sim import LEAN_DIST, Action, MatchConfig, World, angles_to  # noqa: E402
from fpsrl.weapons import PRIMARY_CHOICES, SECONDARY_CHOICES, SLOT_BOMB, SLOT_PISTOL, SLOT_PRIMARY, WEAPONS  # noqa: E402


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


def _shotgun_damage(dist):
    w, shooter, target = _duel()
    target.pos[:] = target.prev_pos[:] = shooter.pos + np.array([dist, 0.0, 0.0])
    target.hp = 1000.0
    w.give(shooter, "nova", select=True)
    _run(w, 1.0)
    mn, mx = target.hitboxes()
    _aim(shooter, (mn[1] + mx[1]) / 2)
    ev = w.step({shooter.id: Action(fire=True)})
    return [e for e in ev if e["type"] == "damage"], [e for e in ev if e["type"] == "shot"]


def test_shotgun_pellets_land_as_one_hit_and_fall_off():
    near, shots = _shotgun_damage(3.0)
    assert len(near) == 1 and near[0]["weapon"] == "nova"
    assert near[0]["amount"] > 100, near[0]["amount"]
    assert len(shots) == 1 and len(shots[0]["pellets"]) == WEAPONS["nova"].pellets - 1
    far, _ = _shotgun_damage(25.0)
    assert sum(e["amount"] for e in far) < near[0]["amount"] * 0.3


def test_tdm_loadout_presets():
    from fpsrl.weapons import LOADOUTS, SLOT_PISTOL
    w = World(MatchConfig(team_size=1, seed=0))
    w.reset(0)
    a = w.agents[0]
    for key, _zh, _en, primary, secondary in LOADOUTS:
        assert w.set_loadout(a, key)
        assert a.inv[SLOT_PRIMARY].name == primary and a.inv[SLOT_PISTOL].name == secondary
    assert not w.set_loadout(a, "nope")
    _run(w, 7.0)
    assert apply_client_cmd(w, a, {"t": "cmd", "c": "loadout", "v": "sniper"})
    assert a.inv[SLOT_PRIMARY].name == "nova"  # too late after spawning: applies next life
    assert a.loadout == {"primary": "awm", "secondary": "glock"}
    a.hp = 0.0
    w._kill(a, None, "frag", False)
    _run(w, w.cfg.respawn_time + 0.5)
    assert a.alive and a.inv[SLOT_PRIMARY].name == "awm"


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
    w = World(MatchConfig(team_size=2, map_name="hp_town", mode="dom", seed=1,
                          mode_params={"prep_time": 0.5, "tick": 0.05, "target": 10, "gap": 0.5}))
    w.reset(1)
    m = w.mode
    a = w.agents[0]
    _run(w, 0.3)
    assert m.phase == "prep" and m.active == -1 and a.frozen
    ev = _run(w, 0.3)
    first = m.active
    assert m.phase == "live" and first >= 0 and not a.frozen
    assert any(e["type"] == "point_active" and e["point"] == first for e in ev)
    _zone_spot(w, a, m.points[first])
    ev = _run(w, 0.7)
    assert any(e["type"] == "point_captured" and e["point"] == first and e["team"] == a.team
               and e["reason"] == "capture" for e in ev)
    assert m.phase == "gap" and m.wins[a.team] == 1
    ev = _run(w, 0.4)
    second = m.active
    assert m.phase == "live" and second not in (-1, first)
    _zone_spot(w, a, m.points[second])
    _run(w, 0.7)
    assert w.done and w.winner == a.team and w.score[a.team] == 2 and m.phase == "over"  # 2:0 ends it


def test_domination_contest_and_timeout():
    w = World(MatchConfig(team_size=2, map_name="hp_town", mode="dom", seed=3,
                          mode_params={"prep_time": 0.2, "tick": 0.05, "target": 100, "point_time": 1.0}))
    w.reset(3)
    m = w.mode
    a = next(x for x in w.agents if x.team == 0)
    b = next(x for x in w.agents if x.team == 1)
    _run(w, 0.3)
    first = m.active
    _zone_spot(w, a, m.points[first])
    _zone_spot(w, b, m.points[first])
    _run(w, 0.3)
    assert m.present == (1, 1) and m.cap == [0, 0]  # contested: nobody scores
    _zone_spot(w, b, m.points[(first + 1) % len(m.points)])
    _run(w, 0.25)
    assert m.cap[0] > 0 and m.cap[1] == 0
    ev = _run(w, 0.6)
    cap = [e for e in ev if e["type"] == "point_captured"]
    assert cap and cap[0]["team"] == 0 and cap[0]["reason"] == "time" and m.wins == [1, 0]
    assert m.phase == "gap"


def test_domination_regen_and_respawn_protect():
    w = World(MatchConfig(team_size=2, map_name="hp_town", mode="dom", seed=2,
                          mode_params={"prep_time": 0.2, "regen_delay": 0.3, "regen_rate": 40,
                                       "respawn_time": 0.4}))
    w.reset(2)
    a = w.agents[0]
    foe = next(x for x in w.agents if x.team != a.team)
    _run(w, 0.25)
    a.hp = 50.0
    a.last_damage_time = w.time
    _run(w, 0.2)
    assert a.hp == 50.0  # still inside the delay
    _run(w, 0.5)
    assert a.hp > 60.0
    w.apply_damage(a, 500, foe, "ak47")
    assert not a.alive
    _run(w, 0.55)
    assert a.alive and a.hp == 100
    assert a.protect_until > w.time + 2.0
    w.apply_damage(a, 40, foe, "ak47")
    assert a.hp == 100  # the 3s after respawn ignore damage


def test_bomb_buy_plant_defuse():
    w = World(MatchConfig(team_size=2, map_name="desert2", mode="bomb", seed=2,
                          mode_params={"freeze_time": 0.2, "defuse_time": 1.0, "plant_time": 1.0}))
    w.reset(2)
    m = w.mode
    carrier = w.agents[m.bomb["carrier"]]
    defender = m.members(m.team_of_side(DEF))[0]
    assert m.phase == "freeze" and carrier.money == m.p["start_money"]
    assert carrier.inv[SLOT_PISTOL].name == "glock"

    carrier.money = 5000
    price = WEAPONS["ak47"].price
    assert not w.buy(carrier, "m4a1"), "attacker bought a CT rifle"
    assert apply_client_cmd(w, carrier, {"t": "cmd", "c": "buy", "v": "ak47"})
    assert carrier.money == 5000 - price and carrier.inv[SLOT_PRIMARY].name == "ak47"
    assert not w.buy(carrier, "ak47"), "bought a gun already owned"
    assert w.buy(carrier, "deagle") and carrier.inv[SLOT_PISTOL].name == "deagle"
    assert carrier.money == 5000 - price - WEAPONS["deagle"].price
    assert w.buy(carrier, "glock")
    assert w.buy(carrier, "kevlar_helmet") and carrier.armor > 0 and carrier.helmet

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


def test_csgo_economy_and_armor():
    w = World(MatchConfig(team_size=1, map_name="desert2", mode="bomb", seed=7))
    w.reset(7)
    m = w.mode
    assert m.p["half_rounds"] == 15 and m.p["win_rounds"] == 16
    assert m.state()["buy_left"] == 20.0
    assert m.loss_streak == [1, 1]  # pistol loser starts at the $1900 tier
    for a in w.agents:
        a.money = 0
    m.on_round_end(0, m.side(0), "elimination")
    assert w.agents[0].money == 3250 and w.agents[1].money == 1900
    assert m.loss_streak == [0, 2]

    for a in w.agents:
        a.money = 0
    m.bomb = {"state": "defused"}
    m.on_round_end(1, m.side(1), "defuse")
    assert w.agents[1].money == 3500
    assert w.agents[0].money == 2200  # $1400 loss + $800 T-side plant
    assert m.loss_streak == [1, 1]

    buyer = w.agents[0]
    buyer.alive = True
    buyer.money = 2000
    m.phase = "freeze"
    assert w.buy(buyer, "kevlar") and buyer.money == 1350 and buyer.armor > 0 and not buyer.helmet
    assert w.buy(buyer, "kevlar_helmet") and buyer.money == 1000 and buyer.helmet
    assert not w.buy(buyer, "kevlar_helmet")
    buyer.money = 1000
    assert w.buy(buyer, "flash") and w.buy(buyer, "flash")
    assert buyer.inv[WEAPONS["flash"].slot].mag == 2 and not w.buy(buyer, "flash")

    victim = w.agents[1]
    victim.alive, victim.protect_until, victim.armor = True, 0.0, 0.2
    m.phase = "live"
    victim.hp, victim.helmet = 100.0, False
    w.apply_damage(victim, 50.0, buyer, "deagle", group=0)
    assert victim.hp == 50.0  # Kevlar alone does not protect the head
    victim.hp, victim.helmet = 100.0, True
    w.apply_damage(victim, 50.0, buyer, "deagle", group=0)
    assert victim.hp == 60.0


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

    target = Bot(w, dfn.id)._wall_target(m, dfn, m.active_sites, outside=False)
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


def test_elimination_wipe_and_draw():
    w = World(MatchConfig(team_size=1, map_name="desert2", mode="elim", seed=1,
                          mode_params={"freeze_time": 0.05, "post_time": 30.0}))
    w.reset(1)
    m = w.mode
    assert m.bomb is None and all(SLOT_BOMB not in a.inv for a in w.agents)
    att = m.members(m.team_of_side(0))[0]
    att.money = 3000
    assert w.buy(att, "ak47") and att.inv[SLOT_PRIMARY].name == "ak47"
    assert not w.buy(att, "defuser")
    for e in w.agents:
        if e is not att:
            e.alive = False
    ev = _run(w, 0.2)
    assert any(e["type"] == "round_end" and e["reason"] == "elimination" and e["side"] == 0 for e in ev)

    w = World(MatchConfig(team_size=1, map_name="desert2", mode="elim", seed=2,
                          mode_params={"freeze_time": 0.05, "round_time": 5.0, "post_time": 30.0}))
    w.reset(2)
    _run(w, 0.1)
    assert w.mode.phase == "live"
    w.mode.phase_end = w.time
    ev = _run(w, 0.1)
    end = [e for e in ev if e["type"] == "round_end"]
    assert end and end[0]["reason"] == "draw" and end[0]["team"] == -1 and w.mode.wins == [0, 0]


def test_mode_selects_its_map():
    assert resolve_map_mode("transport_ship", "r6") == ("bank", "r6")
    assert resolve_map_mode("transport_ship", "dom") == ("hp_town", "dom")
    assert resolve_map_mode("bank", "elim") == ("desert2", "elim")
    assert resolve_map_mode("desert2", "bomb") == ("desert2", "bomb")
    assert resolve_map_mode("hp_town", "tdm") == ("hp_town", "tdm")
    assert resolve_map_mode("pyramid", "bio") == ("pyramid", "bio")


def test_bio_infection_and_rounds():
    from fpsrl.weapons import SLOT_MELEE

    w = World(MatchConfig(team_size=1, map_name="pyramid", mode="bio", seed=1,
                          mode_params={"infect_delay": 0.05, "round_time": 30.0, "post_time": 30.0,
                                       "mother_count": 1, "wins_needed": 2}))
    w.reset(1)
    assert all(getattr(a, "role", "") == "human" for a in w.agents)
    _run(w, 0.12)
    mothers = [a for a in w.agents if a.role == "mother"]
    humans = [a for a in w.agents if a.role == "human"]
    assert w.mode.phase == "live" and len(mothers) == 1 and len(humans) == 1
    mother, human = mothers[0], humans[0]
    assert mother.slot == SLOT_MELEE and SLOT_PRIMARY not in mother.inv and mother.hp == 6000
    assert abs(mother.speed_mult - 1.15) < 1e-6
    bot = Bot(w, mother.id)
    bot.reset(1)
    chase = bot.act()
    assert chase.move_forward != 0.0 or chase.move_right != 0.0 or chase.fire
    human.pos[:] = mother.pos + np.array([0.8, 0.0, 0.0])
    human.prev_pos[:] = human.pos
    mother.yaw = 0.0
    w._melee(mother, mother.weapon.spec, heavy=False)
    ev = _run(w, 0.25)
    assert human.role == "zombie" and human.alive and human.deaths == 0
    assert human.slot == SLOT_MELEE and abs(human.speed_mult - 1.25) < 1e-6
    assert any(e["type"] == "infected" and e["agent"] == human.id for e in ev)
    assert any(e["type"] == "round_end" and e["reason"] == "infected" and e["team"] == 1 for e in ev)
    assert w.mode.wins == [0, 1]
    fallen = w.agents[0]
    fallen.alive = False
    fallen.respawn_at = w.time
    _run(w, 0.3)
    assert not fallen.alive

    w = World(MatchConfig(team_size=1, map_name="transport_ship", mode="bio", seed=2,
                          mode_params={"infect_delay": 10.0, "round_time": 0.2, "post_time": 30.0}))
    w.reset(2)
    assert all(a.role == "human" for a in w.agents)
    ev = _run(w, 0.35)
    assert any(e["type"] == "round_end" and e["reason"] == "time" and e["team"] == 0 for e in ev)
    assert w.mode.wins == [1, 0]


def test_bio_knockback():
    w = World(MatchConfig(team_size=2, map_name="pyramid", mode="bio", seed=3,
                          mode_params={"infect_delay": 0.05, "round_time": 60.0, "mother_count": 1}))
    w.reset(3)
    _run(w, 0.12)
    assert w.mode.phase == "live"
    humans = [a for a in w.agents if a.role == "human"]
    shooter, other = humans[0], humans[1]
    zombie = humans[2]
    w.mode._turn(zombie, mother=False)
    mother = next(a for a in w.agents if a.role == "mother")
    d = np.array([0.0, 1.0, 0.0])

    def shove(victim, weapon, amount=40.0):
        victim.vel[:] = 0.0
        victim.on_ground = True
        w.apply_damage(victim, amount, shooter, weapon, direction=d)
        return victim.vel.copy()

    v = shove(zombie, "ak47")
    assert v[1] > 1.0 and abs(v[0]) < 1e-6 and zombie.knock_until > w.time
    v_shotgun = shove(zombie, "nova")
    assert v_shotgun[1] > v[1] * 1.4 and v_shotgun[2] > 0.0  # heavy hits also lift a little
    v_mother = shove(mother, "ak47")
    assert abs(v_mother[1] - v[1] * 0.5) < 1e-6
    big = shove(zombie, "nova", 400.0)
    assert abs(np.hypot(big[0], big[1]) - w.mode.p["knockback_max"]) < 1e-6
    # the shove carries: a zombie slides several metres before friction wins
    start = zombie.pos.copy()
    zombie.vel[:] = 0.0
    w.apply_damage(zombie, 230.0, shooter, "nova", direction=d)
    _run(w, 0.8)
    assert np.hypot(*(zombie.pos - start)[:2]) > 2.0
    assert shove(other, "ak47")[1] == 0.0  # humans are not shoved
    zombie.vel[:] = 0.0
    w.apply_damage(zombie, 40.0, shooter, "knife", direction=d, kind="melee")
    assert zombie.vel[1] == 0.0


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

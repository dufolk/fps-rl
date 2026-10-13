"""Destruction, gadgets, layered navigation and the Siege round flow on the bank map.

Run with `pytest tests` or `py tests/test_siege.py`.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl.bots import Bot, NavGrid  # noqa: E402
from fpsrl.geometry import poly_center  # noqa: E402
from fpsrl.modes.rounds import ATT, DEF  # noqa: E402
from fpsrl.sim import Action, MatchConfig, World, angles_to  # noqa: E402
from fpsrl.weapons import SLOT_GADGET, SLOT_PRIMARY  # noqa: E402

# basement lockers | CCTV drywall at x = -6 (door at y = -10)
WALL_X, WALL_Y, ZB = -6.0, -14.0, -4.0


def _place(w, a, pos, yaw=0.0):
    a.pos[:] = a.prev_pos[:] = pos
    a.vel[:] = 0.0
    a.yaw, a.pitch = yaw, 0.0
    a.protect_until = 0.0
    a.frozen = False


def _bank(mode="tdm", team_size=1, **params):
    w = World(MatchConfig(team_size=team_size, map_name="bank", mode=mode, seed=0, mode_params=params,
                          primaries=["ak47"] * (2 * team_size)))
    w.reset(0)
    return w


def _wall_panel(w, x=WALL_X, y=WALL_Y, z=ZB + 1.2):
    d = w.destruct
    i = int(np.argmin(np.linalg.norm(d.centers - np.array([x, y, z]), axis=1)))
    return d.panels[i]


def _hold(w, aid, act, seconds):
    events = []
    for _ in range(int(seconds / w.dt) + 2):
        events += w.step({aid: act})
    return events


def test_bullets_pierce_and_break_soft_wall():
    w = _bank()
    shooter, target = w.agents
    _place(w, shooter, (WALL_X - 2.0, WALL_Y, ZB))
    _place(w, target, (WALL_X + 1.5, WALL_Y, ZB), 180.0)
    p = _wall_panel(w, z=ZB + 1.1)
    hp0 = p.hp
    events = []
    for _ in range(240):
        mn, mx = target.hitboxes()
        shooter.yaw, shooter.pitch = angles_to(shooter.eye(), (mn[1] + mx[1]) / 2)
        shooter.punch[:] = 0.0
        events += w.step({shooter.id: Action(fire=True, reload=shooter.weapon.mag == 0)})
        shooter.inv[SLOT_PRIMARY].reserve = 90
    assert any(e["type"] == "damage" and e["victim"] == target.id for e in events), "no wall-bang"
    assert p.hp < hp0 or not p.alive
    assert any(e["type"] == "wall_break" for e in events), "panel never broke"
    assert len(w.smin) == len(w.static_boxes) + len(w.destruct.active_boxes())


def test_reinforced_wall_only_yields_to_thermite():
    w = _bank()
    p = _wall_panel(w)
    done = w.destruct.reinforce_around(p.id)
    assert p.id in done and len(done) >= 4
    box = p.box
    assert not w.soft_box(box)  # bullets stop
    w.destruct.explode(p.center, 1.0, "soft")
    assert p.alive
    w.destruct.damage_panel(p.id, 1e6, "melee")
    assert p.alive
    w.destruct.explode(p.center, 1.0, "hard")
    assert not p.alive
    w.clear_round()
    assert p.alive and not p.reinforced


def test_nav_follows_destruction():
    w = _bank()
    nav = NavGrid.for_world(w)

    def length(path):
        return sum(float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1))

    a, b = np.array([WALL_X - 1.5, WALL_Y, ZB]), np.array([WALL_X + 1.5, WALL_Y, ZB])
    around = length(nav.find_path(a, b))
    assert around > 6.0
    w.destruct.explode((WALL_X, WALL_Y, ZB + 1.0), 1.3, "soft")
    through = nav.find_path(a, b)
    assert through and length(through) < 3.6
    # floor hatch in the 1F open area: drop straight into the basement once it is blown
    top, below = np.array([-15.0, -8.8, 0.0]), np.array([-15.0, -10.0, ZB])
    stairs = length(nav.find_path(top, below))
    w.destruct.explode((-15.0, -10.0, -0.15), 1.2, "soft", floors=True)
    drop = nav.find_path(top, below)
    assert drop and length(drop) < stairs * 0.5
    w.clear_round()
    assert length(nav.find_path(a, b)) > 6.0


def test_all_maps_reachable():
    from tools.map_check import check
    from fpsrl.maps import MAPS
    for name in MAPS:
        assert not check(name, verbose=False), name


def _siege(**params):
    p = {"prep_time": 1.0, "post_time": 0.5, **params}
    w = World(MatchConfig(team_size=1, map_name="bank", mode="r6", seed=3, mode_params=p))
    w.reset(3)
    return w, w.mode


def test_siege_round_flow():
    w, m = _siege()
    att, dfn = w.agents
    assert m.phase == "prep" and m.side(att.team) == ATT and m.side(dfn.team) == DEF
    assert att.frozen and not dfn.frozen and len(m.active_sites) == 2
    assert att.operator and dfn.operator and SLOT_GADGET in att.inv and SLOT_GADGET in dfn.inv
    _hold(w, att.id, Action(), 1.1)
    assert m.phase == "live" and not att.frozen
    zone = m.sites[m.active_sites[0]]
    cx, cy = poly_center(zone["poly"])
    _place(w, att, (cx, cy, zone["z"] + 0.02))
    _place(w, dfn, (cx + 2.0, cy, zone["z"] + 0.02))
    for _ in range(5):
        w.step({})
    ev = _hold(w, att.id, Action(use=True), m.p["plant_time"])
    assert m.phase == "planted" and any(e["type"] == "bomb_planted" for e in ev)
    _place(w, dfn, (att.pos[0] + 0.6, att.pos[1], att.pos[2]))
    ev = []
    for _ in range(int(m.p["defuse_time"] / w.dt) + 4):
        ev += w.step({dfn.id: Action(use=True)})
        if m.phase == "over":
            break
    assert any(e["type"] == "round_end" and e["reason"] == "defuse" for e in ev)
    assert m.wins[dfn.team] == 1
    # overtime at 3:3, swap every 3 rounds
    m.wins = [2, 3]
    m.end_round(m.side(att.team), "elimination")
    assert m.wins == [3, 3] and m.win_target == 5 and not w.done
    m.round_no = 3
    m.next_round()
    assert m.swapped == 1 and m.side(att.team) == DEF and m.phase == "prep"


def test_siege_defuser_completes():
    w, m = _siege(bomb_time=2.0)
    att, dfn = w.agents
    _hold(w, att.id, Action(), 1.1)
    zone = m.sites[m.active_sites[1]]
    cx, cy = poly_center(zone["poly"])
    _place(w, att, (cx, cy, zone["z"] + 0.02))
    _place(w, dfn, (cx - 30.0, cy, 0.02))
    _hold(w, att.id, Action(use=True), m.p["plant_time"])
    ev = _hold(w, att.id, Action(), 2.2)
    assert any(e["type"] == "round_end" and e["team"] == att.team and e["reason"] == "defuser" for e in ev)


def test_drone_spot_and_jammer():
    w, m = _siege(prep_time=30.0)
    att, dfn = w.agents
    w.step({att.id: Action(gadget=True)})
    assert att.view_mode == 1
    drone = w.gadgets.items[att.view_ref]
    start = drone["pos"].copy()
    _hold(w, att.id, Action(move_forward=1.0), 1.0)
    assert np.linalg.norm(drone["pos"][:2] - start[:2]) > 1.0, "drone did not drive"
    # enemy in front of the drone gets spotted for the team
    d = np.array([np.cos(np.radians(drone["yaw"])), np.sin(np.radians(drone["yaw"])), 0.0])
    _place(w, dfn, drone["pos"] + d * 2.5 - np.array([0, 0, drone["pos"][2] - att.pos[2]]))
    drone["pitch"] = 15.0
    w.step({att.id: Action(fire=True)})
    assert dfn.spotted_until > w.time
    # a Mute jammer next to the drone stops it
    w.gadgets._add("jammer", dfn.id, dfn.team, drone["pos"] + [0.5, 0.0, 0.0])
    p0 = drone["pos"].copy()
    _hold(w, att.id, Action(move_forward=1.0), 0.5)
    assert np.linalg.norm(drone["pos"][:2] - p0[:2]) < 1e-6
    w.step({att.id: Action(gadget=True)})
    assert att.view_mode == 0


def test_defender_camera_view():
    w, m = _siege(prep_time=30.0)
    dfn = w.agents[1]
    w.step({dfn.id: Action(gadget=True)})
    assert dfn.view_mode == 2
    cam = w.gadgets.items[dfn.view_ref]
    assert cam["kind"] == "camera" and cam["team"] == dfn.team
    first = dfn.view_ref
    w.step({dfn.id: Action()})
    w.step({dfn.id: Action(alt=True)})
    assert dfn.view_ref != first  # cycle to the next camera
    w.gadgets.shoot(dfn.view_ref, 50.0)
    assert dfn.view_mode == 0  # destroyed camera kicks the viewer out


def test_barricade_blocks_and_breaks():
    w, m = _siege(prep_time=30.0)
    dfn = w.agents[1]
    slot = w.map.objectives["slots"][0]
    mn, mx = np.array(slot["mn"]), np.array(slot["mx"])
    c = (mn + mx) / 2
    axis = 0 if mx[0] - mn[0] < mx[1] - mn[1] else 1
    stand = c.copy()
    stand[axis] -= 0.8
    stand[2] = mn[2] + 0.02
    yaw = 0.0 if axis == 0 else 90.0
    _place(w, dfn, stand, yaw)
    w.step({})
    _hold(w, dfn.id, Action(use=True), 1.6)
    bars = [it for it in w.gadgets.items.values() if it["kind"] == "barricade"]
    assert len(bars) == 1
    probe = c.copy()
    probe[2] = mn[2] + 0.1
    assert w._overlaps(probe, 1.0).any()
    ev = []
    for i in range(240):
        ev += w.step({dfn.id: Action(switch=3, fire=i % 2 == 0)})
        if not any(it["kind"] == "barricade" for it in w.gadgets.items.values()):
            break
    assert not any(it["kind"] == "barricade" for it in w.gadgets.items.values()), "knife did not break it"
    assert not w._overlaps(probe, 1.0).any()


def test_reinforce_during_prep():
    w, m = _siege(prep_time=30.0)
    dfn = w.agents[1]
    left = m.reinforce_left
    _place(w, dfn, (WALL_X + 0.6, WALL_Y, ZB), 180.0)
    w.step({})
    _hold(w, dfn.id, Action(use=True), 3.1)
    assert m.reinforce_left == left - 1
    assert _wall_panel(w).reinforced


def test_siege_bots_play_a_round():
    w, m = _siege(prep_time=8.0, round_time=60.0)
    bots = [Bot(w, a.id, "hard", seed=a.id) for a in w.agents]
    for _ in range(60 * 75):
        w.step({b.aid: b.act() for b in bots})
        if sum(m.wins):
            break
    assert sum(m.wins) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

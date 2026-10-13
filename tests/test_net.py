"""Headless LAN protocol tests (loopback sockets, no rendering)."""
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl.bots import Bot  # noqa: E402
from fpsrl.net import ConnectError, NetClient, NetServer  # noqa: E402
from fpsrl.net.protocol import (agent_state, apply_agent, apply_client_cmd, apply_world, decode_input,  # noqa: E402
                                encode_input, input_to_action, world_state)
from fpsrl.sim import Action, MatchConfig, World  # noqa: E402


def _wait(pred, timeout=3.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        r = pred()
        if r:
            return r
        time.sleep(0.01)
    raise AssertionError("timed out")


def test_input_roundtrip():
    w = World(MatchConfig(team_size=1, seed=1))
    a = w.agents[0]
    act = Action(move_forward=1.0, move_right=-0.5, fire=True, crouch=True, switch=7, lean=-1, use=True)
    inp = decode_input(encode_input(7, act, (a.yaw + 30) % 360, a.pitch - 10, 1.5))
    back = input_to_action(inp, a)
    assert inp["seq"] == 7 and back.fire and back.crouch and not back.jump and back.switch == 7
    assert back.lean == -1 and back.use and not back.gadget
    assert abs(back.look_yaw - 30) < 1e-2 and abs(back.look_pitch + 10) < 1e-2
    bad = decode_input([1, 9, -9, 1000, 500, 0, 99, 0, 5, 1000, -1000])
    assert bad["mf"] == 1.0 and bad["mr"] == -1.0 and bad["pitch"] == 89.0 and bad["sw"] == 0
    assert bad["lean"] == 1 and bad["vy"] == 90.0 and bad["vp"] == -90.0


def test_state_mirror():
    w = World(MatchConfig(team_size=2, seed=3))
    for _ in range(30):
        w.step({0: Action(move_forward=1.0, look_yaw=2.0)})
    m = World(MatchConfig(team_size=2, seed=99))
    st = world_state(w)
    apply_world(m, st, own=0)
    apply_agent(m.agents[0], st["agents"][0])
    for a, b in zip(w.agents, m.agents):
        assert agent_state(a)[2:] == agent_state(b)[2:]
    assert (abs(m.agents[0].pos - w.agents[0].pos) < 1e-3).all()


def test_server_client_session():
    world = World(MatchConfig(team_size=2, seed=5, kill_limit=0, time_limit=0))
    server = NetServer("test-room", port=0)
    try:
        port = server.port
        assert port != 0
        joined = []

        def pump():
            for kind, conn, msg in server.poll():
                if kind == "join":
                    conn.aid = 2
                    joined.append(conn)
                    conn.send({"t": "welcome", "aid": 2, "room": server.room, "w": world_state(world)})
            return joined
        # NetClient blocks on the welcome in its constructor, so the host side runs on a thread
        threading.Thread(target=lambda: _wait(pump), daemon=True).start()
        client = NetClient("127.0.0.1", port, "Tester", {"ak47": "neon"}, team=1, timeout=2.0)
        assert client.welcome["aid"] == 2
        conn = joined[0]
        assert conn.name == "Tester" and conn.team_pref == 1 and conn.skins == {"ak47": "neon"}

        me = world.agents[2]
        start = me.pos.copy()
        yaw = (me.yaw + 90.0) % 360.0
        for seq in range(1, 61):
            client.send({"t": "in", "i": [encode_input(seq, Action(move_forward=1.0), yaw, 0.0, world.time)]})
        _wait(lambda: len(conn.inputs) >= 60)
        while conn.inputs:
            inp = conn.inputs.popleft()
            conn.ack = inp["seq"]
            world.step({2: input_to_action(inp, world.agents[2])})
            server.broadcast({"t": "s", "w": world_state(world), "ev": [], "ack": [[2, conn.ack]]})
        assert abs(((me.yaw - yaw + 180) % 360) - 180) < 0.01
        assert ((me.pos - start)[:2] ** 2).sum() ** 0.5 > 2.0, "server did not apply client movement"

        snaps = []
        _wait(lambda: snaps.extend(m for m in client.poll() if m["t"] == "s") or len(snaps) >= 60)
        assert snaps[-1]["ack"] == [[2, 60]]
        mirror = World(MatchConfig(team_size=2, seed=5))
        apply_world(mirror, snaps[-1]["w"], own=2)
        assert (abs(mirror.agents[2].pos - me.pos) < 1e-3).all()

        client.send({"t": "cmd", "c": "primary", "v": "m4a1"})
        client.send({"t": "cmd", "c": "buy", "v": "m4a1"})  # tdm has no shop
        cmds = []
        _wait(lambda: cmds.extend(m for k, c, m in server.poll() if k == "cmd" and c is conn) or len(cmds) >= 2)
        assert [apply_client_cmd(world, me, m) for m in cmds] == [True, False]
        assert me.primary == "m4a1"

        # room full -> the second client gets a readable reason instead of a dead socket
        def pump_full():
            for kind, c, _ in server.poll():
                if kind == "join":
                    c.kick("房间已满 / room is full")
                    return True
        stop = threading.Event()
        t = threading.Thread(target=lambda: _wait(lambda: pump_full() or stop.is_set()), daemon=True)
        t.start()
        try:
            NetClient("127.0.0.1", port, "Late", timeout=2.0)
            raise AssertionError("expected ConnectError")
        except ConnectError as exc:
            assert "full" in str(exc)
        stop.set()

        # a second host on the same port must fail loudly
        try:
            NetServer("dup", port=port).close()
            raise AssertionError("port was shared")
        except OSError:
            pass

        server.close()
        bye = _wait(lambda: [m for m in client.poll() if m["t"] == "bye"])
        assert "host closed" in bye[0]["reason"]
        client.close()
    finally:
        server.close()


def test_discovery_reply():
    from fpsrl.net.protocol import DISCOVER_MAGIC, DISCOVERY_PORT
    server = NetServer("disc-room", port=0, info=lambda: {"players": 3})
    try:
        if server.udp is None:
            print("  (discovery port busy, skipped)")
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.sendto(DISCOVER_MAGIC, ("127.0.0.1", DISCOVERY_PORT))
        import json
        reply = json.loads(s.recvfrom(4096)[0])
        s.close()
        assert reply["room"] == "disc-room" and reply["port"] == server.port and reply["players"] == 3
    finally:
        server.close()


class _Room:
    """One loopback host + client, driven the same way a LAN match is: inputs and cmds go to the
    host, snapshots rebuild a mirror world."""

    def __init__(self, mode, map_name, mode_params=None, team=0, team_size=1, seed=1):
        from fpsrl.maps import resolve_map_mode
        self.world = World(MatchConfig(team_size=team_size, map_name=map_name, mode=mode, seed=seed,
                                       kill_limit=0, time_limit=0, mode_params=dict(mode_params or {})))
        self.server = NetServer(f"mode-{mode}", port=0)
        self.aid = team * team_size
        self.seq = 0
        self.sent = 0
        self.snaps = []
        self.conn = None
        self.client = None
        self.mirror = None
        try:
            joined = []

            def pump():
                for kind, conn, _msg in self.server.poll():
                    if kind == "join":
                        conn.aid = self.aid
                        joined.append(conn)
                        cfg = self.world.cfg
                        conn.send({"t": "welcome", "aid": self.aid, "room": self.server.room,
                                   "cfg": {"team_size": cfg.team_size, "kill_limit": cfg.kill_limit,
                                           "time_limit": cfg.time_limit, "respawn": cfg.respawn,
                                           "friendly_fire": cfg.friendly_fire, "map_name": cfg.map_name,
                                           "mode": cfg.mode, "mode_params": cfg.mode_params,
                                           "tick_rate": cfg.tick_rate,
                                           "primaries": [a.primary for a in self.world.agents]},
                                   "w": world_state(self.world), "roster": []})
                return joined

            threading.Thread(target=lambda: _wait(pump), daemon=True).start()
            self.client = NetClient("127.0.0.1", self.server.port, "Guest", team=team, timeout=3.0)
            self.conn = joined[0]
            wel = self.client.welcome
            c = wel["cfg"]
            got_map, got_mode = resolve_map_mode(c["map_name"], c["mode"])
            assert (got_map, got_mode) == (map_name, mode), (got_map, got_mode)
            self.mirror = World(MatchConfig(team_size=c["team_size"], kill_limit=c["kill_limit"],
                                            time_limit=c["time_limit"], respawn=c["respawn"],
                                            friendly_fire=c["friendly_fire"], map_name=got_map, mode=got_mode,
                                            mode_params=c["mode_params"], tick_rate=c["tick_rate"],
                                            primaries=c["primaries"], seed=seed + 99))
            self._apply(wel["w"])
        except Exception:
            self.close()
            raise

    def close(self):
        if self.client is not None:
            self.client.close()
        self.server.close()

    def _apply(self, st):
        apply_world(self.mirror, st, own=self.aid)
        for a, s in zip(self.mirror.agents, st["agents"]):
            apply_agent(a, s)

    def _cmds(self):
        done = False
        for kind, conn, msg in self.server.poll():
            if kind == "cmd" and conn.aid is not None:
                self.last_cmd = apply_client_cmd(self.world, self.world.agents[conn.aid], msg)
                done = True
        return done

    def command(self, c, v, p=None):
        msg = {"t": "cmd", "c": c, "v": v}
        if p is not None:
            msg["p"] = p
        self.client.send(msg)
        _wait(self._cmds)
        self.tick(1)

    def play(self, act, n=1):
        me = self.world.agents[self.aid]
        batch = []
        for _ in range(n):
            self.seq += 1
            batch.append(encode_input(self.seq, act, me.yaw, me.pitch, self.world.time))
        self.client.send({"t": "in", "i": batch})
        _wait(lambda: len(self.conn.inputs) >= n)
        self.tick(n)

    def tick(self, n=1):
        w = self.world
        for _ in range(n):
            self._cmds()
            acts = {}
            if self.conn.inputs:
                inp = self.conn.inputs.popleft()
                self.conn.ack = inp["seq"]
                acts[self.aid] = input_to_action(inp, w.agents[self.aid])
            w.step(acts)
            self.sent += 1
            self.server.broadcast({"t": "s", "w": world_state(w), "ev": [], "ack": [[self.aid, self.conn.ack]]})
        def ready():
            self.snaps.extend(m for m in self.client.poll() if m["t"] == "s")
            return len(self.snaps) >= self.sent
        _wait(ready, timeout=4.0)
        self._apply(self.snaps[-1]["w"])

    def same(self):
        h, m = self.world, self.mirror
        assert h.mode.name == m.mode.name
        assert h.mode.state() == m.mode.state(), (h.mode.state(), m.mode.state())
        assert h.gadgets.state() == m.gadgets.state()
        assert h.team_kills == m.team_kills
        assert h.done == m.done and h.winner == m.winner
        for a, b in zip(h.agents, m.agents):
            assert a.alive == b.alive and abs(a.hp - b.hp) < 1e-6
            assert a.money == b.money and a.frozen == b.frozen and a.operator == b.operator
            assert {k: a.inv[k].name for k in a.inv} == {k: b.inv[k].name for k in b.inv}
            assert (abs(a.pos - b.pos) < 1e-2).all()


def test_lan_tdm():
    room = _Room("tdm", "transport_ship")
    try:
        room.same()
        me = room.world.agents[room.aid]
        start = me.pos.copy()
        me.yaw = (me.yaw + 90.0) % 360.0
        room.play(Action(move_forward=1.0), 40)
        assert ((me.pos - start)[:2] ** 2).sum() ** 0.5 > 2.0
        room.command("primary", "m4a1")
        assert me.primary == "m4a1" and me.inv[1].name == "m4a1"
        room.command("buy", "deagle")
        assert room.last_cmd is False
        room.same()
    finally:
        room.close()


def test_lan_dom():
    room = _Room("dom", "hp_town", {"prep_time": 0.2, "tick": 0.05})
    try:
        room.tick(20)
        room.same()
        m = room.world.mode
        assert m.phase == "live" and m.active >= 0
        me = room.world.agents[room.aid]
        c, z = m.centers[m.active], m.points[m.active]["z"]
        nav = Bot(room.world, me.id).nav
        me.pos[:] = me.prev_pos[:] = nav.nodes[nav.nearest_cell((c[0], c[1], z))] + (0.0, 0.0, 0.02)
        me.vel[:] = 0.0
        room.tick(40)
        assert m.cap[me.team] > 0
        room.same()
    finally:
        room.close()


def test_lan_bomb():
    room = _Room("bomb", "desert2", {"freeze_time": 1.0, "plant_time": 0.05, "buy_time": 30.0,
                                     "start_money": 800}, team=0)
    try:
        assert room.world.mode.bomb["state"] == "carried"
        assert room.world.mode.bomb["carrier"] == room.aid
        room.same()
        room.command("buy", "deagle")
        me = room.world.agents[room.aid]
        assert room.last_cmd is True and me.money == 100 and me.inv[2].name == "deagle"
        start = me.pos.copy()
        room.play(Action(move_forward=1.0), 20)  # still in freeze: the host must ignore the walk
        assert ((me.pos - start)[:2] ** 2).sum() ** 0.5 < 0.3
        room.tick(50)
        assert room.world.mode.phase == "live"
        site = room.world.mode.active_sites[0]
        poly = room.world.mode.sites[site]["poly"]
        center = room.world.mode.site_centers[site]
        spots = [center] + [((center[0] + p[0]) * 0.5, (center[1] + p[1]) * 0.5) for p in poly]
        z = room.world.mode.sites[site]["z"] + 0.05
        planted_at = None
        for x, y in spots:
            me.pos[:] = (x, y, z)
            me.vel[:] = 0.0
            me.on_ground = True
            room.tick(6)
            if room.world.mode.site_at(me.pos) == site and me.on_ground and me.alive:
                planted_at = (x, y)
                break
        assert planted_at is not None, "could not stand on a site"
        room.play(Action(switch=7), 1)
        room.play(Action(fire=True), 45)
        assert room.world.mode.phase == "planted", room.world.mode.phase
        assert room.world.mode.bomb["state"] == "planted"
        room.same()
    finally:
        room.close()


def test_lan_elim():
    room = _Room("elim", "desert2", {"freeze_time": 0.05, "post_time": 0.05, "round_time": 30.0,
                                     "start_money": 800})
    try:
        assert room.world.mode.bomb is None
        assert all(ws.name != "c4" for a in room.world.agents for ws in a.inv.values())
        room.same()
        room.command("buy", "defuser")
        assert room.last_cmd is False
        room.command("buy", "deagle")
        me = room.world.agents[room.aid]
        assert room.last_cmd is True and me.inv[2].name == "deagle"
        room.tick(6)  # freeze ignores damage
        assert room.world.mode.phase == "live"
        foe = room.world.agents[1 - room.aid]
        room.world.apply_damage(foe, 200.0, me, "deagle")
        room.tick(2)
        assert room.world.mode.phase == "over"
        assert room.world.mode.last_result["reason"] == "elimination"
        assert room.world.mode.wins == [1, 0]
        room.tick(12)  # post + freeze, then the next round is live
        assert room.world.mode.phase == "live" and room.world.mode.round_no == 2
        room.world.mode.phase_end = room.world.time
        room.tick(1)
        assert room.world.mode.last_result["reason"] == "draw"
        assert room.world.mode.last_result["team"] == -1 and room.world.mode.wins == [1, 0]
        room.same()
    finally:
        room.close()


def test_lan_r6():
    room = _Room("r6", "bank", {"prep_time": 30.0}, team=0)
    try:
        assert room.world.mode.phase == "prep"
        assert room.world.mode.bomb["state"] == "carried"
        assert room.world.agents[room.aid].frozen
        room.same()
        me = room.world.agents[room.aid]
        start = me.pos.copy()
        room.play(Action(move_forward=1.0), 30)
        assert ((me.pos - start)[:2] ** 2).sum() ** 0.5 < 0.3
        room.command("operator", "thermite", "famas")
        assert room.last_cmd is True
        assert me.operator == "thermite" and me.inv[1].name == "famas"
        assert any(ws.name == "thermite" for ws in me.inv.values())
        assert room.world.mode.ops[me.id]["attack"] == "thermite"
        assert room.world.mode.reinforce_left == 2
        room.command("buy", "deagle")
        assert room.last_cmd is False
        room.same()
    finally:
        room.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

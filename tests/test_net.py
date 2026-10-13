"""Headless LAN protocol tests (loopback sockets, no rendering)."""
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

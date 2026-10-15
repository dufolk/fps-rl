"""Headless LAN protocol tests (loopback sockets, no rendering)."""
import collections
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


def _conn():
    from fpsrl.net.server import ClientConn
    c = ClientConn.__new__(ClientConn)
    c.inputs, c.ack, c.debt, c.held, c.folded = collections.deque(), 0, 0, 0, 0
    return c


def _inp(seq, b=0, sw=0):
    return {"seq": seq, "mf": 1.0, "mr": 0.0, "yaw": 0.0, "pitch": 0.0, "b": b, "sw": sw, "rt": 0.0,
            "lean": 0, "vy": 0.0, "vp": 0.0}


def test_inputs_applied_once():
    """Late input -> the agent holds; the input is applied later, never duplicated or dropped."""
    from fpsrl.net.server import JITTER_DEPTH, MAX_BACKLOG
    c = _conn()
    c.inputs.append(_inp(1))
    assert [i["seq"] for i in c.take_inputs()] == [1]
    assert c.take_inputs() == [] and c.debt == 1  # late: hold instead of repeating input 1
    seq = 1
    applied = []
    for _ in range(JITTER_DEPTH + 1):  # the late inputs arrive in a burst
        seq += 1
        c.inputs.append(_inp(seq))
    while c.inputs:
        applied += [i["seq"] for i in c.take_inputs()]
    assert applied == list(range(2, seq + 1)) and c.debt == 0
    # a client that runs ahead without having been held only gets folded past the backlog
    for k in range(MAX_BACKLOG + 3):
        c.inputs.append(_inp(100 + k, b=1 if k == 0 else 0, sw=3 if k == 1 else 0))
    first = c.take_inputs()
    assert len(first) == 1 and c.folded == 3 and first[0]["b"] & 1 and first[0]["sw"] == 3
    assert c.ack == first[0]["seq"]


def test_multi_input_tick_matches_client():
    """Two inputs in one host tick move the host exactly like two client ticks."""
    from fpsrl.net.protocol import inputs_to_actions
    host = World(MatchConfig(team_size=1, seed=4))
    client = World(MatchConfig(team_size=1, seed=4))
    a, b = host.agents[0], client.agents[0]
    yaws = [(a.yaw + 20.0 * k) % 360.0 for k in range(1, 5)]
    for y in yaws:  # client: one input per tick
        b.yaw = y
        client._move(b, Action(move_forward=1.0), client.dt)
    inps = [dict(_inp(k + 1), yaw=y) for k, y in enumerate(yaws)]
    host.step({0: inputs_to_actions(inps[:2], a)})
    host.step({0: inputs_to_actions(inps[2:], a)})
    host.step({0: []})  # held tick: no movement input at all
    assert abs(((a.yaw - yaws[-1] + 180) % 360) - 180) < 1e-6
    assert (abs(a.pos[:2] - b.pos[:2]) < 0.2).all(), (a.pos, b.pos)


def test_snapshot_merge_keeps_events():
    from fpsrl.net.server import merge_snapshots
    old = {"t": "s", "w": {"time": 1.0, "destruct": [1, 2]}, "ev": [{"type": "kill"}], "ack": [[1, 5]]}
    new = {"t": "s", "w": {"time": 1.1}, "ev": [{"type": "shot"}], "ack": [[1, 7]]}
    m = merge_snapshots(old, new)
    assert m["w"]["time"] == 1.1 and m["ack"] == [[1, 7]] and m["w"]["destruct"] == [1, 2]
    assert [e["type"] for e in m["ev"]] == ["kill", "shot"]


def test_slow_link_merges_snapshots():
    """A client that reads slower than the host sends gets fewer, newer snapshots (not a growing
    delay), and every event still arrives exactly once, in order."""
    server = NetServer("slow", port=0)
    try:
        def pump():
            for kind, conn, _ in server.poll():
                if kind == "join":
                    conn.aid = 0
                    conn.send({"t": "welcome", "aid": 0})
                    return conn
        holder = []
        threading.Thread(target=lambda: holder.append(_wait(pump)), daemon=True).start()
        client = NetClient("127.0.0.1", server.port, "Slow", timeout=2.0)
        _wait(lambda: holder)
        blob = "x" * 4000  # snapshot-sized payload so the small send buffer fills up
        n = 600
        for k in range(n):
            server.broadcast_snapshot({"t": "s", "w": {"time": k, "pad": blob}, "ev": [{"type": "e", "k": k}],
                                       "ack": [[0, k]]})
        snaps = []
        _wait(lambda: snaps.extend(m for m in client.poll() if m["t"] == "s") or
              (snaps and snaps[-1]["w"]["time"] == n - 1), timeout=10.0)
        assert [e["k"] for s in snaps for e in s["ev"]] == list(range(n))
        assert [s["w"]["time"] for s in snaps] == sorted(s["w"]["time"] for s in snaps)
        assert len(snaps) < n, "nothing was merged; the queue would grow without bound on a slow link"
        client.close()
    finally:
        server.close()


def test_interp_clock_smooth_under_jitter():
    """Bursty arrivals must not make the playback clock jump or run backwards."""
    import random
    from fpsrl.net.clock import InterpClock
    rng = random.Random(0)
    clk = InterpClock()
    arrivals = []
    for k in range(300):  # 30 Hz snapshots, 50 ms latency, jitter and some 60-90 ms spikes
        st = k / 30.0
        d = 0.05 + rng.expovariate(1 / 0.004) + (rng.uniform(0.03, 0.09) if rng.random() < 0.05 else 0.0)
        arrivals.append((st + d, st))
    arrivals.sort()
    i, t, prev, steps, newest = 0, 0.0, None, [], -1.0
    while t < 9.5:
        t += 1 / 144
        while i < len(arrivals) and arrivals[i][0] <= t:
            clk.on_snapshot(arrivals[i][1], arrivals[i][0])
            newest = max(newest, arrivals[i][1])
            i += 1
        rt = clk.advance(t)
        if rt is not None and prev is not None and t > 2.0:
            steps.append(rt - prev)
            assert rt <= newest + 0.07, "played too far past the data"
        prev = rt
    assert min(steps) > 0.0
    assert max(steps) < (1 / 144) * 1.1 and min(steps) > (1 / 144) * 0.9
    assert InterpClock.MIN_DELAY <= clk.delay <= InterpClock.MAX_DELAY


def test_snapshot_position_precision():
    """Positions must survive the wire finer than collision's 0.1 mm skin, or a replay can start
    inside a floor box and be pushed out the far side (a teleport)."""
    w = World(MatchConfig(team_size=1, seed=2))
    a = w.agents[0]
    a.pos[:] = (1.23456789, -2.3456789, 4.8001)
    s = agent_state(a)
    assert all(abs(x - y) < 2e-5 for x, y in zip(s[0], a.pos))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

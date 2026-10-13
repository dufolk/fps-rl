"""Wire format for LAN play.

TCP stream of length-prefixed JSON messages (4-byte big-endian length). JSON is
used instead of pickle so a malicious peer cannot execute code.

Client -> server: hello, in (inputs), cmd
Server -> client: welcome, full, s (snapshot + events), roster, reset, bye
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np

from ..sim import Action, Fire, Projectile, Smoke, WeaponState
from ..weapons import NUM_SLOTS, PRIMARY_CHOICES, WEAPONS

VERSION = 3
DEFAULT_PORT = 47777
DISCOVERY_PORT = 47778
DISCOVER_MAGIC = b"FPSRL-DISCOVER-1"
MAX_MSG = 4 * 1024 * 1024

# Agent flag bits
F_ALIVE, F_CROUCH, F_GROUND, F_ADS, F_WALK, F_FROZEN = 1, 2, 4, 8, 16, 32
# Input button bits
B_FIRE, B_ALT, B_JUMP, B_CROUCH, B_WALK, B_RELOAD, B_USE, B_GADGET = 1, 2, 4, 8, 16, 32, 64, 128


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def pack(msg: dict) -> bytes:
    data = json.dumps(msg, separators=(",", ":"), default=_default).encode("utf-8")
    return struct.pack("!I", len(data)) + data


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict | None:
    """Blocking read of one message; None when the peer closed the connection."""
    head = _recv_exact(sock, 4)
    if head is None:
        return None
    (n,) = struct.unpack("!I", head)
    if n > MAX_MSG:
        raise ValueError(f"message too large: {n}")
    body = _recv_exact(sock, n)
    if body is None:
        return None
    msg = json.loads(body.decode("utf-8"))
    if not isinstance(msg, dict):
        raise ValueError("bad message")
    return msg


def tune_socket(sock: socket.socket):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


# ----------------------------------------------------------------------------- inputs
def encode_input(seq: int, act: Action, yaw: float, pitch: float, render_time: float) -> list:
    """`yaw` / `pitch` are absolute body angles; while looking through a drone / camera the
    action's look deltas are sent separately."""
    b = (B_FIRE * act.fire) | (B_ALT * act.alt) | (B_JUMP * act.jump) | (B_CROUCH * act.crouch) \
        | (B_WALK * act.walk) | (B_RELOAD * act.reload) | (B_USE * act.use) | (B_GADGET * act.gadget)
    return [seq, round(act.move_forward, 3), round(act.move_right, 3), round(yaw, 3), round(pitch, 3),
            int(b), int(act.switch), round(render_time, 4), int(act.lean),
            round(act.look_yaw, 3), round(act.look_pitch, 3)]


def decode_input(v) -> dict:
    seq, mf, mr, yaw, pitch, b, sw, rt, lean, vy, vp = v
    return {"seq": int(seq), "mf": max(-1.0, min(1.0, float(mf))), "mr": max(-1.0, min(1.0, float(mr))),
            "yaw": float(yaw) % 360.0, "pitch": max(-89.0, min(89.0, float(pitch))),
            "b": int(b), "sw": int(sw) if 0 <= int(sw) <= NUM_SLOTS else 0, "rt": float(rt),
            "lean": max(-1, min(1, int(lean))), "vy": max(-90.0, min(90.0, float(vy))),
            "vp": max(-90.0, min(90.0, float(vp)))}


def input_to_action(inp: dict, agent) -> Action:
    b = inp["b"]
    if agent.view_mode:
        look_yaw, look_pitch = inp["vy"], inp["vp"]
    else:
        look_yaw = (inp["yaw"] - agent.yaw + 180.0) % 360.0 - 180.0
        look_pitch = inp["pitch"] - agent.pitch
    return Action(move_forward=inp["mf"], move_right=inp["mr"], look_yaw=look_yaw, look_pitch=look_pitch,
                  fire=bool(b & B_FIRE), alt=bool(b & B_ALT), jump=bool(b & B_JUMP), crouch=bool(b & B_CROUCH),
                  walk=bool(b & B_WALK), reload=bool(b & B_RELOAD), switch=inp["sw"], lean=inp["lean"],
                  use=bool(b & B_USE), gadget=bool(b & B_GADGET))


def apply_client_cmd(world, agent, msg: dict) -> bool:
    """Host side of a client {"t": "cmd"} message: primary / loadout pick, buy, operator pick."""
    c = msg.get("c")
    v = msg.get("v")
    if not isinstance(v, str):
        return False
    if c == "primary" and v in PRIMARY_CHOICES:
        world.set_primary(agent, v)
        return True
    if c == "loadout":
        world.set_loadout(agent, v)
        return True
    if c == "buy":
        return world.buy(agent, v)
    if c == "operator" and hasattr(world.mode, "choose_operator"):
        p = msg.get("p")
        return world.mode.choose_operator(agent, v, p if isinstance(p, str) else None)
    return False


# ----------------------------------------------------------------------------- world state
def _r(v, n=3):
    return round(float(v), n)


def agent_state(a) -> list:
    flags = (F_ALIVE * a.alive) | (F_CROUCH * a.crouching) | (F_GROUND * a.on_ground) | (F_ADS * a.ads) \
        | (F_WALK * a.walking) | (F_FROZEN * a.frozen)
    inv = [[s, a.inv[s].name, a.inv[s].mag, a.inv[s].reserve] for s in sorted(a.inv)]
    busy = None if a.busy is None else [a.busy[0], _r(a.busy[1]), _r(a.busy[2])]
    return [[_r(a.pos[0]), _r(a.pos[1]), _r(a.pos[2])], [_r(a.vel[0]), _r(a.vel[1]), _r(a.vel[2])],
            _r(a.yaw, 2), _r(a.pitch, 2), _r(a.height), int(flags), _r(a.hp, 1), a.slot, a.last_slot, inv,
            _r(a.reload_until), _r(a.ready_at), _r(a.next_fire), _r(a.protect_until), _r(a.spawn_time),
            _r(a.punch[0]), _r(a.punch[1]), _r(a.bloom), a.kills, a.deaths, a.headshots, a.killer,
            _r(a.respawn_at, 2), _r(a.death_time), _r(a.next_jump), a.primary, _r(a.throw_at), _r(a.step_dist),
            [_r(a.lean, 2), _r(a.lean_off), int(a.money), _r(a.armor, 2), bool(a.helmet),
             busy, _r(a.flash_until), _r(a.flash_t0),
             a.view_mode, a.view_ref, a.operator, _r(a.spotted_until), a.loadout.get("primary"),
             a.loadout.get("secondary"), _r(a.cook_until)]]


def apply_agent(a, s, movement: bool = True, look: bool = True):
    (pos, vel, yaw, pitch, height, flags, hp, slot, last_slot, inv, reload_until, ready_at, next_fire,
     protect_until, spawn_time, p0, p1, bloom, kills, deaths, hs, killer, respawn_at, death_time, next_jump,
     primary, throw_at, step_dist, ext) = s
    (lean, lean_off, money, armor, helmet, busy, flash_until, flash_t0, view_mode, view_ref, operator, spotted,
     lo_primary, lo_secondary, cook_until) = ext
    if movement:
        a.pos[:] = pos
        a.vel[:] = vel
        a.height = height
        a.crouching = bool(flags & F_CROUCH)
        a.on_ground = bool(flags & F_GROUND)
        a.walking = bool(flags & F_WALK)
        a.next_jump = next_jump
        a.step_dist = step_dist
    if look:
        a.yaw, a.pitch = yaw, pitch
    a.alive = bool(flags & F_ALIVE)
    a.ads = bool(flags & F_ADS)
    a.frozen = bool(flags & F_FROZEN)
    a.hp = hp
    present = set()
    for k, name, mag, res in inv:
        if name not in WEAPONS:
            continue
        present.add(k)
        ws = a.inv.get(k)
        if ws is None or ws.name != name:
            ws = a.inv[k] = WeaponState(name)
        ws.mag, ws.reserve = mag, res
    for k in [k for k in a.inv if k not in present]:
        del a.inv[k]
    if movement or look:
        a.lean, a.lean_off = lean, lean_off
    a.money, a.armor, a.helmet = money, armor, helmet
    a.busy = None if busy is None else (busy[0], busy[1], busy[2], None)
    a.flash_until, a.flash_t0 = flash_until, flash_t0
    a.view_mode, a.view_ref, a.operator, a.spotted_until = view_mode, view_ref, operator, spotted
    a.loadout = {"primary": lo_primary, "secondary": lo_secondary}
    a.slot, a.last_slot = slot, last_slot
    a.reload_until, a.ready_at, a.next_fire = reload_until, ready_at, next_fire
    a.protect_until, a.spawn_time = protect_until, spawn_time
    a.prev_punch[:] = a.punch
    a.punch[:] = (p0, p1)
    a.bloom = bloom
    a.kills, a.deaths, a.headshots, a.killer = kills, deaths, hs, killer
    a.respawn_at, a.death_time = respawn_at, death_time
    a.primary, a.throw_at, a.cook_until = primary, throw_at, cook_until


def world_state(world, destruct_version=None) -> dict:
    """Full snapshot. The destruction bitset is only included when its version differs from
    `destruct_version` (what the receivers already have)."""
    st = {
        "time": _r(world.time, 4), "tick": world.tick, "tk": list(world.team_kills),
        "done": world.done, "winner": world.winner,
        "agents": [agent_state(a) for a in world.agents],
        "proj": [[p.id, p.kind, [_r(v) for v in p.pos], [_r(v) for v in p.vel]] for p in world.projectiles],
        "smokes": [[s.id, [_r(v) for v in s.pos], _r(s.t0), _r(s.t1), _r(s.max_radius)] for s in world.smokes],
        "fires": [[f.id, [_r(v) for v in f.pos], _r(f.t0), _r(f.t1), _r(f.radius), f.owner, f.weapon, _r(f.dps)]
                  for f in world.fires],
        "mode": world.mode.state(),
        "gadgets": world.gadgets.state(),
    }
    if world.destruct.panels and destruct_version != world.destruct.version:
        st["destruct"] = world.destruct.state()
    return st


def apply_world(world, st: dict, own: int | None = None):
    """Overwrite the mirror world with server state. Positions of other agents are left to the caller
    (interpolation); the own agent keeps its local look angles."""
    world.time = st["time"]
    world.tick = st["tick"]
    world.team_kills = list(st["tk"])
    world.done = st["done"]
    world.winner = st["winner"]
    world.mode.apply_state(st.get("mode"))
    if "destruct" in st:
        world.destruct.apply_state(st["destruct"])
    world.gadgets.apply_state(st.get("gadgets"))
    for a, s in zip(world.agents, st["agents"]):
        apply_agent(a, s, movement=a.id == own, look=a.id != own)
    projs = []
    for pid, kind, pos, vel in st["proj"]:
        projs.append(Projectile(pid, kind, -1, -1, np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64),
                                0.0, 0.0))
    world.projectiles = projs
    world.smokes = [Smoke(sid, np.array(pos, dtype=np.float64), t0, t1, r) for sid, pos, t0, t1, r in st["smokes"]]
    world.fires = [Fire(fid, np.array(pos, dtype=np.float64), t0, t1, rad, owner, weapon, dps)
                   for fid, pos, t0, t1, rad, owner, weapon, dps in st.get("fires", ())]

"""LAN host: accepts players over TCP and answers UDP discovery broadcasts.

All socket I/O runs on daemon threads; the game loop talks to the server
through thread-safe queues (`poll()`, per-client input deques, `broadcast()`).
"""
from __future__ import annotations

import collections
import json
import queue
import socket
import threading

from .protocol import (B_ALT, B_FIRE, B_RELOAD, DEFAULT_PORT, DISCOVER_MAGIC, DISCOVERY_PORT, VERSION,
                       decode_input, pack, recv_msg, tune_socket)


JITTER_DEPTH = 3  # queued inputs a late arrival may leave behind before the host catches up
MAX_BACKLOG = 8  # queued inputs (~130 ms) beyond which the oldest are folded together
SEND_BUFFER = 32 * 1024  # small kernel buffer: a backlog stays in `out`, where snapshots can be merged


def merge_snapshots(old: dict, new: dict) -> dict:
    """Fold an unsent snapshot into the next one: the newer state wins, nothing in the event
    stream or a pending destruction update is lost."""
    merged = dict(new, ev=list(old.get("ev", ())) + list(new.get("ev", ())))
    if "destruct" in old.get("w", {}) and "destruct" not in new.get("w", {}):
        merged["w"] = dict(new["w"], destruct=old["w"]["destruct"])
    return merged


class ClientConn:
    def __init__(self, server: "NetServer", sock: socket.socket, addr, cid: int):
        self.server = server
        self.sock = sock
        self.addr = addr
        self.cid = cid
        self.name = f"Player{cid}"
        self.team_pref = -1
        self.skins: dict = {}
        self.aid: int | None = None
        self.inputs: collections.deque = collections.deque()
        self.last_seq = 0
        self.ack = 0
        self.debt = 0  # host ticks this client's agent was held waiting for a late input
        self.held = 0  # stats: ticks held / inputs folded because the client ran ahead
        self.folded = 0
        # outgoing items: bytes, or a snapshot dict that may still absorb the next snapshot
        self.out: collections.deque = collections.deque()
        self.cv = threading.Condition()
        self.open = True
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER)
        except OSError:
            pass
        threading.Thread(target=self._reader, daemon=True, name=f"net-read-{cid}").start()
        threading.Thread(target=self._writer, daemon=True, name=f"net-write-{cid}").start()

    # ------------------------------------------------------------------ inputs
    def take_inputs(self) -> list:
        """Inputs to apply this host tick, each exactly once and in order.

        The client predicts its own movement from exactly these inputs, so the host never invents
        or drops one under normal jitter: an empty queue holds the agent for a tick (`[]`). The
        input that arrives late then stays queued, so the queue grows into a small jitter buffer
        by itself; only past `JITTER_DEPTH` does a tick apply two inputs, and only to repay ticks
        that were held, so a client cannot run faster than the host clock. A queue past
        `MAX_BACKLOG` (a fast clock, a speed hack) has its oldest inputs folded together."""
        q = self.inputs
        if not q:
            self.debt = min(self.debt + 1, MAX_BACKLOG)
            self.held += 1
            return []
        while len(q) > MAX_BACKLOG:
            old = q.popleft()
            nxt = q[0]
            if old["sw"] and not nxt["sw"]:
                nxt["sw"] = old["sw"]
            nxt["b"] |= old["b"] & (B_FIRE | B_ALT | B_RELOAD)  # button edges survive the fold
            self.folded += 1
        out = [q.popleft()]
        if self.debt and len(q) >= JITTER_DEPTH:
            self.debt -= 1
            out.append(q.popleft())
        self.ack = out[-1]["seq"]
        return out

    # ------------------------------------------------------------------ outgoing
    def send(self, msg: dict):
        self.send_raw(pack(msg))

    def send_raw(self, data: bytes):
        self._enqueue(data)

    def send_snapshot(self, msg: dict):
        """Queue a snapshot; if the previous one has not left yet (slow link), merge them so the
        client always gets the newest state instead of an ever growing delay."""
        self._enqueue(msg, snapshot=True)

    def _enqueue(self, item, snapshot=False):
        with self.cv:
            if not self.open:
                return
            if snapshot and self.out and isinstance(self.out[-1], dict):
                self.out[-1] = merge_snapshots(self.out[-1], item)
            else:
                if len(self.out) > 900:  # nothing has left for ages: the client is gone
                    self.open = False
                    self.out.clear()
                    self.out.append(None)
                    self.cv.notify()
                    return
                self.out.append(item)
            self.cv.notify()

    def kick(self, reason: str):
        self.send({"t": "bye", "reason": reason})
        self.close(flush=True)

    def close(self, flush: bool = False):
        """Stop the connection; with `flush` the writer sends what is queued (e.g. a "bye") first."""
        with self.cv:
            if not self.open:
                return
            self.open = False
            if not flush:
                self.out.clear()
            self.out.append(None)
            self.cv.notify()
        if not flush:
            self._shutdown()

    def _shutdown(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    def _reader(self):
        events = self.server.events
        joined = False
        try:
            while self.open:
                msg = recv_msg(self.sock)
                if msg is None:
                    break
                t = msg.get("t")
                if t == "hello" and not joined:
                    if msg.get("v") != VERSION:
                        self.send({"t": "bye", "reason": "version mismatch"})
                        break
                    self.name = str(msg.get("name", self.name))[:16] or self.name
                    self.team_pref = int(msg.get("team", -1))
                    skins = msg.get("skins", {})
                    self.skins = {str(k): str(v) for k, v in skins.items()} if isinstance(skins, dict) else {}
                    joined = True
                    events.put(("join", self, None))
                elif t == "in" and joined:
                    for raw in msg.get("i", ()):
                        inp = decode_input(raw)
                        if inp["seq"] > self.last_seq:
                            self.last_seq = inp["seq"]
                            self.inputs.append(inp)
                elif t == "cmd" and joined:
                    events.put(("cmd", self, msg))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass
        finally:
            self.close()
            if joined:
                events.put(("leave", self, None))

    def _writer(self):
        while True:
            with self.cv:
                while not self.out:
                    self.cv.wait()
                data = self.out.popleft()
            if data is None:
                self._shutdown()
                return
            if isinstance(data, dict):
                data = pack(data)
            try:
                self.sock.sendall(data)
            except OSError:
                self.close()
                return


class NetServer:
    def __init__(self, room: str, port: int = DEFAULT_PORT, info=None):
        self.room = room
        self.port = port
        self.info = info or (lambda: {})
        self.events: queue.Queue = queue.Queue()
        self.clients: dict[int, ClientConn] = {}
        self._next = 1
        self.running = True
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # on Windows SO_REUSEADDR would let a second host silently share the port
        excl = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        self.listener.setsockopt(socket.SOL_SOCKET, excl if excl is not None else socket.SO_REUSEADDR, 1)
        self.listener.bind(("", port))
        self.port = self.listener.getsockname()[1]
        self.listener.listen(16)
        self.listener.settimeout(0.5)
        threading.Thread(target=self._accept, daemon=True, name="net-accept").start()
        self.udp = None
        try:
            self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp.bind(("", DISCOVERY_PORT))
            self.udp.settimeout(0.5)
            threading.Thread(target=self._discovery, daemon=True, name="net-discovery").start()
        except OSError as exc:  # another host on this machine already answers discovery
            print("[net] discovery disabled:", exc)
            self.udp = None

    def _accept(self):
        while self.running:
            try:
                sock, addr = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            sock.settimeout(None)
            tune_socket(sock)
            cid = self._next
            self._next += 1
            self.clients[cid] = ClientConn(self, sock, addr, cid)

    def _discovery(self):
        while self.running:
            try:
                data, addr = self.udp.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return
            if data.strip() == DISCOVER_MAGIC:
                reply = {"room": self.room, "port": self.port, "v": VERSION, **self.info()}
                try:
                    self.udp.sendto(json.dumps(reply).encode("utf-8"), addr)
                except OSError:
                    pass

    def poll(self) -> list:
        out = []
        while True:
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "leave":
                self.clients.pop(ev[1].cid, None)
            out.append(ev)
        return out

    def players(self) -> list[ClientConn]:
        return [c for c in list(self.clients.values()) if c.open and c.aid is not None]

    def broadcast(self, msg: dict):
        data = pack(msg)
        for c in self.players():
            c.send_raw(data)

    def broadcast_snapshot(self, msg: dict):
        for c in self.players():
            c.send_snapshot(msg)

    def close(self):
        self.running = False
        for c in list(self.clients.values()):
            c.kick("主机已关闭房间 / host closed the room")
        for s in (self.listener, self.udp):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

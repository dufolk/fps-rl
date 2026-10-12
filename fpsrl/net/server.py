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

from .protocol import (DEFAULT_PORT, DISCOVER_MAGIC, DISCOVERY_PORT, VERSION, decode_input, pack, recv_msg,
                       tune_socket)


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
        self.last_input: dict | None = None
        self.last_seq = 0
        self.ack = 0
        self.out: queue.Queue = queue.Queue()
        self.open = True
        threading.Thread(target=self._reader, daemon=True, name=f"net-read-{cid}").start()
        threading.Thread(target=self._writer, daemon=True, name=f"net-write-{cid}").start()

    def send(self, msg: dict):
        self.send_raw(pack(msg))

    def send_raw(self, data: bytes):
        if not self.open:
            return
        if self.out.qsize() > 900:  # ~15 s of snapshots backed up: the client is gone
            self.close()
            return
        self.out.put(data)

    def kick(self, reason: str):
        self.send({"t": "bye", "reason": reason})
        self.close(flush=True)

    def close(self, flush: bool = False):
        """Stop the connection; with `flush` the writer sends what is queued (e.g. a "bye") first."""
        if not self.open:
            return
        self.open = False
        self.out.put(None)
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
            data = self.out.get()
            if data is None:
                self._shutdown()
                return
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

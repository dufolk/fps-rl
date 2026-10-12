"""LAN client connection + room discovery."""
from __future__ import annotations

import ipaddress
import json
import queue
import socket
import threading
import time

from .protocol import DEFAULT_PORT, DISCOVER_MAGIC, DISCOVERY_PORT, VERSION, pack, recv_msg, tune_socket


class ConnectError(Exception):
    pass


class NetClient:
    def __init__(self, host: str, port: int = DEFAULT_PORT, name: str = "Player", skins: dict | None = None,
                 team: int = -1, timeout: float = 4.0):
        try:
            self.sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            raise ConnectError(f"无法连接 {host}:{port} ({exc})") from exc
        tune_socket(self.sock)
        self.lock = threading.Lock()
        self.open = True
        self.send({"t": "hello", "v": VERSION, "name": name, "skins": skins or {}, "team": team})
        try:
            msg = recv_msg(self.sock)
        except (OSError, ValueError) as exc:
            self.close()
            raise ConnectError(f"握手失败 ({exc})") from exc
        if msg is None or msg.get("t") != "welcome":
            reason = (msg or {}).get("reason", "房间已满或拒绝连接") if msg else "连接被关闭"
            self.close()
            raise ConnectError(reason)
        self.welcome = msg
        self.sock.settimeout(None)
        self.inbox: queue.Queue = queue.Queue()
        self.closed_reason = None
        threading.Thread(target=self._reader, daemon=True, name="net-client-read").start()

    def _reader(self):
        try:
            while self.open:
                msg = recv_msg(self.sock)
                if msg is None:
                    break
                self.inbox.put(msg)
        except (OSError, ValueError):
            pass
        if self.open:
            self.inbox.put({"t": "bye", "reason": "与主机的连接已断开"})
        self.open = False

    def poll(self) -> list:
        out = []
        while True:
            try:
                out.append(self.inbox.get_nowait())
            except queue.Empty:
                return out

    def send(self, msg: dict):
        if not self.open:
            return
        data = pack(msg)
        with self.lock:
            try:
                self.sock.sendall(data)
            except OSError:
                self.open = False
                self.inbox.put({"t": "bye", "reason": "与主机的连接已断开"})

    def close(self):
        self.open = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def _broadcast_targets():
    targets = {"255.255.255.255", "127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127."):
                continue
            net = ipaddress.ip_network(f"{ip}/24", strict=False)
            targets.add(str(net.broadcast_address))
    except OSError:
        pass
    return targets


def discover(timeout: float = 0.6) -> list[dict]:
    """Broadcast a discovery probe and collect room descriptions from LAN hosts."""
    rooms = {}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.settimeout(0.1)
        for target in _broadcast_targets():
            try:
                s.sendto(DISCOVER_MAGIC, (target, DISCOVERY_PORT))
            except OSError:
                pass
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                data, addr = s.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                info = json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(info, dict):
                continue
            ip = addr[0]
            if ip.startswith("127."):
                ip = "127.0.0.1"
            info["ip"] = ip
            key = (ip, info.get("port"))
            if key not in rooms or not ip.startswith("127."):
                rooms[key] = info
    finally:
        s.close()
    # the same host may answer on loopback and its LAN address: keep the LAN one
    lan_ports = {k[1] for k in rooms if k[0] != "127.0.0.1"}
    return [v for k, v in rooms.items() if not (k[0] == "127.0.0.1" and k[1] in lan_ports)]


class Discoverer:
    """Background room scanner for the join menu."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.rooms: list[dict] = []
        self.running = True
        self.version = 0
        threading.Thread(target=self._loop, daemon=True, name="net-discover").start()

    def _loop(self):
        while self.running:
            try:
                rooms = discover()
            except OSError:
                rooms = []
            self.rooms = sorted(rooms, key=lambda r: (r.get("room", ""), r["ip"]))
            self.version += 1
            t = time.monotonic() + self.interval
            while self.running and time.monotonic() < t:
                time.sleep(0.1)

    def stop(self):
        self.running = False

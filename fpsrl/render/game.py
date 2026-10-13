"""Playable sessions.

* `PlayerController` - keyboard / mouse -> `Action` for the local player.
* `LocalSession`     - authoritative world on this machine (single player or LAN host).
* `ClientSession`    - mirror world fed by a LAN host, with client-side prediction of the
                       own movement and interpolation of everybody else.
* `GameApp`          - stand-alone quick match without the menu (`play.py --quick`).
"""
from __future__ import annotations

import collections
import copy
import math
import random
import time

from direct.showbase.DirectObject import DirectObject
from panda3d.core import Filename, KeyboardButton, MouseButton, WindowProperties

from ..bots import Bot
from ..maps import MAPS, default_mode
from ..profile import SENSITIVITY_STEP, clamp_sensitivity
from ..net.protocol import (B_ALT, B_FIRE, B_RELOAD, apply_agent, apply_client_cmd, apply_world, encode_input,
                            input_to_action, world_state)
from ..sim import TEAM_NAMES, Action, MatchConfig, World
from ..weapons import NUM_SLOTS, PRIMARY_CHOICES, random_primary
from .ime import hold as hold_ime
from .viewer import BASE_HFOV, Viewer

HELP_ZH = """[F1] 隐藏帮助
WASD 移动    空格 跳    Ctrl/C 蹲    Shift 静步    Q/E 左右探头
左键 开火/轻刀/投掷    右键 瞄准/开镜/重刀/低抛
R 换弹    1-7 武器    X 上一把    滚轮 切换    F 交互(安包/拆包/加固/路障)·检视
G 无人机/摄像头    B 购买/切换主武器    Tab 计分板
Esc 菜单    F2 瞄准按住/切换    F3 第三人称    F4 FPS    +/- 灵敏度"""
HELP_EN = """[F1] hide help
WASD move    SPACE jump    CTRL/C crouch    SHIFT walk    Q/E lean
LMB fire / slash / throw    RMB aim / scope / heavy stab / underhand
R reload    1-7 weapons    X last weapon    wheel cycle    F use (plant/defuse/reinforce) / inspect
G drone / cameras    B buy menu / cycle primary    TAB scoreboard
ESC menu    F2 aim hold/toggle    F3 third person    F4 fps    +/- sensitivity"""


class _Settings:
    """Fallback settings object when no `Profile` is supplied."""

    def __init__(self, sensitivity=1.6, ads_toggle=False, show_fps=False, name="Player"):
        self.sensitivity = sensitivity
        self.ads_toggle = ads_toggle
        self.show_fps = show_fps
        self.name = name
        self.skins = {}
        self.primary = "ak47"


class PlayerController:
    def __init__(self, viewer: Viewer, settings, on_escape=None, on_primary=None, on_restart=None, on_buy=None,
                 on_operator=None):
        self.viewer = viewer
        self.base = base = viewer.base
        self.settings = settings
        self.on_escape = on_escape
        self.on_primary = on_primary
        self.on_restart = on_restart
        # B menu in round modes (buy in bomb mode, operator pick in Siege)
        self.overlay = None
        if on_buy is not None or on_operator is not None:
            from .buymenu import BuyMenu
            sound = (lambda n: viewer.audio.ui(n, 0.5)) if viewer.audio is not None else None
            self.overlay = BuyMenu(viewer, on_buy or (lambda name: None), on_operator or (lambda n, p: None), sound)
        self.enabled = True
        self.captured = False
        self.fire_latch = self.alt_latch = False
        self.use_latch = self.gadget_latch = False
        self.f_pending = 0
        self.view_look = [0.0, 0.0]
        self.ads_state = False
        self.switch_req = 0
        self.show_help = True
        self.fps_t, self.fps_n = time.perf_counter(), 0
        self.mw = base.mouseWatcherNode
        self.kb = {k: KeyboardButton.asciiKey(k) for k in "wasdcrqefg"}
        self.ev = ev = DirectObject()
        ev.accept("mouse1", self._latch, ["fire"])
        ev.accept("mouse3", self._latch, ["alt"])
        ev.accept("wheel_up", self._cycle, [-1])
        ev.accept("wheel_down", self._cycle, [1])
        for i in range(1, NUM_SLOTS + 1):
            ev.accept(str(i), self._switch, [i])
        ev.accept("x", self._last)
        ev.accept("b", self._buy_or_primary)
        ev.accept("f", self._use_press)
        ev.accept("g", self._gadget_press)
        ev.accept("escape", self._escape)
        ev.accept("f1", self._toggle_help)
        ev.accept("f2", self._toggle_ads_mode)
        ev.accept("f3", self._third_person)
        ev.accept("f4", self._toggle_fps)
        ev.accept("f5", self._restart)
        ev.accept("f12", lambda: base.screenshot(namePrefix="fpsrl_screenshot"))
        for k, d in (("+", SENSITIVITY_STEP), ("=", SENSITIVITY_STEP), ("-", -SENSITIVITY_STEP)):
            ev.accept(k, self._sens, [d])
        ev.accept("tab", self._board, [True])
        ev.accept("tab-up", self._board, [False])
        self._help()

    @property
    def hud(self):
        return self.viewer.hud

    @property
    def agent(self):
        return self.viewer.world.agents[self.viewer.pov]

    @property
    def zh(self):
        return self.hud is not None and self.hud.cjk

    def info(self, text):
        if self.hud is not None:
            self.hud.set_info(text)

    # ------------------------------------------------------------------ callbacks
    def _overlay_open(self) -> bool:
        return self.overlay is not None and self.overlay.visible

    def close_overlay(self):
        if self._overlay_open():
            self.overlay.close()
            if self.enabled:
                self.capture(True)

    def _latch(self, which):
        if not self.enabled or self._overlay_open():
            return
        if not self.captured:
            self.capture(True)
            return
        if which == "fire":
            self.fire_latch = True
        else:
            self.alt_latch = True
            if self.settings.ads_toggle:
                self.ads_state = not self.ads_state

    def _switch(self, slot):
        if self.enabled:
            self.switch_req = slot

    def _last(self):
        if self.enabled:
            self.switch_req = self.agent.last_slot

    def _cycle(self, d):
        if not (self.enabled and self.captured):
            return
        a = self.agent
        slot = a.slot
        for _ in range(NUM_SLOTS):
            slot = (slot - 1 + d) % NUM_SLOTS + 1
            ws = a.inv.get(slot)
            if ws and (ws.spec.kind not in ("grenade", "gadget") or ws.mag > 0):
                break
        self.switch_req = slot

    def _use_press(self):
        if self.enabled and self.captured:
            self.use_latch = True
            self.f_pending = 3

    def _gadget_press(self):
        if self.enabled and self.captured:
            self.gadget_latch = True

    def _buy_or_primary(self):
        if not self.enabled:
            return
        if self._overlay_open():
            self.close_overlay()
        elif self.overlay is not None and not self.viewer.world.mode.free_weapons:
            if self.overlay.open():
                self.capture(False)
        else:
            self._toggle_primary()

    def _toggle_primary(self):
        if not self.enabled:
            return
        a = self.agent
        i = PRIMARY_CHOICES.index(a.primary) if a.primary in PRIMARY_CHOICES else 0
        new = PRIMARY_CHOICES[(i + 1) % len(PRIMARY_CHOICES)]
        self.settings.primary = new
        now = self.on_primary(new) if self.on_primary else False
        if self.zh:
            self.info(f"主武器: {new.upper()}" + ("" if now else " (下次出生生效)"))
        else:
            self.info(f"Primary: {new.upper()}" + ("" if now else " (next spawn)"))

    def _escape(self):
        if self._overlay_open():
            self.close_overlay()
            return
        if self.on_escape is not None:
            self.on_escape()
        else:
            self.capture(False)

    def _help(self):
        if self.hud is not None:
            text = (HELP_ZH if self.zh else HELP_EN) if self.show_help else "[F1] 帮助" if self.zh else "[F1] help"
            self.hud.set_help(text)

    def _toggle_help(self):
        self.show_help = not self.show_help
        self._help()

    def _toggle_ads_mode(self):
        self.settings.ads_toggle = not self.settings.ads_toggle
        self.ads_state = False
        mode = self.settings.ads_toggle
        self.info(("瞄准模式: " + ("切换" if mode else "按住")) if self.zh else
                  ("Aim mode: " + ("TOGGLE" if mode else "HOLD")))

    def _third_person(self):
        if self.enabled:
            self.viewer.toggle_third_person()

    def _toggle_fps(self):
        self.settings.show_fps = not self.settings.show_fps
        if not self.settings.show_fps:
            self.info("")

    def _restart(self):
        if self.enabled and self.on_restart is not None:
            self.on_restart()

    def _sens(self, d):
        self.settings.sensitivity = clamp_sensitivity(self.settings.sensitivity + d)
        self.info(f"{'灵敏度' if self.zh else 'Sensitivity'}: {self.settings.sensitivity:.2f}")
        save = getattr(self.settings, "save", None)
        if save is not None:
            save()

    def _board(self, show):
        self.viewer.show_board = show and self.enabled

    # ------------------------------------------------------------------ mouse
    def capture(self, on: bool):
        win = self.base.win
        props = WindowProperties()
        props.setCursorHidden(on)
        props.setMouseMode(WindowProperties.M_confined if on else WindowProperties.M_absolute)
        win.requestProperties(props)
        self.captured = on
        if on:
            self._recenter()
        self.viewer.mouse_dx = self.viewer.mouse_dy = 0.0

    def _recenter(self):
        win = self.base.win
        win.movePointer(0, win.getXSize() // 2, win.getYSize() // 2)

    def mouse_look(self, agent, autoplay=False):
        if not (self.enabled and self.captured) or not self.base.win.getProperties().getForeground():
            self.viewer.mouse_dx = self.viewer.mouse_dy = 0.0
            return
        win = self.base.win
        md = win.getPointer(0)
        dx, dy = md.getX() - win.getXSize() // 2, md.getY() - win.getYSize() // 2
        self._recenter()
        self.viewer.mouse_dx, self.viewer.mouse_dy = dx, dy
        if autoplay:
            return
        if agent.view_mode:
            k = 0.022 * self.settings.sensitivity * 0.8
            self.view_look[0] -= dx * k
            self.view_look[1] -= dy * k
            return
        if not agent.alive:
            return
        zoom = 1.0
        if agent.ads and agent.slot in agent.inv and agent.spec.ads_fov:
            zoom = math.tan(math.radians(agent.spec.ads_fov / 2)) / math.tan(math.radians(BASE_HFOV / 2))
        k = 0.022 * self.settings.sensitivity * zoom
        agent.yaw = (agent.yaw - dx * k) % 360.0
        agent.pitch = max(-89.0, min(89.0, agent.pitch - dy * k))

    def read_input(self) -> Action:
        act = Action()
        if not (self.enabled and self.captured):
            return act
        down = self.mw.isButtonDown
        kb = self.kb
        act.move_forward = float(down(kb["w"])) - float(down(kb["s"]))
        act.move_right = float(down(kb["d"])) - float(down(kb["a"]))
        act.jump = down(KeyboardButton.space())
        act.crouch = down(KeyboardButton.control()) or down(kb["c"])
        act.walk = down(KeyboardButton.shift())
        act.reload = down(kb["r"])
        act.fire = down(MouseButton.one()) or self.fire_latch
        rmb = down(MouseButton.three())
        act.alt = self.ads_state if self.settings.ads_toggle else (rmb or self.alt_latch)
        a = self.agent
        if a.slot in a.inv and a.spec.kind != "gun":
            act.alt = rmb or self.alt_latch
        act.lean = int(down(kb["e"])) - int(down(kb["q"]))
        act.use = down(kb["f"]) or self.use_latch
        act.gadget = down(kb["g"]) or self.gadget_latch
        act.switch = self.switch_req
        if a.view_mode:
            act.look_yaw, act.look_pitch = self.view_look
        return act

    def after_tick(self):
        self.view_look = [0.0, 0.0]
        self.fire_latch = self.alt_latch = False
        self.use_latch = self.gadget_latch = False
        self.switch_req = 0
        if self.f_pending > 0:
            self.f_pending -= 1
            a = self.agent
            if a.busy is not None or a.view_mode:
                self.f_pending = 0
            elif self.f_pending == 0 and a.alive:
                self.viewer.inspect()

    def update_fps(self, dt: float = 0.0):
        if self.overlay is not None and self.overlay.visible:
            self.overlay.update(dt)
            if not self.overlay.visible and self.enabled:
                self.capture(True)
        self.fps_n += 1
        now = time.perf_counter()
        if now - self.fps_t > 0.5:
            fps = self.fps_n / (now - self.fps_t)
            self.fps_t, self.fps_n = now, 0
            if self.settings.show_fps:
                self.info(f"{fps:.0f} FPS")

    def destroy(self):
        self.ev.ignoreAll()
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None
        self.capture(False)
        self.viewer.show_board = False


# ----------------------------------------------------------------------------- local / host
class LocalSession:
    """Single player against bots, optionally hosting a LAN room (`host={"room":..., "port":...}`)."""

    RESTART_DELAY = 10.0

    def __init__(self, viewer: Viewer, settings, team_size=4, difficulty="normal", team=0, primary="ak47",
                 kill_limit=60, time_limit=420.0, seed=None, autoplay=False, host=None, on_escape=None,
                 map_name="transport_ship", mode=None, mode_params=None):
        rng = random.Random(seed)
        primaries = [random_primary(rng) for _ in range(2 * team_size)]
        self.player = team * team_size
        primaries[self.player] = primary
        map_name = map_name if map_name in MAPS else "transport_ship"
        mode = mode if mode in MAPS[map_name][2] else default_mode(map_name)
        self.world = w = World(MatchConfig(team_size=team_size, kill_limit=kill_limit, time_limit=time_limit,
                                           primaries=primaries, seed=seed, map_name=map_name, mode=mode,
                                           mode_params=dict(mode_params or {})))
        w.agents[self.player].name = settings.name[:16] or w.agents[self.player].name
        self.default_names = {a.id: f"{TEAM_NAMES[a.team]}-{a.id % team_size}" for a in w.agents}
        self.viewer = viewer
        self.settings = settings
        self.difficulty = difficulty
        self.autoplay = autoplay
        self.seed = seed
        self.bots = {a.id: Bot(w, a.id, difficulty, seed=(seed or 0) * 97 + a.id)
                     for a in w.agents if a.id != self.player or autoplay}
        viewer.bind(w, self.player)
        viewer.set_local_skins(settings.skins)
        viewer.set_cinematic(False)
        round_mode = not w.mode.free_weapons
        self.ctrl = PlayerController(viewer, settings, on_escape=on_escape, on_primary=self._set_primary,
                                     on_restart=self.restart, on_buy=self._buy if round_mode else None,
                                     on_operator=self._operator if round_mode else None)
        self.acc = 0.0
        self.done_at = None
        self.server = None
        self.sent_seq = 0
        self.sent_destruct = None  # destruction version the clients already have
        self.skins = {self.player: dict(settings.skins)}
        if host is not None:
            from ..net.server import NetServer
            w.lag_comp = True
            self.server = NetServer(host.get("room", "FPS-RL"), int(host.get("port", 47777)), info=self._room_info)
        if viewer.hud is not None:
            zh = viewer.hud.cjk
            if self.server is not None:
                viewer.hud.end_hint = f"{int(self.RESTART_DELAY)} 秒后开始下一局" if zh else "Next round starting soon"
                viewer.hud.set_info((f"房间已创建: {host.get('room')}  端口 {self.server.port}" if zh else
                                     f"Room open: {host.get('room')}  port {self.server.port}"))
            else:
                viewer.hud.end_hint = "按 F5 重新开始, Esc 返回菜单" if zh else "F5 restart  /  ESC menu"
        if not autoplay:
            self.ctrl.capture(True)

    # ------------------------------------------------------------------ host side
    def _room_info(self):
        w = self.world
        humans = 1 + (len(self.server.players()) if self.server else 0)
        return {"players": humans, "slots": len(w.agents), "team_size": w.cfg.team_size,
                "score": list(w.score), "map": w.cfg.map_name, "mode": w.cfg.mode, "host": self.settings.name}

    def _set_primary(self, name):
        return self.world.set_primary(self.world.agents[self.player], name)

    def _buy(self, name):
        return self.world.buy(self.world.agents[self.player], name)

    def _operator(self, name, primary=None):
        m = self.world.mode
        if hasattr(m, "choose_operator"):
            m.choose_operator(self.world.agents[self.player], name, primary)

    def _human_ids(self):
        ids = {self.player} if not self.autoplay else set()
        if self.server:
            ids |= {c.aid for c in self.server.players()}
        return ids

    def _pick_slot(self, team_pref):
        w = self.world
        humans = self._human_ids()
        counts = [sum(1 for i in humans if w.agents[i].team == t) for t in (0, 1)]
        order = [team_pref] if team_pref in (0, 1) else sorted((0, 1), key=lambda t: counts[t])
        order += [t for t in (0, 1) if t not in order]
        for t in order:
            free = [a for a in w.agents if a.team == t and a.id not in humans]
            if free:
                free.sort(key=lambda a: (a.alive, a.id))
                return free[0].id
        return None

    def _roster(self):
        humans = self._human_ids()
        return [[a.id, a.name, self.skins.get(a.id, {}), a.id in humans] for a in self.world.agents]

    def _welcome(self, conn):
        w = self.world
        cfg = w.cfg
        return {"t": "welcome", "aid": conn.aid, "room": self.server.room,
                "cfg": {"team_size": cfg.team_size, "kill_limit": cfg.kill_limit, "time_limit": cfg.time_limit,
                        "respawn": cfg.respawn, "friendly_fire": cfg.friendly_fire, "map_name": cfg.map_name,
                        "mode": cfg.mode, "mode_params": cfg.mode_params,
                        "tick_rate": cfg.tick_rate, "primaries": [a.primary for a in w.agents]},
                "w": world_state(w), "roster": self._roster()}

    def _net_events(self):
        w = self.world
        changed = False
        for kind, conn, msg in self.server.poll():
            if kind == "join":
                aid = self._pick_slot(conn.team_pref)
                if aid is None:
                    conn.kick("房间已满 / room is full")
                    continue
                conn.aid = aid
                self.bots.pop(aid, None)
                w.agents[aid].name = conn.name
                self.skins[aid] = conn.skins
                conn.send(self._welcome(conn))
                changed = True
                self._notice(f"{conn.name} 加入了房间" if self._zh() else f"{conn.name} joined")
            elif kind == "leave" and conn.aid is not None:
                aid = conn.aid
                conn.aid = None
                w.agents[aid].name = self.default_names[aid]
                self.skins.pop(aid, None)
                self.bots[aid] = Bot(w, aid, self.difficulty, seed=aid)
                w.rewind.pop(aid, None)
                changed = True
                self._notice(f"{conn.name} 离开了房间" if self._zh() else f"{conn.name} left")
            elif kind == "cmd" and conn.aid is not None:
                apply_client_cmd(w, w.agents[conn.aid], msg)
        if changed:
            roster = self._roster()
            for aid, _, skins, _h in roster:
                self.viewer.agent_skins[aid] = skins
            self.viewer.agent_skins[self.player] = self.viewer.local_skins
            self.server.broadcast({"t": "roster", "p": roster})

    def _zh(self):
        return self.viewer.hud is not None and self.viewer.hud.cjk

    def _notice(self, text):
        if self.viewer.hud is not None:
            self.viewer.hud.set_info(text)

    def _remote_actions(self):
        acts = {}
        w = self.world
        for c in self.server.players():
            q = c.inputs
            while len(q) > 4:  # client ran ahead: merge the oldest input into the next one
                old = q.popleft()
                nxt = q[0]
                if old["sw"] and not nxt["sw"]:
                    nxt["sw"] = old["sw"]
                nxt["b"] |= old["b"] & (B_FIRE | B_ALT | B_RELOAD)
            if q:
                inp = q.popleft()
                c.last_input = inp
                c.ack = inp["seq"]
            elif c.last_input is not None:
                inp = dict(c.last_input, sw=0)
            else:
                continue
            acts[c.aid] = input_to_action(inp, w.agents[c.aid])
            w.rewind[c.aid] = inp["rt"]
        return acts

    def _broadcast(self):
        w = self.world
        events = w.events_since(self.sent_seq)
        self.sent_seq = w.seq
        self.server.broadcast({"t": "s", "w": world_state(w, self.sent_destruct), "ev": events,
                               "ack": [[c.aid, c.ack] for c in self.server.players()]})
        self.sent_destruct = w.destruct.version

    # ------------------------------------------------------------------ lifecycle
    def restart(self):
        w = self.world
        w.reset(None if self.seed is None else self.seed + w.tick)
        for b in self.bots.values():
            b.reset()
        self.sent_seq = 0
        self.done_at = None
        self.viewer.bind(w, self.player)
        for aid, skins in self.skins.items():
            self.viewer.agent_skins[aid] = skins
        self.viewer.agent_skins[self.player] = self.viewer.local_skins
        self.acc = 0.0
        if self.server is not None:
            self.server.broadcast({"t": "reset"})

    def update(self, dt):
        w = self.world
        me = w.agents[self.player]
        self.ctrl.mouse_look(me, self.autoplay)
        if self.server is not None:
            self._net_events()
        human = None if self.autoplay else self.ctrl.read_input()
        self.acc += dt
        steps = 0
        while self.acc >= w.dt and steps < 6:
            acts = {aid: b.act() for aid, b in self.bots.items()}
            if human is not None:
                acts[self.player] = human
            if self.server is not None:
                acts.update(self._remote_actions())
            w.step(acts)
            if self.server is not None:
                self._broadcast()
            self.acc -= w.dt
            steps += 1
            if human is not None:
                human.switch = 0
                self.ctrl.after_tick()
        if steps == 6:
            self.acc = 0.0
        if w.done:
            if self.done_at is None:
                self.done_at = time.perf_counter()
            elif self.server is not None and time.perf_counter() - self.done_at > self.RESTART_DELAY:
                self.restart()
        self.viewer.sync(dt, self.acc / w.dt)
        self.ctrl.update_fps(dt)

    def stop(self):
        self.ctrl.destroy()
        if self.server is not None:
            self.server.close()
            self.server = None


# ----------------------------------------------------------------------------- LAN client
class ClientSession:
    INTERP = 0.05  # remote players are shown this far in the past (seconds)

    def __init__(self, viewer: Viewer, settings, client, on_escape=None, on_disconnect=None, autoplay=False):
        wel = client.welcome
        cfg = wel["cfg"]
        self.client = client
        self.viewer = viewer
        self.settings = settings
        self.on_disconnect = on_disconnect
        self.aid = int(wel["aid"])
        self.room = wel.get("room", "")
        map_name = cfg["map_name"] if cfg.get("map_name") in MAPS else "transport_ship"
        mode = cfg.get("mode") if cfg.get("mode") in MAPS[map_name][2] else default_mode(map_name)
        params = cfg.get("mode_params") if isinstance(cfg.get("mode_params"), dict) else {}
        self.world = w = World(MatchConfig(team_size=cfg["team_size"], kill_limit=cfg["kill_limit"],
                                           time_limit=cfg["time_limit"], respawn=cfg["respawn"],
                                           friendly_fire=cfg["friendly_fire"], map_name=map_name, mode=mode,
                                           mode_params=params, tick_rate=cfg["tick_rate"],
                                           primaries=cfg["primaries"]))
        w.event_log.clear()
        w.seq = 0
        st = wel["w"]
        apply_world(w, st, own=self.aid)
        apply_agent(w.agents[self.aid], st["agents"][self.aid])
        for a in w.agents:
            a.prev_pos[:] = a.pos
        self.roster = wel.get("roster", [])
        viewer.bind(w, self.aid)
        viewer.set_local_skins(settings.skins)
        viewer.set_cinematic(False)
        self._apply_roster()
        round_mode = not w.mode.free_weapons
        self.ctrl = PlayerController(viewer, settings, on_escape=on_escape, on_primary=self._set_primary,
                                     on_buy=self._buy if round_mode else None,
                                     on_operator=self._operator if round_mode else None)
        if getattr(settings, "primary", None) in PRIMARY_CHOICES:
            client.send({"t": "cmd", "c": "primary", "v": settings.primary})
        self.seq = 0
        self.pending: collections.deque = collections.deque()
        self.buffers = {a.id: collections.deque(maxlen=90) for a in w.agents if a.id != self.aid}
        self.snap_time = st["time"]
        self.snap_recv = time.perf_counter()
        self.adopt_look = False
        self.acc = 0.0
        self.bot = Bot(w, self.aid, "hard", seed=self.aid) if autoplay else None
        if viewer.hud is not None:
            zh = viewer.hud.cjk
            viewer.hud.end_hint = "等待主机开始下一局" if zh else "Waiting for the host"
            viewer.hud.set_info(f"已加入房间: {self.room}" if zh else f"Joined room: {self.room}")
        if not autoplay:
            self.ctrl.capture(True)

    def _set_primary(self, name):
        self.client.send({"t": "cmd", "c": "primary", "v": name})
        a = self.world.agents[self.aid]
        return a.alive and self.world.time - a.spawn_time < 6.0

    def _buy(self, name):
        self.client.send({"t": "cmd", "c": "buy", "v": name})

    def _operator(self, name, primary=None):
        self.client.send({"t": "cmd", "c": "operator", "v": name, "p": primary})

    def _apply_roster(self):
        for aid, name, skins, _human in self.roster:
            a = self.world.agents[aid]
            a.name = name
            if aid != self.aid:
                self.viewer.agent_skins[aid] = skins
        self.viewer.agent_skins[self.aid] = self.viewer.local_skins

    # ------------------------------------------------------------------ network
    def _render_time(self):
        est = self.snap_time + min(time.perf_counter() - self.snap_recv, 0.1)
        return est - self.INTERP

    def _on_snapshot(self, msg):
        w = self.world
        for e in msg["ev"]:
            w.seq += 1
            e["seq"] = w.seq
            w.event_log.append(e)
            if e["type"] == "spawn" and e.get("agent") == self.aid:
                self.adopt_look = True
        st = msg["w"]
        me = w.agents[self.aid]
        apply_world(w, st, own=self.aid)
        if self.adopt_look:
            me.yaw, me.pitch = st["agents"][self.aid][2], st["agents"][self.aid][3]
            me.prev_pos[:] = me.pos
            self.pending.clear()
            self.adopt_look = False
        t = st["time"]
        for aid, buf in self.buffers.items():
            s = st["agents"][aid]
            buf.append((t, s[0], s[4], s[2], s[3]))
        self.snap_time = t
        self.snap_recv = time.perf_counter()
        ack = 0
        for aid, seq in msg.get("ack", ()):
            if aid == self.aid:
                ack = seq
        while self.pending and self.pending[0][0] <= ack:
            self.pending.popleft()
        if me.alive and self.pending:
            yaw_now, t_save = me.yaw, w.time
            w.muted = True
            for k, (_, act, yaw) in enumerate(self.pending):
                w.time = t + k * w.dt
                me.yaw = yaw
                w._move(me, act, w.dt)
            w.muted = False
            me.yaw, w.time = yaw_now, t_save

    def _interpolate(self):
        rt = self._render_time()
        w = self.world
        for aid, buf in self.buffers.items():
            if not buf:
                continue
            a = w.agents[aid]
            prev = buf[0]
            cur = None
            for item in buf:
                if item[0] >= rt:
                    cur = item
                    break
                prev = item
            if cur is None or cur is prev:
                t, pos, h, yaw, pitch = buf[-1]
                a.pos[:] = pos
                a.height, a.yaw, a.pitch = h, yaw, pitch
            else:
                t0, p0, h0, y0, q0 = prev
                t1, p1, h1, y1, q1 = cur
                k = (rt - t0) / max(t1 - t0, 1e-6)
                if sum((b - c) ** 2 for b, c in zip(p0, p1)) > 9.0:  # respawn teleport
                    k = 1.0
                a.pos[:] = [b + (c - b) * k for b, c in zip(p0, p1)]
                a.height = h0 + (h1 - h0) * k
                a.yaw = (y0 + ((y1 - y0 + 180.0) % 360.0 - 180.0) * k) % 360.0
                a.pitch = q0 + (q1 - q0) * k
            a.prev_pos[:] = a.pos

    def update(self, dt):
        w = self.world
        me = w.agents[self.aid]
        for msg in self.client.poll():
            t = msg.get("t")
            if t == "s":
                self._on_snapshot(msg)
            elif t == "roster":
                self.roster = msg.get("p", [])
                self._apply_roster()
            elif t == "reset":
                self.pending.clear()
                for buf in self.buffers.values():
                    buf.clear()
                self.viewer.bind(w, self.aid)
                self._apply_roster()
            elif t == "bye":
                if self.on_disconnect is not None:
                    self.on_disconnect(msg.get("reason", "disconnected"))
                return
        self.ctrl.mouse_look(me, self.bot is not None)
        act = self.ctrl.read_input()
        self.acc += dt
        steps = 0
        while self.acc >= w.dt and steps < 6:
            if self.bot is not None:
                act = self.bot.act()
                if me.alive:
                    me.yaw = (me.yaw + act.look_yaw) % 360.0
                    me.pitch = max(-89.0, min(89.0, me.pitch + act.look_pitch))
                act.look_yaw = act.look_pitch = 0.0
            self.seq += 1
            self.client.send({"t": "in", "i": [encode_input(self.seq, act, me.yaw, me.pitch, self._render_time())]})
            if me.alive and not w.done:
                t_save = w.time
                w.time = self.snap_time + len(self.pending) * w.dt
                me.prev_pos[:] = me.pos
                w.muted = True
                w._move(me, act, w.dt)
                w.muted = False
                w.time = t_save
                self.pending.append((self.seq, copy.copy(act), me.yaw))
                while len(self.pending) > 120:
                    self.pending.popleft()
            act.switch = 0
            self.ctrl.after_tick()
            self.acc -= w.dt
            steps += 1
        if steps == 6:
            self.acc = 0.0
        self._interpolate()
        self.viewer.sync(dt, self.acc / w.dt)
        self.ctrl.update_fps(dt)

    def stop(self):
        self.ctrl.destroy()
        self.client.close()


# ----------------------------------------------------------------------------- quick play
class GameApp:
    """Stand-alone match without the main menu."""

    def __init__(self, team_size=4, difficulty="normal", primary="ak47", kill_limit=60, time_limit=420.0,
                 size=(1600, 900), fullscreen=False, vsync=True, shadows=True, msaa=4, sensitivity=1.6,
                 autoplay=False, seed=None, screenshot=None, shot_time=6.0, team=0, settings=None,
                 host=None, map_name="transport_ship", mode=None, mode_params=None):
        self.settings = settings or _Settings(sensitivity=sensitivity)
        mode = mode if mode in MAPS[map_name][2] else default_mode(map_name)
        world = World(MatchConfig(team_size=team_size, map_name=map_name, mode=mode))
        self.viewer = Viewer(world, pov=0, onscreen=True, size=size, shadows=shadows, vsync=vsync, msaa=msaa,
                             fullscreen=fullscreen, title=f"FPS-RL  |  {MAPS[map_name][1]}")
        self.base = self.viewer.base
        self.base.ignore("c")
        self.base.ignore("v")
        self.session = LocalSession(self.viewer, self.settings, team_size, difficulty, team, primary, kill_limit,
                                    time_limit, seed, autoplay, host=host, map_name=map_name, mode=mode,
                                    mode_params=mode_params)
        self.world = self.session.world
        self.screenshot = screenshot
        self.shot_time = shot_time
        self.base.taskMgr.add(self.update, "game-update", sort=-10)
        self.base.exitFunc = self._on_exit

    def _on_exit(self):
        hold_ime(self.base.win, True)
        self.session.stop()

    def update(self, task):
        dt = min(globalClock.getDt(), 0.1)  # noqa: F821  (Panda3D builtin)
        hold_ime(self.base.win, False)
        self.session.update(dt)
        if self.screenshot and self.world.time >= self.shot_time:
            self.base.graphicsEngine.renderFrame()
            self.base.win.saveScreenshot(Filename.fromOsSpecific(self.screenshot))
            print("saved", self.screenshot)
            self.screenshot = None
            self.base.userExit()
        return task.cont

    def run(self):
        self.base.run()

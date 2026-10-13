"""Top-level application: main menu <-> game sessions (single player, LAN host, LAN client)."""
from __future__ import annotations

from ..bots import Bot
from ..profile import Profile
from ..sim import MatchConfig, World
from .game import ClientSession, LocalSession
from .ime import hold as hold_ime
from .menu import MainMenu, PauseMenu, set_borderless
from .viewer import Viewer


class App:
    def __init__(self, size=(1600, 900), fullscreen=None, vsync=True, shadows=True, msaa=4):
        self.profile = Profile.load()
        if fullscreen is not None:
            self.profile.borderless = fullscreen
        self.size = size
        # a bot match keeps running behind the menu
        self.menu_world = World(MatchConfig(team_size=4, kill_limit=0, time_limit=0, seed=3,
                                            primaries=["ak47", "awm", "ak47", "ak47"] * 2))
        self.menu_bots = [Bot(self.menu_world, a.id, "hard", seed=a.id) for a in self.menu_world.agents]
        self.viewer = Viewer(self.menu_world, pov=0, onscreen=True, size=size, shadows=shadows, vsync=vsync,
                             msaa=msaa, fullscreen=False, title="FPS-RL | 运输船")
        self.base = base = self.viewer.base
        base.ignore("c")
        base.ignore("v")
        self.apply_volume()
        if self.profile.borderless:
            set_borderless(base, True, size)
        self.session = None
        self.acc = 0.0
        self.menu = MainMenu(self)
        self.pause = PauseMenu(self)
        base.taskMgr.add(self._update, "app-update", sort=-10)
        base.exitFunc = self._on_exit
        self.show_menu()

    # ------------------------------------------------------------------ settings
    def ui_sound(self, name):
        if self.viewer.audio is not None:
            self.viewer.audio.ui(name, 0.5)

    def apply_volume(self):
        if self.viewer.audio is not None:
            self.viewer.audio.master = self.profile.volume

    def apply_borderless(self):
        set_borderless(self.base, self.profile.borderless, self.size)

    # ------------------------------------------------------------------ flow
    def show_menu(self, message=None):
        if self.session is not None:
            self.session.stop()
            self.session = None
        self.pause.hide()
        self.viewer.bind(self.menu_world, 0)
        self.viewer.set_cinematic(True)
        self.menu.show(message)

    def _start(self, session):
        self.menu.hide()
        self.pause.hide()
        self.session = session
        self.profile.save()

    def start_single(self):
        p = self.profile
        s = p.single
        self._start(LocalSession(self.viewer, p, s["team_size"], s["difficulty"], s["team"], p.primary,
                                 s["kill_limit"], s["time_limit"], on_escape=self.toggle_pause,
                                 map_name=s.get("map", "transport_ship"), mode=s.get("mode")))

    def start_watch(self):
        p = self.profile
        s = p.watch
        self._start(LocalSession(self.viewer, p, s["team_size"], s["difficulty"], 0, p.primary,
                                 s["kill_limit"], s["time_limit"], on_escape=self.toggle_pause,
                                 map_name=s.get("map", "transport_ship"), mode=s.get("mode"), spectate=True))

    def start_host(self):
        p = self.profile
        h = p.host
        try:
            session = LocalSession(self.viewer, p, h["team_size"], h["difficulty"], 0, p.primary, h["kill_limit"],
                                   h["time_limit"], host={"room": h["room"], "port": h["port"]},
                                   on_escape=self.toggle_pause, map_name=h.get("map", "transport_ship"),
                                   mode=h.get("mode"))
        except OSError as exc:
            self.show_menu(f"无法创建房间 (端口 {h['port']} 被占用?): {exc}" if self.menu.ui.cjk else
                           f"Cannot host on port {h['port']}: {exc}")
            return
        self._start(session)

    def start_client(self, client):
        self._start(ClientSession(self.viewer, self.profile, client, on_escape=self.toggle_pause,
                                  on_disconnect=self._disconnected))

    def _disconnected(self, reason):
        self.show_menu(("已断开: " if self.menu.ui.cjk else "Disconnected: ") + str(reason))

    def toggle_pause(self):
        s = self.session
        if s is None:
            return
        if self.pause.visible:
            self.pause.hide()
            s.ctrl.enabled = True
            s.ctrl.capture(True)
        else:
            room = ""
            live = isinstance(s, ClientSession) or (isinstance(s, LocalSession) and s.server is not None)
            if isinstance(s, LocalSession) and s.server is not None:
                room = f"{s.server.room}  ·  port {s.server.port}  ·  {len(s.server.players()) + 1} players"
            elif isinstance(s, ClientSession):
                room = s.room
            s.ctrl.enabled = False
            s.ctrl.capture(False)
            self.pause.show(room, live=live)

    def leave_game(self):
        self.show_menu()

    def quit(self):
        self.base.userExit()

    def _on_exit(self):
        hold_ime(self.base.win, True)
        if self.session is not None:
            self.session.stop()
            self.session = None
        self.profile.save()

    # ------------------------------------------------------------------ loop
    def _update(self, task):
        dt = min(globalClock.getDt(), 0.1)  # noqa: F821  (Panda3D builtin)
        hold_ime(self.base.win, self.session is None)
        if self.session is not None:
            self.session.update(dt)
        else:
            w = self.menu_world
            self.acc += dt
            steps = 0
            while self.acc >= w.dt and steps < 3:
                w.step({b.aid: b.act() for b in self.menu_bots})
                self.acc -= w.dt
                steps += 1
            if steps == 3:
                self.acc = 0.0
            self.viewer.sync(dt, self.acc / w.dt)
        self.menu.update(dt)
        return task.cont

    def run(self):
        self.base.run()

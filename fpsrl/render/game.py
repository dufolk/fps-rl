"""Playable / spectator session of the experiment environment.

* `PlayerController` - keyboard / mouse -> `Action` for the local player (or the free camera).
* `LocalSession`     - elimination match against bots; `spectate=True` lets every agent be a bot.
* `GameApp`          - window + session (`play.py`).
"""
from __future__ import annotations

import math
import time

from direct.showbase.DirectObject import DirectObject
from panda3d.core import Filename, KeyboardButton, MouseButton, WindowProperties

from ..bots import Bot
from ..maps import MAPS
from ..sim import Action, MatchConfig, World
from ..weapons import LOADOUT_BY_KEY, LOADOUT_KEYS, NUM_SLOTS
from .ime import hold as hold_ime
from .viewer import BASE_HFOV, Viewer

SENSITIVITY_STEP = 0.1
GRAPHICS_PRESETS = {
    "low": {"shadows": False, "shadow_size": 1024, "msaa": 0},
    "medium": {"shadows": True, "shadow_size": 1024, "msaa": 2},
    "high": {"shadows": True, "shadow_size": 2048, "msaa": 4},
}

HELP_ZH = """[F1] 隐藏帮助
WASD 移动    空格 跳    Ctrl/C 蹲    Shift 静步    Q/E 左右探头
左键 开火/轻刀/按住捏雷松手投掷    右键 瞄准/开镜/重刀/低抛
R 换弹    1-7 武器    X 上一把    滚轮 切换    F 检视
B 切换武器方案（冻结时间内立即生效，否则下回合）    Tab 计分板
Esc 释放鼠标    F2 瞄准按住/切换    F3 第三人称    F4 FPS    F5 重开    +/- 灵敏度
死亡后 左键/右键 切换观战"""
HELP_EN = """[F1] hide help
WASD move    SPACE jump    CTRL/C crouch    SHIFT walk    Q/E lean
LMB fire / slash / hold to cook, release to throw    RMB aim / scope / heavy stab / underhand
R reload    1-7 weapons    X last weapon    wheel cycle    F inspect
B cycle loadout (now during freeze time, else next round)    TAB scoreboard
ESC release mouse    F2 aim hold/toggle    F3 third person    F4 fps    F5 restart    +/- sensitivity
When dead: LMB/RMB cycle spectate"""
WATCH_HELP_ZH = """[F1] 隐藏帮助
观战机器人。默认上帝视角，不受闪光弹影响
WASD 移动    鼠标 转向    Q 上升    E 下降    Shift 加速
F 切换第一人称 / 上帝视角
第一人称时 左键/右键 切换对象
Tab 计分板    Esc 释放鼠标    F5 重开"""
WATCH_HELP_EN = """[F1] hide help
Watch the bots. God view by default; flashbangs do not blind you
WASD move    mouse look    Q up    E down    SHIFT faster
F toggles first person / god view
In first person: LMB/RMB cycle targets
TAB scoreboard    ESC release mouse    F5 restart"""


class Settings:
    def __init__(self, sensitivity=1.6, ads_toggle=False, show_fps=False, name="Player"):
        self.sensitivity = sensitivity
        self.ads_toggle = ads_toggle
        self.show_fps = show_fps
        self.name = name


class PlayerController:
    def __init__(self, viewer: Viewer, settings, on_restart=None, on_loadout=None, watch=False):
        self.viewer = viewer
        self.base = base = viewer.base
        self.settings = settings
        self.on_restart = on_restart
        self.on_loadout = on_loadout
        self.enabled = True
        self.captured = False
        self.watch = watch
        self.fire_latch = self.alt_latch = False
        self.f_pending = 0
        self.ads_state = False
        self.switch_req = 0
        self.show_help = not watch
        self.fps_t, self.fps_n = time.perf_counter(), 0
        self.mw = base.mouseWatcherNode
        self.kb = {k: KeyboardButton.asciiKey(k) for k in "wasdcrqef"}
        self.ev = ev = DirectObject()
        ev.accept("mouse1", self._latch, ["fire"])
        ev.accept("mouse3", self._latch, ["alt"])
        ev.accept("wheel_up", self._cycle, [-1])
        ev.accept("wheel_down", self._cycle, [1])
        for i in range(1, NUM_SLOTS + 1):
            ev.accept(str(i), self._switch, [i])
        ev.accept("x", self._last)
        ev.accept("b", self._next_loadout)
        ev.accept("f", self._f_press)
        ev.accept("escape", lambda: self.capture(False))
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
    def _latch(self, which):
        if not self.enabled:
            return
        if not self.captured:
            self.capture(True)
            return
        if self.watch:
            if not self.viewer.god:
                self.viewer.cycle_follow(1 if which == "fire" else -1)
            return
        if self.viewer.spectate_active():
            self.viewer.cycle_spectate(1 if which == "fire" else -1)
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
            if ws and (ws.spec.kind != "grenade" or ws.mag > 0):
                break
        self.switch_req = slot

    def _f_press(self):
        if not (self.enabled and self.captured):
            return
        if self.watch:
            self.viewer.toggle_god()
        else:
            self.f_pending = 3

    def _next_loadout(self):
        if not self.enabled or self.watch or self.on_loadout is None:
            return
        a = self.viewer.world.agents[self.viewer.owner]
        cur = a.next_loadout or a.loadout
        key = LOADOUT_KEYS[(LOADOUT_KEYS.index(cur) + 1) % len(LOADOUT_KEYS)]
        now = self.on_loadout(key)
        _, zh, en, primary, _ = LOADOUT_BY_KEY[key]
        if self.zh:
            self.info(f"武器方案: {zh} ({primary.upper()})" + ("" if now else "  (下回合生效)"))
        else:
            self.info(f"Loadout: {en} ({primary.upper()})" + ("" if now else "  (next round)"))

    def _help(self):
        if self.hud is not None:
            if self.watch:
                full = WATCH_HELP_ZH if self.zh else WATCH_HELP_EN
            else:
                full = HELP_ZH if self.zh else HELP_EN
            text = full if self.show_help else "[F1] 帮助" if self.zh else "[F1] help"
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
        if self.enabled and not self.watch:
            self.viewer.toggle_third_person()

    def _toggle_fps(self):
        self.settings.show_fps = not self.settings.show_fps
        if not self.settings.show_fps:
            self.info("")

    def _restart(self):
        if self.enabled and self.on_restart is not None:
            self.on_restart()

    def _sens(self, d):
        self.settings.sensitivity = max(0.1, min(12.0, round(self.settings.sensitivity + d, 2)))
        self.info(f"{'灵敏度' if self.zh else 'Sensitivity'}: {self.settings.sensitivity:.2f}")

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
        if self.watch:
            if self.viewer.god:
                self.viewer.fly_look(dx, dy, self.settings.sensitivity)
            return
        if autoplay or not agent.alive:
            return
        zoom = 1.0
        if agent.ads and agent.slot in agent.inv and agent.spec.ads_fov:
            zoom = math.tan(math.radians(agent.spec.ads_fov / 2)) / math.tan(math.radians(BASE_HFOV / 2))
        k = 0.022 * self.settings.sensitivity * zoom
        agent.yaw = (agent.yaw - dx * k) % 360.0
        agent.pitch = max(-89.0, min(89.0, agent.pitch - dy * k))

    def fly_camera(self, dt):
        if not (self.enabled and self.captured and self.viewer.god):
            return
        down = self.mw.isButtonDown
        kb = self.kb
        forward = float(down(kb["w"])) - float(down(kb["s"]))
        right = float(down(kb["d"])) - float(down(kb["a"]))
        up = float(down(kb["q"])) - float(down(kb["e"]))
        self.viewer.fly_move(dt, forward, right, up, down(KeyboardButton.shift()))

    def read_input(self) -> Action:
        act = Action()
        if not (self.enabled and self.captured):
            return act
        if not self.viewer.world.agents[self.viewer.owner].alive:
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
        act.switch = self.switch_req
        return act

    def after_tick(self):
        self.fire_latch = self.alt_latch = False
        self.switch_req = 0
        if self.f_pending > 0:
            self.f_pending -= 1
            if self.f_pending == 0 and self.agent.alive and not self.viewer.spectate_active():
                self.viewer.inspect()

    def update_fps(self):
        self.fps_n += 1
        now = time.perf_counter()
        if now - self.fps_t > 0.5:
            fps = self.fps_n / (now - self.fps_t)
            self.fps_t, self.fps_n = now, 0
            if self.settings.show_fps:
                self.info(f"{fps:.0f} FPS")

    def destroy(self):
        self.ev.ignoreAll()
        self.capture(False)
        self.viewer.show_board = False


class LocalSession:
    """Elimination match on this machine. `spectate=False`: the player controls agent
    `team * team_size`; `spectate=True`: every agent is a bot and the camera is free."""

    def __init__(self, viewer: Viewer, settings, team_size=4, difficulty="normal", team=0, loadout="random",
                 seed=None, autoplay=False, map_name="desert2", mode_params=None, spectate=False):
        self.player = team * team_size
        self.spectate = spectate
        loadouts = [None] * (2 * team_size)  # bots: random fixed loadout each match
        if not spectate and loadout in LOADOUT_BY_KEY:
            loadouts[self.player] = loadout
        self.world = w = World(MatchConfig(team_size=team_size, loadouts=loadouts, seed=seed, map_name=map_name,
                                           mode_params=dict(mode_params or {})))
        if not spectate:
            w.agents[self.player].name = settings.name[:16] or w.agents[self.player].name
        self.viewer = viewer
        self.settings = settings
        self.autoplay = autoplay
        self.seed = seed
        self.bots = {a.id: Bot(w, a.id, difficulty, seed=(seed or 0) * 97 + a.id)
                     for a in w.agents if a.id != self.player or autoplay or spectate}
        viewer.bind(w, self.player)
        if spectate:
            viewer.start_watch()
        self.ctrl = PlayerController(viewer, settings, on_restart=self.restart,
                                     on_loadout=None if spectate else self._loadout, watch=spectate)
        self.acc = 0.0
        if viewer.hud is not None:
            viewer.hud.end_hint = "按 F5 重新开始" if viewer.hud.cjk else "F5 restart"
        if not autoplay:
            self.ctrl.capture(True)

    def _loadout(self, key):
        return self.world.set_loadout(self.world.agents[self.player], key)

    def restart(self):
        w = self.world
        w.reset(None if self.seed is None else self.seed + w.tick)
        for b in self.bots.values():
            b.reset()
        self.viewer.bind(w, self.player)
        if self.spectate:
            self.viewer.start_watch()
        self.acc = 0.0

    def update(self, dt):
        w = self.world
        me = w.agents[self.player]
        self.ctrl.mouse_look(me, self.autoplay)
        if self.spectate:
            self.ctrl.fly_camera(dt)
        human = None if self.autoplay or self.spectate else self.ctrl.read_input()
        self.acc += dt
        steps = 0
        # prefer a smooth frame over catching up every missed sim tick
        budget = time.perf_counter() + 0.010
        while self.acc >= w.dt and steps < 6:
            acts = {aid: b.act() for aid, b in self.bots.items()}
            if human is not None:
                acts[self.player] = human
            w.step(acts)
            self.acc -= w.dt
            steps += 1
            if human is not None:
                human.switch = 0
                self.ctrl.after_tick()
            if steps >= 2 and time.perf_counter() >= budget:
                break
        if steps == 6:
            self.acc = 0.0
        self.viewer.sync(dt, self.acc / w.dt)
        self.ctrl.update_fps()

    def stop(self):
        self.ctrl.destroy()


class GameApp:
    """Window + local session."""

    def __init__(self, team_size=4, difficulty="normal", loadout="random", size=(1600, 900), fullscreen=False,
                 vsync=True, sensitivity=1.6, autoplay=False, seed=None, screenshot=None, shot_time=6.0, team=0,
                 map_name="desert2", mode_params=None, spectate=False, graphics="medium"):
        self.settings = Settings(sensitivity=sensitivity)
        opts = GRAPHICS_PRESETS.get(graphics, GRAPHICS_PRESETS["medium"])
        world = World(MatchConfig(team_size=team_size, map_name=map_name))
        self.viewer = Viewer(world, pov=0, onscreen=True, size=size, shadows=opts["shadows"], vsync=vsync,
                             msaa=opts["msaa"], shadow_size=opts["shadow_size"], fullscreen=fullscreen,
                             title=f"FPS-RL  |  {MAPS[map_name][1]}  |  {'Spectate' if spectate else 'Play'}")
        self.base = self.viewer.base
        self.base.ignore("c")
        self.base.ignore("v")
        self.session = LocalSession(self.viewer, self.settings, team_size, difficulty, team, loadout, seed, autoplay,
                                    map_name=map_name, mode_params=mode_params, spectate=spectate)
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

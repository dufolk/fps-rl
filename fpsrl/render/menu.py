"""Main menu + in-game pause menu (DirectGUI)."""
from __future__ import annotations

import math
import threading

import numpy as np
from direct.gui import DirectGuiGlobals as DGG
from direct.gui.DirectGui import DirectButton, DirectEntry, DirectFrame, DirectLabel, DirectSlider
from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import (AmbientLight, Camera, DirectionalLight, NodePath, PerspectiveLens, TextNode,
                          TransparencyAttrib, Vec4, WindowProperties)

from ..weapons import WEAPONS
from .hud import load_font
from .skins import RARITY, SKIN_WEAPONS, SKINS, get_skin
from .textures import make_texture

ACCENT = (1.0, 0.56, 0.12, 1)
PANEL = (0.04, 0.05, 0.07, 0.84)
BTN = (0.13, 0.14, 0.17, 0.92)
BTN_HOVER = (0.95, 0.5, 0.1, 0.95)
BTN_PRESS = (0.75, 0.38, 0.06, 1)
BTN_OFF = (0.1, 0.1, 0.1, 0.5)
WHITE = (1, 1, 1, 1)
DIM = (0.7, 0.72, 0.76, 1)

DIFFICULTIES = [("easy", "简单", "Easy"), ("normal", "普通", "Normal"), ("hard", "困难", "Hard"),
                ("insane", "地狱", "Insane")]
SPAN = {"ak47": 1.1, "awm": 1.5, "deagle": 0.3, "knife": 0.36, "frag": 0.14, "smoke": 0.14}


def _gradient_tex():
    w = 256
    x = np.linspace(0, 1, w)
    a = np.clip(1.0 - x * 1.6, 0, 1) ** 1.4 * 0.85
    img = np.zeros((4, w, 4))
    img[..., 3] = a[None, :]
    return make_texture(img, "menu_grad", mipmap=False, clamp=True)


class UI:
    """Small widget factory sharing fonts / theme."""

    def __init__(self, base, sound=None):
        self.base = base
        self.font = load_font(base.loader, cjk=True)
        self.cjk = self.font is not None
        if self.font is None:
            self.font = load_font(base.loader)
        self.sound = sound or (lambda name: None)

    def L(self, zh, en):
        return zh if self.cjk else en

    def label(self, parent, text, pos, scale=0.045, fg=WHITE, align=TextNode.ALeft):
        return DirectLabel(parent=parent, text=text, text_font=self.font, text_scale=scale, text_fg=fg,
                           text_align=align, frameColor=(0, 0, 0, 0), pos=(pos[0], 0, pos[1]))

    def button(self, parent, text, pos, command, half=(0.3, 0.05), scale=0.045, args=None, color=BTN,
               hover=BTN_HOVER, align=TextNode.ACenter, fg=WHITE):
        w, h = half
        tx = 0 if align == TextNode.ACenter else -w + 0.04
        b = DirectButton(parent=parent, text=text, text_font=self.font, text_scale=scale, text_fg=fg,
                         text_align=align, text_pos=(tx, -scale * 0.36), frameSize=(-w, w, -h, h),
                         frameColor=(color, BTN_PRESS, hover, BTN_OFF), relief=DGG.FLAT,
                         pos=(pos[0], 0, pos[1]), command=self._wrap(command), extraArgs=args or [],
                         rolloverSound=None, clickSound=None, pressEffect=1)
        b.bind(DGG.ENTER, lambda _e: self.sound("ui_hover"))
        return b

    def _wrap(self, fn):
        def run(*a):
            self.sound("ui_click")
            fn(*a)
        return run

    def entry(self, parent, text, pos, width=12, scale=0.045):
        e = DirectEntry(parent=parent, initialText=text, entryFont=self.font, scale=scale, width=width,
                        pos=(pos[0], 0, pos[1] - scale * 0.3), frameColor=(0.16, 0.17, 0.2, 0.95),
                        text_fg=WHITE, relief=DGG.FLAT, pad=(0.3, 0.3), numLines=1, focus=0)
        return e

    def slider(self, parent, value, rng, pos, command, width=0.42):
        s = DirectSlider(parent=parent, range=rng, value=value, pageSize=(rng[1] - rng[0]) / 20,
                         pos=(pos[0], 0, pos[1]), scale=1, frameSize=(-width, width, -0.012, 0.012),
                         frameColor=(0.25, 0.26, 0.3, 1), thumb_frameColor=ACCENT,
                         thumb_frameSize=(-0.018, 0.018, -0.035, 0.035), thumb_relief=DGG.FLAT,
                         relief=DGG.FLAT)
        s["command"] = lambda: command(s["value"])
        return s


class Cycle:
    """<  value  > option selector."""

    def __init__(self, ui: UI, parent, label, options, value, z, on_change=None, x0=-1.0):
        self.ui = ui
        self.options = options  # [(value, text), ...]
        self.on_change = on_change
        vals = [o[0] for o in options]
        self.i = vals.index(value) if value in vals else 0
        ui.label(parent, label, (x0, z - 0.015), 0.042, DIM)
        ui.button(parent, "<", (x0 + 1.05, z), self._step, (0.04, 0.04), 0.045, [-1])
        ui.button(parent, ">", (x0 + 1.75, z), self._step, (0.04, 0.04), 0.045, [1])
        self.text = ui.label(parent, "", (x0 + 1.4, z - 0.015), 0.045, WHITE, TextNode.ACenter)
        self._refresh()

    def _step(self, d):
        self.i = (self.i + d) % len(self.options)
        self._refresh()
        if self.on_change:
            self.on_change(self.value)

    def _refresh(self):
        self.text["text"] = str(self.options[self.i][1])

    @property
    def value(self):
        return self.options[self.i][0]


class WeaponPreview:
    """Rotating 3D weapon in its own display region (armory page)."""

    def __init__(self, base, models, area):
        self.base = base
        self.models = models
        self.area = area  # aspect2d (x0, x1, z0, z1)
        self.root = NodePath("armory")
        self.root.setShaderAuto()
        sun = DirectionalLight("armory_sun")
        sun.setColor(Vec4(1.35, 1.25, 1.1, 1))
        sn = self.root.attachNewNode(sun)
        sn.setHpr(35, -30, 0)
        self.root.setLight(sn)
        rim = DirectionalLight("armory_rim")
        rim.setColor(Vec4(0.5, 0.6, 0.85, 1))
        rn = self.root.attachNewNode(rim)
        rn.setHpr(-140, 10, 0)
        self.root.setLight(rn)
        amb = AmbientLight("armory_amb")
        amb.setColor(Vec4(0.38, 0.4, 0.46, 1))
        self.root.setLight(self.root.attachNewNode(amb))
        lens = PerspectiveLens()
        lens.setFov(26)
        lens.setNearFar(0.05, 50)
        self.lens = lens
        self.cam = self.root.attachNewNode(Camera("armory_cam", lens))
        self.cam.setPos(0, -4.2, 0.35)
        self.cam.lookAt(0, 0, 0)
        self.dr = base.win.makeDisplayRegion()
        self.dr.setSort(20)
        self.dr.setCamera(self.cam)
        self.dr.setClearDepthActive(True)
        self.dr.setClearColorActive(True)
        self.dr.setClearColor((0.09, 0.1, 0.13, 1))
        self.pivot = self.root.attachNewNode("pivot")
        self.model = None
        self.t = 0.0
        self.key = None

    def show(self, weapon, skin):
        if (weapon, skin) == self.key:
            return
        self.key = (weapon, skin)
        if self.model is not None:
            self.model.removeNode()
        node, _ = self.models.weapon(weapon, skin)
        node.reparentTo(self.pivot)
        node.setScale(1.75 / SPAN.get(weapon, 1.0))
        b = node.getTightBounds()
        if b:
            c = (b[0] + b[1]) * 0.5
            node.setPos(-c)
        self.model = node

    def update(self, dt):
        self.t += dt
        self.pivot.setH(90 + math.sin(self.t * 0.6) * 40 + self.t * 6)
        self.pivot.setP(math.sin(self.t * 0.45) * 8)
        win = self.base.win
        aspect = win.getXSize() / max(1, win.getYSize())
        x0, x1, z0, z1 = self.area
        l, r = (x0 / aspect + 1) / 2, (x1 / aspect + 1) / 2
        b, t = (z0 + 1) / 2, (z1 + 1) / 2
        self.dr.setDimensions(max(0, l), min(1, r), max(0, b), min(1, t))
        self.lens.setAspectRatio(((r - l) * win.getXSize()) / max(1, (t - b) * win.getYSize()))

    def destroy(self):
        self.dr.setActive(False)
        self.base.win.removeDisplayRegion(self.dr)
        self.root.removeNode()


class MainMenu:
    def __init__(self, app):
        self.app = app
        self.base = base = app.base
        self.ui = UI(base, app.ui_sound)
        ui = self.ui
        L = ui.L
        self.root = base.aspect2d.attachNewNode("menu")
        self.overlay = OnscreenImage(_gradient_tex(), scale=(1.0, 1, 1.0), parent=base.a2dLeftCenter)
        self.overlay.setTransparency(TransparencyAttrib.M_alpha)
        self.overlay.setScale(1.9, 1, 1.0)
        self.overlay.setPos(1.9, 0, 0)
        self.overlay.setBin("background", 0)
        self.left = base.a2dLeftCenter.attachNewNode("menu_left")
        self.top = base.a2dTopLeft.attachNewNode("menu_top")
        ui.label(self.top, "FPS-RL", (0.12, -0.2), 0.13, ACCENT)
        ui.label(self.top, L("运输船  ·  Transport Ship", "Transport Ship"), (0.13, -0.27), 0.042, DIM)
        self.nav = {}
        items = [("single", L("单人训练", "Single Player")), ("host", L("创建房间", "Host LAN Room")),
                 ("join", L("加入房间", "Join LAN Room")), ("armory", L("武器仓库", "Armory")),
                 ("settings", L("设置", "Settings")), ("quit", L("退出游戏", "Quit"))]
        for i, (key, text) in enumerate(items):
            self.nav[key] = ui.button(self.left, text, (0.42, 0.36 - i * 0.12), self.open, (0.3, 0.048), 0.048,
                                      [key])
        self.status = ui.label(self.left, "", (0.12, -0.5), 0.036, (1, 0.6, 0.45, 1))
        self.panel = DirectFrame(parent=self.root, frameColor=PANEL, frameSize=(-1.15, 1.15, -0.8, 0.72),
                                 pos=(0.32, 0, -0.02))
        self.page = None
        self.page_key = None
        self.preview = None
        self.discoverer = None
        self.rooms_version = -1
        self.connecting = None
        self.visible = True
        self.open("single")

    # ------------------------------------------------------------------ helpers
    def show(self, message=None):
        self.visible = True
        for n in (self.root, self.left, self.top, self.overlay):
            n.show()
        self.set_status(message or "")
        self.open(self.page_key or "single")

    def hide(self):
        self.visible = False
        self._close_page()
        for n in (self.root, self.left, self.top, self.overlay):
            n.hide()

    def set_status(self, text):
        self.status["text"] = text

    def _close_page(self):
        if self.preview is not None:
            self.preview.destroy()
            self.preview = None
        if self.discoverer is not None:
            self.discoverer.stop()
            self.discoverer = None
        if self.page is not None:
            self.page.destroy()
            self.page = None

    def open(self, key):
        if key == "quit":
            self.app.quit()
            return
        self._close_page()
        self.page_key = key
        for k, b in self.nav.items():
            b["frameColor"] = (ACCENT if k == key else BTN, BTN_PRESS, BTN_HOVER, BTN_OFF)
        self.page = DirectFrame(parent=self.panel, frameColor=(0, 0, 0, 0))
        getattr(self, "_page_" + key)(self.page)

    def _title(self, page, text):
        self.ui.label(page, text, (-1.05, 0.58), 0.065, WHITE)
        DirectFrame(parent=page, frameColor=ACCENT, frameSize=(-1.05, -0.85, 0.535, 0.545))

    # ------------------------------------------------------------------ pages
    def _match_options(self, page, cfg, z0, team=True):
        ui, L = self.ui, self.ui.L
        rows = {}
        z = z0
        rows["team_size"] = Cycle(ui, page, L("每队人数", "Team size"), [(n, f"{n} v {n}") for n in range(1, 7)],
                                  cfg.get("team_size", 4), z)
        z -= 0.12
        rows["difficulty"] = Cycle(ui, page, L("机器人难度", "Bot difficulty"),
                                   [(k, L(zh, en)) for k, zh, en in DIFFICULTIES], cfg.get("difficulty", "normal"), z)
        z -= 0.12
        if team:
            rows["team"] = Cycle(ui, page, L("阵营", "Team"), [(0, "BL"), (1, "GR")], cfg.get("team", 0), z)
            z -= 0.12
        rows["primary"] = Cycle(ui, page, L("主武器", "Primary"), [("ak47", "AK-47"), ("awm", "AWM")],
                                self.app.profile.primary, z)
        z -= 0.12
        rows["kill_limit"] = Cycle(ui, page, L("击杀目标", "Kill limit"),
                                   [(n, str(n)) for n in (20, 40, 60, 100, 150)], cfg.get("kill_limit", 60), z)
        z -= 0.12
        rows["time_limit"] = Cycle(ui, page, L("时间限制", "Time limit"),
                                   [(m * 60, L(f"{m} 分钟", f"{m} min")) for m in (5, 7, 10, 15, 20)],
                                   cfg.get("time_limit", 420), z)
        return rows, z

    def _page_single(self, page):
        ui, L = self.ui, self.ui.L
        self._title(page, L("单人训练", "Single Player"))
        ui.label(page, L("与机器人对战，熟悉地图与枪械手感。", "Fight bots to learn the map and the guns."),
                 (-1.05, 0.44), 0.036, DIM)
        rows, z = self._match_options(page, self.app.profile.single, 0.3)

        def start():
            cfg = {k: r.value for k, r in rows.items()}
            self.app.profile.primary = cfg.pop("primary")
            self.app.profile.single.update(cfg)
            self.app.start_single()
        ui.button(page, L("开始游戏", "START"), (0.75, -0.66), start, (0.3, 0.07), 0.058, color=ACCENT,
                  hover=(1, 0.7, 0.3, 1))

    def _page_host(self, page):
        ui, L = self.ui, self.ui.L
        prof = self.app.profile
        self._title(page, L("创建房间", "Host LAN Room"))
        ui.label(page, L("局域网内的玩家可在“加入房间”中看到你的房间；空位由机器人补齐。",
                         "Players on your LAN will see this room; empty slots are filled with bots."),
                 (-1.05, 0.44), 0.034, DIM)
        ui.label(page, L("房间名称", "Room name"), (-1.0, 0.285), 0.042, DIM)
        room = ui.entry(page, prof.host.get("room", "FPS-RL"), (0.05, 0.3), 14)
        ui.label(page, L("端口", "Port"), (-1.0, 0.165), 0.042, DIM)
        port = ui.entry(page, str(prof.host.get("port", 47777)), (0.05, 0.18), 6)
        rows, z = self._match_options(page, prof.host, 0.06, team=False)

        def start():
            try:
                p = int(port.get().strip())
            except ValueError:
                self.set_status(L("端口必须是数字", "Port must be a number"))
                return
            cfg = {k: r.value for k, r in rows.items()}
            prof.primary = cfg.pop("primary")
            prof.host.update(cfg, room=room.get().strip()[:24] or "FPS-RL", port=p)
            self.app.start_host()
        ui.button(page, L("创建并开始", "HOST"), (0.75, -0.66), start, (0.3, 0.07), 0.058, color=ACCENT,
                  hover=(1, 0.7, 0.3, 1))

    def _page_join(self, page):
        from ..net.client import Discoverer
        ui, L = self.ui, self.ui.L
        prof = self.app.profile
        self._title(page, L("加入房间", "Join LAN Room"))
        ui.label(page, L("正在搜索局域网房间…", "Searching the LAN for rooms..."), (-1.05, 0.44), 0.034, DIM)
        self.room_frame = DirectFrame(parent=page, frameColor=(0, 0, 0, 0.25), frameSize=(-1.07, 1.07, -0.2, 0.38))
        self.room_widgets = []
        self.rooms_version = -1
        self.discoverer = Discoverer()
        ui.label(page, L("手动连接 IP", "Direct IP"), (-1.0, -0.33), 0.04, DIM)
        self.ip_entry = ui.entry(page, prof.last_ip or "127.0.0.1", (-0.45, -0.315), 11)
        ui.label(page, L("端口", "Port"), (0.2, -0.33), 0.04, DIM)
        self.port_entry = ui.entry(page, str(prof.last_port), (0.36, -0.315), 5)
        ui.button(page, L("连接", "Connect"), (0.85, -0.32), self._join_manual, (0.17, 0.045), 0.042)
        self.team_cycle = Cycle(ui, page, L("偏好阵营", "Team preference"),
                                [(-1, L("自动", "Auto")), (0, "BL"), (1, "GR")], prof.team_pref, -0.47)
        self.join_status = ui.label(page, "", (-1.0, -0.64), 0.04, ACCENT)

    def _refresh_rooms(self):
        ui, L = self.ui, self.ui.L
        for w in self.room_widgets:
            w.destroy()
        self.room_widgets = []
        rooms = self.discoverer.rooms if self.discoverer else []
        if not rooms:
            self.room_widgets.append(ui.label(self.room_frame, L("暂无房间。确认主机已创建房间，且防火墙允许 Python 通信。",
                                                                 "No rooms found. Check the host and firewall."),
                                              (-1.0, 0.27), 0.036, DIM))
        for i, r in enumerate(rooms[:6]):
            text = f"{r.get('room', '?')}   {r['ip']}:{r.get('port')}   " \
                   f"{L('玩家', 'players')} {r.get('players', '?')}/{r.get('slots', '?')}   " \
                   f"{L('比分', 'score')} {'-'.join(str(s) for s in r.get('score', []))}"
            self.room_widgets.append(ui.button(self.room_frame, text, (0.0, 0.31 - i * 0.095), self._join,
                                               (1.03, 0.04), 0.038, [r['ip'], int(r.get('port', 47777))],
                                               align=TextNode.ALeft))

    def _join_manual(self):
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            self.join_status["text"] = self.ui.L("端口必须是数字", "Port must be a number")
            return
        self._join(self.ip_entry.get().strip(), port)

    def _join(self, ip, port):
        if self.connecting is not None:
            return
        prof = self.app.profile
        prof.last_ip, prof.last_port = ip, port
        prof.team_pref = self.team_cycle.value
        self.join_status["text"] = self.ui.L(f"正在连接 {ip}:{port} …", f"Connecting to {ip}:{port} ...")
        result = {}

        def work():
            from ..net.client import ConnectError, NetClient
            try:
                result["client"] = NetClient(ip, port, prof.name, prof.skins, prof.team_pref)
            except ConnectError as exc:
                result["error"] = str(exc)
        self.connecting = (threading.Thread(target=work, daemon=True), result)
        self.connecting[0].start()

    def _page_armory(self, page):
        ui, L = self.ui, self.ui.L
        self._title(page, L("武器仓库", "Armory"))
        ui.label(page, L("选择皮肤后自动装备；游戏中按 F 检视武器。", "Click a skin to equip it. Press F in game to inspect."),
                 (-1.05, 0.44), 0.034, DIM)
        self.arm_weapon = getattr(self, "arm_weapon", "ak47")
        self.arm_tabs = {}
        for i, w in enumerate(SKIN_WEAPONS):
            self.arm_tabs[w] = ui.button(page, WEAPONS[w].display, (-0.88 + i * 0.37, 0.33), self._arm_select,
                                         (0.17, 0.04), 0.036, [w])
        self.arm_list = DirectFrame(parent=page, frameColor=(0, 0, 0, 0))
        self.arm_caption = ui.label(page, "", (0.45, -0.66), 0.045, WHITE, TextNode.ACenter)
        self.arm_rarity = ui.label(page, "", (0.45, -0.72), 0.034, DIM, TextNode.ACenter)
        px, pz = self.panel.getPos()[0], self.panel.getPos()[2]
        self.preview = WeaponPreview(self.base, self.app.viewer.models, (px - 0.08, px + 1.1, pz - 0.6, pz + 0.26))
        self._arm_select(self.arm_weapon)

    def _arm_select(self, weapon):
        ui, L = self.ui, self.ui.L
        self.arm_weapon = weapon
        for w, b in self.arm_tabs.items():
            b["frameColor"] = (ACCENT if w == weapon else BTN, BTN_PRESS, BTN_HOVER, BTN_OFF)
        for c in self.arm_list.getChildren():
            c.removeNode()
        equipped = self.app.profile.skins.get(weapon, "default")
        for i, sk in enumerate(SKINS[weapon]):
            col = RARITY[sk.rarity][2]
            mark = self.ui.L("  · 已装备", "  · equipped") if sk.key == equipped else ""
            b = ui.button(self.arm_list, f"{L(sk.zh, sk.en)}{mark}", (-0.75, 0.2 - i * 0.105), self._arm_equip,
                          (0.3, 0.043), 0.04, [weapon, sk.key], align=TextNode.ALeft,
                          color=(0.2, 0.21, 0.25, 0.95) if sk.key == equipped else BTN)
            DirectFrame(parent=b, frameColor=col, frameSize=(-0.3, -0.288, -0.043, 0.043))
        self._arm_show(weapon, equipped)

    def _arm_show(self, weapon, key):
        sk = get_skin(weapon, key)
        rz, ren, col = RARITY[sk.rarity]
        self.arm_caption["text"] = f"{WEAPONS[weapon].display}  |  {self.ui.L(sk.zh, sk.en)}"
        self.arm_caption["text_fg"] = col
        self.arm_rarity["text"] = self.ui.L(f"{rz}  ·  {sk.en}", ren)
        if self.preview is not None:
            self.preview.show(weapon, sk.key)

    def _arm_equip(self, weapon, key):
        self.app.profile.skins[weapon] = key
        self.app.profile.save()
        self._arm_select(weapon)

    def _page_settings(self, page):
        ui, L = self.ui, self.ui.L
        prof = self.app.profile
        self._title(page, L("设置", "Settings"))
        ui.label(page, L("玩家名称", "Player name"), (-1.0, 0.385), 0.042, DIM)
        name = ui.entry(page, prof.name, (0.05, 0.4), 12)

        def set_name(_=None):
            prof.name = name.get().strip()[:16] or prof.name
            prof.save()
        name["command"] = set_name
        name["focusOutCommand"] = set_name
        sens_lbl = ui.label(page, "", (0.98, 0.265), 0.042, WHITE, TextNode.ARight)

        def on_sens(v):
            prof.sensitivity = round(v, 2)
            sens_lbl["text"] = f"{prof.sensitivity:.2f}"
        ui.label(page, L("鼠标灵敏度", "Mouse sensitivity"), (-1.0, 0.265), 0.042, DIM)
        ui.slider(page, prof.sensitivity, (0.2, 6.0), (0.4, 0.28), on_sens)
        on_sens(prof.sensitivity)
        vol_lbl = ui.label(page, "", (0.98, 0.145), 0.042, WHITE, TextNode.ARight)

        def on_vol(v):
            prof.volume = round(v, 2)
            vol_lbl["text"] = f"{int(prof.volume * 100)}%"
            self.app.apply_volume()
        ui.label(page, L("主音量", "Master volume"), (-1.0, 0.145), 0.042, DIM)
        ui.slider(page, prof.volume, (0.0, 1.0), (0.4, 0.16), on_vol)
        on_vol(prof.volume)

        def flag(attr):
            def set_(v):
                setattr(prof, attr, v)
                prof.save()
                if attr == "fullscreen":
                    self.app.apply_fullscreen()
            return set_
        Cycle(ui, page, L("开镜方式", "Aim mode"), [(False, L("按住", "Hold")), (True, L("切换", "Toggle"))],
              prof.ads_toggle, 0.03, flag("ads_toggle"))
        Cycle(ui, page, L("显示帧率", "Show FPS"), [(False, L("关", "Off")), (True, L("开", "On"))],
              prof.show_fps, -0.09, flag("show_fps"))
        Cycle(ui, page, L("全屏", "Fullscreen"), [(False, L("关", "Off")), (True, L("开", "On"))],
              prof.fullscreen, -0.21, flag("fullscreen"))
        ui.label(page, L("设置自动保存到 ~/.fpsrl/profile.json", "Settings are saved to ~/.fpsrl/profile.json"),
                 (-1.0, -0.66), 0.032, DIM)
        self._settings_save = lambda: (set_name(), prof.save())

    # ------------------------------------------------------------------ per frame
    def update(self, dt):
        if not self.visible:
            return
        if self.preview is not None:
            self.preview.update(dt)
        if self.discoverer is not None and self.discoverer.version != self.rooms_version:
            self.rooms_version = self.discoverer.version
            self._refresh_rooms()
        if self.connecting is not None and not self.connecting[0].is_alive():
            _, result = self.connecting
            self.connecting = None
            if "client" in result:
                self.app.start_client(result["client"])
            else:
                if self.page_key == "join":
                    self.join_status["text"] = result.get("error", "?")
                self.set_status(result.get("error", ""))


class PauseMenu:
    def __init__(self, app):
        self.app = app
        base = app.base
        self.ui = ui = UI(base, app.ui_sound)
        L = ui.L
        self.root = base.aspect2d.attachNewNode("pause")
        self.root.setBin("gui-popup", 50)
        DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0.55), frameSize=(-4, 4, -1, 1))
        DirectFrame(parent=self.root, frameColor=PANEL, frameSize=(-0.6, 0.6, -0.52, 0.5))
        ui.label(self.root, L("暂停", "PAUSED"), (0, 0.36), 0.075, ACCENT, TextNode.ACenter)
        self.room_lbl = ui.label(self.root, "", (0, 0.29), 0.034, DIM, TextNode.ACenter)
        ui.button(self.root, L("继续游戏", "Resume"), (0, 0.17), app.toggle_pause, (0.4, 0.055), 0.048)
        prof = app.profile
        self.sens_lbl = ui.label(self.root, "", (0.5, 0.035), 0.04, WHITE, TextNode.ARight)
        ui.label(self.root, L("灵敏度", "Sensitivity"), (-0.5, 0.035), 0.04, DIM)
        self.sens = ui.slider(self.root, prof.sensitivity, (0.2, 6.0), (0.05, 0.05), self._on_sens, 0.22)
        self.vol_lbl = ui.label(self.root, "", (0.5, -0.075), 0.04, WHITE, TextNode.ARight)
        ui.label(self.root, L("音量", "Volume"), (-0.5, -0.075), 0.04, DIM)
        self.vol = ui.slider(self.root, prof.volume, (0.0, 1.0), (0.05, -0.06), self._on_vol, 0.22)
        ui.button(self.root, L("返回主菜单", "Leave to main menu"), (0, -0.25), app.leave_game, (0.4, 0.055), 0.048)
        ui.button(self.root, L("退出游戏", "Quit"), (0, -0.38), app.quit, (0.4, 0.055), 0.048)
        self.visible = False
        self.root.hide()

    def _on_sens(self, v):
        self.app.profile.sensitivity = round(v, 2)
        self.sens_lbl["text"] = f"{self.app.profile.sensitivity:.2f}"

    def _on_vol(self, v):
        self.app.profile.volume = round(v, 2)
        self.vol_lbl["text"] = f"{int(v * 100)}%"
        self.app.apply_volume()

    def show(self, room_text=""):
        prof = self.app.profile
        self.sens["value"] = prof.sensitivity
        self.vol["value"] = prof.volume
        self._on_sens(prof.sensitivity)
        self._on_vol(prof.volume)
        self.room_lbl["text"] = room_text
        self.visible = True
        self.root.show()

    def hide(self):
        self.visible = False
        self.root.hide()


def set_fullscreen(base, on: bool, windowed_size=(1600, 900)):
    props = WindowProperties()
    if on:
        di = base.pipe.getDisplayInformation()
        w, h = di.getDisplayModeWidth(0), di.getDisplayModeHeight(0)
        if w <= 0:
            w, h = 1920, 1080
        props.setSize(w, h)
    else:
        props.setSize(*windowed_size)
    props.setFullscreen(on)
    base.win.requestProperties(props)

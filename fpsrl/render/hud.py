"""2D heads-up display (PUBG-style layout: top-left squad, top-right minimap, bottom weapon dock)."""
from __future__ import annotations

import math
import os

import numpy as np
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (CardMaker, CullFaceAttrib, Filename, NodePath, Point3, SamplerState, TextNode,
                          Texture, TransparencyAttrib)

from ..sim import LEAN_DIST, LEAN_ROLL, TEAM_NAMES
from ..weapons import LOADOUT_BY_KEY, NUM_SLOTS, SLOT_PISTOL, SLOT_PRIMARY, item_name
from .textures import make_texture

TEAM_COLORS = {0: (0.28, 0.78, 0.95, 1), 1: (0.95, 0.42, 0.32, 1)}
SQUAD_BADGE_COLORS = (
    (0.88, 0.78, 0.18, 0.95),
    (0.92, 0.46, 0.18, 0.95),
    (0.24, 0.68, 0.92, 0.95),
    (0.32, 0.82, 0.36, 0.95),
    (0.75, 0.42, 0.92, 0.95),
)
CROSS_COLOR = (0.35, 1.0, 0.45, 0.95)
ROUND_REASONS = {"elimination": ("全歼对方", "Elimination"), "time": ("时间耗尽", "Time ran out"),
                 "draw": ("双方同归于尽", "Both teams wiped")}


def scope_reticle_roll(lean_off: float) -> float:
    """Degrees the scope cross turns while the view stays upright. Positive lean rolls it clockwise."""
    k = 0.0 if not LEAN_DIST else float(lean_off) / LEAN_DIST
    return -k * LEAN_ROLL


def _rect(parent, color, l, r, b, t, name="rect"):
    cm = CardMaker(name)
    cm.setFrame(l, r, b, t)
    n = parent.attachNewNode(cm.generate())
    n.setColor(*color)
    n.setTransparency(TransparencyAttrib.M_alpha)
    return n


LATIN_FONTS = ("C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/arialbd.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
CJK_FONTS = ("C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
             "/System/Library/Fonts/PingFang.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
             "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
_FONTS = {}
_MINIMAP_CACHE: dict[str, tuple[Texture, tuple[float, float, float, float]]] = {}


def load_font(loader, cjk=False):
    """Bold UI font; cjk=True returns a font with Chinese glyphs (None if unavailable)."""
    key = "cjk" if cjk else "latin"
    if key in _FONTS:
        return _FONTS[key]
    font = None
    for path in (CJK_FONTS if cjk else LATIN_FONTS):
        if os.path.exists(path):
            try:
                font = loader.loadFont(Filename.fromOsSpecific(path).getFullpath())
                if font is None or not font.isValid():
                    font = None
                    continue
                font.setPixelsPerUnit(64)
                break
            except Exception:
                font = None
    _FONTS[key] = font
    return font


def loadout_label(key: str, zh: bool) -> str:
    lo = LOADOUT_BY_KEY.get(key)
    return (lo[1] if zh else lo[2]) if lo else ""


def _build_minimap_texture(m) -> tuple[Texture, tuple[float, float, float, float]]:
    if m.name in _MINIMAP_CACHE:
        return _MINIMAP_CACHE[m.name]
    (x0, y0), (x1, y1) = m.nav_bounds or m.bounds
    pad = 8.0
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    S = 256
    mn = np.array([b.mn for b in m.boxes if b.solid], dtype=np.float64)
    mx = np.array([b.mx for b in m.boxes if b.solid], dtype=np.float64)
    zb, zt = float(mn[:, 2].min()), float(mx[:, 2].max())
    floor = np.full((S, S), np.nan)
    roof = np.zeros((S, S), dtype=bool)
    dx = (x1 - x0) / S
    dy = (y1 - y0) / S
    ground = mn[:, 2] <= zb + 0.25
    for k in range(len(mn)):
        if mx[k, 2] >= zt - 0.15:
            continue
        i0 = max(0, min(S, int((mn[k, 0] - x0) / dx)))
        i1 = max(0, min(S, int(math.ceil((mx[k, 0] - x0) / dx))))
        j0 = max(0, min(S, int((mn[k, 1] - y0) / dy)))
        j1 = max(0, min(S, int(math.ceil((mx[k, 1] - y0) / dy))))
        if i0 >= i1 or j0 >= j1:
            continue
        if ground[k]:
            sub = floor[j0:j1, i0:i1]
            sub[np.isnan(sub) | (sub < mx[k, 2])] = mx[k, 2]
        elif mx[k, 2] > 1.8:
            roof[j0:j1, i0:i1] = True
    rgb = np.zeros((S, S, 3), dtype=np.float32)
    rgb[:] = (0.20, 0.25, 0.19)
    walk = ~np.isnan(floor)
    rgb[walk] = (0.34, 0.36, 0.32)
    rgb[roof] = (0.58, 0.47, 0.36)
    # grid lines
    for g in range(32, S, 32):
        rgb[g, :] = rgb[g, :] * 0.8 + 0.08
        rgb[:, g] = rgb[:, g] * 0.8 + 0.08
    # poison zone circle
    zone = (m.objectives or {}).get("zone")
    if zone:
        yy, xx = np.mgrid[0:S, 0:S]
        wx = x0 + (xx + 0.5) * dx
        wy = y0 + (yy + 0.5) * dy
        cx, cy = zone["center"]
        dist = np.hypot(wx - cx, wy - cy)
        ring = np.abs(dist - zone["radius"]) < max(dx, dy) * 1.6
        outside = dist > zone["radius"]
        rgb[outside] = rgb[outside] * 0.55 + np.array([0.08, 0.18, 0.32], dtype=np.float32)
        rgb[ring] = (0.30, 0.78, 1.0)
    # row 0 in make_texture is top (+Y), so flip vertically before passing
    tex = make_texture(rgb[::-1], f"minimap_{m.name}", mipmap=False, clamp=True)
    tex.setMagfilter(SamplerState.FT_linear)
    _MINIMAP_CACHE[m.name] = (tex, (x0, y0, x1, y1))
    return _MINIMAP_CACHE[m.name]


class HUD:
    def __init__(self, base, textures, local_id: int):
        self.base = base
        self.textures = textures
        self.local_id = local_id
        self.font = load_font(base.loader)
        self.font_cjk = load_font(base.loader, cjk=True)
        self.cjk = self.font_cjk is not None
        a2d = base.aspect2d
        self.root = a2d.attachNewNode("hud")
        f = self.font
        cf = self.font_cjk or f

        def text(txt="", pos=(0, 0), scale=0.05, fg=(1, 1, 1, 1), align=TextNode.ACenter, parent=None, shadow=True,
                 font=None):
            return OnscreenText(text=txt, pos=pos, scale=scale, fg=fg, align=align, font=font or f,
                                shadow=(0, 0, 0, 0.75) if shadow else None, mayChange=True,
                                parent=parent or self.root)

        self._text = text
        # crosshair
        self.cross = self.root.attachNewNode("cross")
        self.cross_bars = []
        for _ in range(4):
            self.cross_bars.append(_rect(self.cross, CROSS_COLOR, -0.5, 0.5, -0.5, 0.5, "bar"))
        self.dot = _rect(self.cross, CROSS_COLOR, -0.003, 0.003, -0.003, 0.003, "dot")
        # hitmarker
        self.hitmarker = self.root.attachNewNode("hitmarker")
        for ang in (45, 135, 225, 315):
            b = _rect(self.hitmarker, (1, 1, 1, 1), 0.018, 0.042, -0.0028, 0.0028)
            b.setR(ang)
        self.hitmarker.hide()
        self.hit_t = 0.0
        self.hit_kill = False

        # ------------------------------------------------------------------ Top-left: Squad panel (队友信息)
        tl = base.a2dTopLeft
        self.squad_panel = tl.attachNewNode("squad")
        _rect(self.squad_panel, (0.05, 0.06, 0.08, 0.65), 0.02, 0.52, -0.075, -0.02)
        self.top_stat_text = text("淘汰 0   战损率 0.0", (0.035, -0.058), 0.028, (0.95, 0.95, 0.95, 1),
                                  TextNode.ALeft, parent=self.squad_panel, font=cf)
        self.squad_rows = []
        for i in range(5):
            y = -0.095 - i * 0.072
            row = self.squad_panel.attachNewNode(f"mate_{i}")
            _rect(row, (0.04, 0.05, 0.07, 0.58), 0.02, 0.44, -0.058, 0.0)
            badge_col = SQUAD_BADGE_COLORS[i % len(SQUAD_BADGE_COLORS)]
            badge = _rect(row, badge_col, 0.02, 0.062, -0.042, 0.0)
            num_t = text(str(i + 1), (0.041, -0.031), 0.027, (0.08, 0.08, 0.08, 1),
                         TextNode.ACenter, parent=row, shadow=False)
            name_t = text("", (0.074, -0.030), 0.028, (0.96, 0.96, 0.96, 1),
                          TextNode.ALeft, parent=row, font=cf)
            _rect(row, (0.18, 0.18, 0.18, 0.85), 0.02, 0.44, -0.058, -0.044)
            hp_fill = _rect(row, (0.95, 0.96, 0.96, 0.95), 0.0, 0.42, 0.0, 0.014)
            hp_fill.setPos(0.02, 0, -0.058)
            row.setPos(0, 0, y)
            self.squad_rows.append((row, badge, num_t, name_t, hp_fill))

        # ------------------------------------------------------------------ Top-center: Match score & alive pips
        top = base.a2dTopCenter
        self.score_panel = top.attachNewNode("score")
        _rect(self.score_panel, (0.06, 0.08, 0.1, 0.65), -0.46, 0.46, -0.135, -0.015)
        _rect(self.score_panel, (0.18, 0.52, 0.65, 0.82), -0.30, -0.09, -0.125, -0.025)
        _rect(self.score_panel, (0.72, 0.25, 0.22, 0.82), 0.09, 0.30, -0.125, -0.025)
        _rect(self.score_panel, (0.12, 0.14, 0.18, 0.9), -0.09, 0.09, -0.125, -0.025)
        self.side_l_label = text("己方" if self.cjk else "ALLY", (-0.23, -0.085), 0.030, (0.9, 0.96, 1, 1),
                                 parent=self.score_panel, font=cf)
        self.side_r_label = text("对方" if self.cjk else "ENEMY", (0.23, -0.085), 0.030, (1, 0.92, 0.9, 1),
                                 parent=self.score_panel, font=cf)
        self.score_l = text("0", (-0.135, -0.092), 0.055, (1, 1, 1, 1), parent=self.score_panel)
        self.score_r = text("0", (0.135, -0.092), 0.055, (1, 1, 1, 1), parent=self.score_panel)
        self.goal_text = text("歼灭" if self.cjk else "ELIM", (0, -0.055), 0.024, (0.95, 0.85, 0.45, 1),
                              parent=self.score_panel, font=cf)
        self.timer_text = text("0:00", (0, -0.105), 0.036, (1, 1, 1, 1), parent=self.score_panel)
        self.pips_l = [_rect(self.score_panel, (0.28, 0.78, 0.95, 0.95),
                             -0.33 - i * 0.025, -0.312 - i * 0.025, -0.085, -0.062) for i in range(5)]
        self.pips_r = [_rect(self.score_panel, (0.95, 0.42, 0.32, 0.95),
                             0.312 + i * 0.025, 0.33 + i * 0.025, -0.085, -0.062) for i in range(5)]
        self.phase_text = text("", (0, -0.185), 0.036, (1, 0.9, 0.55, 1), parent=self.score_panel, font=cf)

        # ------------------------------------------------------------------ Top-right: Minimap & Killfeed
        tr = base.a2dTopRight
        self.map_panel = tr.attachNewNode("minimap")
        self.map_size = 0.36
        self.map_pad = 0.025
        ml, mr = -self.map_pad - self.map_size, -self.map_pad
        mb, mt = -self.map_pad - self.map_size, -self.map_pad
        _rect(self.map_panel, (0.04, 0.05, 0.07, 0.85), ml - 0.008, mr + 0.008, mb - 0.052, mt + 0.008)
        cm = CardMaker("map_tex")
        cm.setFrame(ml, mr, mb, mt)
        self.map_card = self.map_panel.attachNewNode(cm.generate())
        self._map_name = None
        self._map_bounds = (-85.0, -85.0, 85.0, 85.0)
        self.map_fov = self.map_panel.attachNewNode("map_fov")
        cm_fov = CardMaker("fov_tri")
        cm_fov.setFrame(-0.028, 0.028, 0.0, 0.055)
        fov_q = self.map_fov.attachNewNode(cm_fov.generate())
        fov_q.setColor(1.0, 1.0, 1.0, 0.24)
        fov_q.setTransparency(TransparencyAttrib.M_alpha)
        self.map_dots = []
        for _ in range(10):
            d = _rect(self.map_panel, (1, 1, 1, 1), -0.009, 0.009, -0.009, 0.009, "dot")
            d.hide()
            self.map_dots.append(d)
        self.map_label = text("", ((ml + mr) * 0.5, mb - 0.036), 0.026, (0.9, 0.95, 1.0, 1),
                              TextNode.ACenter, parent=self.map_panel, font=cf)

        self.feed_root = tr.attachNewNode("feed")
        self.feed = []
        self.feed_texts = [text("", (-0.03, -0.46 - i * 0.052), 0.032, align=TextNode.ARight, parent=self.feed_root,
                                font=self.font_cjk) for i in range(6)]

        # ------------------------------------------------------------------ Bottom-center: Health bar & Weapon dock (下方枪械)
        bc = base.a2dBottomCenter
        self.bottom_panel = bc.attachNewNode("bottom_dock")
        # main backdrop
        _rect(self.bottom_panel, (0.04, 0.05, 0.07, 0.68), -0.46, 0.46, 0.025, 0.235)
        # wide HP bar across top of dock
        _rect(self.bottom_panel, (0.14, 0.15, 0.17, 0.9), -0.44, 0.44, 0.192, 0.222)
        self.hp_bar = _rect(self.bottom_panel, (0.95, 0.96, 0.96, 0.95), 0.0, 0.88, 0.0, 0.030)
        self.hp_bar.setPos(-0.44, 0, 0.192)
        self.hp_text = text("100", (0.40, 0.197), 0.028, (1, 1, 1, 1), TextNode.ARight, parent=self.bottom_panel)
        self.hp_label = text("HP", (-0.42, 0.197), 0.026, (0.75, 0.95, 0.8, 1), TextNode.ALeft,
                             parent=self.bottom_panel)
        self.loadout_text = text("", (-0.34, 0.197), 0.026, (0.65, 0.85, 1.0, 1), TextNode.ALeft,
                                 parent=self.bottom_panel, font=cf)
        # active ammo & weapon name (left half of dock)
        _rect(self.bottom_panel, (0.08, 0.10, 0.13, 0.78), -0.44, -0.02, 0.038, 0.178)
        self.ammo_text = text("30", (-0.28, 0.105), 0.072, align=TextNode.ARight, parent=self.bottom_panel)
        self.reserve_text = text("/ 90", (-0.265, 0.115), 0.036, (0.82, 0.84, 0.86, 1), TextNode.ALeft,
                                 parent=self.bottom_panel)
        self.weapon_text = text("AK-47", (-0.04, 0.132), 0.036, (1, 0.9, 0.55, 1), TextNode.ARight,
                                parent=self.bottom_panel, font=cf)
        self.slot1_text = text("", (-0.42, 0.052), 0.027, (0.95, 0.95, 0.95, 1), TextNode.ALeft,
                               parent=self.bottom_panel, font=cf)
        self.slot2_text = text("", (-0.20, 0.052), 0.027, (0.72, 0.74, 0.78, 1), TextNode.ALeft,
                               parent=self.bottom_panel, font=cf)
        # right half of dock: utility slots & match stats
        _rect(self.bottom_panel, (0.08, 0.10, 0.13, 0.78), 0.0, 0.44, 0.038, 0.178)
        self.nade_text = text("", (0.02, 0.125), 0.027, (0.92, 0.92, 0.92, 1), TextNode.ALeft,
                              parent=self.bottom_panel, font=cf)
        self.bottom_stat_text = text("", (0.02, 0.055), 0.026, (0.78, 0.86, 0.95, 1), TextNode.ALeft,
                                     parent=self.bottom_panel, font=cf)
        # compatibility aliases for any external code touching hp_panel / ammo_panel
        self.hp_panel = self.bottom_panel
        self.ammo_panel = self.bottom_panel

        # kill medals (bottom centre)
        from .killbanner import KillBanner
        self.banner = KillBanner(base, cf, self.cjk)
        self.kill_fx = None
        # center messages
        self.center_text = text("", (0, 0.25), 0.068, font=self.font_cjk)
        self.sub_text = text("", (0, 0.17), 0.042, (0.9, 0.9, 0.9, 1), font=self.font_cjk)
        self.help_text = text("", (0.03, -0.48), 0.028, (1, 1, 1, 0.85), TextNode.ALeft,
                              parent=base.a2dTopLeft, font=self.font_cjk)
        self.info_text = text("", (0.04, 0.08), 0.030, (0.85, 0.85, 0.85, 0.9), TextNode.ALeft,
                              parent=base.a2dBottomLeft, font=self.font_cjk)
        self.view_text = text("", (0, 0.255), 0.032, (0.95, 0.85, 0.35, 1), parent=self.bottom_panel, font=cf)
        # damage arcs
        self.arcs = []
        for _ in range(4):
            img = OnscreenImage(textures.sprite("damage_arc"), pos=(0, 0, 0), scale=0.92, parent=self.root)
            img.setTransparency(TransparencyAttrib.M_alpha)
            img.setBin("fixed", 40)
            img.hide()
            self.arcs.append([img, 0.0, 0.0])
        self.arc_idx = 0
        # vignettes
        self.vignette = OnscreenImage(textures.sprite("vignette"), scale=(2.0, 1, 1.0), parent=base.render2d)
        self.vignette.setTransparency(TransparencyAttrib.M_alpha)
        self.vignette.setScale(1, 1, 1)
        self.vignette.setColorScale(0, 0, 0, 0.8)
        self.hurt = OnscreenImage(textures.sprite("vignette"), parent=base.render2d)
        self.hurt.setTransparency(TransparencyAttrib.M_alpha)
        self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
        self.hurt_level = 0.0
        # scope
        self.scope = NodePath("scope")
        self.scope.reparentTo(base.aspect2d, -100)
        img = OnscreenImage(textures.sprite("scope"), scale=1.0, parent=self.scope)
        img.setTransparency(TransparencyAttrib.M_alpha)
        _rect(self.scope, (0, 0, 0, 1), -4, -0.999, -1.2, 1.2)
        _rect(self.scope, (0, 0, 0, 1), 0.999, 4, -1.2, 1.2)
        self.scope_cross = base.aspect2d.attachNewNode("scope_cross")
        self.scope_cross.reparentTo(base.aspect2d, -99)
        self._reticle_roll = None
        self.scope.hide()
        self.scope_cross.hide()
        # scoreboard
        self.board = self.root.attachNewNode("board")
        _rect(self.board, (0, 0, 0, 0.65), -0.95, 0.95, -0.62, 0.62)
        self.board_text = [text("", (-0.88 + 0.95 * k, 0.5), 0.042, TEAM_COLORS[k], TextNode.ALeft, parent=self.board,
                                font=self.font_cjk) for k in (0, 1)]
        self.board.hide()
        self.end_hint = "按 F5 重新开始" if self.cjk else "Press F5 to restart"
        # round banner
        self.round_text = text("", (0, 0.42), 0.085, font=cf)
        self.round_sub = text("", (0, 0.34), 0.045, (0.9, 0.9, 0.9, 1), font=cf)
        self.round_t = 0.0
        # flashbang
        self.flash_card = _rect(base.render2d, (1, 1, 1, 1), -1, 1, -1, 1, "flash_card")
        self.flash_card.setColorScale(1, 1, 1, 0)
        self.watch = False
        self.god_view = False

    def clear_transient(self):
        """Forget hit feedback, kill feed and banners from the previous world."""
        for slot in self.arcs:
            slot[1] = 0.0
            slot[0].hide()
        self.hurt_level = 0.0
        self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
        self.hit_t = 0.0
        self.hit_kill = False
        self.kill_fx = None
        self.feed = []
        self.round_t = 0.0
        self.banner.reset_streak()

    def show_round(self, title, sub="", duration=3.5, color=(1, 1, 1, 1)):
        self.round_text.setText(title)
        self.round_text.setFg(color)
        self.round_sub.setText(sub)
        self.round_t = duration

    # ------------------------------------------------------------------ events
    def on_event(self, e, world):
        t = e["type"]
        me = self.local_id
        if t == "damage" and not self.watch:
            if e["attacker"] == me and e["victim"] != me:
                self.hit_t = 0.22
                self.hit_kill = False
            if e["victim"] == me and e["attacker"] >= 0 and e["attacker"] != me:
                att = world.agents[e["attacker"]]
                v = world.agents[me]
                dx, dy = att.pos[0] - v.pos[0], att.pos[1] - v.pos[1]
                slot = self.arcs[self.arc_idx]
                self.arc_idx = (self.arc_idx + 1) % len(self.arcs)
                slot[1] = 1.6
                slot[2] = math.degrees(math.atan2(dy, dx))
                self.hurt_level = min(1.0, self.hurt_level + e["amount"] / 60.0)
            elif e["victim"] == me:
                self.hurt_level = min(1.0, self.hurt_level + e["amount"] / 60.0)
        elif t == "kill":
            a = world.agents[e["attacker"]].name if e["attacker"] >= 0 else "world"
            v = world.agents[e["victim"]].name
            w = item_name(e["weapon"], self.cjk)
            hs = "  [HS]" if e["headshot"] else ""
            col = TEAM_COLORS[world.agents[e["attacker"]].team] if e["attacker"] >= 0 else (1, 1, 1, 1)
            if not self.watch and (e["attacker"] == me or e["victim"] == me):
                col = (1, 1, 0.5, 1)
            self.feed.insert(0, [f"{a}  [{w}]{hs}  {v}", col, 6.0])
            self.feed = self.feed[:6]
            if not self.watch and e["attacker"] == me and e["victim"] != me:
                self.hit_t = 0.4
                self.hit_kill = True
                if world.agents[e["victim"]].team != world.agents[me].team:
                    self.kill_fx = self.banner.on_kill(e["weapon"], e["headshot"], v)
            elif not self.watch and e["victim"] == me:
                self.banner.reset_streak()
        elif t == "spawn" and e["agent"] == me:
            self.banner.reset_streak()
            self.hurt_level = 0.0
            self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
            for slot in self.arcs:
                slot[1] = 0.0
                slot[0].hide()
        else:
            self._round_event(e, world)

    def _round_event(self, e, world):
        t = e["type"]
        zh = self.cjk
        me = world.agents[self.local_id]
        if t == "round_start":
            n = e["round"]
            title = f"第 {n} 回合" if zh else f"ROUND {n}"
            if self.watch:
                self.show_round(title, "", 3.0)
            else:
                lo = loadout_label(me.loadout, zh)
                self.show_round(title, (f"歼灭模式   ·   方案: {lo}" if zh else f"ELIMINATION   ·   Loadout: {lo}"),
                                3.0)
        elif t == "round_end":
            r = ROUND_REASONS.get(e["reason"], (e["reason"], e["reason"]))
            sub = r[0] if zh else r[1]
            if e["team"] not in (0, 1):
                self.show_round("平局" if zh else "DRAW", sub, 4.0)
            elif self.watch:
                name = TEAM_NAMES[e["team"]]
                self.show_round(f"{name} 拿下回合" if zh else f"{name} TAKES THE ROUND", sub, 4.0,
                                TEAM_COLORS[e["team"]])
            else:
                won = e["team"] == me.team
                title = ("回合胜利" if won else "回合失败") if zh else ("ROUND WON" if won else "ROUND LOST")
                self.show_round(title, sub, 4.0, (0.5, 1, 0.55, 1) if won else (1, 0.45, 0.4, 1))
        elif t == "side_swap":
            self.show_round("交换攻守" if zh else "SWITCHING SIDES", "", 3.0)

    def _paint_reticle(self, roll_deg: float):
        """Scope cross in the XZ plane, rotated so a lean tilts the lines and not the view."""
        if self._reticle_roll is not None and abs(roll_deg - self._reticle_roll) < 0.05:
            return
        self._reticle_roll = roll_deg
        for child in self.scope_cross.getChildren():
            child.removeNode()
        ang = math.radians(roll_deg)
        c, s = math.cos(ang), math.sin(ang)

        def rot(x, z):
            return x * c - z * s, x * s + z * c

        def seg(x0, z0, x1, z1, half):
            dx, dz = x1 - x0, z1 - z0
            length = math.hypot(dx, dz) or 1.0
            px, pz = -dz / length * half, dx / length * half
            corners = ((x0 - px, z0 - pz), (x0 + px, z0 + pz), (x1 + px, z1 + pz), (x1 - px, z1 - pz))
            cm = CardMaker("reticle")
            cm.setFrame(*(Point3(x, 0, z) for x, z in corners))
            n = self.scope_cross.attachNewNode(cm.generate())
            n.setColor(0, 0, 0, 0.92)
            n.setTransparency(TransparencyAttrib.M_alpha)
            n.setAttrib(CullFaceAttrib.make(CullFaceAttrib.M_cull_none))
            n.show()
        self.scope_cross.show()

        arms = ((0, -0.90, 0, 0.90, 0.0015), (-0.90, 0, 0.90, 0, 0.0015),
                (0, 0.46, 0, 0.90, 0.0045), (0, -0.90, 0, -0.46, 0.0045),
                (0.46, 0, 0.90, 0, 0.0045), (-0.90, 0, -0.46, 0, 0.0045))
        for x0, z0, x1, z1, half in arms:
            seg(*rot(x0, z0), *rot(x1, z1), half)

    def _update_damage_arcs(self, dt, agent):
        """Red wedge on the view rim. World angle 0 is +X; the wedge's top points at the shooter."""
        if self.watch:
            for slot in self.arcs:
                slot[0].hide()
            return
        for slot in self.arcs:
            img, life, ang = slot
            if life <= 0:
                img.hide()
                continue
            slot[1] -= dt
            rel = (ang - agent.yaw + 180.0) % 360.0 - 180.0
            img.setR(-rel)
            img.setColorScale(1.6, 0.12, 0.08, min(1.0, slot[1]))
            img.show()

    def _update_squad_and_minimap(self, world, agent):
        zh = self.cjk
        kd = agent.kills / max(1, agent.deaths)
        self.top_stat_text.setText(f"淘汰 {agent.kills}   战损率 {kd:.1f}" if zh else f"ELIM {agent.kills}   K/D {kd:.1f}")
        mates = [a for a in world.agents if a.team == agent.team]
        for i, (row, badge, num_t, name_t, hp_fill) in enumerate(self.squad_rows):
            if i >= len(mates):
                row.hide()
                continue
            row.show()
            m = mates[i]
            mark = " <" if m.id == agent.id else ""
            name_t.setText(f"{m.name}{mark}")
            frac = max(0.0, min(1.0, m.hp / 100.0)) if m.alive else 0.0
            hp_fill.setSx(max(0.001, frac))
            if not m.alive:
                name_t.setFg((0.55, 0.55, 0.55, 0.85))
                hp_fill.setColor(0.35, 0.35, 0.35, 0.7)
            elif frac < 0.35:
                name_t.setFg((1.0, 0.85, 0.85, 1))
                hp_fill.setColor(1.0, 0.28, 0.22, 0.95)
            else:
                name_t.setFg((0.98, 0.98, 0.98, 1))
                hp_fill.setColor(0.95, 0.97, 0.98, 0.95)

        # Top-center alive pips relative to focused agent's team
        my_team = agent.team
        en_team = 1 - my_team
        my_mates = [a for a in world.agents if a.team == my_team]
        en_mates = [a for a in world.agents if a.team == en_team]
        for i, pip in enumerate(self.pips_l):
            if i < len(my_mates):
                pip.show()
                pip.setColor(*(TEAM_COLORS[0] if my_mates[i].alive else (0.3, 0.32, 0.35, 0.6)))
            else:
                pip.hide()
        for i, pip in enumerate(self.pips_r):
            if i < len(en_mates):
                pip.show()
                pip.setColor(*(TEAM_COLORS[1] if en_mates[i].alive else (0.3, 0.32, 0.35, 0.6)))
            else:
                pip.hide()
        score = world.score
        self.score_l.setText(str(score[my_team]))
        self.score_r.setText(str(score[en_team]))
        win_r = world.mode.p["win_rounds"]
        self.goal_text.setText(f"先赢 {win_r} 局" if zh else f"FIRST TO {win_r}")

        # Top-right minimap
        if self._map_name != world.map.name:
            tex, bounds = _build_minimap_texture(world.map)
            self.map_card.setTexture(tex)
            self._map_bounds = bounds
            self._map_name = world.map.name
        x0, y0, x1, y1 = self._map_bounds
        ml = -self.map_pad - self.map_size
        mb = -self.map_pad - self.map_size

        def to_map(px, py):
            u = max(0.02, min(0.98, (px - x0) / max(1e-6, x1 - x0)))
            v = max(0.02, min(0.98, (py - y0) / max(1e-6, y1 - y0)))
            return ml + u * self.map_size, mb + v * self.map_size

        vis = world.visibility()
        for i, dot in enumerate(self.map_dots):
            if i >= len(world.agents):
                dot.hide()
                continue
            a = world.agents[i]
            if not a.alive:
                dot.hide()
                continue
            same = a.team == agent.team
            can_see = same or self.watch or bool(vis[agent.id, a.id])
            if not can_see:
                dot.hide()
                continue
            mx, my = to_map(a.pos[0], a.pos[1])
            dot.setPos(mx, 0, my)
            if a.id == agent.id:
                dot.setColor(1.0, 0.95, 0.25, 1.0)
                dot.setScale(1.25)
            elif same:
                idx = mates.index(a) if a in mates else 0
                dot.setColor(*SQUAD_BADGE_COLORS[idx % len(SQUAD_BADGE_COLORS)])
                dot.setScale(1.0)
            else:
                dot.setColor(1.0, 0.25, 0.2, 0.95)
                dot.setScale(0.95)
            dot.show()
        if agent.alive:
            fx, fy = to_map(agent.pos[0], agent.pos[1])
            self.map_fov.setPos(fx, 0, fy)
            self.map_fov.setR(90.0 - agent.yaw)
            self.map_fov.show()
        else:
            self.map_fov.hide()
        zd = max(0.0, -world.zone_dist(agent.pos))
        self.map_label.setText((f"安全区  {int(zd)}m" if zh else f"ZONE  {int(zd)}m")
                               if not world.outside_zone(agent.pos) else
                               ("毒圈外!" if zh else "OUTSIDE ZONE!"))

    # ------------------------------------------------------------------ update
    def update(self, dt, world, agent, vfov_deg, show_board=False, scope_active=False, spectating=None,
               first_person=True):
        now = world.time
        zh = self.cjk
        self.banner.update(dt)
        self._update_damage_arcs(dt, agent)
        if self.watch:
            self.flash_card.setColorScale(1, 1, 1, 0)
            self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
        if agent.slot not in agent.inv:
            return
        self._update_squad_and_minimap(world, agent)
        self._update_round(dt, world, agent)
        # crosshair
        spec = agent.spec
        sights = agent.ads and spec.ads_fov and first_person
        show_cross = (agent.alive and not self.god_view and not scope_active
                      and not (spec.kind == "gun" and sights and spec.scope_overlay))
        if show_cross and spec.kind == "gun" and not sights:
            self.cross.show()
            spread = world.current_spread(agent)
            gap = math.tan(math.radians(spread)) / math.tan(math.radians(vfov_deg / 2)) + 0.012
            gap = min(gap, 0.35)
            ln, th = 0.026, 0.0028
            b = self.cross_bars
            b[0].setPos(0, 0, gap + ln / 2); b[0].setScale(th, 1, ln)
            b[1].setPos(0, 0, -gap - ln / 2); b[1].setScale(th, 1, ln)
            b[2].setPos(gap + ln / 2, 0, 0); b[2].setScale(ln, 1, th)
            b[3].setPos(-gap - ln / 2, 0, 0); b[3].setScale(ln, 1, th)
            for bar in b:
                bar.show()
            self.dot.hide()
        elif show_cross:
            self.cross.show()
            for bar in self.cross_bars:
                bar.hide()
            self.dot.show()
        else:
            self.cross.hide()
        if scope_active and not self.god_view:
            self.scope.show()
            self.scope_cross.show()
            self._paint_reticle(scope_reticle_roll(agent.lean_off))
        else:
            self.scope.hide()
            self.scope_cross.hide()
        # hitmarker
        if self.hit_t > 0:
            self.hit_t -= dt
            self.hitmarker.show()
            k = max(0.0, self.hit_t / 0.22)
            self.hitmarker.setColorScale(1, 0.25, 0.2, min(1, k + 0.2)) if self.hit_kill else \
                self.hitmarker.setColorScale(1, 1, 1, min(1, k + 0.2))
            self.hitmarker.setScale(1.0 + (0.3 if self.hit_kill else 0.0) * k)
        else:
            self.hitmarker.hide()
        # bottom weapon & health dock
        hp = max(0, int(math.ceil(agent.hp)))
        self.hp_text.setText(str(hp))
        frac = max(0.0, agent.hp / 100.0)
        self.hp_bar.setSx(max(frac, 0.001))
        col = (0.95, 0.97, 0.98, 0.95) if frac > 0.5 else (1.0, 0.8, 0.2, 0.95) if frac > 0.25 else (1.0, 0.25, 0.2, 0.95)
        self.hp_bar.setColor(*col)
        self.loadout_text.setText(loadout_label(agent.loadout, zh))
        ws = agent.weapon
        self.weapon_text.setText(item_name(spec.name, zh))
        w1 = agent.inv.get(SLOT_PRIMARY)
        w2 = agent.inv.get(SLOT_PISTOL)
        self.slot1_text.setText(f"[1] {item_name(w1.name, zh)} ({w1.mag})" if w1 else "")
        self.slot2_text.setText(f"[2] {item_name(w2.name, zh)} ({w2.mag})" if w2 else "")
        self.slot1_text.setFg((1, 0.92, 0.45, 1) if agent.slot == SLOT_PRIMARY else (0.72, 0.74, 0.78, 1))
        self.slot2_text.setFg((1, 0.92, 0.45, 1) if agent.slot == SLOT_PISTOL else (0.72, 0.74, 0.78, 1))
        if spec.kind == "gun":
            self.ammo_text.setText(str(ws.mag))
            self.reserve_text.setText(f"/ {ws.reserve}")
            self.reserve_text.setFg((0.82, 0.84, 0.86, 1))
            low = ws.mag <= max(1, spec.mag // 5)
            self.ammo_text.setFg((1, 0.35, 0.3, 1) if low else (1, 1, 1, 1))
            if agent.reloading:
                p = 1.0 - (agent.reload_until - now) / spec.reload_time
                self.reserve_text.setText(f"换弹 {int(p * 100)}%" if zh else f"RELOAD {int(p * 100)}%")
        elif spec.kind == "grenade":
            self.ammo_text.setText(str(ws.mag))
            self.ammo_text.setFg((1, 1, 1, 1))
            if agent.throw_at >= 0 and agent.cook_until > 0:
                left = max(0.0, agent.cook_until - now)
                self.reserve_text.setText(f"{left:.1f}s")
                hot = spec.name == "frag" and left < 0.75
                self.reserve_text.setFg((1, 0.3, 0.22, 1) if hot else (1, 0.85, 0.35, 1))
            else:
                self.reserve_text.setText("")
        else:
            self.ammo_text.setText("--")
            self.ammo_text.setFg((1, 1, 1, 1))
            self.reserve_text.setText("")
        items = []
        for s in range(4, NUM_SLOTS + 1):
            w = agent.inv.get(s)
            if w is not None:
                n = f"x{w.mag}" if w.spec.kind == "grenade" else ""
                label = item_name(w.name, zh)
                items.append(f"[{s}]{label}{n}")
        self.nade_text.setText("  ".join(items))
        self.bottom_stat_text.setText(
            f"淘汰 {agent.kills}    助攻 {agent.headshots}    伤害 {int(agent.damage_dealt)}" if zh else
            f"K {agent.kills}    HS {agent.headshots}    DMG {int(agent.damage_dealt)}"
        )
        left = world.mode.time_left()
        self.timer_text.setText(f"{int(left // 60)}:{int(left % 60):02d}")
        # killfeed
        for item in self.feed:
            item[2] -= dt
        self.feed = [i for i in self.feed if i[2] > 0]
        for i, tn in enumerate(self.feed_texts):
            if i < len(self.feed):
                txt, col, life = self.feed[i]
                tn.setText(txt)
                tn.setFg((col[0], col[1], col[2], min(1.0, life)))
            else:
                tn.setText("")
        self.hurt_level = max(0.0, self.hurt_level - dt * 0.8)
        if self.watch:
            self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
        else:
            low_hp = 0.35 * max(0.0, 1.0 - agent.hp / 35.0) if agent.alive else 0.0
            self.hurt.setColorScale(0.8, 0.0, 0.0, min(0.9, self.hurt_level * 0.7 + low_hp))
        # death / spectate
        if self.watch and not world.done:
            self.center_text.setText("")
            self.sub_text.setText("")
        elif spectating is not None and spectating.alive:
            who = spectating.name
            self.center_text.setText("")
            self.sub_text.setText(f"观战 {who}（第三人称）    左键/右键切换" if zh else
                                  f"Spectating {who} (3rd person)    LMB/RMB cycle")
        elif not agent.alive and not world.done:
            killer = world.agents[agent.killer].name if agent.killer >= 0 else ("环境" if zh else "the world")
            self.center_text.setText(f"被 {killer} 击杀" if zh else f"Killed by {killer}")
            self.sub_text.setText("等待下一回合" if zh else "Waiting for the next round")
        elif world.done:
            w = world.winner
            if w < 0:
                self.center_text.setText("平局" if zh else "DRAW")
            else:
                self.center_text.setText(f"{TEAM_NAMES[w]} 获胜" if zh else f"{TEAM_NAMES[w]} WINS")
            self.sub_text.setText(self.end_hint)
        else:
            self.center_text.setText("")
            self.sub_text.setText("")
        if show_board:
            self.board.show()
            for k in (0, 1):
                rows = [f"{TEAM_NAMES[k]}   {world.score[k]}", "", "Name        K    D    HS   " +
                        ("方案" if zh else "KIT")]
                for a in sorted((a for a in world.agents if a.team == k), key=lambda a: -a.kills):
                    mark = " <" if a.id == self.local_id else ""
                    lo = loadout_label(a.loadout, zh)
                    rows.append(f"{a.name:<10} {a.kills:>3}  {a.deaths:>3}  {a.headshots:>3}   {lo}{mark}")
                self.board_text[k].setText("\n".join(rows))
        else:
            self.board.hide()

    def _update_round(self, dt, world, agent):
        zh = self.cjk
        mode = world.mode
        now = world.time
        if self.round_t > 0:
            self.round_t -= dt
            a = min(1.0, self.round_t / 0.5)
            self.round_text.setAlphaScale(a)
            self.round_sub.setAlphaScale(a)
            if self.round_t <= 0:
                self.round_text.setText("")
                self.round_sub.setText("")
        names = {"freeze": ("冻结时间", "FREEZE TIME"), "live": ("", ""), "over": ("回合结束", "ROUND OVER")}
        phase = names.get(mode.phase, ("", ""))[0 if zh else 1]
        if mode.phase == "freeze" and not self.watch:
            phase += "   [B] 切换方案" if zh else "   [B] switch loadout"
        elif mode.phase == "live" and not self.god_view and agent.alive and world.outside_zone(agent.pos):
            phase = "毒圈外 — 持续扣血" if zh else "OUTSIDE SAFE ZONE — TAKING DAMAGE"
        self.phase_text.setText(phase)
        self.bottom_panel.show()
        self.squad_panel.show()
        self.map_panel.show()
        if self.watch:
            if self.god_view:
                self.view_text.setText("上帝视角   WASD 移动   鼠标转向   Q/E 升降   F 切换第三人称"
                                       if zh else
                                       "GOD VIEW   WASD move   mouse look   Q/E up/down   F third person")
                self.cross.hide()
                self.scope.hide()
                self.scope_cross.hide()
            else:
                self.view_text.setText(f"第三人称观战  {agent.name}   左键/右键切换   F 上帝视角" if zh else
                                       f"THIRD PERSON  {agent.name}   LMB/RMB cycle   F god view")
        else:
            self.view_text.setText("")
        # flashbang - a spectator camera is never blinded
        if not self.watch and now < agent.flash_until:
            total = max(agent.flash_until - agent.flash_t0, 1e-3)
            left = agent.flash_until - now
            a = min(1.0, left / min(1.2, total)) if left < 1.2 else 1.0
            self.flash_card.setColorScale(1, 1, 1, a * 0.97)
        else:
            self.flash_card.setColorScale(1, 1, 1, 0)

    def set_help(self, text: str):
        self.help_text.setText(text)

    def set_info(self, text: str):
        self.info_text.setText(text)

    def destroy(self):
        self.flash_card.removeNode()
        self.banner.destroy()
        self.root.removeNode()
        self.scope.removeNode()
        self.scope_cross.removeNode()
        self.bottom_panel.removeNode()
        self.squad_panel.removeNode()
        self.map_panel.removeNode()
        self.score_panel.removeNode()
        self.feed_root.removeNode()
        self.vignette.destroy()
        self.hurt.destroy()

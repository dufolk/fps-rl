"""2D heads-up display."""
from __future__ import annotations

import math
import os
import random

from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (CardMaker, CullFaceAttrib, Filename, NodePath, Point3, TextNode,
                          TransparencyAttrib)

from ..geometry import poly_center
from ..sim import LEAN_DIST, LEAN_ROLL, TEAM_NAMES
from ..weapons import NUM_SLOTS, WEAPONS, item_name

TEAM_COLORS = {0: (1.0, 0.55, 0.3, 1), 1: (0.45, 0.75, 1.0, 1)}
DOM_COLORS = {0: (1.0, 0.3, 0.26, 1), 1: (0.3, 0.6, 1.0, 1)}  # domination: red side / blue side
CROSS_COLOR = (0.35, 1.0, 0.45, 0.95)


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


class HUD:
    def __init__(self, base, textures, local_id: int):
        self.base = base
        self.local_id = local_id
        self.font = load_font(base.loader)
        self.font_cjk = load_font(base.loader, cjk=True)
        self.cjk = self.font_cjk is not None
        a2d = base.aspect2d
        self.root = a2d.attachNewNode("hud")
        f = self.font

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
        # health panel
        bl = base.a2dBottomLeft
        self.hp_panel = bl.attachNewNode("hp")
        _rect(self.hp_panel, (0, 0, 0, 0.35), 0.04, 0.62, 0.04, 0.2)
        self.hp_text = text("100", (0.17, 0.085), 0.085, parent=self.hp_panel)
        self.hp_label = text("HP", (0.075, 0.09), 0.04, (0.8, 0.8, 0.8, 1), parent=self.hp_panel)
        _rect(self.hp_panel, (0.15, 0.15, 0.15, 0.8), 0.27, 0.6, 0.095, 0.125)
        self.hp_bar = _rect(self.hp_panel, (0.35, 0.95, 0.45, 0.95), 0.0, 0.33, 0.0, 0.03)
        self.hp_bar.setPos(0.27, 0, 0.095)
        self.protect_text = text("", (0.33, 0.055), 0.032, (0.6, 0.9, 1, 1), TextNode.ALeft, parent=self.hp_panel)
        # ammo panel
        br = base.a2dBottomRight
        self.ammo_panel = br.attachNewNode("ammo")
        _rect(self.ammo_panel, (0, 0, 0, 0.35), -0.66, -0.04, 0.04, 0.24)
        self.ammo_text = text("30", (-0.38, 0.085), 0.1, align=TextNode.ARight, parent=self.ammo_panel)
        self.reserve_text = text("/ 90", (-0.36, 0.085), 0.05, (0.8, 0.8, 0.8, 1), TextNode.ALeft,
                                 parent=self.ammo_panel)
        self.weapon_text = text("AK-47", (-0.07, 0.19), 0.042, (1, 0.9, 0.6, 1), TextNode.ARight,
                                parent=self.ammo_panel, font=self.font_cjk)
        self.nade_text = text("", (-0.07, 0.055), 0.034, (0.85, 0.85, 0.85, 1), TextNode.ARight,
                              parent=self.ammo_panel, font=self.font_cjk)
        # score
        top = base.a2dTopCenter
        self.score_panel = top.attachNewNode("score")
        _rect(self.score_panel, (0, 0, 0, 0.4), -0.3, 0.3, -0.13, -0.02)
        self.score_l = text("0", (-0.12, -0.095), 0.07, TEAM_COLORS[0], parent=self.score_panel)
        self.score_r = text("0", (0.12, -0.095), 0.07, TEAM_COLORS[1], parent=self.score_panel)
        self.timer_text = text("0:00", (0, -0.085), 0.045, parent=self.score_panel)
        text(TEAM_NAMES[0], (-0.24, -0.09), 0.035, TEAM_COLORS[0], parent=self.score_panel)
        text(TEAM_NAMES[1], (0.24, -0.09), 0.035, TEAM_COLORS[1], parent=self.score_panel)
        # killfeed
        tr = base.a2dTopRight
        self.feed_root = tr.attachNewNode("feed")
        self.feed = []
        self.feed_texts = [text("", (-0.05, -0.08 - i * 0.06), 0.04, align=TextNode.ARight, parent=self.feed_root,
                                font=self.font_cjk) for i in range(6)]
        # kill medals (bottom centre) + weapon inspect caption (above the ammo panel)
        from .killbanner import KillBanner
        self.banner = KillBanner(base, self.font_cjk or f, self.cjk)
        self.kill_fx = None
        self.inspect_root = base.a2dBottomRight.attachNewNode("inspect")
        self.inspect_title = text("", (-0.07, 0.33), 0.05, align=TextNode.ARight, parent=self.inspect_root,
                                  font=self.font_cjk)
        self.inspect_sub = text("", (-0.07, 0.285), 0.032, (0.85, 0.85, 0.85, 1), TextNode.ARight,
                                parent=self.inspect_root, font=self.font_cjk)
        self.inspect_t = 0.0
        # center messages
        self.center_text = text("", (0, 0.25), 0.07, font=self.font_cjk)
        self.sub_text = text("", (0, 0.17), 0.045, (0.9, 0.9, 0.9, 1), font=self.font_cjk)
        self.help_text = text("", (-1.7, 0.85), 0.035, (1, 1, 1, 0.9), TextNode.ALeft,
                              parent=base.a2dTopLeft, font=self.font_cjk)
        self.help_text.setPos(0.05, -0.08)
        self.info_text = text("", (0.05, 0.25), 0.032, (0.85, 0.85, 0.85, 0.9), TextNode.ALeft,
                              parent=base.a2dBottomLeft, font=self.font_cjk)
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
        # low-health rim: a frame, transparent across the middle of the screen
        self.bleed = OnscreenImage(textures.sprite("edge"), scale=(2.0, 1, 1.0), parent=base.render2d)
        self.bleed.setTransparency(TransparencyAttrib.M_alpha)
        self.bleed.setScale(1, 1, 1)
        self.bleed.setColorScale(1, 1, 1, 0)
        # scope
        self.scope = NodePath("scope")
        self.scope.reparentTo(base.aspect2d, -100)
        img = OnscreenImage(textures.sprite("scope"), scale=1.0, parent=self.scope)
        img.setTransparency(TransparencyAttrib.M_alpha)
        _rect(self.scope, (0, 0, 0, 1), -4, -0.999, -1.2, 1.2)
        _rect(self.scope, (0, 0, 0, 1), 0.999, 4, -1.2, 1.2)
        # its own node: `self.reticle` is the drone overlay, and hiding the scope
        # node leaves quads created afterwards invisible
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
        self.flash_white = 0.0
        self.end_hint = "按 F5 重新开始" if self.cjk else "Press F5 to restart"
        # domination bar (replaces the score panel): match score as squares, open point's score
        cf = self.font_cjk or f
        self.obj_root = top.attachNewNode("objectives")
        red, blue = DOM_COLORS[0], DOM_COLORS[1]
        _rect(self.obj_root, (0, 0, 0, 0.45), -0.4, 0.4, -0.15, -0.02)
        self.dom_wins = {0: [], 1: []}
        for team, sign in ((0, -1), (1, 1)):
            col = DOM_COLORS[team]
            for k in range(2):
                x = sign * (0.25 + 0.065 * k)
                _rect(self.obj_root, (0.05, 0.05, 0.05, 0.85), x - 0.026, x + 0.026, -0.112, -0.06)
                cell = _rect(self.obj_root, col, x - 0.021, x + 0.021, -0.107, -0.065)
                self.dom_wins[team].append(cell)
            label = ("红方", "RED") if team == 0 else ("蓝方", "BLUE")
            text(label[0] if self.cjk else label[1], (sign * 0.282, -0.137), 0.028, col, parent=self.obj_root, font=cf)
        self.dom_cap = (text("0", (-0.07, -0.1), 0.075, red, TextNode.ACenter, parent=self.obj_root),
                        text("0", (0.07, -0.1), 0.075, blue, TextNode.ACenter, parent=self.obj_root))
        self.dom_colon = text(":", (0, -0.098), 0.065, parent=self.obj_root)
        _rect(self.obj_root, (0.15, 0.15, 0.15, 0.85), -0.18, 0.18, -0.138, -0.124)
        self.dom_bar = (_rect(self.obj_root, red, -0.18, 0.0, -0.137, -0.125),
                        _rect(self.obj_root, blue, 0.0, 0.18, -0.137, -0.125))
        self.dom_point = text("", (0, -0.19), 0.04, (1, 1, 1, 0.95), parent=self.obj_root, font=cf)
        self.dom_info = text("", (0, -0.24), 0.036, (1, 0.9, 0.55, 1), parent=self.obj_root, font=cf)
        self.obj_root.hide()
        self.phase_text = text("", (0, -0.2), 0.04, (1, 0.9, 0.55, 1), parent=self.score_panel, font=cf)
        self.money_text = text("", (0.06, 0.24), 0.05, (0.45, 1.0, 0.5, 1), TextNode.ALeft, parent=self.hp_panel)
        self.armor_text = text("", (0.33, 0.165), 0.03, (0.6, 0.8, 1.0, 1), TextNode.ALeft, parent=self.hp_panel,
                               font=cf)
        # hold-action progress
        self.busy_root = self.root.attachNewNode("busy")
        _rect(self.busy_root, (0, 0, 0, 0.55), -0.3, 0.3, -0.33, -0.27)
        self.busy_bar = _rect(self.busy_root, (1.0, 0.75, 0.25, 0.95), 0.0, 0.58, 0.0, 0.04)
        self.busy_bar.setPos(-0.29, 0, -0.32)
        self.busy_label = text("", (0, -0.24), 0.04, parent=self.busy_root, font=cf)
        self.busy_root.hide()
        self.prompt_text = text("", (0, -0.42), 0.042, (1, 1, 1, 0.95), font=cf)
        self.view_text = text("", (0, -0.85), 0.04, (0.7, 1.0, 0.7, 1), font=cf)
        # round banner
        self.round_text = text("", (0, 0.42), 0.085, font=cf)
        self.round_sub = text("", (0, 0.34), 0.045, (0.9, 0.9, 0.9, 1), font=cf)
        self.round_t = 0.0
        # objective / spotted-enemy markers
        self.marks = [text("", (0, 0), 0.036, font=cf) for _ in range(12)]
        # drone / camera feed
        self.feed_fx = base.render2d.attachNewNode("feed_fx")
        self.feed_tint = _rect(self.feed_fx, (1, 1, 1, 1), -1, 1, -1, 1, "tint")
        self.scan = self.feed_fx.attachNewNode("scan")
        for i in range(42):
            y = -1.05 + i * 0.05
            _rect(self.scan, (0, 0, 0, 0.16), -1, 1, y, y + 0.014, "line")
        self.noise = [_rect(self.feed_fx, (1, 1, 1, 1), -1, 1, -0.006, 0.006, "noise") for _ in range(6)]
        self.feed_fx.hide()
        self.reticle = self.root.attachNewNode("reticle")
        for x, z, sx, sz in ((0, 0.06, 0.003, 0.03), (0, -0.06, 0.003, 0.03), (0.06, 0, 0.03, 0.003),
                             (-0.06, 0, 0.03, 0.003)):
            _rect(self.reticle, (1, 1, 1, 0.8), x - sx / 2, x + sx / 2, z - sz / 2, z + sz / 2)
        self.reticle.hide()
        self.rec_text = text("", (0, -0.3), 0.045, (1, 0.3, 0.25, 1), parent=top, font=cf)
        # flashbang
        self.flash_card = _rect(base.render2d, (1, 1, 1, 1), -1, 1, -1, 1, "flash_card")
        self.flash_card.setColorScale(1, 1, 1, 0)
        self.watch = False
        self.god_view = False

    def clear_transient(self):
        """Forget hit feedback, kill feed and banners from the previous world (e.g. the menu match)."""
        for slot in self.arcs:
            slot[1] = 0.0
            slot[0].hide()
        self.hurt_level = 0.0
        self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
        self.bleed.setColorScale(1, 1, 1, 0)
        self.hit_t = 0.0
        self.hit_kill = False
        self.kill_fx = None
        self.feed = []
        self.round_t = 0.0
        self.banner.reset_streak()
        self.hide_inspect()

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
                self.hide_inspect()
        elif t == "spawn" and e["agent"] == me:
            self.banner.reset_streak()
        else:
            self._mode_event(e, world)

    def _mode_event(self, e, world):
        t = e["type"]
        zh = self.cjk
        me = world.agents[self.local_id]
        mode = world.mode
        if t == "round_start":
            n = e["round"]
            if self.watch:
                self.show_round(f"第 {n} 回合" if zh else f"ROUND {n}", "", 3.0)
                return
            side = mode.side(me.team)
            if mode.name == "r6":
                role = ("进攻方" if side == 0 else "防守方") if zh else ("ATTACK" if side == 0 else "DEFEND")
                sub = ("准备阶段: 无人机侦查 / 加固墙体" if zh else "Prep phase: drone / reinforce")
            else:
                role = ("潜伏者 (T)" if side == 0 else "保卫者 (CT)") if zh else ("TERRORISTS" if side == 0 else "COUNTER-TERRORISTS")
                sub = ("B 键购买武器" if zh else "Press B to buy")
            self.show_round((f"第 {n} 回合" if zh else f"ROUND {n}"), f"{role}   ·   {sub}", 3.0)
        elif t == "round_end":
            if self.watch:
                reasons = {"elimination": ("全歼对方", "Elimination"), "time": ("时间耗尽", "Time ran out"),
                           "bomb": ("目标已摧毁", "Target destroyed"), "defuse": ("炸弹已拆除", "Bomb defused"),
                           "defuser": ("拆弹器已完成", "Defuser secured")}
                r = reasons.get(e["reason"], (e["reason"], e["reason"]))
                if e["team"] in (0, 1):
                    name = TEAM_NAMES[e["team"]]
                    title = f"{name} 拿下回合" if zh else f"{name} TAKES THE ROUND"
                    col = TEAM_COLORS[e["team"]]
                else:
                    title = "平局" if zh else "DRAW"
                    col = (1, 1, 1, 1)
                self.show_round(title, r[0] if zh else r[1], 4.0, col)
                return
            won = e["team"] == me.team
            reasons = {"elimination": ("全歼敌人", "Enemies eliminated"), "time": ("时间耗尽", "Time ran out"),
                       "bomb": ("目标已摧毁", "Target destroyed"), "defuse": ("炸弹已拆除", "Bomb defused"),
                       "defuser": ("拆弹器已完成", "Defuser secured")}
            r = reasons.get(e["reason"], (e["reason"], e["reason"]))
            title = ("回合胜利" if won else "回合失败") if zh else ("ROUND WON" if won else "ROUND LOST")
            self.show_round(title, r[0] if zh else r[1], 4.0, (0.5, 1, 0.55, 1) if won else (1, 0.45, 0.4, 1))
        elif t == "bomb_planted":
            label = "拆弹器已安放" if mode.name == "r6" else "炸弹已安放"
            self.show_round(label if zh else ("DEFUSER PLANTED" if mode.name == "r6" else "BOMB PLANTED"), "", 2.5,
                            (1, 0.6, 0.3, 1))
        elif t == "side_swap":
            self.show_round("交换攻守" if zh else "SWITCHING SIDES", "", 3.0)
        elif t == "overtime":
            self.show_round("加时赛" if zh else "OVERTIME", "", 3.0)
        elif t == "point_active":
            name = e.get("name", "?")
            self.show_round(f"{name} 点已激活" if zh else f"POINT {name} IS LIVE",
                            "前往占领" if zh else "Go capture it", 2.0, (1, 0.9, 0.55, 1))
        elif t == "point_captured":
            name = e.get("name", "?")
            c0, c1 = e.get("cap", (0, 0))
            w0, w1 = e.get("wins", (0, 0))
            sub = f"{c0} : {c1}    总比分 {w0} : {w1}" if zh else f"{c0} : {c1}    MATCH {w0} : {w1}"
            if e["team"] < 0:
                self.show_round(f"{name} 点平局" if zh else f"POINT {name} DRAWN", sub, 2.5, (0.9, 0.9, 0.9, 1))
            else:
                ours = e["team"] == me.team
                msg = (f"已占领 {name} 点" if ours else f"{name} 点失守") if zh else \
                    (f"Captured {name}" if ours else f"Lost {name}")
                self.show_round(msg, sub, 2.5, (0.5, 1, 0.55, 1) if ours else (1, 0.45, 0.4, 1))
        elif t == "hint" and e["agent"] == self.local_id:
            hints = {"not_in_site": ("需在包点内安放", "Must be in a bomb site"),
                     "need_soft_wall": ("需对准可破坏墙体", "Aim at a breakable wall"),
                     "need_door": ("需靠近门窗", "Move to a doorway or window")}
            h = hints.get(e["text"], (e["text"], e["text"]))
            self.set_info(h[0] if zh else h[1])

    def show_inspect(self, title: str, sub: str, color, duration: float = 3.0):
        self.inspect_title.setText(title)
        self.inspect_title.setFg(color)
        self.inspect_sub.setText(sub)
        self.inspect_t = duration
        self.inspect_root.show()

    def hide_inspect(self):
        self.inspect_t = 0.0
        self.inspect_root.hide()

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

    # ------------------------------------------------------------------ update
    def update(self, dt, world, agent, vfov_deg, show_board=False, scope_active=False, spectating=None,
               first_person=True):
        now = world.time
        self.banner.update(dt)
        if self.inspect_t > 0:
            self.inspect_t -= dt
            self.inspect_root.setAlphaScale(min(1.0, self.inspect_t / 0.4))
            if self.inspect_t <= 0:
                self.inspect_root.hide()
        self._update_damage_arcs(dt, agent)
        if self.watch:
            self.flash_card.setColorScale(1, 1, 1, 0)
            self.hurt.setColorScale(0.8, 0.0, 0.0, 0.0)
            self.bleed.setColorScale(1, 1, 1, 0)
        if agent.slot not in agent.inv:
            return
        self._update_mode(dt, world, agent)
        # crosshair
        spec = agent.spec
        sights = agent.ads and spec.ads_fov and first_person
        show_cross = agent.alive and not scope_active and not agent.view_mode \
            and not (spec.kind == "gun" and sights and spec.scope_overlay)
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
        if scope_active:
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
        # panels
        hp = max(0, int(math.ceil(agent.hp)))
        self.hp_text.setText(str(hp))
        frac = max(0.0, agent.hp / 100.0)
        self.hp_bar.setSx(max(frac, 0.001))
        col = (0.35, 0.95, 0.45, 0.95) if frac > 0.5 else (1.0, 0.8, 0.2, 0.95) if frac > 0.25 else (1.0, 0.25, 0.2, 0.95)
        self.hp_bar.setColor(*col)
        self.protect_text.setText("PROTECTED" if now < agent.protect_until and agent.alive else "")
        ws = agent.weapon
        if getattr(agent, "role", "human") != "human" and spec.kind == "melee":
            self.weapon_text.setText("利爪" if self.cjk else "CLAWS")
        else:
            self.weapon_text.setText(item_name(spec.name, self.cjk))
        if spec.kind == "gun":
            self.ammo_text.setText(str(ws.mag))
            self.reserve_text.setText(f"/ {ws.reserve}")
            low = ws.mag <= max(1, spec.mag // 5)
            self.ammo_text.setFg((1, 0.35, 0.3, 1) if low else (1, 1, 1, 1))
            if agent.reloading:
                p = 1.0 - (agent.reload_until - now) / spec.reload_time
                self.reserve_text.setText(f"RELOADING {int(p * 100)}%")
        elif spec.kind in ("grenade", "gadget", "shield"):
            self.ammo_text.setText(str(ws.mag))
            self.ammo_text.setFg((1, 1, 1, 1))
            if spec.kind == "grenade" and agent.throw_at >= 0 and agent.cook_until > 0:
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
                n = f" x{w.mag}" if w.spec.kind in ("grenade", "gadget", "shield") else ""
                label = item_name(w.name, self.cjk)
                items.append(f"[{s}] {label if self.cjk else label.upper()}{n}")
        self.nade_text.setText("   ".join(items))
        score = world.score
        self.score_l.setText(str(score[0]))
        self.score_r.setText(str(score[1]))
        mode = world.mode
        if hasattr(mode, "time_left") and mode.name in ("bomb", "elim", "r6", "bio"):
            left = mode.time_left()
            if mode.phase == "planted":
                self.timer_text.setText("C4")
                self.timer_text.setFg((1, 0.35, 0.3, 1) if int(now * 2) % 2 else (1, 0.8, 0.3, 1))
            else:
                self.timer_text.setText(f"{int(left // 60)}:{int(left % 60):02d}")
                self.timer_text.setFg((1, 1, 1, 1))
        elif world.cfg.time_limit:
            left = max(0.0, world.cfg.time_limit - now)
            self.timer_text.setText(f"{int(left // 60)}:{int(left % 60):02d}")
        else:
            self.timer_text.setText(f"{int(now // 60)}:{int(now % 60):02d}")
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
            self.bleed.setColorScale(1, 1, 1, 0)
        else:
            # domination keeps this card for the hit flash only; the steady low-HP red is the edge frame
            low_hp = 0.0 if world.mode.name == "dom" else (
                0.35 * max(0.0, 1.0 - agent.hp / 35.0) if agent.alive else 0.0)
            self.hurt.setColorScale(0.8, 0.0, 0.0, min(0.9, self.hurt_level * 0.7 + low_hp))
            rim = 0.0
            if world.mode.name == "dom" and agent.alive:
                rim = max(0.0, min(1.0, (60.0 - agent.hp) / 50.0))
                # breathing lamp: fades out and back in; lower HP breathes deeper and a little faster
                breath = 0.5 - 0.5 * math.cos(now * (1.7 + 1.6 * rim))
                rim *= 0.15 + 0.85 * breath
            self.bleed.setColorScale(1, 1, 1, rim)
        # death / spectate
        zh = self.cjk
        if self.watch and not world.done:
            self.center_text.setText("")
            self.sub_text.setText("")
        elif spectating is not None and spectating.alive:
            who = spectating.name
            self.center_text.setText("")
            self.sub_text.setText(f"观战 {who}    左键/右键切换    F3 切换人称" if zh else
                                  f"Spectating {who}    LMB/RMB cycle    F3 camera")
        elif not agent.alive and not world.done:
            killer = world.agents[agent.killer].name if agent.killer >= 0 else ("环境" if zh else "the world")
            self.center_text.setText(f"被 {killer} 击杀" if zh else f"Killed by {killer}")
            left = max(0.0, agent.respawn_at - now)
            if world.cfg.respawn:
                self.sub_text.setText(f"{left:.1f} 秒后复活" if zh else f"Respawn in {left:.1f}s")
            else:
                self.sub_text.setText("观战中" if zh else "Spectating")
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
                eco = getattr(world.mode, "economy", False)
                rows = [f"{TEAM_NAMES[k]}   {world.score[k]}", "",
                        "Name        K    D    HS" + ("     $" if eco else "")]
                mine = world.agents[self.local_id].team
                for a in sorted((a for a in world.agents if a.team == k), key=lambda a: -a.kills):
                    mark = " <" if a.id == self.local_id else ""
                    money = f"  {a.money:>5}" if eco and a.team == mine else ""
                    op = f" {a.operator}" if a.operator and world.mode.name == "r6" else ""
                    rows.append(f"{a.name:<10} {a.kills:>3}  {a.deaths:>3}  {a.headshots:>3}{money}{op}{mark}")
                self.board_text[k].setText("\n".join(rows))
        else:
            self.board.hide()

    def _update_mode(self, dt, world, agent):
        zh = self.cjk
        mode = world.mode
        now = world.time
        # round banner
        if self.round_t > 0:
            self.round_t -= dt
            a = min(1.0, self.round_t / 0.5)
            self.round_text.setAlphaScale(a)
            self.round_sub.setAlphaScale(a)
            if self.round_t <= 0:
                self.round_text.setText("")
                self.round_sub.setText("")
        if mode.name == "dom" and mode.points:
            self._update_dom(mode)
        else:
            self.obj_root.hide()
        # phase / economy
        phase = ""
        if mode.name in ("bomb", "elim", "r6", "bio"):
            names = {"freeze": ("冻结时间", "FREEZE TIME"), "prep": ("准备阶段", "PREPARATION"),
                     "live": ("", ""), "planted": ("炸弹已安放" if mode.name == "bomb" else "拆弹器已安放",
                                                  "BOMB PLANTED" if mode.name == "bomb" else "DEFUSER PLANTED"),
                     "over": ("回合结束", "ROUND OVER")}
            phase = names.get(mode.phase, ("", ""))[0 if zh else 1]
            if mode.name == "bio":
                info = mode.hud(agent.team)
                count = (f"人类 {info['humans']}  僵尸 {info['zombies']}" if zh else
                         f"HUMANS {info['humans']}  ZOMBIES {info['zombies']}")
                phase = (phase + "   " if phase else "") + count
            if getattr(mode, "economy", False) and mode.can_buy_now(agent):
                left = mode.state().get("buy_left", 0.0)
                phase = (phase + "   " if phase else "") + (f"购买 {left:.0f}s [B]" if zh else f"BUY {left:.0f}s [B]")
            if mode.name == "r6":
                side = mode.side(agent.team)
                extra = []
                if side == 1:
                    extra.append((f"加固 {mode.reinforce_left}" if zh else f"REINFORCE {mode.reinforce_left}"))
                else:
                    extra.append((f"无人机 {world.gadgets.drones_left.get(agent.id, 0)}" if zh else
                                  f"DRONES {world.gadgets.drones_left.get(agent.id, 0)}"))
                phase = (phase + "   " if phase else "") + "  ".join(extra)
        self.phase_text.setText(phase)
        self.money_text.setText(f"${agent.money}" if getattr(mode, "economy", False) else "")
        op = ""
        if mode.name == "r6" and agent.operator:
            from ..operators import OPERATORS
            o = OPERATORS.get(agent.operator)
            op = (o.zh if zh else o.name.upper()) if o else ""
        armor = (("护甲+头盔" if agent.helmet else "护甲") if zh else
                 ("ARMOR+HELMET" if agent.helmet else "ARMOR")) if agent.armor > 0 else ""
        self.armor_text.setText(" ".join(x for x in (op, armor) if x))
        # hold action progress
        if agent.busy is not None:
            kind, t0, dur = agent.busy[0], agent.busy[1], agent.busy[2]
            k = min(1.0, max(0.0, (now - t0) / max(dur, 1e-3)))
            labels = {"plant": ("安放中", "PLANTING"), "defuse": ("拆除中", "DEFUSING"),
                      "reinforce": ("加固中", "REINFORCING"), "barricade": ("设置路障", "BARRICADING"),
                      "place": ("部署中", "DEPLOYING")}
            self.busy_label.setText(labels.get(kind, (kind, kind))[0 if zh else 1])
            self.busy_bar.setSx(max(k, 0.001))
            self.busy_root.show()
        else:
            self.busy_root.hide()
        # contextual prompt
        prompt = ""
        if not self.watch and agent.alive and agent.busy is None and mode.name in ("bomb", "elim", "r6"):
            b = mode.bomb
            side = mode.side(agent.team)
            if b and b["state"] == "planted" and side == 1 and \
                    math.hypot(agent.pos[0] - b["pos"][0], agent.pos[1] - b["pos"][1]) < 1.5:
                prompt = "按住 F 拆除" if zh else "Hold F to defuse"
            elif b and b["state"] == "carried" and b["carrier"] == agent.id and mode.phase == "live" \
                    and mode.site_at(agent.pos) >= 0:
                prompt = "按住 F 或左键 安放" if zh else "Hold F / LMB to plant"
        self.prompt_text.setText(prompt)
        # gadget view overlay
        if agent.view_mode == 1:
            locked = getattr(world.mode, "view_locked", None)
            if locked is not None and locked(agent):
                tail = "准备阶段结束后返回本体" if zh else "body view after prep"
            else:
                tail = "G 返回" if zh else "G exit"
            self.view_text.setText(("无人机 · WASD 移动 · 空格 跳 · 左键 标记 · " if zh else
                                    "DRONE · WASD move · SPACE jump · LMB spot · ") + tail)
        elif agent.view_mode == 2:
            self.view_text.setText("摄像头 · 左键 标记 · 右键 下一个 · G 返回" if zh else
                                   "CAMERA · LMB spot · RMB next · G exit")
        else:
            self.view_text.setText("")
        if self.watch:
            if self.god_view:
                self.view_text.setText("上帝视角   WASD 移动   鼠标转向   Q 上升  E 下降   Shift 加速   F 第一人称"
                                       if zh else
                                       "GOD VIEW   WASD move   mouse look   Q up  E down   SHIFT fast   F first person")
                self.cross.hide()
                self.scope.hide()
                self.scope_cross.hide()
                self.hp_panel.hide()
                self.ammo_panel.hide()
            else:
                self.view_text.setText(f"第一人称  {agent.name}   左键/右键切换   F 上帝视角" if zh else
                                       f"FIRST PERSON  {agent.name}   LMB/RMB cycle   F god view")
                self.hp_panel.show()
                self.ammo_panel.show()
        self._update_feed(now, agent)
        self._update_markers(world, agent)
        # flashbang — a spectator camera is never blinded
        if not self.watch and now < agent.flash_until:
            total = max(agent.flash_until - agent.flash_t0, 1e-3)
            left = agent.flash_until - now
            a = min(1.0, left / min(1.2, total)) if left < 1.2 else 1.0
            self.flash_card.setColorScale(1, 1, 1, a * 0.97)
        else:
            self.flash_card.setColorScale(1, 1, 1, 0)

    def _update_dom(self, mode):
        zh = self.cjk
        self.score_panel.hide()
        self.obj_root.show()
        for team in (0, 1):
            for k, cell in enumerate(self.dom_wins[team]):
                cell.show() if mode.wins[team] > k else cell.hide()
        target = mode.p["target"]
        for team in (0, 1):
            self.dom_cap[team].setText(str(mode.cap[team]))
            self.dom_bar[team].setSx(max(mode.cap[team] / target, 0.001))
        left = mode.time_left()
        clock = f"{int(left // 60)}:{int(left % 60):02d}"
        if mode.phase == "live":
            name = mode.points[mode.active]["name"]
            n0, n1 = mode.present
            self.dom_point.setText(f"{name} 点  {clock}" if zh else f"POINT {name}  {clock}")
            if n0 and n1:
                self.dom_info.setText("争夺中" if zh else "CONTESTED")
                self.dom_colon.setFg((1, 0.85, 0.2, 1))
            else:
                self.dom_info.setText("")
                self.dom_colon.setFg(DOM_COLORS[0 if n0 else 1] if (n0 or n1) else (1, 1, 1, 1))
        else:
            self.dom_colon.setFg((1, 1, 1, 1))
            self.dom_point.setText("")
            if mode.phase == "prep":
                self.dom_info.setText(f"准备阶段 {left:.0f}s" if zh else f"PREPARATION {left:.0f}s")
            elif mode.phase == "gap":
                self.dom_info.setText(f"下一个点 {left:.0f}s 后开启" if zh else f"NEXT POINT IN {left:.0f}s")
            else:
                self.dom_info.setText("")

    def _update_feed(self, now, agent):
        if self.god_view or not agent.view_mode:
            self.feed_fx.hide()
            self.reticle.hide()
            self.rec_text.setText("")
            return
        drone = agent.view_mode == 1
        self.feed_fx.show()
        self.reticle.show()
        flicker = random.uniform(0.0, 0.03)
        if drone:
            self.feed_tint.setColor(0.45, 0.85, 0.55, 0.1 + flicker)
        else:
            self.feed_tint.setColor(0.55, 0.65, 0.95, 0.16 + flicker)
        self.scan.setZ((now * 0.04) % 0.05)
        for band in self.noise:
            band.setZ(random.uniform(-1, 1))
            band.setSz(random.uniform(0.3, 2.5))
            band.setColorScale(1, 1, 1, random.uniform(0.0, 0.1))
        label = ("无人机" if self.cjk else "DRONE") if drone else (("摄像头" if self.cjk else "CAM") + f" {agent.view_ref}")
        self.rec_text.setText(("● " if int(now * 2) % 2 else "   ") + "REC  " + label)

    def _update_markers(self, world, agent):
        mode = world.mode
        now = world.time
        zh = self.cjk
        items = []
        if not world.done and mode.name in ("bomb", "elim", "r6") and mode.phase != "over":
            b = mode.bomb
            if b and b["state"] == "planted":
                label = ("拆弹器" if zh else "DEFUSER") if mode.name == "r6" else "C4"
                items.append(((b["pos"][0], b["pos"][1], b["pos"][2] + 0.6), label, (1, 0.35, 0.3, 1)))
            else:
                if self.god_view:
                    site_ids = mode.active_sites
                else:
                    site_ids = mode.known_sites(agent) if hasattr(mode, "known_sites") else mode.active_sites
                for i in site_ids:
                    s = mode.sites[i]
                    c = mode.site_centers[i]
                    items.append(((c[0], c[1], s.get("z", s["zmin"] + 0.5) + 1.6), s["name"], (1, 0.85, 0.4, 0.9)))
        elif not world.done and mode.name == "dom" and mode.phase == "live":
            p = mode.points[mode.active]
            c = poly_center(p["poly"])
            n0, n1 = mode.present
            col = (1, 0.85, 0.2, 1) if n0 and n1 else DOM_COLORS[0 if n0 else 1] if n0 or n1 else (1, 1, 1, 0.9)
            items.append(((c[0], c[1], p.get("z", p.get("zmin", 0.0) + 0.5) + 2.0), p["name"], col))
        for e in world.agents:
            enemy = True if self.god_view else e.team != agent.team
            if enemy and e.alive and e.spotted_until > now:
                items.append(((e.pos[0], e.pos[1], e.pos[2] + e.height + 0.35), "!", (1, 0.25, 0.2, 1)))
        cam = self.base.cam
        hfov, vfov = self.base.camLens.getFov()
        tx, ty = math.tan(math.radians(hfov / 2)), math.tan(math.radians(vfov / 2))
        ar = self.base.getAspectRatio()
        eye = cam.getPos(self.base.render)
        placed = []
        for i, node in enumerate(self.marks):
            if i >= len(items):
                node.setText("")
                continue
            pos, label, col = items[i]
            wp = Point3(float(pos[0]), float(pos[1]), float(pos[2]))
            p = cam.getRelativePoint(self.base.render, wp)
            if p.y > 0.1:
                fx, fy = p.x / p.y / tx, p.z / p.y / ty
            else:
                n = max(abs(p.x), abs(p.z), 1e-6)
                fx, fy = p.x / n * 2.0, p.z / n * 2.0
            k = max(abs(fx) / 0.94, abs(fy) / 0.86, 1.0)
            x, y = fx / k * ar, fy / k
            while any(abs(x - px) < 0.16 and abs(y - py) < 0.09 for px, py in placed):
                y -= 0.09
            placed.append((x, y))
            node.setPos(x, y)
            node.setText(f"{label}\n{(wp - eye).length():.0f}m")
            node.setFg((col[0], col[1], col[2], col[3] * (0.55 if k > 1.0 else 1.0)))

    def set_help(self, text: str):
        self.help_text.setText(text)

    def set_info(self, text: str):
        self.info_text.setText(text)

    def set_visible(self, visible: bool):
        for n in (self.root, self.hp_panel, self.ammo_panel, self.score_panel, self.feed_root, self.banner.root,
                  self.inspect_root, self.help_text, self.info_text, self.feed_fx, self.rec_text):
            if visible:
                n.show()
            else:
                n.hide()
        if not visible:
            self.obj_root.hide()
            self.scope.hide()
            self.scope_cross.hide()
            self.hurt.setColorScale(0.8, 0, 0, 0)
            self.bleed.setColorScale(1, 1, 1, 0)
            self.vignette.hide()
        else:
            self.vignette.show()
            self.banner.root.hide()
            self.inspect_root.hide()

    def destroy(self):
        self.flash_card.removeNode()
        self.feed_fx.removeNode()
        self.rec_text.destroy()
        self.obj_root.removeNode()
        self.banner.destroy()
        self.inspect_root.removeNode()
        self.root.removeNode()
        self.scope.removeNode()
        self.scope_cross.removeNode()
        self.hp_panel.removeNode()
        self.ammo_panel.removeNode()
        self.score_panel.removeNode()
        self.feed_root.removeNode()
        self.vignette.destroy()
        self.hurt.destroy()
        self.bleed.destroy()

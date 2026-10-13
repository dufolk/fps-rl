"""2D heads-up display."""
from __future__ import annotations

import math
import os
import random

from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import CardMaker, Filename, NodePath, Point3, TextNode, TransparencyAttrib

from ..geometry import poly_center
from ..sim import TEAM_NAMES
from ..weapons import SLOT_FRAG, SLOT_SMOKE, WEAPONS

TEAM_COLORS = {0: (1.0, 0.55, 0.3, 1), 1: (0.45, 0.75, 1.0, 1)}
CROSS_COLOR = (0.35, 1.0, 0.45, 0.95)


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
                                parent=self.ammo_panel)
        self.nade_text = text("", (-0.07, 0.055), 0.034, (0.85, 0.85, 0.85, 1), TextNode.ARight,
                              parent=self.ammo_panel)
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
            img = OnscreenImage(textures.sprite("damage_arc"), pos=(0, 0, 0), scale=0.42, parent=self.root)
            img.setTransparency(TransparencyAttrib.M_alpha)
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
        self.scope.hide()
        # scoreboard
        self.board = self.root.attachNewNode("board")
        _rect(self.board, (0, 0, 0, 0.65), -0.95, 0.95, -0.62, 0.62)
        self.board_text = [text("", (-0.88 + 0.95 * k, 0.5), 0.042, TEAM_COLORS[k], TextNode.ALeft, parent=self.board,
                                font=self.font_cjk) for k in (0, 1)]
        self.board.hide()
        self.flash_white = 0.0
        self.end_hint = "按 F5 重新开始" if self.cjk else "Press F5 to restart"
        # objectives (under the score panel)
        cf = self.font_cjk or f
        self.obj_root = top.attachNewNode("objectives")
        self.obj_cells = []
        for i in range(3):
            x = (i - 1) * 0.13
            bg = _rect(self.obj_root, (0.1, 0.1, 0.1, 0.6), x - 0.05, x + 0.05, -0.25, -0.15)
            fill = _rect(self.obj_root, (1, 1, 1, 0.8), 0.0, 0.1, 0.0, 0.1)
            fill.setPos(x - 0.05, 0, -0.25)
            label = text("ABC"[i], (x, -0.218), 0.05, parent=self.obj_root)
            self.obj_cells.append((bg, fill, label))
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

    def show_round(self, title, sub="", duration=3.5, color=(1, 1, 1, 1)):
        self.round_text.setText(title)
        self.round_text.setFg(color)
        self.round_sub.setText(sub)
        self.round_t = duration

    # ------------------------------------------------------------------ events
    def on_event(self, e, world):
        t = e["type"]
        me = self.local_id
        if t == "damage":
            if e["attacker"] == me and e["victim"] != me:
                self.hit_t = 0.22
                self.hit_kill = False
            if e["victim"] == me and e["attacker"] >= 0 and e["attacker"] != me:
                att = world.agents[e["attacker"]]
                v = world.agents[me]
                dx, dy = att.pos[0] - v.pos[0], att.pos[1] - v.pos[1]
                slot = self.arcs[self.arc_idx]
                self.arc_idx = (self.arc_idx + 1) % len(self.arcs)
                slot[1] = 1.2
                slot[2] = math.degrees(math.atan2(dy, dx))
                self.hurt_level = min(1.0, self.hurt_level + e["amount"] / 60.0)
            elif e["victim"] == me:
                self.hurt_level = min(1.0, self.hurt_level + e["amount"] / 60.0)
        elif t == "kill":
            a = world.agents[e["attacker"]].name if e["attacker"] >= 0 else "world"
            v = world.agents[e["victim"]].name
            w = WEAPONS[e["weapon"]].display if e["weapon"] in WEAPONS else e["weapon"]
            hs = "  [HS]" if e["headshot"] else ""
            col = TEAM_COLORS[world.agents[e["attacker"]].team] if e["attacker"] >= 0 else (1, 1, 1, 1)
            if e["attacker"] == me or e["victim"] == me:
                col = (1, 1, 0.5, 1)
            self.feed.insert(0, [f"{a}  [{w}]{hs}  {v}", col, 6.0])
            self.feed = self.feed[:6]
            if e["attacker"] == me and e["victim"] != me:
                self.hit_t = 0.4
                self.hit_kill = True
                if world.agents[e["victim"]].team != world.agents[me].team:
                    self.kill_fx = self.banner.on_kill(e["weapon"], e["headshot"], v)
            if e["victim"] == me:
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
            side = mode.side(me.team)
            if mode.name == "r6":
                role = ("进攻方" if side == 0 else "防守方") if zh else ("ATTACK" if side == 0 else "DEFEND")
                sub = ("准备阶段: 无人机侦查 / 加固墙体" if zh else "Prep phase: drone / reinforce")
            else:
                role = ("潜伏者 (T)" if side == 0 else "保卫者 (CT)") if zh else ("TERRORISTS" if side == 0 else "COUNTER-TERRORISTS")
                sub = ("B 键购买武器" if zh else "Press B to buy")
            self.show_round((f"第 {n} 回合" if zh else f"ROUND {n}"), f"{role}   ·   {sub}", 3.0)
        elif t == "round_end":
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
        elif t == "point_captured":
            name = e.get("name", "?")
            ours = e["team"] == me.team
            msg = (f"已占领 {name} 点" if ours else f"{name} 点失守") if zh else \
                (f"Captured {name}" if ours else f"Lost {name}")
            self.show_round(msg, "", 1.8, (0.5, 1, 0.55, 1) if ours else (1, 0.45, 0.4, 1))
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
        else:
            self.scope.hide()
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
        self.weapon_text.setText(spec.display)
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
            self.reserve_text.setText("")
        else:
            self.ammo_text.setText("--")
            self.ammo_text.setFg((1, 1, 1, 1))
            self.reserve_text.setText("")
        items = []
        for s in range(4, 8):
            w = agent.inv.get(s)
            if w is not None:
                n = f" x{w.mag}" if w.spec.kind in ("grenade", "gadget", "shield") else ""
                items.append(f"[{s}] {w.spec.display.upper()}{n}")
        self.nade_text.setText("   ".join(items))
        score = world.score
        self.score_l.setText(str(score[0]))
        self.score_r.setText(str(score[1]))
        mode = world.mode
        if hasattr(mode, "time_left") and mode.name in ("bomb", "r6"):
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
        # damage arcs
        for slot in self.arcs:
            img, life, ang = slot
            if life <= 0:
                img.hide()
                continue
            slot[1] -= dt
            rel = ang - agent.yaw
            img.setR(-rel)
            img.setColorScale(1, 1, 1, min(1.0, slot[1]))
            img.show()
        self.hurt_level = max(0.0, self.hurt_level - dt * 0.8)
        low_hp = 0.35 * max(0.0, 1.0 - agent.hp / 35.0) if agent.alive else 0.0
        self.hurt.setColorScale(0.8, 0.0, 0.0, min(0.9, self.hurt_level * 0.7 + low_hp))
        # death / spectate
        zh = self.cjk
        if not agent.alive and not world.done:
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
                eco = world.mode.name == "bomb"
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
        # domination cells
        if mode.name == "dom" and mode.points:
            self.obj_root.show()
            for i, (bg, fill, label) in enumerate(self.obj_cells):
                for n in (bg, fill, label):
                    n.show() if i < len(mode.points) else n.hide()
                if i >= len(mode.points):
                    continue
                label.setText(mode.points[i]["name"])
                own = mode.owner[i]
                prog = mode.progress[i]
                k = abs(prog)
                team = 0 if prog > 0 else 1
                c = TEAM_COLORS[team]
                fill.setColor(c[0], c[1], c[2], 0.85 if own >= 0 else 0.5)
                fill.setSz(max(k, 0.001))
                n0, n1 = mode.present[i]
                bg.setColor(*(((0.9, 0.9, 0.2, 0.7) if n0 and n1 else (0.1, 0.1, 0.1, 0.6))))
        else:
            self.obj_root.hide()
        # phase / economy
        phase = ""
        if mode.name in ("bomb", "r6"):
            names = {"freeze": ("冻结时间", "FREEZE TIME"), "prep": ("准备阶段", "PREPARATION"),
                     "live": ("", ""), "planted": ("炸弹已安放" if mode.name == "bomb" else "拆弹器已安放",
                                                  "BOMB PLANTED" if mode.name == "bomb" else "DEFUSER PLANTED"),
                     "over": ("回合结束", "ROUND OVER")}
            phase = names.get(mode.phase, ("", ""))[0 if zh else 1]
            if mode.name == "bomb" and mode.can_buy_now(agent):
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
        self.money_text.setText(f"${agent.money}" if mode.name == "bomb" else "")
        op = ""
        if mode.name == "r6" and agent.operator:
            from ..operators import OPERATORS
            o = OPERATORS.get(agent.operator)
            op = (o.zh if zh else o.name.upper()) if o else ""
        self.armor_text.setText(" ".join(x for x in (op, ("护甲" if zh else "ARMOR") if agent.armor > 0 else "") if x))
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
        if agent.alive and agent.busy is None and mode.name in ("bomb", "r6"):
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
            self.view_text.setText("无人机 · WASD 移动 · 空格 跳 · 左键 标记 · G 返回" if zh else
                                   "DRONE · WASD move · SPACE jump · LMB spot · G exit")
        elif agent.view_mode == 2:
            self.view_text.setText("摄像头 · 左键 标记 · 右键 下一个 · G 返回" if zh else
                                   "CAMERA · LMB spot · RMB next · G exit")
        else:
            self.view_text.setText("")
        self._update_feed(now, agent)
        self._update_markers(world, agent)
        # flashbang
        if now < agent.flash_until:
            total = max(agent.flash_until - agent.flash_t0, 1e-3)
            left = agent.flash_until - now
            a = min(1.0, left / min(1.2, total)) if left < 1.2 else 1.0
            self.flash_card.setColorScale(1, 1, 1, a * 0.97)
        else:
            self.flash_card.setColorScale(1, 1, 1, 0)

    def _update_feed(self, now, agent):
        if not agent.view_mode:
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
        if not world.done and mode.name in ("bomb", "r6") and mode.phase != "over":
            b = mode.bomb
            if b and b["state"] == "planted":
                label = ("拆弹器" if zh else "DEFUSER") if mode.name == "r6" else "C4"
                items.append(((b["pos"][0], b["pos"][1], b["pos"][2] + 0.6), label, (1, 0.35, 0.3, 1)))
            else:
                for i in mode.active_sites:
                    s = mode.sites[i]
                    c = mode.site_centers[i]
                    items.append(((c[0], c[1], s.get("z", s["zmin"] + 0.5) + 1.6), s["name"], (1, 0.85, 0.4, 0.9)))
        elif not world.done and mode.name == "dom":
            for i, p in enumerate(mode.points):
                c = poly_center(p["poly"])
                own = mode.owner[i]
                col = TEAM_COLORS[own] if own >= 0 else (1, 1, 1, 0.9)
                items.append(((c[0], c[1], p.get("z", p.get("zmin", 0.0) + 0.5) + 2.0), p["name"], col))
        for e in world.agents:
            if e.team != agent.team and e.alive and e.spotted_until > now:
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
            self.scope.hide()
            self.hurt.setColorScale(0.8, 0, 0, 0)
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
        self.hp_panel.removeNode()
        self.ammo_panel.removeNode()
        self.score_panel.removeNode()
        self.feed_root.removeNode()
        self.vignette.destroy()
        self.hurt.destroy()

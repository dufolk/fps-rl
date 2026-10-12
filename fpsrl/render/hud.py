"""2D heads-up display."""
from __future__ import annotations

import math
import os

from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import CardMaker, Filename, NodePath, TextNode, TransparencyAttrib

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
    def update(self, dt, world, agent, vfov_deg, show_board=False, scope_active=False, spectating=None):
        now = world.time
        self.banner.update(dt)
        if self.inspect_t > 0:
            self.inspect_t -= dt
            self.inspect_root.setAlphaScale(min(1.0, self.inspect_t / 0.4))
            if self.inspect_t <= 0:
                self.inspect_root.hide()
        # crosshair
        spec = agent.spec
        show_cross = agent.alive and not scope_active and not (spec.kind == "gun" and agent.ads and spec.name == "awm")
        if show_cross and spec.kind == "gun" and not (agent.ads and spec.ads_fov):
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
        elif spec.kind == "grenade":
            self.ammo_text.setText(str(ws.mag))
            self.ammo_text.setFg((1, 1, 1, 1))
            self.reserve_text.setText("")
        else:
            self.ammo_text.setText("--")
            self.ammo_text.setFg((1, 1, 1, 1))
            self.reserve_text.setText("")
        self.nade_text.setText(f"[4] HE x{agent.inv[SLOT_FRAG].mag}    [5] SMOKE x{agent.inv[SLOT_SMOKE].mag}")
        self.score_l.setText(str(world.team_kills[0]))
        self.score_r.setText(str(world.team_kills[1]))
        if world.cfg.time_limit:
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
                rows = [f"{TEAM_NAMES[k]}   {world.team_kills[k]}", "", "Name        K    D    HS"]
                for a in sorted((a for a in world.agents if a.team == k), key=lambda a: -a.kills):
                    mark = " <" if a.id == self.local_id else ""
                    rows.append(f"{a.name:<10} {a.kills:>3}  {a.deaths:>3}  {a.headshots:>3}{mark}")
                self.board_text[k].setText("\n".join(rows))
        else:
            self.board.hide()

    def set_help(self, text: str):
        self.help_text.setText(text)

    def set_info(self, text: str):
        self.info_text.setText(text)

    def set_visible(self, visible: bool):
        for n in (self.root, self.hp_panel, self.ammo_panel, self.score_panel, self.feed_root, self.banner.root,
                  self.inspect_root, self.help_text, self.info_text):
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

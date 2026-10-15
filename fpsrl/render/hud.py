"""2D heads-up display."""
from __future__ import annotations

import math
import os

from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import CardMaker, CullFaceAttrib, Filename, NodePath, Point3, TextNode, TransparencyAttrib

from ..sim import LEAN_DIST, LEAN_ROLL, TEAM_NAMES
from ..weapons import LOADOUT_BY_KEY, NUM_SLOTS, item_name

TEAM_COLORS = {0: (1.0, 0.55, 0.3, 1), 1: (0.45, 0.75, 1.0, 1)}
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
        # health panel
        bl = base.a2dBottomLeft
        self.hp_panel = bl.attachNewNode("hp")
        _rect(self.hp_panel, (0, 0, 0, 0.35), 0.04, 0.62, 0.04, 0.2)
        self.hp_text = text("100", (0.17, 0.085), 0.085, parent=self.hp_panel)
        self.hp_label = text("HP", (0.075, 0.09), 0.04, (0.8, 0.8, 0.8, 1), parent=self.hp_panel)
        _rect(self.hp_panel, (0.15, 0.15, 0.15, 0.8), 0.27, 0.6, 0.095, 0.125)
        self.hp_bar = _rect(self.hp_panel, (0.35, 0.95, 0.45, 0.95), 0.0, 0.33, 0.0, 0.03)
        self.hp_bar.setPos(0.27, 0, 0.095)
        self.loadout_text = text("", (0.33, 0.155), 0.032, (0.6, 0.8, 1.0, 1), TextNode.ALeft, parent=self.hp_panel,
                                 font=cf)
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
        self.phase_text = text("", (0, -0.2), 0.04, (1, 0.9, 0.55, 1), parent=self.score_panel, font=cf)
        # killfeed
        tr = base.a2dTopRight
        self.feed_root = tr.attachNewNode("feed")
        self.feed = []
        self.feed_texts = [text("", (-0.05, -0.08 - i * 0.06), 0.04, align=TextNode.ARight, parent=self.feed_root,
                                font=self.font_cjk) for i in range(6)]
        # kill medals (bottom centre)
        from .killbanner import KillBanner
        self.banner = KillBanner(base, cf, self.cjk)
        self.kill_fx = None
        # center messages
        self.center_text = text("", (0, 0.25), 0.07, font=self.font_cjk)
        self.sub_text = text("", (0, 0.17), 0.045, (0.9, 0.9, 0.9, 1), font=self.font_cjk)
        self.help_text = text("", (-1.7, 0.85), 0.035, (1, 1, 1, 0.9), TextNode.ALeft,
                              parent=base.a2dTopLeft, font=self.font_cjk)
        self.help_text.setPos(0.05, -0.08)
        self.info_text = text("", (0.05, 0.25), 0.032, (0.85, 0.85, 0.85, 0.9), TextNode.ALeft,
                              parent=base.a2dBottomLeft, font=self.font_cjk)
        self.view_text = text("", (0, -0.85), 0.04, (0.7, 1.0, 0.7, 1), font=cf)
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
        # its own node: hiding the scope node leaves quads created afterwards invisible
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
        # site markers (god view)
        self.marks = [text("", (0, 0), 0.036, font=cf) for _ in range(4)]
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
        self._update_round(dt, world, agent)
        # crosshair
        spec = agent.spec
        sights = agent.ads and spec.ads_fov and first_person
        show_cross = agent.alive and not scope_active and not (spec.kind == "gun" and sights and spec.scope_overlay)
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
        self.loadout_text.setText(loadout_label(agent.loadout, zh))
        ws = agent.weapon
        self.weapon_text.setText(item_name(spec.name, zh))
        if spec.kind == "gun":
            self.ammo_text.setText(str(ws.mag))
            self.reserve_text.setText(f"/ {ws.reserve}")
            self.reserve_text.setFg((0.8, 0.8, 0.8, 1))
            low = ws.mag <= max(1, spec.mag // 5)
            self.ammo_text.setFg((1, 0.35, 0.3, 1) if low else (1, 1, 1, 1))
            if agent.reloading:
                p = 1.0 - (agent.reload_until - now) / spec.reload_time
                self.reserve_text.setText(f"RELOADING {int(p * 100)}%")
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
                n = f" x{w.mag}" if w.spec.kind == "grenade" else ""
                label = item_name(w.name, zh)
                items.append(f"[{s}] {label if zh else label.upper()}{n}")
        self.nade_text.setText("   ".join(items))
        score = world.score
        self.score_l.setText(str(score[0]))
        self.score_r.setText(str(score[1]))
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
            self.sub_text.setText(f"观战 {who}    左键/右键切换    F3 切换人称" if zh else
                                  f"Spectating {who}    LMB/RMB cycle    F3 camera")
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
        self.phase_text.setText(phase)
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
        else:
            self.view_text.setText("")
        self._update_markers(world)
        # flashbang - a spectator camera is never blinded
        if not self.watch and now < agent.flash_until:
            total = max(agent.flash_until - agent.flash_t0, 1e-3)
            left = agent.flash_until - now
            a = min(1.0, left / min(1.2, total)) if left < 1.2 else 1.0
            self.flash_card.setColorScale(1, 1, 1, a * 0.97)
        else:
            self.flash_card.setColorScale(1, 1, 1, 0)

    def _update_markers(self, world):
        """Site labels, only in the spectator's god view."""
        mode = world.mode
        items = []
        if self.god_view and not world.done:
            for s, c in zip(mode.sites, mode.site_centers):
                items.append(((c[0], c[1], s.get("z", s.get("zmin", 0.0) + 0.5) + 1.6), s["name"],
                              (1, 0.85, 0.4, 0.9)))
        cam = self.base.cam
        hfov, vfov = self.base.camLens.getFov()
        tx, ty = math.tan(math.radians(hfov / 2)), math.tan(math.radians(vfov / 2))
        ar = self.base.getAspectRatio()
        eye = cam.getPos(self.base.render)
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
            node.setPos(fx / k * ar, fy / k)
            node.setText(f"{label}\n{(wp - eye).length():.0f}m")
            node.setFg((col[0], col[1], col[2], col[3] * (0.55 if k > 1.0 else 1.0)))

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
        self.hp_panel.removeNode()
        self.ammo_panel.removeNode()
        self.score_panel.removeNode()
        self.feed_root.removeNode()
        self.vignette.destroy()
        self.hurt.destroy()

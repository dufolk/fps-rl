"""CF-style kill medals shown at the bottom centre of the screen."""
from __future__ import annotations

import math

import numpy as np
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, TransparencyAttrib

from .textures import make_texture

KINDS = {  # fill colour, rim colour, glow colour
    "kill": ((0.62, 0.06, 0.05), (0.85, 0.86, 0.9), (1.0, 0.25, 0.15)),
    "headshot": ((0.7, 0.05, 0.04), (1.0, 0.78, 0.25), (1.0, 0.7, 0.2)),
    "knife": ((0.1, 0.22, 0.42), (0.8, 0.88, 1.0), (0.35, 0.65, 1.0)),
    "grenade": ((0.55, 0.22, 0.03), (1.0, 0.75, 0.3), (1.0, 0.5, 0.1)),
}
STREAK_LABELS = {1: ("击杀", "KILL"), 2: ("双杀", "DOUBLE KILL"), 3: ("三杀", "TRIPLE KILL"),
                 4: ("四杀", "QUADRA KILL"), 5: ("五杀", "PENTA KILL"), 6: ("超神", "GODLIKE")}
KIND_TAGS = {"headshot": ("爆头", "HEADSHOT"), "knife": ("刀杀", "KNIFE KILL"), "grenade": ("手雷击杀", "GRENADE KILL")}


# ----------------------------------------------------------------------------- medal textures
def _cov(d, px):
    """Anti-aliased coverage of a signed distance field (negative = inside)."""
    return np.clip(0.5 - d / px, 0.0, 1.0)


def _over(dst, rgb, a):
    a = a[..., None]
    dst[..., :3] = dst[..., :3] * (1 - a) + np.asarray(rgb) * a
    dst[..., 3:] = dst[..., 3:] * (1 - a) + a


def _hexagon(x, y, r):
    return np.maximum(np.abs(x) * 0.866 + np.abs(y) * 0.5, np.abs(y)) - r


def _box(x, y, w, h):
    return np.maximum(np.abs(x) - w, np.abs(y) - h)


def _glyph(kind, x, y):
    """Signed distance of the white glyph and of its cut-outs."""
    if kind == "headshot":
        cranium = np.hypot(x, y - 0.08) - 0.33
        jaw = _box(x, y + 0.24, 0.19, 0.12) - 0.03
        skull = np.minimum(cranium, jaw)
        eyes = np.minimum(np.hypot(x - 0.135, y - 0.03) - 0.095, np.hypot(x + 0.135, y - 0.03) - 0.095)
        nose = _box(x, y + 0.1, 0.03, 0.045)
        teeth = np.minimum(_box(x, y + 0.29, 0.006, 0.07), np.minimum(_box(x - 0.08, y + 0.29, 0.006, 0.07),
                                                                     _box(x + 0.08, y + 0.29, 0.006, 0.07)))
        holes = np.minimum(np.minimum(eyes, nose), teeth)
        return skull, holes
    if kind == "knife":
        c, s = math.cos(math.radians(40)), math.sin(math.radians(40))
        u, v = x * c + y * s, -x * s + y * c
        blade = np.maximum(_box(u - 0.08, v, 0.3, 0.075), (v - 0.075) + (u - 0.2) * 0.45)
        guard = _box(u + 0.24, v, 0.025, 0.13)
        handle = _box(u + 0.4, v, 0.13, 0.055) - 0.015
        return np.minimum(np.minimum(blade, guard), handle), _box(u - 0.05, v - 0.03, 0.22, 0.008)
    if kind == "grenade":
        ang = np.arctan2(y, x)
        rad = np.hypot(x, y)
        star = rad - (0.2 + 0.2 * (0.5 + 0.5 * np.cos(ang * 8)) ** 2)
        return star, np.hypot(x, y) - 0.1
    ring = np.abs(np.hypot(x, y) - 0.3) - 0.045
    ticks = np.minimum(_box(x, y, 0.03, 0.45), _box(x, y, 0.45, 0.03))
    gap = np.hypot(x, y) - 0.14
    cross = np.maximum(ticks, -gap)
    dot = np.hypot(x, y) - 0.05
    return np.minimum(np.minimum(ring, cross), dot), np.full_like(x, 1.0)


def medal_image(kind: str, size: int = 256) -> np.ndarray:
    fill, rim, _ = KINDS[kind]
    px = 2.0 / size
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size * 2 - 1
    y = 1 - (yy + 0.5) / size * 2
    img = np.zeros((size, size, 4))
    outer = _hexagon(x, y, 0.86)
    shadow = _hexagon(x + 0.02, y + 0.04, 0.88)
    _over(img, (0, 0, 0), _cov(shadow, px * 6) * 0.55)
    # metallic rim: vertical gradient + bevel highlight
    shade = 0.65 + 0.35 * y
    rim_rgb = np.asarray(rim)[None, None, :] * shade[..., None]
    a = _cov(outer, px)
    img[..., :3] = img[..., :3] * (1 - a[..., None]) + rim_rgb * a[..., None]
    img[..., 3] = img[..., 3] * (1 - a) + a
    inner = _hexagon(x, y, 0.72)
    grad = 0.55 + 0.6 * np.clip(0.5 + 0.5 * y, 0, 1) ** 1.5
    fill_rgb = np.asarray(fill)[None, None, :] * grad[..., None]
    a = _cov(inner, px)
    img[..., :3] = img[..., :3] * (1 - a[..., None]) + fill_rgb * a[..., None]
    _over(img, (1, 1, 1), _cov(np.abs(inner + 0.02) - 0.008, px) * 0.35)
    # radial streaks inside
    ang = np.arctan2(y, x)
    streak = (np.cos(ang * 12) * 0.5 + 0.5) ** 6 * np.clip(1 - np.hypot(x, y) / 0.72, 0, 1)
    _over(img, (1, 1, 1), streak * a * 0.12)
    g, holes = _glyph(kind, x, y)
    _over(img, (0, 0, 0), _cov(g - 0.025, px * 3) * 0.6)
    glyph = np.maximum(g, -holes)
    _over(img, (1.0, 0.97, 0.9), _cov(glyph, px))
    _over(img, fill, _cov(holes, px) * _cov(g, px))
    # top gloss
    gloss = np.clip((y - 0.1) * 1.4, 0, 1) * _cov(inner, px) * 0.18
    _over(img, (1, 1, 1), gloss)
    return img


def glow_image(size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size * 2 - 1
    y = 1 - (yy + 0.5) / size * 2
    r = np.hypot(x, y)
    ang = np.arctan2(y, x)
    rays = (np.cos(ang * 9) * 0.5 + 0.5) ** 8 * np.clip(1 - r, 0, 1) ** 1.5
    soft = np.clip(1 - r, 0, 1) ** 2.5
    a = np.clip(soft * 0.8 + rays * 0.7, 0, 1)
    img = np.ones((size, size, 4))
    img[..., 3] = a
    return img


# ----------------------------------------------------------------------------- banner widget
class KillBanner:
    HOLD = 2.6
    FADE = 0.45
    MAX_PIPS = 8

    def __init__(self, base, font=None, cjk=True):
        self.cjk = cjk
        self.root = base.a2dBottomCenter.attachNewNode("kill_banner")
        self.root.setZ(0.43)
        self.tex = {k: make_texture(medal_image(k), "medal_" + k, clamp=True) for k in KINDS}
        self.glow = OnscreenImage(make_texture(glow_image(), "medal_glow", clamp=True), scale=0.34, parent=self.root)
        self.glow.setTransparency(TransparencyAttrib.M_alpha)
        self.glow.setBin("fixed", 0)
        self.medal = OnscreenImage(self.tex["kill"], scale=0.15, parent=self.root)
        self.medal.setTransparency(TransparencyAttrib.M_alpha)
        self.medal.setBin("fixed", 1)
        self.flash = OnscreenImage(make_texture(glow_image(), "medal_flash", clamp=True), scale=0.2, parent=self.root)
        self.flash.setTransparency(TransparencyAttrib.M_alpha)
        self.flash.setBin("fixed", 2)
        self.label = OnscreenText(text="", pos=(0, -0.24), scale=0.068, fg=(1, 0.92, 0.75, 1), font=font,
                                  shadow=(0, 0, 0, 0.9), align=TextNode.ACenter, parent=self.root, mayChange=True)
        self.sub = OnscreenText(text="", pos=(0, -0.3), scale=0.036, fg=(0.92, 0.92, 0.92, 1), font=font,
                                shadow=(0, 0, 0, 0.9), align=TextNode.ACenter, parent=self.root, mayChange=True)
        self.pips = []
        for i in range(self.MAX_PIPS):
            p = OnscreenImage(self.tex["kill"], scale=0.028, parent=self.root)
            p.setTransparency(TransparencyAttrib.M_alpha)
            p.hide()
            self.pips.append([p, 0.0])
        self.streak = 0
        self.t = 99.0
        self.kind = "kill"
        self.root.hide()

    def _text(self, pair):
        return pair[0] if self.cjk else pair[1]

    def reset_streak(self):
        self.streak = 0

    def on_kill(self, weapon: str, headshot: bool, victim_name: str):
        self.streak += 1
        kind = "knife" if weapon == "knife" else "grenade" if weapon == "frag" else \
            "headshot" if headshot else "kill"
        self.kind = kind
        self.t = 0.0
        self.medal.setImage(self.tex[kind])
        self.medal.setTransparency(TransparencyAttrib.M_alpha)
        g = KINDS[kind][2]
        self.glow.setColorScale(g[0], g[1], g[2], 1)
        self.flash.setColorScale(1, 1, 1, 1)
        n = min(self.streak, 6)
        main = self._text(STREAK_LABELS[n])
        if n == 1 and kind in KIND_TAGS:
            main = self._text(KIND_TAGS[kind])
            tag = ""
        else:
            tag = self._text(KIND_TAGS[kind]) + "  " if kind in KIND_TAGS else ""
        self.label.setText(main)
        self.label.setFg((1.0, 0.85, 0.35, 1) if kind == "headshot" or n >= 3 else (1, 0.95, 0.85, 1))
        self.sub.setText(f"{tag}{'击败' if self.cjk else 'Eliminated'} {victim_name}")
        shown = min(self.streak, self.MAX_PIPS)
        width = 0.072
        for i, slot in enumerate(self.pips):
            p = slot[0]
            if i < shown:
                p.setPos((i - (shown - 1) / 2) * width, 0, -0.36)
                if i == shown - 1:
                    p.setImage(self.tex[kind])
                    p.setTransparency(TransparencyAttrib.M_alpha)
                    slot[1] = 0.0
                p.show()
            else:
                p.hide()
        self.root.show()
        self.root.setAlphaScale(1)
        return self.streak, kind

    def update(self, dt):
        if self.t > self.HOLD + self.FADE:
            return
        self.t += dt
        t = self.t
        if t < 0.12:
            s = 1.6 - 0.75 * (t / 0.12)
        elif t < 0.26:
            s = 0.85 + 0.15 * ((t - 0.12) / 0.14)
        else:
            s = 1.0
        self.medal.setScale(0.15 * s)
        self.glow.setScale(0.34 * (0.9 + 0.1 * math.sin(t * 6)))
        self.glow.setR(t * 25)
        self.glow.setAlphaScale(min(1.0, t * 6) * 0.9)
        self.flash.setScale(0.2 + t * 1.6)
        self.flash.setAlphaScale(max(0.0, 1 - t / 0.25))
        self.label.setScale(0.068 * (1.25 - 0.25 * min(1.0, t / 0.18)))
        for slot in self.pips:
            slot[1] += dt
            k = min(1.0, slot[1] / 0.15)
            slot[0].setScale(0.028 * (1.8 - 0.8 * k))
        if t > self.HOLD:
            self.root.setAlphaScale(max(0.0, 1 - (t - self.HOLD) / self.FADE))
        if t > self.HOLD + self.FADE:
            self.root.hide()

    def destroy(self):
        self.root.removeNode()

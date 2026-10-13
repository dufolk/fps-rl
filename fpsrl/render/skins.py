"""Weapon skins: procedural pattern textures + per-surface overrides.

A skin re-colours the surface groups of a weapon model (`PartSet` children are
named "<weapon>_<surface>"). Mesh UVs are planar projections in metres, so a
texture scale is enough to wrap a pattern around any part.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from panda3d.core import GeomVertexReader, GeomVertexRewriter, Material, TextureStage

from .textures import make_texture, noise

RARITY = {  # name, colour
    0: ("普通", "Common", (0.82, 0.84, 0.86, 1)),
    1: ("稀有", "Rare", (0.35, 0.6, 1.0, 1)),
    2: ("史诗", "Epic", (0.72, 0.4, 1.0, 1)),
    3: ("传说", "Legendary", (1.0, 0.72, 0.18, 1)),
}


@dataclass(frozen=True)
class Skin:
    key: str
    zh: str
    en: str
    rarity: int = 0
    pattern: str | None = None          # texture key in PATTERNS
    surfaces: tuple = ()                 # surfaces that get the pattern
    tint: dict = field(default_factory=dict)  # surface -> flat colour override
    shine: tuple | None = None           # (specular rgb, shininess) for patterned / tinted surfaces
    scale: float = 6.0                   # pattern repeats per metre


GOLD_SHINE = ((1.0, 0.85, 0.5), 48.0)
_ALL = ("metal", "wood", "poly", "chrome", "paint")


def _gold(scale=8.0):
    return Skin("gold", "黄金", "Gold", 3, "gold", _ALL, {}, GOLD_SHINE, scale)


SKINS: dict[str, list[Skin]] = {
    "ak47": [
        Skin("default", "原厂", "Factory New"),
        Skin("fire_serpent", "火麒麟", "Fire Serpent", 3, "flame", ("wood", "metal"), {},
             ((0.6, 0.4, 0.3), 30.0), 4.0),
        Skin("neon", "霓虹骑士", "Neon Rider", 2, "neon", ("wood", "metal"), {}, ((0.7, 0.7, 0.9), 40.0), 7.0),
        Skin("arctic", "极地迷彩", "Arctic Camo", 1, "arctic", ("wood", "metal"), {}, None, 6.0),
        _gold(),
    ],
    "awm": [
        Skin("default", "原厂", "Factory New"),
        Skin("dragon", "龙狙", "Dragon Lore", 3, "dragon", ("poly",), {}, ((0.8, 0.7, 0.4), 36.0), 7.0),
        Skin("digital", "城市数码", "Urban Digital", 1, "digital", ("poly",), {}, None, 5.0),
        Skin("carbon", "碳纤维", "Carbon Fiber", 2, "carbon", ("poly", "metal"), {}, ((0.6, 0.6, 0.65), 50.0), 22.0),
        _gold(),
    ],
    "glock": [
        Skin("default", "原厂", "Factory New"),
        Skin("fade", "渐变之色", "Fade", 2, "fade", ("poly",), {}, ((0.8, 0.8, 0.85), 40.0), 8.0),
        _gold(12.0),
    ],
    "deagle": [
        Skin("default", "原厂", "Factory New"),
        Skin("blaze", "猛虎", "Tiger Blaze", 2, "blaze", ("chrome",), {}, ((0.9, 0.7, 0.5), 50.0), 7.0),
        Skin("doppler", "多普勒", "Doppler", 3, "doppler", ("chrome",), {}, ((0.9, 0.9, 1.0), 70.0), 7.0),
        _gold(14.0),
    ],
    "knife": [
        Skin("default", "原厂", "Factory New"),
        Skin("fade", "渐变之色", "Fade", 3, "fade", ("chrome",), {}, ((1.0, 1.0, 1.0), 80.0), 2.6),
        Skin("doppler", "多普勒", "Doppler", 3, "doppler", ("chrome",), {}, ((0.9, 0.9, 1.0), 80.0), 6.0),
        Skin("crimson", "深红之网", "Crimson Web", 2, "web", ("chrome",), {}, ((0.7, 0.5, 0.5), 40.0), 8.0),
        _gold(14.0),
    ],
    "frag": [
        Skin("default", "原厂", "Factory New"),
        _gold(20.0),
    ],
    "smoke": [Skin("default", "原厂", "Factory New")],
    "m4a4": [
        Skin("default", "原厂", "Factory New"),
        Skin("howl", "咆哮", "Howl", 3, "flame", ("metal", "poly"), {}, ((0.6, 0.4, 0.3), 30.0), 5.0),
        Skin("digital", "城市数码", "Urban Digital", 1, "digital", ("poly", "metal"), {}, None, 5.0),
        _gold(),
    ],
    "m4a1": [
        Skin("default", "原厂", "Factory New"),
        Skin("hyper", "次时代", "Hyper Beast", 3, "neon", ("metal", "poly"), {}, ((0.7, 0.7, 0.9), 40.0), 6.0),
        Skin("arctic", "极地迷彩", "Arctic Camo", 1, "arctic", ("metal", "poly"), {}, None, 6.0),
        _gold(),
    ],
    "famas": [
        Skin("default", "原厂", "Factory New"),
        Skin("carbon", "碳纤维", "Carbon Fiber", 2, "carbon", ("poly",), {}, ((0.6, 0.6, 0.65), 50.0), 22.0),
        _gold(),
    ],
    "mp5": [
        Skin("default", "原厂", "Factory New"),
        Skin("web", "深红之网", "Crimson Web", 2, "web", ("metal", "poly"), {}, ((0.7, 0.5, 0.5), 40.0), 8.0),
        _gold(),
    ],
    "mp9": [
        Skin("default", "原厂", "Factory New"),
        Skin("doppler", "多普勒", "Doppler", 3, "doppler", ("poly",), {}, ((0.9, 0.9, 1.0), 60.0), 7.0),
        _gold(),
    ],
    "uzi": [
        Skin("default", "原厂", "Factory New"),
        Skin("blaze", "猛虎", "Tiger Blaze", 2, "blaze", ("metal",), {}, ((0.9, 0.7, 0.5), 50.0), 7.0),
        _gold(),
    ],
    "dualies": [
        Skin("default", "原厂", "Factory New"),
        Skin("fade", "渐变之色", "Fade", 3, "fade", ("chrome",), {}, ((1.0, 1.0, 1.0), 80.0), 4.0),
        _gold(14.0),
    ],
    "scar20": [
        Skin("default", "原厂", "Factory New"),
        Skin("dragon", "翡翠龙鳞", "Jade Scales", 3, "dragon", ("poly",), {}, ((0.8, 0.7, 0.4), 36.0), 7.0),
        _gold(),
    ],
    "m249": [
        Skin("default", "原厂", "Factory New"),
        Skin("arctic", "极地迷彩", "Arctic Camo", 1, "arctic", ("metal", "poly"), {}, None, 6.0),
        _gold(),
    ],
    "nova": [
        Skin("default", "原厂", "Factory New"),
        Skin("blaze", "猛虎", "Tiger Blaze", 2, "blaze", ("wood",), {}, ((0.9, 0.7, 0.5), 50.0), 7.0),
        _gold(),
    ],
}
SKIN_WEAPONS = ("ak47", "m4a4", "m4a1", "famas", "mp5", "mp9", "uzi", "awm", "scar20", "m249", "nova", "glock",
                "deagle",
                "dualies", "knife", "frag")


def get_skin(weapon: str, key: str | None) -> Skin:
    for s in SKINS.get(weapon, ()):
        if s.key == key:
            return s
    return SKINS[weapon][0] if weapon in SKINS else Skin("default", "原厂", "Factory New")


# ----------------------------------------------------------------------------- patterns
def _rng(seed):
    return np.random.default_rng(seed)


def _lerp(a, b, t):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a + (b - a) * t[..., None]


def _ramp(t, stops):
    """Piecewise-linear colour ramp; stops = [(pos, rgb), ...]."""
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros(t.shape + (3,))
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        m = (t >= p0) & (t <= p1)
        k = (t[m] - p0) / max(p1 - p0, 1e-9)
        out[m] = _lerp(c0, c1, k)
    return out


def pat_flame(size=256):
    r = _rng(31)
    yy = np.linspace(0, 1, size)[:, None] * np.ones((1, size))
    n = noise(size, size, 3, 5, 0.55, rng=r)
    warp = noise(size, size, 6, 3, 0.5, rng=r)
    tongue = np.sin((np.linspace(0, 1, size)[None, :] * 6 + warp * 1.4) * 2 * np.pi) * 0.5 + 0.5
    heat = np.clip((1 - yy) * 1.25 - 0.25 + (n - 0.5) * 0.9 + tongue * 0.35, 0, 1)
    col = _ramp(heat, [(0, (0.05, 0.02, 0.02)), (0.35, (0.45, 0.04, 0.02)), (0.6, (0.95, 0.3, 0.03)),
                       (0.8, (1.0, 0.7, 0.12)), (1.0, (1.0, 0.95, 0.6))])
    scale = noise(size, size, 16, 2, 0.5, rng=r)
    return col * (0.85 + 0.15 * scale[..., None])


def pat_blaze(size=256):
    """Orange / black tiger stripes."""
    r = _rng(37)
    yy, xx = np.mgrid[0:size, 0:size] / size
    n = noise(size, size, 4, 4, 0.55, rng=r)
    s = np.sin((xx * 3 + yy * 1 + n * 0.9) * 2 * np.pi)
    stripe = np.clip((s - 0.35) * 5, 0, 1)
    warm = _ramp(noise(size, size, 3, 3, 0.5, rng=r), [(0, (0.95, 0.35, 0.03)), (1, (1.0, 0.7, 0.15))])
    return warm * (1 - stripe[..., None]) + np.array((0.05, 0.04, 0.04)) * stripe[..., None]


def pat_gold(size=256):
    r = _rng(71)
    brushed = noise(size, size, 2, 6, 0.6, aspect=(8, 1), rng=r)
    yy, xx = np.mgrid[0:size, 0:size] / size
    n = noise(size, size, 4, 4, 0.5, rng=r)
    engr = np.abs(np.sin((xx * 4 + n * 1.6) * 2 * np.pi) * np.sin((yy * 4 + n) * 2 * np.pi)) < 0.06
    col = _ramp(brushed, [(0, (0.6, 0.4, 0.12)), (0.5, (0.85, 0.62, 0.24)), (1, (1.0, 0.84, 0.45))])
    col[engr] *= 0.55
    return col


def pat_neon(size=256):
    r = _rng(41)
    base = np.zeros((size, size, 3)) + (0.05, 0.05, 0.08)
    yy, xx = np.mgrid[0:size, 0:size] / size
    # hex grid
    s = 8.0
    qx = xx * s * 2 / np.sqrt(3)
    qy = yy * s - (np.floor(qx) % 2) * 0.5
    fx = np.abs((qx % 1.0) - 0.5)
    fy = np.abs((qy % 1.0) - 0.5)
    edge = np.clip(1 - np.minimum(np.abs(fx - 0.5) * 18, np.abs(fy - 0.5) * 18), 0, 1)
    glow = np.exp(-np.minimum(np.abs(fx - 0.5), np.abs(fy - 0.5)) * 14)
    band = noise(size, size, 3, 3, 0.5, rng=r)
    c1 = np.array((0.1, 0.95, 1.0))
    c2 = np.array((1.0, 0.15, 0.75))
    col = c1 * (1 - band[..., None]) + c2 * band[..., None]
    out = base + col * (edge[..., None] * 0.9 + glow[..., None] * 0.35)
    stripe = (np.abs(((xx + yy) * 3) % 1.0 - 0.5) < 0.04)
    out[stripe] = out[stripe] * 0.4 + col[stripe] * 0.8
    return out


def pat_arctic(size=256):
    r = _rng(43)
    out = np.zeros((size, size, 3)) + (0.86, 0.88, 0.9)
    for color, thr, base in (((0.62, 0.66, 0.7), 0.55, 4), ((0.36, 0.4, 0.45), 0.62, 5), ((0.18, 0.2, 0.24), 0.7, 6)):
        m = noise(size, size, base, 4, 0.5, rng=r) > thr
        out[m] = color
    return out * (0.92 + 0.08 * noise(size, size, 32, 2, 0.5, rng=r)[..., None])


def pat_digital(size=256):
    r = _rng(47)
    cells = 32
    out = np.zeros((cells, cells, 3)) + (0.5, 0.56, 0.64)
    for color, thr, base in (((0.28, 0.34, 0.45), 0.5, 3), ((0.75, 0.78, 0.82), 0.62, 4), ((0.1, 0.12, 0.16), 0.68, 5)):
        m = noise(cells, cells, base, 3, 0.5, rng=r) > thr
        out[m] = color
    return np.repeat(np.repeat(out, size // cells, 0), size // cells, 1)


def pat_carbon(size=256):
    yy, xx = np.mgrid[0:size, 0:size] / size
    n = 8
    u, v = (xx * n) % 1.0, (yy * n) % 1.0
    checker = ((np.floor(xx * n) + np.floor(yy * n)) % 2).astype(bool)
    weave = np.where(checker, np.sin(u * np.pi), np.sin(v * np.pi)) ** 0.6
    g = 0.05 + 0.16 * weave
    return np.stack([g, g, g * 1.08], -1)


def pat_dragon(size=256):
    r = _rng(53)
    yy, xx = np.mgrid[0:size, 0:size] / size
    s = 10.0
    row = np.floor(yy * s)
    u = (xx * s + row * 0.5) % 1.0
    v = (yy * s) % 1.0
    d = np.sqrt((u - 0.5) ** 2 + (v - 0.15) ** 2)  # overlapping scales
    rim = np.clip(1 - np.abs(d - 0.5) * 14, 0, 1)
    inner = np.clip(1 - d * 1.4, 0, 1)
    n = noise(size, size, 3, 4, 0.5, rng=r)
    jade = _ramp(n, [(0, (0.05, 0.25, 0.15)), (0.5, (0.12, 0.5, 0.3)), (1, (0.5, 0.8, 0.55))])
    gold = np.array((0.95, 0.75, 0.3))
    out = jade * (0.6 + 0.4 * inner[..., None]) * (1 - rim[..., None]) + gold * rim[..., None]
    swirl = np.sin((xx * 3 + n * 2.5) * 2 * np.pi) > 0.94
    out[swirl] = gold
    return out


def pat_doppler(size=256):
    r = _rng(59)
    n = noise(size, size, 3, 6, 0.6, rng=r)
    m = noise(size, size, 5, 4, 0.5, rng=r)
    col = _ramp(n, [(0, (0.03, 0.02, 0.12)), (0.35, (0.15, 0.05, 0.4)), (0.55, (0.55, 0.1, 0.5)),
                    (0.75, (0.1, 0.35, 0.75)), (1.0, (0.85, 0.9, 1.0))])
    col *= (0.7 + 0.5 * m[..., None])
    stars = r.random((size, size)) > 0.996
    col[stars] = 1.0
    return col


def pat_fade(size=256):
    xx = np.linspace(0, 1, size)[None, :] * np.ones((size, 1))
    n = noise(size, size, 3, 3, 0.5, rng=_rng(61))
    t = np.clip(xx + (n - 0.5) * 0.12, 0, 1)
    return _ramp(t, [(0, (0.95, 0.85, 0.2)), (0.35, (0.95, 0.35, 0.55)), (0.65, (0.55, 0.2, 0.85)),
                     (1.0, (0.2, 0.25, 0.9))])


def pat_web(size=256):
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    ang = np.arctan2(yy, xx)
    rad = np.sqrt(xx ** 2 + yy ** 2)
    spokes = np.abs(np.sin(ang * 6)) < 0.04
    rings = np.abs(((rad * 9 + 0.04 * np.sin(ang * 12)) % 1.0) - 0.5) > 0.47
    out = np.zeros((size, size, 3)) + (0.5, 0.04, 0.04)
    out *= (0.75 + 0.25 * noise(size, size, 8, 3, 0.5, rng=_rng(67))[..., None])
    out[spokes | rings] = (0.06, 0.03, 0.03)
    return out


PATTERNS = {"flame": pat_flame, "blaze": pat_blaze, "neon": pat_neon, "arctic": pat_arctic,
            "digital": pat_digital, "carbon": pat_carbon, "dragon": pat_dragon, "doppler": pat_doppler,
            "fade": pat_fade, "web": pat_web, "gold": pat_gold}
_TEX_CACHE = {}


def pattern_texture(key: str):
    if key not in _TEX_CACHE:
        _TEX_CACHE[key] = make_texture(PATTERNS[key](), "skin_" + key, clamp=key == "fade")
    return _TEX_CACHE[key]


def _project_uvs(np_, s: float):
    """Model-space planar UVs: u along the barrel (+Y), v up the side (+Z, plus some X so top faces vary)."""
    for gn_np in np_.findAllMatches("**/+GeomNode"):
        gn = gn_np.node()
        for i in range(gn.getNumGeoms()):
            vdata = gn.modifyGeom(i).modifyVertexData()
            pos = GeomVertexReader(vdata, "vertex")
            uv = GeomVertexRewriter(vdata, "texcoord")
            while not pos.isAtEnd():
                p = pos.getData3()
                uv.setData2(p.y * s + 0.5, (p.z + 0.45 * p.x) * s + 0.5)


def apply_skin(root, skin: Skin):
    """Apply `skin` to a weapon model whose children are per-surface mesh nodes."""
    if skin.key == "default":
        return
    stage = TextureStage("skin")
    for child in root.getChildren():
        surface = child.getName().rsplit("_", 1)[-1]
        if surface == "glass":
            continue
        styled = False
        if skin.pattern and surface in skin.surfaces:
            _project_uvs(child, skin.scale)
            child.setColor(1, 1, 1, 1, 1)
            child.setTexture(stage, pattern_texture(skin.pattern), 1)
            styled = True
        elif surface in skin.tint:
            child.setColor(*skin.tint[surface], 1)
            styled = True
        if styled and skin.shine is not None:
            spec, shin = skin.shine
            m = Material(f"skin_{skin.key}_{surface}")
            m.setSpecular((*spec, 1.0))
            m.setShininess(shin)
            child.setMaterial(m, 2)

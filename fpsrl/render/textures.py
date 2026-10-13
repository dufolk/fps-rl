"""Procedural textures generated with numpy (no external assets needed)."""
from __future__ import annotations

import numpy as np
from panda3d.core import SamplerState, Texture

_RNG = np.random.default_rng(1234)


def _upsample_periodic(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    gh, gw = grid.shape
    ys = np.arange(h) * gh / h
    xs = np.arange(w) * gw / w
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    y1 = (y0 + 1) % gh
    x1 = (x0 + 1) % gw
    a = grid[y0][:, x0]
    b = grid[y0][:, x1]
    c = grid[y1][:, x0]
    d = grid[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def noise(h: int, w: int, base: int = 4, octaves: int = 5, persistence: float = 0.5,
          aspect: tuple = (1, 1), rng=None) -> np.ndarray:
    """Tileable fractal value noise in [0, 1]."""
    rng = rng or _RNG
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    f = base
    for _ in range(octaves):
        gh = max(1, int(f * aspect[0]))
        gw = max(1, int(f * aspect[1]))
        out += _upsample_periodic(rng.random((gh, gw)), h, w) * amp
        total += amp
        amp *= persistence
        f *= 2
    out /= total
    out -= out.min()
    out /= max(out.max(), 1e-9)
    return out


def make_texture(arr: np.ndarray, name: str = "tex", mipmap: bool = True, clamp: bool = False) -> Texture:
    """arr: (h, w, 3|4) float [0,1] or uint8, row 0 = top."""
    a = arr
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    a = np.ascontiguousarray(a[::-1])
    h, w, c = a.shape
    tex = Texture(name)
    if c == 4:
        tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgba8)
        tex.setRamImageAs(a.tobytes(), "RGBA")
    else:
        tex.setup2dTexture(w, h, Texture.T_unsigned_byte, Texture.F_rgb8)
        tex.setRamImageAs(a.tobytes(), "RGB")
    if mipmap:
        tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
        tex.setAnisotropicDegree(8)
    else:
        tex.setMinfilter(SamplerState.FT_linear)
    tex.setMagfilter(SamplerState.FT_linear)
    wrap = SamplerState.WM_clamp if clamp else SamplerState.WM_repeat
    tex.setWrapU(wrap)
    tex.setWrapV(wrap)
    return tex


def normal_from_height(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, 1, axis=0) - np.roll(height, -1, axis=0)) * 0.5
    n = np.stack([-dx * strength * height.shape[1] / 64.0, -dy * strength * height.shape[0] / 64.0,
                  np.ones_like(height)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


# ----------------------------------------------------------------------------- materials
def container(color, size=256):
    """Corrugated steel. Texture spans 2 m (uv scale 0.5)."""
    h = w = size
    u = np.arange(w) / w
    wave = np.clip(np.sin(u * 2 * np.pi * 7) * 1.8, -1, 1)
    height = np.tile(wave[None, :], (h, 1)) * 0.5 + 0.5
    n1 = noise(h, w, 4, 5)
    streak = noise(h, w, 3, 4, aspect=(4, 0.5))
    rust = np.clip((noise(h, w, 6, 5) - 0.62) * 4.0, 0, 1) * (0.5 + 0.5 * streak)
    col = np.asarray(color)[None, None, :] * (0.82 + 0.18 * height[..., None]) * (0.85 + 0.25 * n1[..., None])
    col = col * (1 - 0.25 * streak[..., None] * 0.6)
    rust_col = np.array([0.36, 0.2, 0.1])[None, None, :] * (0.7 + 0.5 * n1[..., None])
    col = col * (1 - rust[..., None] * 0.75) + rust_col * rust[..., None] * 0.75
    grime = np.linspace(0.75, 1.0, h)[:, None, None]  # darker towards the bottom of each tile
    col = col * (0.9 + 0.1 * grime[::-1])
    return col, normal_from_height(height * 0.6 + n1 * 0.05, 3.0)


def deck(size=512):
    """Painted steel deck plates, 4 m per tile (uv scale 0.25)."""
    h = w = size
    n1 = noise(h, w, 8, 6)
    n2 = noise(h, w, 3, 4)
    base = np.array([0.33, 0.38, 0.34])
    col = base[None, None, :] * (0.8 + 0.35 * n1[..., None]) * (0.85 + 0.25 * n2[..., None])
    height = np.ones((h, w)) * 0.5
    yy, xx = np.mgrid[0:h, 0:w]
    seam = ((xx % (w // 2)) < 3) | ((yy % (h // 2)) < 3)
    height[seam] = 0.0
    col[seam] *= 0.45
    wear = np.clip((noise(h, w, 16, 4) - 0.6) * 3.0, 0, 1)
    col = col * (1 - wear[..., None] * 0.4) + np.array([0.45, 0.43, 0.38]) * wear[..., None] * 0.4
    rust = np.clip((noise(h, w, 10, 5) - 0.7) * 4.0, 0, 1)
    near_seam = np.zeros((h, w))
    near_seam[seam] = 1.0
    for _ in range(3):
        near_seam = np.maximum(near_seam, (np.roll(near_seam, 2, 0) + np.roll(near_seam, -2, 0)
                                           + np.roll(near_seam, 2, 1) + np.roll(near_seam, -2, 1)) * 0.25)
    rust = np.clip(rust + near_seam * 0.35 * n1, 0, 1)
    col = col * (1 - rust[..., None] * 0.6) + np.array([0.35, 0.2, 0.11]) * rust[..., None] * 0.6
    # weld beads / rivets
    rivet = (((xx % 32) - 16) ** 2 + ((yy % (h // 2)) - 8) ** 2) < 6
    height[rivet] = 1.0
    return col, normal_from_height(height + n1 * 0.08, 2.0)


def steel_plate(size=256):
    """Diamond tread plate, 1 m per tile."""
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w] / size
    k = 12
    a = ((xx + yy) * k) % 1.0
    b = ((xx - yy) * k) % 1.0
    bump_a = np.exp(-((a - 0.5) ** 2) / 0.004) * (((b * 2) % 1.0) < 0.6)
    bump_b = np.exp(-((b - 0.5) ** 2) / 0.004) * ((((a + 0.5) * 2) % 1.0) < 0.6)
    height = np.clip(bump_a + bump_b, 0, 1)
    n1 = noise(h, w, 6, 5)
    col = np.array([0.5, 0.52, 0.54])[None, None, :] * (0.75 + 0.35 * n1[..., None]) + height[..., None] * 0.06
    return col, normal_from_height(height, 1.2)


def wood(size=256, base=(0.62, 0.45, 0.26)):
    h = w = size
    n1 = noise(h, w, 2, 5, aspect=(1, 10))
    grain = (np.sin((np.arange(h)[:, None] / h * 60 + n1 * 18)) * 0.5 + 0.5)
    col = np.asarray(base)[None, None, :] * (0.72 + 0.28 * grain[..., None]) * (0.85 + 0.2 * noise(h, w, 8, 4)[..., None])
    planks = 4
    yy = np.arange(h)[:, None] % (h // planks)
    gap = (yy < 3).repeat(w, axis=1)
    col[gap] *= 0.35
    border = np.zeros((h, w), bool)
    bw = int(size * 0.09)
    border[:, :bw] = border[:, -bw:] = True
    border[:bw, :] = border[-bw:, :] = True
    col[border] = col[border] * 0.78
    height = 0.5 + 0.1 * grain
    height[gap] = 0.0
    height[border] += 0.25
    return col, normal_from_height(height, 1.5)


def painted(base, size=256, rust_amount=0.35, streaks=True):
    h = w = size
    n1 = noise(h, w, 6, 5)
    col = np.asarray(base)[None, None, :] * (0.88 + 0.18 * n1[..., None])
    if streaks:
        st = noise(h, w, 2, 4, aspect=(6, 0.4))
        rust = np.clip((st - 0.55) * 3.0, 0, 1) * rust_amount
        col = col * (1 - rust[..., None]) + np.array([0.38, 0.22, 0.12]) * rust[..., None]
    height = n1 * 0.2
    return col, normal_from_height(height, 1.0)


def superstructure(size=512):
    """White painted wall with portholes; 3 m tiles (uv scale 1/3)."""
    h = w = size
    col, _ = painted((0.86, 0.86, 0.83), size, 0.25)
    yy, xx = np.mgrid[0:h, 0:w] / size
    win = (np.abs(xx - 0.5) < 0.22) & (np.abs(yy - 0.38) < 0.13)
    frame = (np.abs(xx - 0.5) < 0.24) & (np.abs(yy - 0.38) < 0.15) & ~win
    glass = np.array([0.12, 0.18, 0.26])[None, None, :] + (1 - yy[..., None]) * 0.25 * np.array([0.6, 0.7, 0.9])
    col = np.where(win[..., None], glass, col)
    col = np.where(frame[..., None], col * 0.55, col)
    height = np.zeros((h, w))
    height[frame] = 1.0
    height[win] = -0.3
    stripe = (yy > 0.97)
    col[stripe] *= 0.7
    return col, normal_from_height(height, 1.5)


def hull(size=256):
    col, nrm = painted((0.12, 0.13, 0.17), size, 0.5)
    return col, nrm


def hatch(size=256):
    h = w = size
    col, _ = painted((0.2, 0.33, 0.3), size, 0.3)
    yy, xx = np.mgrid[0:h, 0:w]
    lines = ((xx % 64) < 4)
    col[lines] *= 0.6
    height = np.zeros((h, w))
    height[lines] = -1.0
    rivets = (((xx % 64) - 32) ** 2 + ((yy % 32) - 16) ** 2) < 7
    height[rivets] = 1.0
    stripe = (((xx + yy) // 24) % 2 == 0)
    edge = (yy < 20) | (yy > h - 20)
    col[edge & stripe] = np.array([0.85, 0.7, 0.1])
    col[edge & ~stripe] = np.array([0.08, 0.08, 0.08])
    return col, normal_from_height(height, 1.5)


def water(size=512):
    h = w = size
    n1 = noise(h, w, 8, 6, persistence=0.55)
    n2 = noise(h, w, 24, 3)
    col = (np.array([0.05, 0.16, 0.24])[None, None, :] * (0.7 + 0.5 * n1[..., None])
           + np.array([0.04, 0.08, 0.08])[None, None, :] * n2[..., None])
    return col, normal_from_height(n1 * 0.7 + n2 * 0.3, 6.0)


def sand(size=512, base=(0.78, 0.66, 0.46)):
    h = w = size
    n1 = noise(h, w, 6, 6, 0.55)
    n2 = noise(h, w, 32, 3)
    ripples = np.sin((np.mgrid[0:h, 0:w][0] / h * 40 + noise(h, w, 3, 3) * 12)) * 0.5 + 0.5
    col = np.asarray(base)[None, None, :] * (0.82 + 0.22 * n1[..., None]) * (0.92 + 0.1 * n2[..., None])
    col *= (0.96 + 0.06 * ripples[..., None])
    return col, normal_from_height(n1 * 0.4 + ripples * 0.15 + n2 * 0.2, 1.5)


def blocks(base, size=512, rows=8, cols=4, mortar=(0.6, 0.55, 0.45), jitter=0.12, stagger=True, mortar_px=3):
    """Masonry: stone blocks / bricks with mortar lines."""
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    rh = h // rows
    cw = w // cols
    row = yy // rh
    off = (row % 2) * (cw // 2) if stagger else 0
    col_idx = (xx + off) // cw
    rng = np.random.default_rng(int(sum(base) * 1000) % 9973)
    tint = rng.uniform(1 - jitter, 1 + jitter, (rows + 1, cols + 2))
    t = tint[row % (rows + 1), col_idx % (cols + 2)]
    n1 = noise(h, w, 8, 5)
    col = np.asarray(base)[None, None, :] * t[..., None] * (0.85 + 0.25 * n1[..., None])
    gap = ((yy % rh) < mortar_px) | (((xx + off) % cw) < mortar_px)
    col[gap] = np.asarray(mortar) * (0.8 + 0.2 * n1[gap][..., None])
    height = 0.6 + 0.2 * n1
    height[gap] = 0.0
    return col, normal_from_height(height, 2.0)


def plaster(base, size=256, grime=0.25):
    h = w = size
    n1 = noise(h, w, 6, 6, 0.55)
    st = noise(h, w, 2, 4, aspect=(6, 0.4))
    col = np.asarray(base)[None, None, :] * (0.88 + 0.16 * n1[..., None])
    dirt = np.clip((st - 0.5) * 2.5, 0, 1) * grime
    col = col * (1 - dirt[..., None] * 0.5)
    yy = np.arange(h)[:, None] / h
    col = col * (0.9 + 0.1 * (1 - yy[..., None]) ** 0.2)
    chips = noise(h, w, 16, 3) > 0.78
    col[chips] *= 0.82
    return col, normal_from_height(n1 * 0.3 - chips * 0.3, 1.2)


def concrete(size=256, base=(0.55, 0.55, 0.53), seams=True):
    h = w = size
    n1 = noise(h, w, 8, 6, 0.6)
    n2 = noise(h, w, 32, 3)
    col = np.asarray(base)[None, None, :] * (0.82 + 0.25 * n1[..., None]) * (0.92 + 0.1 * n2[..., None])
    height = n1 * 0.3 + n2 * 0.1
    if seams:
        yy, xx = np.mgrid[0:h, 0:w]
        s = (xx < 2) | (yy < 2)
        col[s] *= 0.6
        height[s] = 0
    return col, normal_from_height(height, 1.5)


def asphalt(size=512):
    h = w = size
    n2 = noise(h, w, 64, 2)
    grit = np.random.default_rng(3).random((h, w))
    col = np.array([0.2, 0.2, 0.21])[None, None, :] * (0.8 + 0.3 * n2[..., None]) + (grit > 0.97)[..., None] * 0.12
    patch = noise(h, w, 3, 4) > 0.62
    col[patch] *= 0.8
    return col, normal_from_height(n2 * 0.4 + grit * 0.1, 1.0)


def grass(size=512):
    h = w = size
    n1 = noise(h, w, 6, 6, 0.55)
    n2 = noise(h, w, 48, 3)
    col = np.array([0.24, 0.36, 0.14])[None, None, :] * (0.75 + 0.4 * n1[..., None]) * (0.85 + 0.3 * n2[..., None])
    dry = np.clip((noise(h, w, 4, 4) - 0.55) * 3, 0, 1)
    col = col * (1 - dry[..., None] * 0.5) + np.array([0.45, 0.4, 0.22]) * dry[..., None] * 0.5
    return col, normal_from_height(n2 * 0.5 + n1 * 0.2, 2.0)


def tiles(base, size=256, n=4, grout=(0.3, 0.3, 0.3), checker=None):
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    c = size // n
    n1 = noise(h, w, 8, 5)
    col = np.asarray(base)[None, None, :] * (0.9 + 0.12 * n1[..., None])
    if checker is not None:
        chk = (((xx // c) + (yy // c)) % 2) == 1
        col[chk] = np.asarray(checker) * (0.9 + 0.12 * n1[chk][..., None])
    g = ((xx % c) < 2) | ((yy % c) < 2)
    col[g] = grout
    height = np.ones((h, w)) * 0.5
    height[g] = 0
    return col, normal_from_height(height, 1.0)


def marble(size=512, base=(0.85, 0.84, 0.8)):
    h = w = size
    n = noise(h, w, 3, 6, 0.6)
    veins = np.abs(np.sin((np.mgrid[0:h, 0:w][1] / w * 6 + n * 9) * np.pi)) ** 0.15
    col = np.asarray(base)[None, None, :] * (0.75 + 0.25 * veins[..., None])
    yy, xx = np.mgrid[0:h, 0:w]
    g = ((xx % (w // 2)) < 2) | ((yy % (h // 2)) < 2)
    col[g] *= 0.7
    return col, normal_from_height(np.where(g, 0.0, 0.5), 0.8)


def carpet(size=256, base=(0.42, 0.12, 0.1)):
    h = w = size
    n2 = noise(h, w, 64, 2)
    yy, xx = np.mgrid[0:h, 0:w] / size
    pattern = (np.abs(((xx * 4) % 1) - 0.5) < 0.04) | (np.abs(((yy * 4) % 1) - 0.5) < 0.04)
    col = np.asarray(base)[None, None, :] * (0.8 + 0.3 * n2[..., None])
    col[pattern] = col[pattern] * 0.6 + np.array([0.6, 0.5, 0.2]) * 0.4
    return col, normal_from_height(n2 * 0.6, 1.0)


def roof(size=256, base=(0.48, 0.2, 0.14)):
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w]
    rows = 8
    rh = h // rows
    off = ((yy // rh) % 2) * (w // 8)
    wave = np.sin(((xx + off) % (w // 4)) / (w // 4) * np.pi)
    shade = (yy % rh) / rh
    n1 = noise(h, w, 8, 4)
    col = np.asarray(base)[None, None, :] * (0.7 + 0.25 * wave[..., None] + 0.15 * shade[..., None]) \
        * (0.85 + 0.25 * n1[..., None])
    return col, normal_from_height(wave * 0.6 + shade * 0.4, 2.5)


def reinforce(size=256):
    h = w = size
    col, _ = painted((0.32, 0.34, 0.36), size, 0.25)
    yy, xx = np.mgrid[0:h, 0:w]
    bars = (np.abs((xx % (w // 2)) - w // 4) < 10)
    col[bars] *= 0.65
    rivets = (((xx % 32) - 16) ** 2 + ((yy % 32) - 16) ** 2) < 10
    col[rivets] = (0.55, 0.55, 0.58)
    height = np.zeros((h, w))
    height[bars] = 0.8
    height[rivets] = 1.0
    stripe = (((xx + yy) // 20) % 2 == 0) & (yy > h - 24)
    col[stripe] = (0.85, 0.65, 0.1)
    return col, normal_from_height(height, 2.0)


THEME_TEXTURES = {
    "sand": lambda: sand(),
    "sand_dark": lambda: sand(base=(0.66, 0.54, 0.37)),
    "sandstone": lambda: blocks((0.76, 0.62, 0.42), rows=6, cols=3, mortar=(0.62, 0.52, 0.38)),
    "sandstone_dark": lambda: blocks((0.6, 0.48, 0.32), rows=6, cols=3, mortar=(0.5, 0.42, 0.3)),
    "plaster_tan": lambda: plaster((0.8, 0.7, 0.52)),
    "plaster_white": lambda: plaster((0.86, 0.85, 0.8)),
    "plaster_yellow": lambda: plaster((0.85, 0.72, 0.42)),
    "plaster_blue": lambda: plaster((0.45, 0.58, 0.7)),
    "brick": lambda: blocks((0.55, 0.26, 0.18), rows=16, cols=6, mortar=(0.62, 0.6, 0.55), mortar_px=2),
    "brick_dark": lambda: blocks((0.32, 0.22, 0.18), rows=16, cols=6, mortar=(0.45, 0.43, 0.4), mortar_px=2),
    "concrete": lambda: concrete(),
    "concrete_dark": lambda: concrete(base=(0.38, 0.38, 0.37)),
    "asphalt": lambda: asphalt(),
    "grass": lambda: grass(),
    "tile_floor": lambda: tiles((0.78, 0.76, 0.7), checker=(0.3, 0.32, 0.34)),
    "tile_white": lambda: tiles((0.85, 0.86, 0.86), n=8, grout=(0.6, 0.6, 0.6)),
    "marble": lambda: marble(),
    "carpet": lambda: carpet(),
    "carpet_blue": lambda: carpet(base=(0.12, 0.16, 0.32)),
    "wood_floor": lambda: wood(base=(0.5, 0.33, 0.18)),
    "wood_door": lambda: wood(base=(0.45, 0.28, 0.14)),
    "wood_planks": lambda: wood(base=(0.58, 0.42, 0.24)),
    "roof": lambda: roof(),
    "roof_grey": lambda: roof(base=(0.3, 0.32, 0.34)),
    "drywall": lambda: plaster((0.82, 0.8, 0.74), grime=0.1),
    "drywall_green": lambda: plaster((0.55, 0.62, 0.52), grime=0.1),
    "reinforce": lambda: reinforce(),
    "metal_dark": lambda: painted((0.18, 0.19, 0.2), rust_amount=0.1),
    "metal_door": lambda: painted((0.35, 0.4, 0.38), rust_amount=0.4),
    "metal": lambda: painted((0.45, 0.46, 0.47), rust_amount=0.15),
    "vault": lambda: steel_plate(),
    "stone_dark": lambda: blocks((0.4, 0.38, 0.35), rows=4, cols=2, mortar=(0.3, 0.3, 0.28)),
}


def flat_normal(size=4):
    n = np.zeros((size, size, 3))
    n[..., 2] = 1.0
    return n * 0.5 + 0.5


# ----------------------------------------------------------------------------- sprites
def soft_circle(size=64, power=2.0):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    a = np.clip(1 - r, 0, 1) ** power
    rgba = np.ones((size, size, 4))
    rgba[..., 3] = a
    return rgba


def smoke_puff(size=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    n = noise(size, size, 4, 5, rng=rng)
    a = np.clip(1 - r * (0.9 + 0.5 * n), 0, 1) ** 1.6
    shade = 0.75 + 0.25 * noise(size, size, 6, 3, rng=rng)
    rgba = np.ones((size, size, 4))
    rgba[..., :3] = shade[..., None]
    rgba[..., 3] = a * (0.6 + 0.4 * n)
    return rgba


def muzzle_flash(size=128):
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx - size / 2 + 0.5) / (size / 2)
    dy = (yy - size / 2 + 0.5) / (size / 2)
    r = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)
    star = np.clip(1 - r / (0.35 + 0.65 * np.abs(np.cos(ang * 2.5)) ** 6), 0, 1)
    core = np.clip(1 - r * 2.2, 0, 1)
    a = np.clip(star ** 1.5 + core, 0, 1)
    rgba = np.zeros((size, size, 4))
    rgba[..., 0] = 1.0
    rgba[..., 1] = 0.65 + 0.35 * core
    rgba[..., 2] = 0.25 + 0.75 * core ** 2
    rgba[..., 3] = a
    return rgba


def flash_side(size=128):
    """Side-on flame cone pointing to +x (texture u)."""
    yy, xx = np.mgrid[0:size, 0:size]
    u = xx / size
    v = (yy - size / 2 + 0.5) / (size / 2)
    width = 0.15 + 0.85 * np.sin(np.clip(u, 0, 1) * np.pi) ** 0.7 * (1 - u) ** 0.5
    a = np.clip(1 - np.abs(v) / np.maximum(width, 1e-3), 0, 1) ** 1.5 * (1 - u) ** 0.8
    rgba = np.zeros((size, size, 4))
    rgba[..., 0] = 1.0
    rgba[..., 1] = 0.7
    rgba[..., 2] = 0.3
    rgba[..., 3] = a
    return rgba


def spark(size=32):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    a = np.clip(1 - r, 0, 1) ** 3
    rgba = np.ones((size, size, 4))
    rgba[..., 2] = 0.6
    rgba[..., 3] = a
    return rgba


def tracer(size=64):
    yy, xx = np.mgrid[0:size, 0:size]
    v = np.abs((xx - size / 2 + 0.5) / (size / 2))
    u = yy / size
    a = np.clip(1 - v, 0, 1) ** 2.5 * np.clip(u * 1.5, 0, 1)
    rgba = np.ones((size, size, 4))
    rgba[..., 2] = 0.55
    rgba[..., 1] = 0.9
    rgba[..., 3] = a
    return rgba


def bullet_hole(size=64):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    n = noise(size, size, 6, 3)
    rgba = np.zeros((size, size, 4))
    core = r < 0.18
    ring = np.clip(1 - (r - 0.18) / (0.5 + 0.3 * n), 0, 1)
    rgba[..., :3] = 0.06
    rgba[..., 3] = np.where(core, 1.0, ring * 0.75)
    return rgba


def scorch(size=128):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    n = noise(size, size, 5, 4)
    a = np.clip(1 - r * (0.8 + 0.6 * n), 0, 1) ** 0.8
    rgba = np.zeros((size, size, 4))
    rgba[..., :3] = 0.04
    rgba[..., 3] = a * 0.9
    return rgba


def blood(size=64):
    rgba = smoke_puff(size, seed=7)
    rgba[..., 0] = 0.55 * rgba[..., 0]
    rgba[..., 1] = 0.03
    rgba[..., 2] = 0.03
    return rgba


def scope_overlay(size=1024):
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx - size / 2 + 0.5) / (size / 2)
    dy = (yy - size / 2 + 0.5) / (size / 2)
    r = np.hypot(dx, dy)
    rgba = np.zeros((size, size, 4))
    edge = np.clip((r - 0.93) / 0.03, 0, 1)
    rgba[..., 3] = edge
    vign = np.clip((r - 0.6) / 0.35, 0, 1) ** 2 * 0.55
    rgba[..., 3] = np.maximum(rgba[..., 3], vign)
    line_w = 1.6 / size * 2
    thick_w = 5.0 / size * 2
    cross = ((np.abs(dx) < line_w) | (np.abs(dy) < line_w)) & (r < 0.93)
    thick = (((np.abs(dx) < thick_w) & (np.abs(dy) > 0.45)) | ((np.abs(dy) < thick_w) & (np.abs(dx) > 0.45))) & (r < 0.93)
    rgba[cross | thick, 3] = 1.0
    return rgba


def vignette(size=256, strength=0.55):
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx - size / 2 + 0.5) / (size / 2)
    dy = (yy - size / 2 + 0.5) / (size / 2)
    r = np.hypot(dx, dy) / 1.41
    rgba = np.zeros((size, size, 4))
    rgba[..., 3] = np.clip((r - 0.45) / 0.55, 0, 1) ** 1.8 * strength
    return rgba


def damage_arc(size=256):
    yy, xx = np.mgrid[0:size, 0:size]
    dx = (xx - size / 2 + 0.5) / (size / 2)
    dy = (size / 2 - yy - 0.5) / (size / 2)
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dx, dy))
    band = np.clip(1 - np.abs(r - 0.8) / 0.08, 0, 1)
    arc = np.clip(1 - np.abs(ang) / 28.0, 0, 1)
    rgba = np.zeros((size, size, 4))
    rgba[..., 0] = 0.9
    rgba[..., 1] = 0.08
    rgba[..., 2] = 0.05
    rgba[..., 3] = band * arc
    return rgba


def sun_glow(size=256):
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    core = np.clip(1 - r / 0.12, 0, 1) ** 0.5
    glow = np.clip(1 - r, 0, 1) ** 3 * 0.6
    rgba = np.zeros((size, size, 4))
    rgba[..., 0] = 1.0
    rgba[..., 1] = 0.85 + 0.15 * core
    rgba[..., 2] = 0.6 + 0.4 * core
    rgba[..., 3] = np.clip(core + glow, 0, 1)
    return rgba


def cloud_layer(size=512):
    n = noise(size, size, 3, 6, persistence=0.55)
    a = np.clip((n - 0.45) * 2.4, 0, 1)
    rgba = np.ones((size, size, 4))
    rgba[..., :3] = (0.9 + 0.1 * n)[..., None]
    rgba[..., 3] = a * 0.8
    return rgba


class TextureLibrary:
    """Lazily generated & cached textures."""

    CONTAINER_COLORS = {
        "red": (0.6, 0.14, 0.1), "blue": (0.11, 0.27, 0.52), "orange": (0.82, 0.38, 0.08),
        "green": (0.16, 0.38, 0.22), "yellow": (0.82, 0.62, 0.1), "white": (0.78, 0.78, 0.74),
        "teal": (0.08, 0.42, 0.45), "gray": (0.42, 0.44, 0.45), "maroon": (0.38, 0.1, 0.12),
    }

    def __init__(self):
        self._cache = {}

    def get(self, key: str):
        if key in self._cache:
            return self._cache[key]
        if key.startswith("container_"):
            color = self.CONTAINER_COLORS.get(key.split("_", 1)[1], (0.5, 0.5, 0.5))
            col, nrm = container(color)
        elif key == "deck":
            col, nrm = deck()
        elif key == "steel_plate":
            col, nrm = steel_plate()
        elif key in ("crate", "crate_big"):
            col, nrm = wood(base=(0.62, 0.45, 0.26) if key == "crate" else (0.5, 0.38, 0.22))
        elif key == "superstructure":
            col, nrm = superstructure()
        elif key == "hull":
            col, nrm = hull()
        elif key == "bulwark":
            col, nrm = painted((0.62, 0.64, 0.6), rust_amount=0.45)
        elif key == "parapet":
            col, nrm = painted((0.75, 0.6, 0.15), rust_amount=0.3)
        elif key == "mast":
            col, nrm = painted((0.82, 0.62, 0.1), rust_amount=0.4)
        elif key == "funnel":
            col, nrm = painted((0.62, 0.12, 0.1), rust_amount=0.2)
        elif key == "hatch":
            col, nrm = hatch()
        elif key == "water":
            col, nrm = water()
        elif key in THEME_TEXTURES:
            col, nrm = THEME_TEXTURES[key]()
        else:
            col, nrm = painted((0.5, 0.5, 0.5))
        pair = (make_texture(col, key), make_texture(nrm, key + "_n"))
        self._cache[key] = pair
        return pair

    def sprite(self, key: str):
        if key in self._cache:
            return self._cache[key]
        fn = {
            "soft": soft_circle, "smoke": smoke_puff, "flash": muzzle_flash, "flash_side": flash_side,
            "spark": spark, "tracer": tracer, "hole": bullet_hole, "scorch": scorch, "blood": blood,
            "scope": scope_overlay, "vignette": vignette, "damage_arc": damage_arc, "sun": sun_glow,
            "clouds": cloud_layer,
        }[key]
        tex = make_texture(fn(), key, mipmap=key not in ("scope",), clamp=key not in ("clouds",))
        self._cache[key] = tex
        return tex

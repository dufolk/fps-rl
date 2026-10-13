"""Top-down collision map of a map with spawns, sites and buy zones drawn on.

    py tools/top_view.py <map> [out.png]

Colour is the lowest walkable floor (blue low, red high); covered floors are darker.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpsrl.maps import load_map  # noqa: E402

RES = 0.2
SCALE = 2


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "desert2"
    out = sys.argv[2] if len(sys.argv) > 2 else f"shots/{name}_top.png"
    m = load_map(name)
    mn = np.array([b.mn for b in m.boxes if b.solid])
    mx = np.array([b.mx for b in m.boxes if b.solid])
    (x0, y0), (x1, y1) = m.bounds
    W, H = int((x1 - x0) / RES), int((y1 - y0) / RES)
    zb, zt = mn[:, 2].min(), mx[:, 2].max()
    floor = np.full((W, H), np.nan)
    ceil = np.full((W, H), np.inf)

    def span(k):
        i0, i1 = (np.clip(np.round((np.array([mn[k, 0], mx[k, 0]]) - x0) / RES), 0, W)).astype(int)
        j0, j1 = (np.clip(np.round((np.array([mn[k, 1], mx[k, 1]]) - y0) / RES), 0, H)).astype(int)
        return slice(i0, i1), slice(j0, j1)

    ground = mn[:, 2] <= zb + 0.01
    for k in np.nonzero(ground)[0]:
        s = span(k)
        sub = floor[s]
        sub[np.isnan(sub) | (sub < mx[k, 2])] = mx[k, 2]
    for k in np.nonzero(~ground)[0]:
        s = span(k)
        ceil[s] = np.minimum(ceil[s], mn[k, 2])
    wall = np.isnan(floor) | (floor >= zt - 0.01)
    v = ~wall
    lo, hi = np.nanmin(floor[v]), np.nanmax(floor[v])
    t = np.clip((floor - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([40 + 215 * t, 60 + 170 * (1 - np.abs(t - 0.5) * 2), 220 - 190 * t], -1)
    rgb[np.isfinite(ceil) & v] *= 0.55
    rgb[~v] = 0
    img = Image.fromarray(rgb.transpose(1, 0, 2)[::-1].astype(np.uint8)).resize((W * SCALE, H * SCALE),
                                                                                  Image.NEAREST)
    dr = ImageDraw.Draw(img)

    def px(x, y):
        return (x - x0) / RES * SCALE, (H - (y - y0) / RES) * SCALE

    for gx in range(int(np.ceil(x0 / 5)) * 5, int(x1), 5):
        a = px(gx, y0)[0]
        dr.line([(a, 0), (a, H * SCALE)], fill=(130, 130, 130) if gx % 10 == 0 else (60, 60, 60))
        if gx % 10 == 0:
            dr.text((a + 2, 2), str(gx), fill=(255, 255, 255))
    for gy in range(int(np.ceil(y0 / 5)) * 5, int(y1), 5):
        b = px(x0, gy)[1]
        dr.line([(0, b), (W * SCALE, b)], fill=(130, 130, 130) if gy % 10 == 0 else (60, 60, 60))
        if gy % 10 == 0:
            dr.text((2, b + 2), str(gy), fill=(255, 255, 255))
    for zone in m.objectives.get("sites", []):
        pts = [px(*p) for p in zone["poly"]]
        dr.polygon(pts, outline=(255, 60, 60))
        cx, cy = np.mean(pts, axis=0)
        dr.text((cx, cy), zone["name"], fill=(255, 255, 255))
    for side, polys in m.buy_zones.items():
        for poly in polys:
            dr.polygon([px(*p) for p in poly], outline=(255, 200, 40) if side == 0 else (60, 200, 255))
    for team, pts in m.spawns.items():
        col = (255, 150, 60) if team == 0 else (90, 170, 255)
        for p in pts:
            a, b = px(p[0], p[1])
            dr.ellipse([a - 4, b - 4, a + 4, b + 4], fill=col, outline=(0, 0, 0))
    for p in m.strategic_points:
        a, b = px(p[0], p[1])
        dr.rectangle([a - 2, b - 2, a + 2, b + 2], outline=(255, 255, 255))
    img.save(out)
    print("saved", out, img.size, f"floor {lo:.2f}..{hi:.2f} m")


if __name__ == "__main__":
    main()

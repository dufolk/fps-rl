"""Static environment: map geometry, sky, lighting."""
from __future__ import annotations

import math

import numpy as np
from panda3d.core import (AmbientLight, BitMask32, CompassEffect, CullFaceAttrib, DirectionalLight, Fog,
                          RenderState, TextureStage, TransparencyAttrib, Vec3, Vec4)

from ..maps import MapData
from . import models as M
from .meshes import MeshBuilder
from .textures import TextureLibrary

SHADOW_MASK = BitMask32.bit(3)
MAIN_MASK = BitMask32.bit(0)

UV_SCALE = {"sand": 0.15, "sand_dark": 0.15, "sandstone": 0.4, "sandstone_dark": 0.4, "brick": 0.33,
            "brick_dark": 0.33, "concrete": 0.25, "concrete_dark": 0.25, "wood_planks": 0.8, "crate": 1 / 0.85,
            "crate_big": 1 / 1.2, "steel_plate": 1.0}
SURFACE_FOR = {"steel_plate": "metal", "crate": "wood", "crate_big": "wood", "sand": "cloth", "sand_dark": "cloth",
               "sandstone": "cloth", "sandstone_dark": "cloth", "brick": "cloth", "brick_dark": "cloth",
               "wood_planks": "wood", "metal": "metal", "metal_dark": "metal"}
THEME = dict(horizon=(0.78, 0.82, 0.84), zenith=(0.32, 0.5, 0.74), fog=0.0045, sun=(1.45, 1.35, 1.15),
             amb=(0.44, 0.45, 0.48), ground="grass", hills=(0.36, 0.42, 0.32), blocks=(0.55, 0.55, 0.52))


class WorldScene:
    def __init__(self, base, game_map: MapData, textures: TextureLibrary, shadows: bool = True,
                 shadow_size: int = 2048):
        self.base = base
        self.map = game_map
        self.tex = textures
        self.root = base.render.attachNewNode("world")
        self.ts_normal = TextureStage("normal")
        self.ts_normal.setMode(TextureStage.M_normal)
        self.theme = THEME
        (x0, y0), (x1, y1) = game_map.bounds
        self.center = Vec3((x0 + x1) / 2, (y0 + y1) / 2, 0)
        self.extent = max(x1 - x0, y1 - y0)
        self._build_lights(shadows, shadow_size)
        self._build_sky()
        self._build_ground()
        self._build_map()
        if game_map.mesh:
            from .map_mesh import build as build_mesh
            build_mesh(self.root, game_map.mesh, self.ts_normal, self.tex)
        self._build_distant_land()
        self._build_zone()
        fog = Fog("fog")
        fog.setColor(*self.theme["horizon"])
        fog.setExpDensity(self.theme["fog"])
        base.render.setFog(fog)
        self.fog = fog

    # ------------------------------------------------------------------ lights
    def _build_lights(self, shadows, size):
        d = Vec3(*self.map.sun_dir)
        d.normalize()
        self.sun_dir = d
        sun = DirectionalLight("sun")
        sun.setColor(Vec4(*self.theme["sun"], 1))
        sun.setSpecularColor(Vec4(1.2, 1.1, 0.9, 1))
        if shadows:
            sun.setShadowCaster(True, size, size)
            sun.setCameraMask(SHADOW_MASK)
            sun.setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
            lens = sun.getLens()
            (sx0, sy0), (sx1, sy1) = self.map.nav_bounds or self.map.bounds
            film = max(118.0, max(sx1 - sx0, sy1 - sy0) * 1.25)
            lens.setFilmSize(film, film)
            lens.setNearFar(10, 130 + film)
        self.sun_np = self.base.render.attachNewNode(sun)
        self.sun_np.setPos(self.center - d * (130 + self.extent * 0.3))
        self.sun_np.lookAt(self.center)
        self.base.render.setLight(self.sun_np)
        amb = AmbientLight("ambient")
        amb.setColor(Vec4(*self.theme["amb"], 1))
        self.amb_np = self.base.render.attachNewNode(amb)
        self.base.render.setLight(self.amb_np)
        fill = DirectionalLight("fill")
        fill.setColor(Vec4(0.16, 0.19, 0.25, 1))
        self.fill_np = self.base.render.attachNewNode(fill)
        self.fill_np.lookAt(Vec3(d.x, d.y, 0.6) * 1.0)
        self.fill_np.setHpr(self.sun_np.getH() + 180, -25, 0)
        self.base.render.setLight(self.fill_np)

    # ------------------------------------------------------------------ sky
    def _build_sky(self):
        sun = -self.sun_dir
        mb = MeshBuilder("sky")
        horizon = np.array(self.theme["horizon"])
        zenith = np.array(self.theme["zenith"])

        def color(nx, ny, nz):
            up = max(nz, 0.0)
            t = up ** 0.45
            c = horizon * (1 - t) + zenith * t
            s = max(0.0, nx * sun.x + ny * sun.y + nz * sun.z)
            c = c + np.array([0.55, 0.38, 0.18]) * (s ** 8) * (1 - t * 0.6)
            if nz < 0:
                c = horizon * 0.92
            return (float(c[0]), float(c[1]), float(c[2]), 1.0)

        mb.sphere((0, 0, 0), 1500.0, rings=24, segments=36, inward=True, color_fn=color)
        sky = mb.node()
        sky.reparentTo(self.base.camera)
        sky.setEffect(CompassEffect.make(self.base.render))
        sky.setBin("background", 0)
        sky.setDepthWrite(False)
        sky.setLightOff(1)
        sky.setFogOff(1)
        sky.setShaderOff(1)
        sky.hide(SHADOW_MASK)
        self.sky = sky
        # sun disc
        cm_sun = MeshBuilder("sun")
        cm_sun.quad((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1), uvs=((0, 0), (1, 0), (1, 1), (0, 1)))
        sun_np = cm_sun.node()
        sun_np.reparentTo(sky)
        sun_np.setPos(sun * 1400)
        sun_np.setScale(160)
        sun_np.setBillboardPointEye()
        sun_np.setTexture(self.tex.sprite("sun"), 1)
        sun_np.setTransparency(TransparencyAttrib.M_alpha)
        sun_np.setBin("background", 1)
        sun_np.setDepthWrite(False)
        sun_np.setDepthTest(False)
        # cloud plane
        cl = MeshBuilder("clouds")
        R = 1400.0
        cl.quad((-R, -R, 0), (R, -R, 0), (R, R, 0), (-R, R, 0), uvs=((0, 0), (3, 0), (3, 3), (0, 3)))
        cl.quad((-R, R, 0), (R, R, 0), (R, -R, 0), (-R, -R, 0), uvs=((0, 3), (3, 3), (3, 0), (0, 0)))
        clouds = cl.node()
        clouds.reparentTo(sky)
        clouds.setZ(260)
        clouds.setTexture(self.tex.sprite("clouds"), 1)
        clouds.setTransparency(TransparencyAttrib.M_alpha)
        clouds.setColorScale(1.0, 0.94, 0.88, 0.75)
        clouds.setBin("background", 2)
        clouds.setDepthWrite(False)
        clouds.setDepthTest(False)
        self.clouds = clouds

    # ------------------------------------------------------------------ map
    def _build_map(self):
        groups: dict[str, MeshBuilder] = {}
        for b in self.map.boxes:
            if not b.visible:
                continue
            mat = b.mat
            mb = groups.setdefault(mat, MeshBuilder(mat))
            scale = UV_SCALE.get(mat, 0.5)
            skip = ()
            if b.tag in ("deck", "ground", "slab"):
                skip = ("-z",)
            elif b.tag == "ceiling":
                skip = ("+z",)
            mb.box(b.mn, b.mx, (1, 1, 1, 1), uv_mode="world" if not mat.startswith("crate") else "local",
                   uv_scale=scale, skip=skip)
        for mat, mb in groups.items():
            node = mb.node()
            node.reparentTo(self.root)
            col, nrm = self.tex.get(mat)
            node.setTexture(col, 1)
            node.setTexture(self.ts_normal, nrm, 1)
            node.setMaterial(M.make_material(SURFACE_FOR.get(mat, "paint")), 1)
            node.flattenStrong()

    def _build_ground(self):
        """Large textured ground plane around (and under) the playable area."""
        mat = self.theme["ground"]
        col, nrm = self.tex.get(mat)
        mb = MeshBuilder("ground_far")
        R = 900.0
        cx, cy = self.center.x, self.center.y
        z = self.map.ground_z
        mb.quad((cx - R, cy - R, z), (cx + R, cy - R, z), (cx + R, cy + R, z), (cx - R, cy + R, z),
                uv_mode="world", uv_scale=UV_SCALE.get(mat, 0.2))
        node = mb.node()
        node.reparentTo(self.root)
        node.setTexture(col, 1)
        node.setTexture(self.ts_normal, nrm, 1)
        node.setMaterial(M.make_material("cloth"), 1)
        node.hide(SHADOW_MASK)  # far plane is huge; drawing it into the shadow map just burns fill rate

    def _build_distant_land(self):
        rng = np.random.default_rng(11)
        mb = MeshBuilder("distant")
        cx, cy = self.center.x, self.center.y
        z0 = self.map.ground_z
        r0 = self.extent * 0.75 + 60
        hills, blocks = (*self.theme["hills"], 1), (*self.theme["blocks"], 1)
        for k in range(36):
            a = k / 36 * math.tau + rng.uniform(-0.05, 0.05)
            r = r0 + rng.uniform(110, 220)
            h = rng.uniform(10, 34)
            rad = rng.uniform(40, 70)
            mb.sphere((cx + math.cos(a) * r, cy + math.sin(a) * r, z0 - h * 0.4), rad, hills, 6, 10,
                      scale=(1.0, 1.0, h / rad))
        for k in range(14):
            a = rng.uniform(0, math.tau)
            r = r0 + rng.uniform(5, 40)
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            w, d, h = rng.uniform(6, 14), rng.uniform(6, 14), rng.uniform(4, 10)
            mb.box((x, y, z0), (x + w, y + d, z0 + h), blocks)
        node = mb.node()
        node.reparentTo(self.root)
        node.setMaterial(M.make_material("paint"), 1)
        node.hide(SHADOW_MASK)

    def _build_zone(self):
        """Glowing circular poison-zone boundary (毒圈) along the OBJ outer edge."""
        zone = (self.map.objectives or {}).get("zone")
        if not zone:
            return
        cx, cy = zone["center"]
        r = float(zone["radius"])
        low = [b for b in self.map.boxes if b.solid and b.mx[2] < 4.0]
        bmn = np.array([b.mn for b in low], dtype=np.float64).reshape(-1, 3)
        bmx = np.array([b.mx for b in low], dtype=np.float64).reshape(-1, 3)
        n = 96
        pts = []
        for k in range(n):
            a = k / n * math.tau
            x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
            ins = (bmn[:, 0] <= x) & (x <= bmx[:, 0]) & (bmn[:, 1] <= y) & (y <= bmx[:, 1])
            z = float(bmx[ins, 2].max()) if ins.any() else 0.0
            pts.append((x, y, z))
        line = MeshBuilder("zone_line")
        wall = MeshBuilder("zone_wall")
        w, h = 0.32, 10.0
        for k in range(n):
            ax, ay, az = pts[k]
            bx, by, bz = pts[(k + 1) % n]
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L * w, dx / L * w
            line.quad((ax - nx, ay - ny, az + 0.04), (bx - nx, by - ny, bz + 0.04),
                      (bx + nx, by + ny, bz + 0.04), (ax + nx, ay + ny, az + 0.04),
                      color=(0.1, 0.85, 1.0, 0.95))
            wall.quad((ax, ay, az), (bx, by, bz), (bx, by, bz + h), (ax, ay, az + h),
                      color=(0.12, 0.48, 1.0, 0.34))
            for zh in (2.5, 5.0, 7.5, h):
                wall.quad((ax, ay, az + zh - 0.08), (bx, by, bz + zh - 0.08),
                          (bx, by, bz + zh + 0.08), (ax, ay, az + zh + 0.08),
                          color=(0.2, 0.88, 1.0, 0.78))
            mx, my = ax + dx * 0.06, ay + dy * 0.06
            wall.quad((ax, ay, az), (mx, my, az), (mx, my, az + h), (ax, ay, az + h),
                      color=(0.2, 0.88, 1.0, 0.65))
        ln = line.node()
        ln.reparentTo(self.root)
        ln.setLightOff(1)
        ln.setShaderOff(1)
        ln.setDepthOffset(2)
        ln.setTransparency(TransparencyAttrib.M_alpha)
        ln.hide(SHADOW_MASK)
        wn = wall.node()
        wn.reparentTo(self.root)
        wn.setTwoSided(True)
        wn.setLightOff(1)
        wn.setShaderOff(1)
        wn.setTransparency(TransparencyAttrib.M_alpha)
        wn.setDepthWrite(False)
        wn.setBin("transparent", 10)
        wn.hide(SHADOW_MASK)

    def destroy(self):
        r = self.base.render
        for np_ in (self.sun_np, self.amb_np, self.fill_np):
            r.clearLight(np_)
            np_.removeNode()
        r.clearFog()
        self.sky.removeNode()
        self.root.removeNode()

    # ------------------------------------------------------------------ update
    def update(self, t: float):
        self.clouds.setTexOffset(TextureStage.getDefault(), t * 0.0015, 0)

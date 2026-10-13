"""Static environment: map geometry, sky, sea, lighting."""
from __future__ import annotations

import math

import numpy as np
from panda3d.core import (AmbientLight, BitMask32, CompassEffect, CullFaceAttrib, DirectionalLight,
                          Fog, NodePath, RenderState, TextureStage, TransparencyAttrib, Vec3,
                          Vec4)

from ..maps import MapData
from . import models as M
from .meshes import MeshBuilder
from .textures import TextureLibrary

SHADOW_MASK = BitMask32.bit(3)
MAIN_MASK = BitMask32.bit(0)

UV_SCALE = {"deck": 0.25, "superstructure": 1 / 3.0, "steel_plate": 1.0, "crate": 1 / 0.85,
            "crate_big": 1 / 1.2, "hull": 0.25, "water": 0.02, "sand": 0.15, "sand_dark": 0.15,
            "sandstone": 0.4, "sandstone_dark": 0.4, "brick": 0.33, "brick_dark": 0.33, "asphalt": 0.12,
            "grass": 0.15, "tile_floor": 0.5, "tile_white": 0.5, "marble": 0.25, "carpet": 0.33,
            "carpet_blue": 0.33, "roof": 0.5, "roof_grey": 0.5, "concrete": 0.25, "concrete_dark": 0.25,
            "wood_floor": 0.5, "wood_door": 0.8, "wood_planks": 0.8, "reinforce": 0.66, "vault": 1.0,
            "stone_dark": 0.5}

SURFACE_FOR = {"deck": "paint", "steel_plate": "metal", "crate": "wood", "crate_big": "wood",
               "superstructure": "paint", "hull": "paint", "bulwark": "paint", "parapet": "paint",
               "mast": "paint", "funnel": "paint", "hatch": "paint", "sand": "cloth", "sand_dark": "cloth",
               "sandstone": "cloth", "sandstone_dark": "cloth", "brick": "cloth", "brick_dark": "cloth",
               "asphalt": "cloth", "grass": "cloth", "carpet": "cloth", "carpet_blue": "cloth",
               "marble": "glass", "tile_floor": "paint", "tile_white": "paint", "wood_floor": "wood",
               "wood_door": "wood", "wood_planks": "wood", "reinforce": "metal", "vault": "metal",
               "metal": "metal", "metal_dark": "metal", "metal_door": "metal"}

THEMES = {
    "ship": dict(horizon=(0.78, 0.74, 0.7), zenith=(0.22, 0.4, 0.66), fog=0.0042, sun=(1.35, 1.18, 0.95),
                 amb=(0.36, 0.4, 0.48), ground=None),
    "desert": dict(horizon=(0.92, 0.82, 0.66), zenith=(0.3, 0.52, 0.8), fog=0.0032, sun=(1.5, 1.3, 1.0),
                   amb=(0.46, 0.42, 0.4), ground="sand_dark"),
    "town": dict(horizon=(0.8, 0.82, 0.84), zenith=(0.3, 0.48, 0.74), fog=0.0048, sun=(1.3, 1.22, 1.05),
                 amb=(0.42, 0.44, 0.5), ground="grass"),
    "bank": dict(horizon=(0.66, 0.68, 0.74), zenith=(0.22, 0.3, 0.48), fog=0.0055, sun=(1.1, 1.04, 0.96),
                 amb=(0.56, 0.56, 0.6), ground="asphalt"),
}

HORIZON = THEMES["ship"]["horizon"]
ZENITH = THEMES["ship"]["zenith"]
SITE_COLORS = [(0.95, 0.25, 0.15, 1), (0.2, 0.55, 0.95, 1), (0.95, 0.8, 0.2, 1), (0.3, 0.85, 0.35, 1)]


def _darker(c, k):
    return (c[0] * k, c[1] * k, c[2] * k, 1.0)


class WorldScene:
    def __init__(self, base, game_map: MapData, textures: TextureLibrary, shadows: bool = True,
                 shadow_size: int = 4096):
        self.base = base
        self.map = game_map
        self.tex = textures
        self.root = base.render.attachNewNode("world")
        self.ts_normal = TextureStage("normal")
        self.ts_normal.setMode(TextureStage.M_normal)
        self.animated = []
        self.theme = THEMES.get(game_map.theme, THEMES["ship"])
        (x0, y0), (x1, y1) = game_map.bounds
        self.center = Vec3((x0 + x1) / 2, (y0 + y1) / 2, 0)
        self.extent = max(x1 - x0, y1 - y0)
        self._build_lights(shadows, shadow_size)
        self._build_sky()
        ship = game_map.theme == "ship"
        if ship:
            self._build_water()
        else:
            self.water = None
            self._build_ground()
        self._build_map()
        self._build_props()
        if ship:
            self._build_container_details()
            self._build_markings()
            self._build_distant()
        else:
            self._build_distant_land()
        self._build_objectives()
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
            film = max(118.0, self.extent * 1.25)
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

    # ------------------------------------------------------------------ water
    def _build_water(self):
        col, nrm = self.tex.get("water")
        mb = MeshBuilder("water")
        R = 1600.0
        z = self.map.water_z
        mb.quad((-R, -R, z), (R, -R, z), (R, R, z), (-R, R, z), uv_mode="world", uv_scale=0.06)
        water = mb.node()
        water.reparentTo(self.root)
        water.setTexture(col, 1)
        water.setTexture(self.ts_normal, nrm, 1)
        mat = M.make_material("glass")
        water.setMaterial(mat, 1)
        water.hide(SHADOW_MASK)
        self.water = water
        # hull waterline foam strip
        foam = MeshBuilder("foam")
        fz = z + 0.02
        for y0, y1 in ((14.0, 15.6), (-15.6, -14.0)):
            foam.quad((-48, y0, fz), (48, y0, fz), (48, y1, fz), (-48, y1, fz),
                      color=(0.85, 0.9, 0.92, 0.55), uv_scale=0.3)
        fnp = foam.node()
        fnp.reparentTo(self.root)
        fnp.setTransparency(TransparencyAttrib.M_alpha)
        fnp.setTexture(self.tex.sprite("smoke"), 1)
        fnp.setLightOff(1)
        fnp.hide(SHADOW_MASK)
        self.animated.append(("water", water))

    # ------------------------------------------------------------------ map
    def _build_map(self):
        groups: dict[str, MeshBuilder] = {}
        for b in self.map.boxes:
            if not b.visible or b.params.get("destruct"):
                continue
            mat = b.mat
            mb = groups.setdefault(mat, MeshBuilder(mat))
            scale = UV_SCALE.get(mat, 0.5)
            skip = ()
            if b.tag in ("deck", "ground"):
                skip = ("-z",)
            mb.box(b.mn, b.mx, (1, 1, 1, 1), uv_mode="world" if not mat.startswith("crate") else "local",
                   uv_scale=scale, skip=skip)
        for mat, mb in groups.items():
            node = mb.node()
            node.reparentTo(self.root)
            col, nrm = self.tex.get(mat)
            node.setTexture(col, 1)
            node.setTexture(self.ts_normal, nrm, 1)
            surface = "paint" if mat.startswith("container_") else SURFACE_FOR.get(mat, "paint")
            node.setMaterial(M.make_material(surface), 1)

    def _build_container_details(self):
        mb = MeshBuilder("container_detail")
        for b in self.map.boxes:
            if b.tag != "container":
                continue
            color = TextureLibrary.CONTAINER_COLORS.get(b.mat.split("_", 1)[1], (0.5, 0.5, 0.5))
            dark = _darker(color, 0.55)
            x0, y0, z0 = b.mn
            x1, y1, z1 = b.mx
            e = 0.025
            c = 0.16
            for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                sx = 1 if cx == x0 else -1
                sy = 1 if cy == y0 else -1
                ax0, ax1 = sorted((cx - sx * e, cx + sx * c))
                ay0, ay1 = sorted((cy - sy * e, cy + sy * c))
                mb.box((ax0, ay0, z0), (ax1, ay1, z1), dark)
            for z in (z0, z1 - 0.14):
                mb.box((x0 - e, y0 - e, z), (x1 + e, y0 + 0.08, z + 0.14), dark)
                mb.box((x0 - e, y1 - 0.08, z), (x1 + e, y1 + e, z + 0.14), dark)
                mb.box((x0 - e, y0 - e, z), (x0 + 0.08, y1 + e, z + 0.14), dark)
                mb.box((x1 - 0.08, y0 - e, z), (x1 + e, y1 + e, z + 0.14), dark)
            along = b.params.get("along", "x")
            door = b.params.get("door", 1)
            bar = _darker(color, 0.4)
            if along == "x":
                xd = x1 + 0.03 if door > 0 else x0 - 0.06
                w = y1 - y0
                for k in range(4):
                    y = y0 + w * (0.14 + 0.24 * k)
                    mb.box((xd, y - 0.025, z0 + 0.15), (xd + 0.03, y + 0.025, z1 - 0.15), bar)
                    mb.box((xd - 0.01, y - 0.05, z0 + 1.0), (xd + 0.05, y + 0.12, z0 + 1.08), bar)
                yc = (y0 + y1) / 2
                mb.box((xd, yc - 0.012, z0 + 0.14), (xd + 0.02, yc + 0.012, z1 - 0.14), dark)
            else:
                yd = y1 + 0.03 if door > 0 else y0 - 0.06
                w = x1 - x0
                for k in range(4):
                    x = x0 + w * (0.14 + 0.24 * k)
                    mb.box((x - 0.025, yd, z0 + 0.15), (x + 0.025, yd + 0.03, z1 - 0.15), bar)
                    mb.box((x - 0.05, yd - 0.01, z0 + 1.0), (x + 0.12, yd + 0.05, z0 + 1.08), bar)
                xc = (x0 + x1) / 2
                mb.box((xc - 0.012, yd, z0 + 0.14), (xc + 0.012, yd + 0.02, z1 - 0.14), dark)
        node = mb.node()
        node.reparentTo(self.root)
        node.setMaterial(M.make_material("paint"), 1)
        # railings along the bulwarks
        rail = MeshBuilder("railings")
        white = (0.86, 0.86, 0.82, 1)
        for s in (1, -1):
            y = s * 13.3
            rail.cylinder((-40, y, 1.62), (40, y, 1.62), 0.035, white, 8)
            rail.cylinder((-40, y, 1.36), (40, y, 1.36), 0.025, white, 8)
            for k in range(41):
                x = -40 + k * 2.0
                rail.cylinder((x, y, 1.1), (x, y, 1.64), 0.03, white, 6)
        rn = rail.node()
        rn.reparentTo(self.root)
        rn.setMaterial(M.make_material("paint"), 1)

    def _build_props(self):
        self.lamps = []
        self.radars = []
        for p in self.map.props:
            yaw = p.params.get("yaw", 0.0)
            if p.kind == "barrel":
                n = M.build_barrel(p.params.get("color", "blue"))
            elif p.kind == "lamp":
                n, bulb = M.build_lamp()
                bulb.setLightOff(1)
                bulb.setShaderOff(1)
                bulb.setColorScale(1.6, 1.4, 1.0, 1)
                self.lamps.append(bulb)
            elif p.kind == "crane_boom":
                n = M.build_crane_boom()
            elif p.kind == "antenna":
                n = M.build_antenna()
            elif p.kind == "radar":
                n, bar = M.build_radar()
                self.radars.append(bar)
            elif p.kind == "door":
                n = M.build_door()
            elif p.kind == "tree":
                n = M.build_tree(p.params.get("seed", int(abs(p.pos[0] * 7 + p.pos[1] * 13))))
            elif p.kind == "palm":
                n = M.build_palm(p.params.get("seed", int(abs(p.pos[0] * 7 + p.pos[1] * 13))))
            elif p.kind == "car":
                n = M.build_car(p.params.get("color", (0.6, 0.12, 0.1)))
            elif p.kind == "desk":
                n = M.build_desk()
            elif p.kind == "ceiling_light":
                n, bulb = M.build_ceiling_light()
                bulb.setLightOff(1)
                bulb.setShaderOff(1)
                bulb.setColorScale(1.5, 1.45, 1.3, 1)
            elif p.kind == "ship_bow_lights":
                continue
            else:
                continue
            n.reparentTo(self.root)
            n.setPos(*p.pos)
            n.setH(yaw)
            n.flattenLight()

    def _build_ground(self):
        """Large textured ground plane around (and under) the playable area."""
        mat = self.theme["ground"]
        col, nrm = self.tex.get(mat)
        mb = MeshBuilder("ground_far")
        R = 900.0
        cx, cy = self.center.x, self.center.y
        z = -0.03
        mb.quad((cx - R, cy - R, z), (cx + R, cy - R, z), (cx + R, cy + R, z), (cx - R, cy + R, z),
                uv_mode="world", uv_scale=UV_SCALE.get(mat, 0.2))
        node = mb.node()
        node.reparentTo(self.root)
        node.setTexture(col, 1)
        node.setTexture(self.ts_normal, nrm, 1)
        node.setMaterial(M.make_material("cloth"), 1)

    def _build_distant_land(self):
        theme = self.map.theme
        rng = np.random.default_rng(11)
        mb = MeshBuilder("distant")
        cx, cy = self.center.x, self.center.y
        r0 = self.extent * 0.75 + 30
        if theme == "desert":
            for k in range(36):
                a = k / 36 * math.tau + rng.uniform(-0.05, 0.05)
                r = r0 + rng.uniform(110, 220)
                h = rng.uniform(10, 34)
                rad = rng.uniform(40, 70)
                mb.sphere((cx + math.cos(a) * r, cy + math.sin(a) * r, -h * 0.4), rad,
                          (0.62, 0.5, 0.36, 1), 6, 10, scale=(1.0, 1.0, h / rad))
            for k in range(14):
                a = rng.uniform(0, math.tau)
                r = r0 + rng.uniform(5, 40)
                x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
                w, d, h = rng.uniform(6, 14), rng.uniform(6, 14), rng.uniform(4, 10)
                mb.box((x, y, 0), (x + w, y + d, h), (0.78, 0.66, 0.48, 1))
        elif theme == "town":
            for k in range(30):
                a = k / 30 * math.tau
                r = r0 + rng.uniform(60, 160)
                h = rng.uniform(20, 55)
                mb.sphere((cx + math.cos(a) * r, cy + math.sin(a) * r, -h * 0.3), rng.uniform(50, 90),
                          (0.22, 0.34, 0.18, 1), 6, 10, scale=(1.0, 1.0, h / 70))
            for k in range(40):
                a = rng.uniform(0, math.tau)
                r = r0 + rng.uniform(0, 50)
                x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
                w, d, h = rng.uniform(5, 10), rng.uniform(5, 10), rng.uniform(4, 9)
                mb.box((x, y, 0), (x + w, y + d, h), (0.8, 0.76, 0.66, 1))
                mb.box((x - 0.3, y - 0.3, h), (x + w + 0.3, y + d + 0.3, h + 0.6), (0.45, 0.2, 0.14, 1))
                if k % 2:
                    mb.sphere((x - 3, y - 3, 3), 2.6, (0.2, 0.34, 0.14, 1), 5, 8)
        else:
            for k in range(46):
                a = k / 46 * math.tau + rng.uniform(-0.04, 0.04)
                r = r0 + rng.uniform(10, 70)
                x, y = cx + math.cos(a) * r, cy + math.sin(a) * r
                w, d, h = rng.uniform(12, 30), rng.uniform(12, 30), rng.uniform(15, 70)
                g = rng.uniform(0.4, 0.62)
                mb.box((x, y, 0), (x + w, y + d, h), (g, g, g * 1.04, 1))
                for z in np.arange(3.0, h - 2, 3.5):
                    mb.box((x - 0.05, y + 1, z), (x + w + 0.05, y + d - 1, z + 1.4), (0.18, 0.22, 0.28, 1))
        node = mb.node()
        node.reparentTo(self.root)
        node.setMaterial(M.make_material("paint"), 1)
        node.hide(SHADOW_MASK)

    def _build_objectives(self):
        """Painted zone outlines + letters for domination points / bomb sites."""
        from panda3d.core import TextNode
        obj = self.map.objectives or {}
        self.point_nodes = []
        for kind in ("points", "sites"):
            for i, z in enumerate(obj.get(kind, [])):
                poly = z["poly"]
                gz = z.get("z", z.get("zmin", 0.0)) + 0.02
                color = (0.95, 0.95, 0.95, 1) if kind == "points" else SITE_COLORS[i % len(SITE_COLORS)]
                mb = MeshBuilder(f"zone_{kind}_{i}")
                n = len(poly)
                w = 0.12
                for k in range(n):
                    (ax, ay), (bx, by) = poly[k], poly[(k + 1) % n]
                    dx, dy = bx - ax, by - ay
                    L = math.hypot(dx, dy) or 1.0
                    nx, ny = -dy / L * w, dx / L * w
                    mb.quad((ax - nx, ay - ny, gz), (bx - nx, by - ny, gz), (bx + nx, by + ny, gz),
                            (ax + nx, ay + ny, gz), color=color)
                node = mb.node()
                node.reparentTo(self.root)
                node.setLightOff(1)
                node.setShaderOff(1)
                node.setDepthOffset(2)
                node.hide(SHADOW_MASK)
                cxp = sum(p[0] for p in poly) / n
                cyp = sum(p[1] for p in poly) / n
                tn = TextNode(f"zone_label_{i}")
                tn.setText(z.get("name", "?"))
                tn.setAlign(TextNode.ACenter)
                tn.setTextColor(*color[:3], 0.85)
                t = self.root.attachNewNode(tn)
                t.setPos(cxp, cyp - 0.6, gz + 0.01)
                t.setP(-90)
                t.setScale(2.4 if kind == "sites" else 1.6)
                t.setLightOff(1)
                t.setShaderOff(1)
                t.setDepthOffset(2)
                t.setTransparency(TransparencyAttrib.M_alpha)
                t.hide(SHADOW_MASK)
                if kind == "points":
                    self.point_nodes.append((node, t))

    def set_point_colors(self, colors):
        for (node, label), c in zip(self.point_nodes, colors):
            node.setColorScale(*c)
            label.setColorScale(*c)

    def _build_markings(self):
        mb = MeshBuilder("markings")
        yel = (0.88, 0.7, 0.12, 1)
        wht = (0.85, 0.85, 0.82, 1)
        z = 0.004
        for sx in (1, -1):
            x0, x1 = sorted((sx * 34.8, sx * 34.6))
            mb.box((x0, -12.8, 0), (x1, 12.8, z), yel)
            for edge, top in ((33.0, 0.4), (34.0, 0.8), (35.0, 1.2)):
                x0, x1 = sorted((sx * edge, sx * (edge + 0.12)))
                mb.box((x0, -2.5, top), (x1, 2.5, top + z), yel)
        for s in (1, -1):
            mb.box((-34.0, s * 12.4 - 0.08, 0), (34.0, s * 12.4 + 0.08, z), yel)
        for x in range(-28, 29, 4):
            if abs(x) > 3:
                mb.box((x - 0.7, -0.1, 0), (x + 0.7, 0.1, z), wht)
        node = mb.node()
        node.reparentTo(self.root)
        node.setMaterial(M.make_material("paint"), 1)
        node.setDepthOffset(1)

    def _build_distant(self):
        mb = MeshBuilder("distant")
        rng = np.random.default_rng(7)
        quay = (0.42, 0.42, 0.4, 1)
        mb.box((-400, 170, -6), (400, 260, -2), quay)
        for k in range(7):
            x = -180 + k * 60 + rng.uniform(-10, 10)
            col = (0.75, 0.2, 0.15, 1) if k % 2 else (0.85, 0.85, 0.82, 1)
            for sx in (-6, 6):
                mb.box((x + sx - 0.8, 176, -2), (x + sx + 0.8, 177.6, 38), col)
                mb.box((x + sx - 0.8, 196, -2), (x + sx + 0.8, 197.6, 38), col)
            mb.box((x - 7, 150, 38), (x + 7, 205, 42), col)
            mb.box((x - 4, 190, 42), (x + 4, 200, 48), (0.3, 0.3, 0.3, 1))
        for k in range(40):
            x = rng.uniform(-300, 300)
            y = rng.uniform(205, 250)
            hgt = 2.6 * rng.integers(1, 5)
            c = list(TextureLibrary.CONTAINER_COLORS.values())[rng.integers(9)]
            mb.box((x, y, -2), (x + 12, y + 2.5, -2 + hgt), (*c, 1))
        for k in range(10):
            x = rng.uniform(-500, 500)
            hgt = rng.uniform(25, 80)
            mb.box((x, 240 + rng.uniform(0, 40), -2), (x + rng.uniform(15, 40), 280 + rng.uniform(0, 30), hgt),
                   (0.55, 0.57, 0.6, 1))
        # another ship far away
        mb.box((-260, -320, -6), (-120, -296, 6), (0.18, 0.2, 0.25, 1))
        mb.box((-140, -318, 6), (-122, -298, 22), (0.85, 0.85, 0.82, 1))
        for k in range(8):
            c = list(TextureLibrary.CONTAINER_COLORS.values())[k % 9]
            mb.box((-250 + k * 13, -316, 6), (-238 + k * 13, -300, 6 + 2.6 * (1 + k % 3)), (*c, 1))
        node = mb.node()
        node.reparentTo(self.root)
        node.setMaterial(M.make_material("paint"), 1)
        node.hide(SHADOW_MASK)

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
        if self.water is not None:
            self.water.setTexOffset(TextureStage.getDefault(), t * 0.004, t * 0.002)
            self.water.setTexOffset(self.ts_normal, -t * 0.006, t * 0.003)
        self.clouds.setTexOffset(TextureStage.getDefault(), t * 0.0015, 0)
        for r in self.radars:
            r.setH(t * 90.0)

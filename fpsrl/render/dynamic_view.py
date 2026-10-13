"""Renderers for world state that changes during a match: destructible panels, gadgets, the bomb
and domination point colours."""
from __future__ import annotations

import math

from panda3d.core import NodePath, TextureStage, TransparencyAttrib

from . import models as M
from .hud import TEAM_COLORS
from .meshes import MeshBuilder
from .world_scene import SHADOW_MASK, SURFACE_FOR, UV_SCALE

GADGET_MATS = {"barricade": "wood_planks", "castle": "reinforce", "dshield": "metal"}


class DestructView:
    """One mesh per destructible wall, rebuilt from its live panels whenever the wall changes."""

    def __init__(self, root: NodePath, world, textures, ts_normal: TextureStage):
        self.w = world
        self.tex = textures
        self.ts_normal = ts_normal
        self.root = root.attachNewNode("destructibles")
        self.nodes: dict[int, NodePath] = {}
        self.sig: dict[int, tuple] = {}
        self.version = None
        self.update(force=True)

    def _signature(self, wall):
        ps = self.w.destruct.panels
        return tuple((ps[i].alive, ps[i].reinforced) for i in self.w.destruct.walls[wall])

    def _build(self, wall):
        old = self.nodes.pop(wall, None)
        if old is not None:
            old.removeNode()
        d = self.w.destruct
        groups: dict[str, MeshBuilder] = {}
        for i in d.walls[wall]:
            p = d.panels[i]
            if not p.alive:
                continue
            mat = p.box.mat
            groups.setdefault(mat, MeshBuilder(mat)).box(p.box.mn, p.box.mx, uv_mode="world",
                                                         uv_scale=UV_SCALE.get(mat, 0.5))
            if p.reinforced:
                mn, mx = list(p.box.mn), list(p.box.mx)
                for k in range(3):
                    if k != p.axis:
                        mn[k] += 0.005
                        mx[k] -= 0.005
                    else:
                        mn[k] -= 0.02
                        mx[k] += 0.02
                groups.setdefault("reinforce", MeshBuilder("reinforce")).box(mn, mx, uv_mode="world",
                                                                            uv_scale=0.66)
        node = self.root.attachNewNode(f"wall{wall}")
        for mat, mb in groups.items():
            if mb.empty():
                continue
            n = mb.node()
            n.reparentTo(node)
            col, nrm = self.tex.get(mat)
            n.setTexture(col, 1)
            n.setTexture(self.ts_normal, nrm, 1)
            n.setMaterial(M.make_material(SURFACE_FOR.get(mat, "paint")), 1)
        self.nodes[wall] = node

    def update(self, force=False):
        d = self.w.destruct
        if not force and d.version == self.version:
            return
        self.version = d.version
        for wall in range(len(d.walls)):
            s = self._signature(wall)
            if force or self.sig.get(wall) != s:
                self.sig[wall] = s
                self._build(wall)


class GadgetView:
    def __init__(self, root: NodePath, world, textures, ts_normal, models: M.ModelCache):
        self.w = world
        self.tex = textures
        self.ts_normal = ts_normal
        self.models = models
        self.root = root.attachNewNode("gadgets")
        self.nodes: dict[int, tuple] = {}
        self.bomb_np = None
        self.bomb_light = None
        self.bomb_key = None

    def _box_node(self, it):
        mat = GADGET_MATS.get(it["kind"], "metal")
        mb = MeshBuilder(it["kind"])
        mb.box(tuple(it["mn"]), tuple(it["mx"]), uv_mode="world", uv_scale=UV_SCALE.get(mat, 0.8))
        n = mb.node()
        col, nrm = self.tex.get(mat)
        n.setTexture(col, 1)
        n.setTexture(self.ts_normal, nrm, 1)
        n.setMaterial(M.make_material(SURFACE_FOR.get(mat, "paint")), 1)
        if it["kind"] == "barricade":
            n.setTransparency(TransparencyAttrib.M_alpha)
            n.setColorScale(1, 1, 1, 0.92)
        return n

    def _model_node(self, it):
        kind = it["kind"]
        name = {"charge": it.get("item") or "breach", "edd": "edd", "jammer": "jammer", "drone": "drone",
                "camera": "camera", "barbed": "barbed", "armor_bag": "armor_bag"}.get(kind)
        if name is None:
            return None
        try:
            n, _ = self.models.weapon(name)
        except KeyError:
            return None
        if kind == "barbed":
            n.setScale(4.0, 4.0, 1.5)
        return n

    def update(self, t: float):
        g = self.w.gadgets
        seen = set()
        for gid, it in g.items.items():
            seen.add(gid)
            key = (it["kind"], tuple(it["mn"]) if "mn" in it else None)
            cur = self.nodes.get(gid)
            if cur is None or cur[0] != key:
                if cur is not None:
                    cur[1].removeNode()
                n = self._box_node(it) if "mn" in it else self._model_node(it)
                if n is None:
                    n = NodePath("empty")
                n.reparentTo(self.root)
                self.nodes[gid] = (key, n)
            n = self.nodes[gid][1]
            if "mn" not in it:
                n.setPos(*it["pos"])
                n.setHpr(it.get("yaw", 0.0) - 90.0, it.get("pitch", 0.0), 0)
                if it["kind"] == "camera":
                    n.setHpr(it.get("yaw", 0.0) - 90.0, it.get("pitch", 0.0), 0)
                elif it["kind"] in ("charge", "edd", "jammer"):
                    n.setColorScale(1, 1 - 0.5 * (int(t * 4) % 2), 1 - 0.5 * (int(t * 4) % 2), 1)
        for gid in list(self.nodes):
            if gid not in seen:
                self.nodes.pop(gid)[1].removeNode()
        self._update_bomb(t)

    def _update_bomb(self, t):
        mode = self.w.mode
        bomb = getattr(mode, "bomb", None)
        show = bomb is not None and bomb.get("state") in ("dropped", "planted", "defused")
        if not show:
            if self.bomb_np is not None:
                self.bomb_np.hide()
            return
        if self.bomb_np is None:
            n, _ = self.models.weapon(getattr(mode, "bomb_item", "c4") or "c4")
            n.reparentTo(self.root)
            n.setScale(1.6)
            self.bomb_np = n
            lm = MeshBuilder("bomb_led")
            lm.sphere((0, 0, 0), 0.03, (1, 0.1, 0.05, 1), 4, 6)
            self.bomb_light = lm.node()
            self.bomb_light.reparentTo(n)
            self.bomb_light.setPos(0, 0, 0.1)
            self.bomb_light.setLightOff(1)
            self.bomb_light.setShaderOff(1)
        self.bomb_np.show()
        self.bomb_np.setPos(bomb["pos"][0], bomb["pos"][1], bomb["pos"][2] + 0.02)
        planted = bomb["state"] == "planted"
        if planted:
            left = max(0.0, getattr(mode, "phase_end", self.w.time) - self.w.time)
            rate = 1.0 if left > 10 else (2.5 if left > 4 else 6.0)
            on = (self.w.time * rate) % 1.0 < 0.3
            self.bomb_light.setColorScale(2.0, 1, 1, 1) if on else self.bomb_light.setColorScale(0.2, 0.2, 0.2, 1)
        else:
            self.bomb_light.setColorScale(0.2, 0.2, 0.2, 1)


def point_colors(world):
    """Colour scale per domination point (owner tint, blinking while contested)."""
    m = world.mode
    if getattr(m, "name", "") != "dom":
        return []
    out = []
    for owner, prog in zip(m.owner, m.progress):
        if owner in (0, 1):
            c = TEAM_COLORS[owner]
            col = (c[0] * 1.3, c[1] * 1.3, c[2] * 1.3, 1)
        else:
            col = (1, 1, 1, 1)
        if 0.01 < abs(prog) < 0.999:
            blink = 0.5 + 0.5 * math.sin(world.time * 10)
            c2 = TEAM_COLORS[0 if prog > 0 else 1]
            col = tuple(col[k] * (1 - blink) + c2[k] * 1.3 * blink for k in range(3)) + (1,)
        out.append(col)
    return out

"""Procedural 3D models: weapons, grenades, soldiers and map props.

Weapon frame: +Y = barrel direction, +Z = up, origin near the trigger.
"""
from __future__ import annotations

import math

import numpy as np
from panda3d.core import Material, NodePath, Point3, Quat, Vec3

from .meshes import MeshBuilder

SURFACES = {
    # name: (specular, shininess)
    "metal": (0.55, 38.0),
    "chrome": (0.9, 70.0),
    "wood": (0.12, 10.0),
    "poly": (0.2, 18.0),
    "cloth": (0.04, 4.0),
    "skin": (0.1, 8.0),
    "glass": (1.0, 110.0),
    "paint": (0.3, 24.0),
}


def make_material(surface: str) -> Material:
    spec, shin = SURFACES[surface]
    m = Material(surface)
    m.setSpecular((spec, spec, spec, 1.0))
    m.setShininess(shin)
    return m


class PartSet:
    """Groups mesh builders by surface type so each gets its own material."""

    def __init__(self, name: str):
        self.name = name
        self.builders: dict[str, MeshBuilder] = {}

    def b(self, surface: str) -> MeshBuilder:
        if surface not in self.builders:
            self.builders[surface] = MeshBuilder(f"{self.name}_{surface}")
        return self.builders[surface]

    def node(self) -> NodePath:
        root = NodePath(self.name)
        for surface, mb in self.builders.items():
            if mb.empty():
                continue
            np_ = mb.node()
            np_.setMaterial(make_material(surface), 1)
            np_.reparentTo(root)
        return root


def obox(mb: MeshBuilder, center, size, hpr=(0, 0, 0), color=(1, 1, 1, 1)):
    """Oriented box."""
    q = Quat()
    q.setHpr(Vec3(*hpr))
    c = Vec3(*center)
    sx, sy, sz = (s / 2 for s in size)
    corners = []
    for x, y, z in ((-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
                    (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz)):
        corners.append(c + q.xform(Vec3(x, y, z)))
    mb.hexa(corners, color)


def _trigger_group(m: MeshBuilder, dark, y0=0.0, y1=0.06):
    m.box((-0.004, y0, -0.032), (0.004, y1, -0.026), dark)
    m.box((-0.004, y1 - 0.007, -0.032), (0.004, y1, 0.0), dark)
    m.box((-0.0025, y0 + 0.02, -0.022), (0.0025, y0 + 0.028, 0.0), dark)


# ----------------------------------------------------------------------------- weapons
def build_ak47():
    ps = PartSet("ak47")
    m, w = ps.b("metal"), ps.b("wood")
    steel = (0.16, 0.16, 0.17, 1)
    dark = (0.07, 0.07, 0.075, 1)
    wood = (0.48, 0.24, 0.1, 1)
    woodd = (0.38, 0.18, 0.08, 1)
    m.box((-0.022, -0.11, 0.0), (0.022, 0.17, 0.062), steel)
    m.box((-0.019, -0.11, 0.062), (0.019, 0.15, 0.08), dark)
    m.box((-0.016, -0.13, 0.008), (0.016, -0.11, 0.07), dark)
    m.cylinder((0, 0.17, 0.045), (0, 0.62, 0.045), 0.0105, dark, 12)
    m.cylinder((0, 0.6, 0.045), (0, 0.665, 0.045), 0.0155, steel, 12)
    m.cylinder((0, 0.17, 0.078), (0, 0.42, 0.078), 0.011, steel, 12)
    m.box((-0.012, 0.4, 0.03), (0.012, 0.44, 0.09), steel)
    m.box((-0.01, 0.54, 0.03), (0.01, 0.575, 0.07), steel)
    m.box((-0.0022, 0.555, 0.07), (0.0022, 0.561, 0.099), dark)
    m.box((-0.0085, 0.553, 0.07), (-0.006, 0.565, 0.097), dark)
    m.box((0.006, 0.553, 0.07), (0.0085, 0.565, 0.097), dark)
    m.box((-0.012, 0.135, 0.08), (0.012, 0.175, 0.091), steel)
    m.box((-0.009, 0.15, 0.091), (-0.002, 0.168, 0.098), dark)
    m.box((0.002, 0.15, 0.091), (0.009, 0.168, 0.098), dark)
    w.box((-0.026, 0.17, 0.012), (0.026, 0.4, 0.066), wood)
    w.box((-0.017, 0.19, 0.066), (0.017, 0.4, 0.092), woodd)
    for k in range(4):  # handguard grooves
        y = 0.2 + k * 0.05
        m.box((-0.0265, y, 0.03), (0.0265, y + 0.006, 0.05), woodd)
    obox(w, (0, -0.035, -0.05), (0.028, 0.036, 0.11), (0, -20, 0), woodd)
    _trigger_group(m, dark)
    # curved magazine
    t = Vec3(0, 0.085, 0.004)
    for k in range(4):
        p = 8 + 9 * k
        d = Vec3(0, math.sin(math.radians(p)), -math.cos(math.radians(p)))
        seg = 0.06
        obox(m, t + d * (seg / 2), (0.026, 0.07, seg + 0.006), (0, p, 0), (0.42, 0.2, 0.08, 1) if k % 2 else (0.4, 0.19, 0.08, 1))
        t = t + d * seg
    w.tapered(-0.42, -0.13, 0.04, 0.135, 0.034, 0.062, -0.085, 0.0, wood)
    m.box((-0.021, -0.432, -0.087), (0.021, -0.42, 0.052), dark)
    m.box((0.022, 0.08, 0.05), (0.048, 0.095, 0.06), steel)
    m.box((0.019, -0.05, 0.02), (0.025, 0.1, 0.03), steel)
    node = ps.node()
    return node, {"muzzle": Point3(0, 0.67, 0.045), "sight_z": 0.1005, "grip": Point3(0, -0.035, -0.03),
                  "fore": Point3(0, 0.29, 0.02), "eject": Point3(0.03, 0.06, 0.06)}


def build_awm():
    ps = PartSet("awm")
    m, p = ps.b("metal"), ps.b("poly")
    g = (0.3, 0.34, 0.2, 1)
    gd = (0.24, 0.27, 0.16, 1)
    black = (0.05, 0.05, 0.055, 1)
    p.box((-0.03, -0.13, -0.005), (0.03, 0.26, 0.055), g)
    m.box((-0.022, -0.1, 0.055), (0.022, 0.15, 0.075), black)
    m.cylinder((0, 0.26, 0.04), (0, 0.88, 0.04), 0.0125, black, 14, radius1=0.0105)
    m.cylinder((0, 0.865, 0.04), (0, 0.955, 0.04), 0.019, black, 14)
    for k in range(3):
        m.box((-0.0195, 0.885 + k * 0.022, 0.028), (0.0195, 0.893 + k * 0.022, 0.052), (0.02, 0.02, 0.02, 1))
    p.tapered(0.26, 0.6, 0.06, 0.05, 0.05, 0.04, 0.0, 0.006, g)
    p.tapered(-0.5, -0.13, 0.045, 0.15, 0.055, 0.06, -0.09, -0.005, g)
    p.box((-0.021, -0.42, 0.055), (0.021, -0.2, 0.082), gd)
    m.box((-0.024, -0.52, -0.096), (0.024, -0.5, 0.065), black)
    obox(p, (0, -0.06, -0.055), (0.03, 0.042, 0.11), (0, -15, 0), gd)
    m.box((-0.018, 0.02, -0.06), (0.018, 0.1, -0.005), black)
    _trigger_group(m, black, -0.01, 0.05)
    m.cylinder((0.022, -0.06, 0.066), (0.075, -0.078, 0.06), 0.005, black, 8)
    m.sphere((0.079, -0.08, 0.06), 0.012, black, 6, 10)
    # scope
    m.cylinder((0, -0.08, 0.115), (0, 0.2, 0.115), 0.016, black, 16)
    m.cylinder((0, 0.17, 0.115), (0, 0.27, 0.115), 0.016, black, 16, radius1=0.027)
    m.cylinder((0, 0.27, 0.115), (0, 0.292, 0.115), 0.027, black, 16)
    m.cylinder((0, -0.152, 0.115), (0, -0.08, 0.115), 0.022, black, 16, radius1=0.016)
    m.cylinder((0, 0.06, 0.128), (0, 0.06, 0.152), 0.012, black, 12)
    m.cylinder((0.014, 0.06, 0.115), (0.04, 0.06, 0.115), 0.012, black, 12)
    m.box((-0.02, -0.04, 0.07), (0.02, -0.02, 0.118), black)
    m.box((-0.02, 0.12, 0.07), (0.02, 0.14, 0.118), black)
    gl = ps.b("glass")
    gl.cylinder((0, 0.292, 0.115), (0, 0.294, 0.115), 0.024, (0.15, 0.32, 0.5, 1), 16)
    gl.cylinder((0, -0.154, 0.115), (0, -0.152, 0.115), 0.019, (0.1, 0.2, 0.32, 1), 16)
    node = ps.node()
    return node, {"muzzle": Point3(0, 0.96, 0.04), "sight_z": 0.115, "grip": Point3(0, -0.06, -0.035),
                  "fore": Point3(0, 0.36, 0.0), "eject": Point3(0.03, 0.0, 0.07)}


def build_deagle():
    ps = PartSet("deagle")
    m, c, p = ps.b("metal"), ps.b("chrome"), ps.b("poly")
    silver = (0.5, 0.5, 0.52, 1)
    dark = (0.08, 0.08, 0.085, 1)
    c.box((-0.014, -0.035, 0.03), (0.014, 0.2, 0.068), silver)
    for i in range(6):
        y = -0.03 + i * 0.008
        m.box((-0.0148, y, 0.036), (0.0148, y + 0.003, 0.064), dark)
    c.box((-0.013, -0.03, 0.0), (0.013, 0.18, 0.03), (0.38, 0.38, 0.4, 1))
    m.cylinder((0, 0.2, 0.05), (0, 0.2015, 0.05), 0.007, (0.01, 0.01, 0.01, 1), 10)
    obox(p, (0, -0.022, -0.05), (0.03, 0.052, 0.115), (0, -14, 0), (0.06, 0.06, 0.06, 1))
    _trigger_group(m, dark, 0.0, 0.055)
    m.box((-0.003, 0.185, 0.068), (0.003, 0.195, 0.079), dark)
    m.box((-0.009, -0.03, 0.068), (-0.003, -0.02, 0.078), dark)
    m.box((0.003, -0.03, 0.068), (0.009, -0.02, 0.078), dark)
    m.box((-0.005, -0.048, 0.04), (0.005, -0.035, 0.06), dark)
    node = ps.node()
    return node, {"muzzle": Point3(0, 0.205, 0.05), "sight_z": 0.0795, "grip": Point3(0, -0.022, -0.03),
                  "fore": Point3(-0.01, -0.01, -0.05), "eject": Point3(0.02, 0.06, 0.06)}


def build_knife():
    ps = PartSet("knife")
    m, c, p = ps.b("metal"), ps.b("chrome"), ps.b("poly")
    blade = (0.72, 0.74, 0.77, 1)
    p.box((-0.012, -0.12, -0.016), (0.012, 0.0, 0.016), (0.07, 0.07, 0.07, 1))
    for i in range(5):
        y = -0.11 + i * 0.022
        p.box((-0.0128, y, -0.0165), (0.0128, y + 0.006, 0.0165), (0.12, 0.12, 0.12, 1))
    m.box((-0.008, 0.0, -0.026), (0.008, 0.012, 0.026), (0.3, 0.3, 0.32, 1))
    m.box((-0.01, -0.13, -0.014), (0.01, -0.12, 0.014), (0.3, 0.3, 0.32, 1))
    c.tapered(0.012, 0.15, 0.006, 0.032, 0.005, 0.024, -0.014, -0.008, blade)
    c.tapered(0.15, 0.205, 0.005, 0.024, 0.0012, 0.003, -0.008, 0.003, blade)
    node = ps.node()
    return node, {"muzzle": Point3(0, 0.2, 0.0), "sight_z": 0.0, "grip": Point3(0, -0.06, 0.0),
                  "fore": Point3(0, -0.06, 0.0), "eject": Point3(0, 0, 0)}


def build_frag():
    ps = PartSet("frag")
    m, p = ps.b("metal"), ps.b("paint")
    olive = (0.27, 0.32, 0.17, 1)
    p.sphere((0, 0, 0), 0.034, olive, 10, 14, scale=(1, 1, 1.15))
    for z in (-0.012, 0.012):
        p.cylinder((0, 0, z - 0.002), (0, 0, z + 0.002), 0.0345, (0.2, 0.24, 0.12, 1), 14)
    m.cylinder((0, 0, 0.034), (0, 0, 0.058), 0.011, (0.5, 0.5, 0.48, 1), 10)
    obox(m, (0.016, 0, 0.03), (0.007, 0.016, 0.07), (0, 0, -14), (0.5, 0.5, 0.48, 1))
    for dz, dy, sx, sz in ((0.072, 0, 0.003, 0.003), (0.052, 0, 0.003, 0.003)):
        m.box((-0.032, -0.012, dz - sz), (-0.02, 0.012, dz), (0.6, 0.6, 0.6, 1))
    m.box((-0.032, -0.012, 0.049), (-0.029, -0.009, 0.072), (0.6, 0.6, 0.6, 1))
    m.box((-0.032, 0.009, 0.049), (-0.029, 0.012, 0.072), (0.6, 0.6, 0.6, 1))
    return ps.node(), {}


def build_smoke():
    ps = PartSet("smoke")
    m, p = ps.b("metal"), ps.b("paint")
    p.cylinder((0, 0, -0.055), (0, 0, 0.05), 0.03, (0.32, 0.36, 0.32, 1), 16)
    p.cylinder((0, 0, 0.018), (0, 0, 0.034), 0.0305, (0.85, 0.85, 0.82, 1), 16)
    m.cylinder((0, 0, 0.05), (0, 0, 0.066), 0.016, (0.5, 0.5, 0.48, 1), 12)
    obox(m, (0.03, 0, 0.02), (0.006, 0.016, 0.08), (0, 0, -8), (0.5, 0.5, 0.48, 1))
    m.box((-0.03, -0.012, 0.06), (-0.018, 0.012, 0.064), (0.6, 0.6, 0.6, 1))
    return ps.node(), {}


def build_molotov():
    ps = PartSet("molotov")
    g, p, c = ps.b("glass"), ps.b("paint"), ps.b("cloth")
    glass = (0.45, 0.18, 0.05, 1)
    g.cylinder((0, 0, -0.055), (0, 0, 0.02), 0.024, glass, 14)
    g.cylinder((0, 0, 0.02), (0, 0, 0.055), 0.012, (0.35, 0.16, 0.05, 1), 12)
    p.cylinder((0, 0, -0.02), (0, 0, 0.012), 0.022, (0.85, 0.35, 0.08, 1), 12)
    p.box((-0.02, -0.004, -0.01), (0.02, 0.004, 0.012), (0.9, 0.75, 0.2, 1))
    c.cylinder((0, 0, 0.05), (0, 0, 0.078), 0.007, (0.25, 0.14, 0.06, 1), 8)
    p.box((-0.004, -0.004, 0.072), (0.004, 0.004, 0.095), (1.0, 0.45, 0.08, 1))
    return ps.node(), {}


WEAPON_BUILDERS = {"ak47": build_ak47, "awm": build_awm, "deagle": build_deagle,
                   "knife": build_knife, "frag": build_frag, "smoke": build_smoke, "molotov": build_molotov}


def _builder(name: str):
    if name in WEAPON_BUILDERS:
        return WEAPON_BUILDERS[name]
    from .guns import GUN_BUILDERS
    return GUN_BUILDERS[name]


class ModelCache:
    def __init__(self):
        self._cache = {}

    def weapon(self, name: str, skin: str | None = None):
        from .skins import apply_skin, get_skin
        sk = get_skin(name, skin)
        key = (name, sk.key)
        if key not in self._cache:
            node, info = _builder(name)()
            if sk.key == "default":
                node.flattenStrong()
            else:
                for child in node.getChildren():
                    child.flattenStrong()
                apply_skin(node, sk)
            self._cache[key] = (node, info)
        node, info = self._cache[key]
        return node.copyTo(NodePath(name)), info


# ----------------------------------------------------------------------------- soldier
TEAM_STYLE = {
    0: dict(shirt=(0.52, 0.46, 0.34, 1), pants=(0.47, 0.41, 0.31, 1), vest=(0.16, 0.16, 0.15, 1),
            head=(0.1, 0.1, 0.1, 1), helmet=None, accent=(0.85, 0.25, 0.1, 1), skin=(0.72, 0.55, 0.42, 1)),
    1: dict(shirt=(0.22, 0.28, 0.38, 1), pants=(0.19, 0.23, 0.31, 1), vest=(0.1, 0.12, 0.15, 1),
            head=(0.78, 0.6, 0.48, 1), helmet=(0.18, 0.22, 0.28, 1), accent=(0.15, 0.55, 0.95, 1),
            skin=(0.78, 0.6, 0.48, 1)),
}

GLOVE = (0.07, 0.07, 0.07, 1)
BOOT = (0.12, 0.09, 0.07, 1)


def _limb(name, length, w, d, color, surface="cloth"):
    """Unit-length (along +Y) limb mesh: scale Y to stretch."""
    ps = PartSet(name)
    ps.b(surface).box((-w / 2, 0.0, -d / 2), (w / 2, 1.0, d / 2), color)
    return ps.node()


class SoldierModel:
    """Articulated low-poly soldier. Feet at origin, facing +Y."""

    def __init__(self, team: int, models: ModelCache):
        st = TEAM_STYLE[team]
        self.style = st
        self.models = models
        self.root = NodePath(f"soldier_{team}")
        self.body = self.root.attachNewNode("body")
        self.pelvis = self.body.attachNewNode("pelvis")
        self.pelvis.setZ(0.95)
        ps = PartSet("pelvis")
        ps.b("cloth").box((-0.17, -0.1, -0.08), (0.17, 0.1, 0.1), st["pants"])
        ps.b("poly").box((-0.18, -0.11, 0.05), (0.18, 0.11, 0.1), (0.12, 0.1, 0.08, 1))  # belt
        ps.node().reparentTo(self.pelvis)

        self.spine = self.pelvis.attachNewNode("spine")
        self.spine.setZ(0.08)
        ps = PartSet("torso")
        c = ps.b("cloth")
        c.tapered(-0.11, 0.11, 0.32, 0.5, 0.32, 0.5, 0.0, 0.0, st["shirt"])
        c.box((-0.215, -0.12, 0.3), (0.215, 0.12, 0.5), st["shirt"])
        v = ps.b("poly")
        v.box((-0.2, -0.135, 0.06), (0.2, 0.135, 0.46), st["vest"])
        for k in range(3):
            v.box((-0.16 + k * 0.11, 0.135, 0.1), (-0.07 + k * 0.11, 0.17, 0.24), st["vest"])  # mag pouches
        v.box((-0.12, -0.19, 0.18), (0.12, -0.135, 0.42), (0.2, 0.2, 0.18, 1))  # back pack
        ps.b("paint").box((0.212, -0.06, 0.34), (0.226, 0.06, 0.4), st["accent"])  # armband
        ps.b("paint").box((-0.226, -0.06, 0.34), (-0.212, 0.06, 0.4), st["accent"])
        ps.node().reparentTo(self.spine)

        self.head = self.spine.attachNewNode("head")
        self.head.setZ(0.52)
        ps = PartSet("head")
        ps.b("skin").box((-0.045, -0.045, -0.02), (0.045, 0.045, 0.05), st["skin"])
        ps.b("cloth" if st["helmet"] is None else "skin").box((-0.1, -0.11, 0.04), (0.1, 0.11, 0.28), st["head"])
        if st["helmet"] is not None:
            ps.b("paint").sphere((0, 0, 0.22), 0.125, st["helmet"], 8, 14, scale=(1.0, 1.05, 0.75))
            ps.b("glass").box((-0.085, 0.105, 0.17), (0.085, 0.125, 0.22), (0.1, 0.12, 0.12, 1))
            ps.b("poly").box((-0.1, 0.1, 0.165), (0.1, 0.11, 0.225), (0.05, 0.05, 0.05, 1))
        else:
            ps.b("skin").box((-0.075, 0.105, 0.15), (0.075, 0.115, 0.2), st["skin"])  # eye slit
            ps.b("glass").box((-0.06, 0.112, 0.165), (-0.015, 0.118, 0.19), (0.05, 0.05, 0.05, 1))
            ps.b("glass").box((0.015, 0.112, 0.165), (0.06, 0.118, 0.19), (0.05, 0.05, 0.05, 1))
        ps.node().reparentTo(self.head)

        self.arms = self.spine.attachNewNode("arms")
        self.arms.setZ(0.42)
        self.gun_mount = self.arms.attachNewNode("gun_mount")
        self.limbs = {}
        for side in ("r", "l"):
            self.limbs[side + "_upper"] = _limb("upper", 1, 0.1, 0.1, st["shirt"])
            self.limbs[side + "_upper"].reparentTo(self.arms)
            self.limbs[side + "_fore"] = _limb("fore", 1, 0.085, 0.085, st["shirt"])
            self.limbs[side + "_fore"].reparentTo(self.arms)
            hand = PartSet("hand")
            hand.b("cloth").box((-0.04, -0.04, -0.045), (0.04, 0.05, 0.04), GLOVE)
            self.limbs[side + "_hand"] = hand.node()
            self.limbs[side + "_hand"].reparentTo(self.arms)

        self.legs = {}
        for side, x in (("r", 0.095), ("l", -0.095)):
            hip = self.body.attachNewNode(side + "_hip")
            hip.setPos(x, 0, 0.92)
            ps = PartSet("thigh")
            ps.b("cloth").box((-0.075, -0.08, -0.47), (0.075, 0.08, 0.02), st["pants"])
            ps.b("poly").box((-0.085, -0.02, -0.33), (0.085, 0.09, -0.22), st["vest"])  # thigh pouch / knee pad
            ps.node().reparentTo(hip)
            knee = hip.attachNewNode(side + "_knee")
            knee.setZ(-0.46)
            ps = PartSet("shin")
            ps.b("cloth").box((-0.065, -0.07, -0.36), (0.065, 0.07, 0.0), st["pants"])
            ps.b("poly").box((-0.07, -0.08, -0.08), (0.07, 0.085, 0.02), (0.1, 0.1, 0.1, 1))
            ps.b("poly").box((-0.07, -0.1, -0.46), (0.07, 0.16, -0.36), BOOT)
            ps.node().reparentTo(knee)
            self.legs[side] = (hip, knee)

        self.weapon_np = None
        self.offhand_np = None
        self.weapon_name = None
        self.weapon_skin = None
        self.phase = 0.0
        self.set_weapon("ak47")

    def _place_limb(self, upper, fore, shoulder, elbow, hand):
        for node, a, b in ((upper, shoulder, elbow), (fore, elbow, hand)):
            a, b = Point3(*a), Point3(*b)
            node.setPos(a)
            node.lookAt(b)
            node.setSy(max((b - a).length(), 1e-3))

    def set_weapon(self, name: str, skin: str | None = None):
        if (name, skin) == (self.weapon_name, self.weapon_skin):
            return
        if self.weapon_np is not None:
            self.weapon_np.removeNode()
        self.weapon_name = name
        self.weapon_skin = skin
        from ..weapons import SLOT_PRIMARY, WEAPONS
        node, info = self.models.weapon(name, skin)
        node.reparentTo(self.gun_mount)
        self.weapon_np = node
        if self.offhand_np is not None:
            self.offhand_np.removeNode()
            self.offhand_np = None
        spec = WEAPONS.get(name)
        style = "item"
        if spec is not None:
            if spec.kind == "gun":
                style = "dual" if spec.dual else ("long" if spec.slot == SLOT_PRIMARY else "pistol")
            elif name == "hammer":
                style = "long"
            elif spec.kind == "shield":
                style = "shield"
        grip = info.get("grip", Point3(0, 0, 0))
        if style == "long":
            hand_r = Point3(0.12, 0.3, -0.08)
        elif style in ("pistol", "shield"):
            hand_r = Point3(0.06, 0.42, 0.0)
        elif style == "dual":
            hand_r = Point3(0.14, 0.42, 0.0)
        else:
            hand_r = Point3(0.2, 0.3, -0.12)
        self.gun_mount.setPos(hand_r - grip)
        fore = info.get("fore", grip) + self.gun_mount.getPos()
        if style == "long":
            hand_l = fore
            elbow_l = Point3(-0.12, 0.2, -0.22)
        elif style == "pistol":
            hand_l = hand_r + Point3(-0.05, -0.01, -0.02)
            elbow_l = Point3(-0.18, 0.18, -0.2)
        elif style == "dual":
            hand_l = Point3(-0.14, 0.42, 0.0)
            elbow_l = Point3(-0.24, 0.16, -0.2)
            off, _ = self.models.weapon(name, skin)
            off.reparentTo(self.arms)
            off.setPos(hand_l - grip)
            self.offhand_np = off
        elif style == "shield":
            hand_l = Point3(-0.12, 0.38, -0.05)
            elbow_l = Point3(-0.24, 0.14, -0.2)
            self.gun_mount.setPos(hand_l - grip + Point3(0.08, 0.06, 0))
            hand_r = Point3(0.12, 0.3, -0.15)
        else:
            hand_l = Point3(-0.24, 0.12, -0.42)
            elbow_l = Point3(-0.25, 0.02, -0.2)
        self._place_limb(self.limbs["r_upper"], self.limbs["r_fore"], (0.21, 0.0, 0.0),
                         (0.26, 0.08, -0.24), hand_r)
        self._place_limb(self.limbs["l_upper"], self.limbs["l_fore"], (-0.21, 0.0, 0.0), elbow_l, hand_l)
        self.limbs["r_hand"].setPos(hand_r)
        self.limbs["l_hand"].setPos(hand_l)
        if name in ("frag", "smoke", "flash", "molotov"):
            self.gun_mount.setPos(hand_r + Point3(0, 0, 0.03))

    def pose(self, dt: float, speed: float, crouch: float, pitch: float, on_ground: bool, lean: float = 0.0):
        """crouch in [0,1]; pitch in degrees; lean in [-1, 1] (+ = right)."""
        self.spine.setR(lean * 16.0)
        self.pelvis.setX(lean * 0.05)
        self.phase += dt * (speed * 2.1 + 0.0001)
        amp = min(1.0, speed / 5.0) if on_ground else 0.3
        hip_z = 0.92 - 0.3 * crouch
        bend = 49.0 * crouch
        swing = math.sin(self.phase) * 32.0 * amp * (1 - 0.6 * crouch)
        for side, sgn in (("r", 1.0), ("l", -1.0)):
            hip, knee = self.legs[side]
            hip.setZ(hip_z)
            s = swing * sgn
            hip.setP(bend + s)
            knee_bend = max(0.0, -math.sin(self.phase * 1.0 * sgn + (0.0 if sgn > 0 else math.pi)) ) * 40 * amp
            knee.setP(-2 * bend - knee_bend * (1 - crouch))
        if not on_ground:
            for side in ("r", "l"):
                hip, knee = self.legs[side]
                hip.setP(25)
                knee.setP(-50)
        self.pelvis.setZ(hip_z + 0.03 + abs(math.sin(self.phase)) * 0.02 * amp)
        self.spine.setP(pitch * 0.35 + 8 * crouch)
        self.arms.setP(pitch * 0.65 - 8 * crouch)
        self.head.setP(pitch * 0.3)


# ----------------------------------------------------------------------------- props
def build_barrel(color: str) -> NodePath:
    cols = {"blue": (0.12, 0.25, 0.55, 1), "red": (0.6, 0.12, 0.08, 1)}
    c = cols.get(color, (0.3, 0.3, 0.3, 1))
    ps = PartSet("barrel")
    p = ps.b("paint")
    p.cylinder((0, 0, 0), (0, 0, 0.9), 0.29, c, 20)
    for z in (0.02, 0.3, 0.6, 0.88):
        p.cylinder((0, 0, z - 0.012), (0, 0, z + 0.012), 0.3, (c[0] * 0.7, c[1] * 0.7, c[2] * 0.7, 1), 20)
    p.cylinder((0.12, 0, 0.9), (0.12, 0, 0.915), 0.03, (0.4, 0.4, 0.4, 1), 8)
    return ps.node()


def build_lamp() -> tuple[NodePath, NodePath]:
    ps = PartSet("lamp")
    m = ps.b("paint")
    gray = (0.55, 0.56, 0.55, 1)
    m.cylinder((0, 0, 0), (0, 0, 2.6), 0.045, gray, 10)
    m.cylinder((0, 0, 2.6), (0, -0.7, 2.75), 0.035, gray, 8)
    m.box((-0.14, -0.95, 2.66), (0.14, -0.6, 2.8), (0.25, 0.25, 0.25, 1))
    node = ps.node()
    bulb = MeshBuilder("bulb")
    bulb.box((-0.12, -0.93, 2.645), (0.12, -0.62, 2.665), (1.0, 0.92, 0.7, 1))
    bulb_np = bulb.node()
    bulb_np.reparentTo(node)
    return node, bulb_np


def build_crane_boom() -> NodePath:
    ps = PartSet("crane")
    m = ps.b("paint")
    y = (0.85, 0.65, 0.1, 1)
    m.box((-0.9, -0.9, 0), (0.9, 0.9, 1.8), (0.8, 0.6, 0.1, 1))
    m.box((-0.6, 0.9, 0.3), (0.6, 1.9, 1.5), (0.85, 0.85, 0.82, 1))  # cab
    gl = ps.b("glass")
    gl.box((-0.5, 1.9, 0.8), (0.5, 1.92, 1.4), (0.1, 0.15, 0.2, 1))
    # lattice boom along +Y, raised 25 degrees
    L = 16.0
    ang = math.radians(22)
    for sx in (-0.35, 0.35):
        for sz in (0.0, 0.6):
            p0 = (sx, 0.5, 1.2 + sz)
            p1 = (sx, 0.5 + L * math.cos(ang), 1.2 + sz + L * math.sin(ang))
            m.cylinder(p0, p1, 0.06, y, 8)
    for k in range(1, 16):
        t = k / 16
        yy = 0.5 + L * math.cos(ang) * t
        zz = 1.2 + L * math.sin(ang) * t
        m.cylinder((-0.35, yy, zz), (0.35, yy, zz + 0.6), 0.03, y, 6)
        m.cylinder((0.35, yy, zz), (-0.35, yy, zz + 0.6), 0.03, y, 6)
    m.cylinder((0, 0.5 + L * math.cos(ang), 1.2 + L * math.sin(ang)),
               (0, 0.5 + L * math.cos(ang), -4.0), 0.012, (0.1, 0.1, 0.1, 1), 4)
    m.box((-0.2, 0.3 + L * math.cos(ang), -4.4), (0.2, 0.7 + L * math.cos(ang), -4.0), (0.8, 0.2, 0.1, 1))
    return ps.node()


def build_antenna() -> NodePath:
    ps = PartSet("antenna")
    m = ps.b("paint")
    w = (0.85, 0.85, 0.82, 1)
    m.cylinder((0, 0, 0), (0, 0, 6.0), 0.08, w, 8)
    for z, l in ((3.0, 1.6), (4.5, 1.2), (5.6, 0.8)):
        m.cylinder((-l, 0, z), (l, 0, z), 0.04, w, 6)
    m.sphere((0, 0, 6.1), 0.15, (0.9, 0.15, 0.1, 1), 6, 8)
    return ps.node()


def build_radar() -> tuple[NodePath, NodePath]:
    ps = PartSet("radar")
    m = ps.b("paint")
    m.cylinder((0, 0, 0), (0, 0, 1.6), 0.18, (0.85, 0.85, 0.82, 1), 10)
    node = ps.node()
    bar = PartSet("radar_bar")
    bar.b("paint").box((-1.6, -0.12, -0.08), (1.6, 0.12, 0.08), (0.9, 0.9, 0.88, 1))
    bar_np = bar.node()
    bar_np.reparentTo(node)
    bar_np.setZ(1.7)
    return node, bar_np


def build_tree(seed: int = 0, autumn: bool = False) -> NodePath:
    rng = np.random.default_rng(seed)
    ps = PartSet("tree")
    ps.b("wood").cylinder((0, 0, 0), (0, 0, 2.4), 0.16, (0.32, 0.22, 0.14, 1), 10, radius1=0.11)
    leaf = ps.b("cloth")
    for _ in range(5):
        if autumn:
            c = (0.82 + rng.uniform(0, 0.1), 0.6 + rng.uniform(0, 0.12), 0.12, 1)
        else:
            c = (0.18 + rng.uniform(0, 0.08), 0.32 + rng.uniform(0, 0.12), 0.1, 1)
        leaf.sphere((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 2.6 + rng.uniform(0, 1.2)),
                    rng.uniform(0.8, 1.2), c, 6, 9)
    return ps.node()


def build_palm(seed: int = 0) -> NodePath:
    rng = np.random.default_rng(seed)
    ps = PartSet("palm")
    trunk = ps.b("wood")
    lean = rng.uniform(-0.5, 0.5, 2)
    top = (lean[0], lean[1], 5.5)
    trunk.cylinder((0, 0, 0), top, 0.18, (0.48, 0.38, 0.26, 1), 8, radius1=0.12)
    leaf = ps.b("cloth")
    for k in range(8):
        a = k * math.pi / 4 + rng.uniform(-0.2, 0.2)
        end = (top[0] + math.cos(a) * 2.2, top[1] + math.sin(a) * 2.2, top[2] - 0.9)
        mid = (top[0] + math.cos(a) * 1.1, top[1] + math.sin(a) * 1.1, top[2] + 0.2)
        g = (0.2, 0.38 + rng.uniform(0, 0.1), 0.12, 1)
        leaf.cylinder(top, mid, 0.22, g, 4, radius1=0.18)
        leaf.cylinder(mid, end, 0.18, g, 4, radius1=0.02)
    return ps.node()


def build_car(color=(0.6, 0.12, 0.1)) -> NodePath:
    ps = PartSet("car")
    p = ps.b("paint")
    c = (*color, 1)
    p.box((-2.1, -0.85, 0.3), (2.1, 0.85, 0.95), c)
    p.box((-1.1, -0.78, 0.95), (1.0, 0.78, 1.45), c)
    ps.b("glass").box((-1.05, -0.8, 1.0), (0.95, 0.8, 1.4), (0.1, 0.13, 0.18, 1))
    dark = (0.06, 0.06, 0.06, 1)
    for x in (-1.35, 1.35):
        for y in (-0.82, 0.82):
            ps.b("poly").cylinder((x, y - 0.1, 0.32), (x, y + 0.1, 0.32), 0.32, dark, 12)
    p.box((2.1, -0.7, 0.6), (2.14, -0.4, 0.75), (1.0, 0.95, 0.8, 1))
    p.box((2.1, 0.4, 0.6), (2.14, 0.7, 0.75), (1.0, 0.95, 0.8, 1))
    return ps.node()


def build_ceiling_light() -> tuple[NodePath, NodePath]:
    ps = PartSet("clight")
    ps.b("metal").box((-0.6, -0.15, 0.0), (0.6, 0.15, 0.06), (0.6, 0.6, 0.6, 1))
    node = ps.node()
    bulb = MeshBuilder("cbulb")
    bulb.box((-0.55, -0.11, -0.01), (0.55, 0.11, 0.0), (1.0, 0.97, 0.88, 1))
    b = bulb.node()
    b.reparentTo(node)
    return node, b


def build_desk() -> NodePath:
    ps = PartSet("desk")
    w = ps.b("wood")
    col = (0.42, 0.28, 0.16, 1)
    w.box((-0.8, -0.4, 0.72), (0.8, 0.4, 0.77), col)
    w.box((-0.78, -0.38, 0.0), (-0.7, 0.38, 0.72), col)
    w.box((0.7, -0.38, 0.0), (0.78, 0.38, 0.72), col)
    ps.b("poly").box((-0.25, -0.05, 0.77), (0.25, 0.02, 1.1), (0.08, 0.08, 0.09, 1))
    return ps.node()


def build_door() -> NodePath:
    ps = PartSet("door")
    m = ps.b("paint")
    m.box((0.0, -0.65, 0.0), (0.06, 0.65, 2.2), (0.3, 0.33, 0.35, 1))
    m.box((0.06, -0.55, 0.1), (0.09, 0.55, 2.05), (0.4, 0.43, 0.45, 1))
    m.box((0.09, 0.3, 1.0), (0.14, 0.42, 1.06), (0.7, 0.7, 0.7, 1))
    ps.b("glass").box((0.09, -0.25, 1.4), (0.1, 0.25, 1.85), (0.1, 0.14, 0.2, 1))
    return ps.node()

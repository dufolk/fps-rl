"""Procedural 3D models: weapons, grenades and soldiers.

Weapon frame: +Y = barrel direction, +Z = up, origin near the trigger.
"""
from __future__ import annotations

import math

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


_BLACK = (0.07, 0.07, 0.075, 1)
_DARK = (0.12, 0.12, 0.13, 1)
_STEEL = (0.2, 0.2, 0.21, 1)


def _info(muzzle, sight_z, grip=(0, -0.035, -0.03), fore=(0, 0.25, 0.0), eject=(0.03, 0.05, 0.06)):
    return {"muzzle": Point3(*muzzle), "sight_z": sight_z, "grip": Point3(*grip), "fore": Point3(*fore),
            "eject": Point3(*eject)}


def build_glock():
    ps = PartSet("glock")
    m, p = ps.b("metal"), ps.b("poly")
    poly = (0.13, 0.13, 0.14, 1)
    slide = (0.18, 0.18, 0.19, 1)
    m.box((-0.013, -0.025, 0.03), (0.013, 0.15, 0.056), slide)          # squared slide
    for i in range(4):
        y = -0.022 + i * 0.006
        m.box((-0.0136, y, 0.034), (0.0136, y + 0.002, 0.052), _BLACK)  # rear serrations
    m.cylinder((0, 0.15, 0.043), (0, 0.172, 0.043), 0.006, _BLACK, 10)  # barrel
    p.box((-0.014, -0.01, 0.0), (0.014, 0.135, 0.032), poly)            # frame
    p.box((-0.01, 0.07, -0.008), (0.01, 0.125, 0.0), _BLACK)            # accessory rail
    obox(p, (0, -0.012, -0.048), (0.028, 0.042, 0.1), (0, -20, 0), poly)
    p.box((-0.015, -0.028, -0.105), (0.015, 0.012, -0.092), _BLACK)     # magazine floorplate
    _trigger_group(m, _BLACK, 0.03, 0.075)
    m.box((-0.003, 0.155, 0.056), (0.003, 0.163, 0.066), _BLACK)        # front sight
    m.box((-0.011, -0.012, 0.056), (-0.004, -0.002, 0.064), _BLACK)     # rear sight notch
    m.box((0.004, -0.012, 0.056), (0.011, -0.002, 0.064), _BLACK)
    return ps.node(), _info((0, 0.175, 0.043), 0.066, grip=(0, -0.012, -0.03), fore=(-0.01, 0.04, -0.04),
                            eject=(0.016, 0.05, 0.05))


def build_nova():
    ps = PartSet("nova")
    m, w = ps.b("metal"), ps.b("wood")
    wood = (0.38, 0.22, 0.11, 1)
    m.box((-0.024, -0.1, 0.0), (0.024, 0.14, 0.068), _BLACK)             # receiver
    m.box((0.024, -0.02, 0.025), (0.03, 0.08, 0.055), _STEEL)            # loading / ejection port
    m.cylinder((0, 0.14, 0.05), (0, 0.66, 0.05), 0.013, _BLACK, 14)      # barrel
    m.cylinder((0, 0.14, 0.018), (0, 0.6, 0.018), 0.012, _DARK, 12)      # magazine tube
    m.cylinder((0, 0.6, 0.018), (0, 0.615, 0.018), 0.014, _BLACK, 12)    # tube cap
    m.box((-0.006, 0.585, 0.024), (0.006, 0.62, 0.05), _BLACK)           # barrel clamp
    m.box((-0.003, 0.64, 0.063), (0.003, 0.65, 0.074), (0.75, 0.6, 0.2, 1))  # bead sight
    m.box((-0.01, -0.08, 0.068), (0.01, 0.12, 0.076), _DARK)             # top rib
    w.box((-0.026, 0.2, -0.004), (0.026, 0.36, 0.04), wood)              # pump
    for k in range(5):
        y = 0.215 + k * 0.03
        w.box((-0.0275, y, 0.0), (0.0275, y + 0.01, 0.036), (0.3, 0.17, 0.08, 1))
    obox(w, (0, -0.035, -0.05), (0.03, 0.038, 0.11), (0, -22, 0), wood)  # pistol grip
    _trigger_group(m, _BLACK)
    w.tapered(-0.42, -0.1, 0.045, 0.125, 0.04, 0.07, -0.08, -0.005, wood)  # stock
    m.box((-0.024, -0.43, -0.08), (0.024, -0.415, 0.05), _BLACK)         # recoil pad
    return ps.node(), _info((0, 0.665, 0.05), 0.076, fore=(0, 0.28, -0.0), eject=(0.035, 0.03, 0.04))


def build_flash():
    ps = PartSet("flash")
    m, p = ps.b("metal"), ps.b("paint")
    p.cylinder((0, 0, -0.055), (0, 0, 0.05), 0.028, (0.25, 0.27, 0.3, 1), 16)
    for z in (-0.03, 0.0, 0.03):
        p.cylinder((0, 0, z - 0.003), (0, 0, z + 0.003), 0.029, (0.12, 0.14, 0.16, 1), 16)
    m.cylinder((0, 0, 0.05), (0, 0, 0.066), 0.015, (0.5, 0.5, 0.48, 1), 12)
    obox(m, (0.028, 0, 0.02), (0.006, 0.016, 0.08), (0, 0, -8), (0.5, 0.5, 0.48, 1))
    return ps.node(), {}


WEAPON_BUILDERS = {"ak47": build_ak47, "awm": build_awm, "nova": build_nova, "glock": build_glock,
                   "knife": build_knife, "frag": build_frag, "smoke": build_smoke, "flash": build_flash,
                   "molotov": build_molotov}


class ModelCache:
    def __init__(self):
        self._cache = {}

    def weapon(self, name: str):
        if name not in self._cache:
            node, info = WEAPON_BUILDERS[name]()
            node.flattenStrong()
            self._cache[name] = (node, info)
        node, info = self._cache[name]
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

    def __init__(self, team: int, models: ModelCache, style: dict | None = None):
        st = style or TEAM_STYLE[team]
        gear = st.get("gear", True)  # vest, pouches, back pack, armbands
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
        if gear:
            v = ps.b("poly")
            v.box((-0.2, -0.135, 0.06), (0.2, 0.135, 0.46), st["vest"])
            for k in range(3):
                v.box((-0.16 + k * 0.11, 0.135, 0.1), (-0.07 + k * 0.11, 0.17, 0.24), st["vest"])  # mag pouches
            v.box((-0.12, -0.19, 0.18), (0.12, -0.135, 0.42), (0.2, 0.2, 0.18, 1))  # back pack
            ps.b("paint").box((0.212, -0.06, 0.34), (0.226, 0.06, 0.4), st["accent"])  # armband
            ps.b("paint").box((-0.226, -0.06, 0.34), (-0.212, 0.06, 0.4), st["accent"])
        else:  # torn shirt: bare skin showing through rips on the chest and back
            sk = ps.b("skin")
            for x0, z0, z1 in ((-0.16, 0.12, 0.3), (0.02, 0.2, 0.42), (0.1, 0.06, 0.2)):
                sk.box((x0, 0.105, z0), (x0 + 0.1, 0.125, z1), st["skin"])
                sk.box((-x0 - 0.1, -0.125, z0), (-x0, -0.105, z1), st["skin"])
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
            hand.b("cloth" if gear else "skin").box((-0.04, -0.04, -0.045), (0.04, 0.05, 0.04),
                                                    GLOVE if gear else st["skin"])
            self.limbs[side + "_hand"] = hand.node()
            self.limbs[side + "_hand"].reparentTo(self.arms)

        self.legs = {}
        for side, x in (("r", 0.095), ("l", -0.095)):
            hip = self.body.attachNewNode(side + "_hip")
            hip.setPos(x, 0, 0.92)
            ps = PartSet("thigh")
            ps.b("cloth").box((-0.075, -0.08, -0.47), (0.075, 0.08, 0.02), st["pants"])
            if gear:
                ps.b("poly").box((-0.085, -0.02, -0.33), (0.085, 0.09, -0.22), st["vest"])  # thigh pouch
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
        self.weapon_name = None
        self.phase = 0.0
        self.set_weapon("ak47")

    def _place_limb(self, upper, fore, shoulder, elbow, hand):
        for node, a, b in ((upper, shoulder, elbow), (fore, elbow, hand)):
            a, b = Point3(*a), Point3(*b)
            node.setPos(a)
            node.lookAt(b)
            node.setSy(max((b - a).length(), 1e-3))

    def set_weapon(self, name: str):
        if name == self.weapon_name:
            return
        if self.weapon_np is not None:
            self.weapon_np.removeNode()
        self.weapon_name = name
        from ..weapons import SLOT_PRIMARY, WEAPONS
        node, info = self.models.weapon(name)
        node.reparentTo(self.gun_mount)
        self.weapon_np = node
        spec = WEAPONS.get(name)
        style = "item"
        if spec is not None and spec.kind == "gun":
            style = "long" if spec.slot == SLOT_PRIMARY else "pistol"
        grip = info.get("grip", Point3(0, 0, 0))
        if style == "long":
            hand_r = Point3(0.12, 0.3, -0.08)
        elif style == "pistol":
            hand_r = Point3(0.06, 0.42, 0.0)
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


def body_model(team: int, models: ModelCache) -> SoldierModel:
    return SoldierModel(team, models)

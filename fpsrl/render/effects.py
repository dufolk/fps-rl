"""Visual effects: particles, tracers, decals, muzzle flashes, explosions, smoke."""
from __future__ import annotations

import math
import random

from panda3d.core import (ColorBlendAttrib, NodePath, PointLight, Point3, TransparencyAttrib, Vec3,
                          Vec4)

from .meshes import MeshBuilder
from .world_scene import SHADOW_MASK


def _card(name="card"):
    mb = MeshBuilder(name)
    mb.quad((-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5),
            uvs=((0, 0), (1, 0), (1, 1), (0, 1)))
    return mb.node()


def _additive(np_):
    np_.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.M_add, ColorBlendAttrib.O_incoming_alpha,
                                        ColorBlendAttrib.O_one))
    np_.setDepthWrite(False)
    np_.setBin("fixed", 10)


class _P:
    __slots__ = ("node", "card", "vel", "age", "life", "s0", "s1", "c0", "c1", "grav", "drag", "spin",
                 "hold", "active")


class ParticleSystem:
    def __init__(self, parent, texture, additive: bool, count: int, name: str = "ps"):
        self.root = parent.attachNewNode(name)
        self.root.setLightOff(1)
        self.root.setShaderOff(1)
        self.root.hide(SHADOW_MASK)
        self.root.setTexture(texture, 1)
        if additive:
            _additive(self.root)
            self.root.setFogOff(1)
        else:
            self.root.setTransparency(TransparencyAttrib.M_alpha)
            self.root.setDepthWrite(False)
        proto = _card(name)
        self.pool = []
        for _ in range(count):
            p = _P()
            p.node = self.root.attachNewNode("p")
            p.node.setBillboardPointEye()
            p.card = proto.copyTo(p.node)
            p.node.hide()
            p.active = False
            self.pool.append(p)
        self.free = list(self.pool)
        self.live = []

    def emit(self, pos, vel=(0, 0, 0), life=1.0, size0=0.2, size1=None, c0=(1, 1, 1, 1), c1=None,
             grav=0.0, drag=0.0, spin=0.0, hold=0.0):
        if not self.free:
            p = self.live.pop(0)
        else:
            p = self.free.pop()
        p.node.setPos(*pos)
        p.vel = Vec3(*vel)
        p.age = 0.0
        p.life = max(life, 1e-3)
        p.s0 = size0
        p.s1 = size0 if size1 is None else size1
        p.c0 = Vec4(*c0)
        p.c1 = Vec4(*(c0 if c1 is None else c1))
        p.grav = grav
        p.drag = drag
        p.spin = spin
        p.hold = hold
        p.card.setR(random.uniform(0, 360))
        p.node.setScale(size0)
        p.node.setColorScale(p.c0)
        p.node.show()
        p.active = True
        self.live.append(p)
        return p

    def update(self, dt: float):
        keep = []
        for p in self.live:
            p.age += dt
            if p.age >= p.life:
                p.node.hide()
                p.active = False
                self.free.append(p)
                continue
            t = p.age / p.life
            if p.drag:
                p.vel *= max(0.0, 1.0 - p.drag * dt)
            p.vel.z -= p.grav * dt
            p.node.setPos(p.node.getPos() + p.vel * dt)
            p.node.setScale(p.s0 + (p.s1 - p.s0) * min(1.0, t * 3.0 if p.hold else t))
            tc = max(0.0, (t - p.hold) / (1.0 - p.hold)) if p.hold else t
            p.node.setColorScale(p.c0 + (p.c1 - p.c0) * tc)
            if p.spin:
                p.card.setR(p.card.getR() + p.spin * dt)
            keep.append(p)
        self.live = keep

    def clear(self):
        for p in self.live:
            p.node.hide()
            self.free.append(p)
        self.live = []


class TracerSystem:
    def __init__(self, parent, texture, count=48):
        self.root = parent.attachNewNode("tracers")
        self.root.setLightOff(1)
        self.root.setShaderOff(1)
        self.root.hide(SHADOW_MASK)
        self.root.setTexture(texture, 1)
        _additive(self.root)
        mb = MeshBuilder("tracer")
        w = 0.5
        mb.quad((-w, 0, 0), (w, 0, 0), (w, 1, 0), (-w, 1, 0), uvs=((0, 0), (1, 0), (1, 1), (0, 1)))
        mb.quad((w, 0, 0), (-w, 0, 0), (-w, 1, 0), (w, 1, 0), uvs=((1, 0), (0, 0), (0, 1), (1, 1)))
        mb.quad((0, 0, -w), (0, 0, w), (0, 1, w), (0, 1, -w), uvs=((0, 0), (1, 0), (1, 1), (0, 1)))
        mb.quad((0, 0, w), (0, 0, -w), (0, 1, -w), (0, 1, w), uvs=((1, 0), (0, 0), (0, 1), (1, 1)))
        proto = mb.node()
        self.pool = []
        for _ in range(count):
            n = proto.copyTo(self.root)
            n.hide()
            self.pool.append([n, None, None, 0.0, 0.0, False, 0.0, 0.0])
        self.idx = 0

    def fire(self, start, end, speed=420.0, length=3.0, width=0.025, color=(1, 0.85, 0.5, 1)):
        s, e = Point3(*start), Point3(*end)
        d = e - s
        dist = d.length()
        if dist < 1.0:
            return
        slot = self.pool[self.idx]
        self.idx = (self.idx + 1) % len(self.pool)
        n = slot[0]
        n.setPos(s)
        n.lookAt(e)
        n.setScale(width, min(length, dist), width)
        n.setColorScale(*color)
        n.show()
        slot[:] = [n, s, d / dist, dist, 0.0, True, speed, length]

    def update(self, dt):
        for slot in self.pool:
            if not slot[5]:
                continue
            n, s, d, dist, trav, _ = slot[:6]
            speed, length = slot[6], slot[7]
            trav += speed * dt
            slot[4] = trav
            if trav - length > dist:
                n.hide()
                slot[5] = False
                continue
            head = min(trav, dist)
            tail = max(0.0, trav - length)
            n.setPos(s + d * tail)
            n.setSy(max(0.01, head - tail))


class DecalSystem:
    def __init__(self, parent, texture, count=160, size=0.09, name="decals"):
        self.root = parent.attachNewNode(name)
        self.root.setTexture(texture, 1)
        self.root.setTransparency(TransparencyAttrib.M_alpha)
        self.root.setDepthWrite(False)
        self.root.setDepthOffset(2)
        self.root.setLightOff(1)
        self.root.setShaderOff(1)
        self.root.hide(SHADOW_MASK)
        self.root.setBin("fixed", 5)
        proto = _card(name)
        self.pool = []
        for _ in range(count):
            n = proto.copyTo(self.root)
            n.hide()
            self.pool.append(n)
        self.idx = 0
        self.size = size

    def place(self, point, normal, size=None, color=(1, 1, 1, 1)):
        n = self.pool[self.idx]
        self.idx = (self.idx + 1) % len(self.pool)
        p = Point3(*point)
        nv = Vec3(*normal)
        if nv.length() < 1e-6:
            return
        nv.normalize()
        n.setPos(p + nv * 0.012)
        up = Vec3(0, 0, 1) if abs(nv.z) < 0.9 else Vec3(0, 1, 0)
        n.lookAt(p - nv, up)
        n.setR(random.uniform(0, 360))
        n.setScale(size or self.size)
        n.setColorScale(*color)
        n.show()

    def clear(self):
        for n in self.pool:
            n.hide()


class FlashLights:
    """Small pool of always-attached point lights (avoids shader regeneration)."""

    def __init__(self, render, count=2):
        self.lights = []
        for i in range(count):
            pl = PointLight(f"flash{i}")
            pl.setColor(Vec4(0, 0, 0, 1))
            pl.setAttenuation(Vec3(1.0, 0.0, 0.35))
            np_ = render.attachNewNode(pl)
            render.setLight(np_)
            self.lights.append([np_, pl, 0.0, 0.0, Vec4(0, 0, 0, 1)])
        self.idx = 0

    def flash(self, pos, color=(4.0, 2.6, 1.2), duration=0.06, atten=0.35):
        slot = self.lights[self.idx]
        self.idx = (self.idx + 1) % len(self.lights)
        slot[0].setPos(*pos)
        slot[1].setAttenuation(Vec3(1.0, 0.0, atten))
        slot[2] = duration
        slot[3] = duration
        slot[4] = Vec4(*color, 1)

    def update(self, dt):
        for slot in self.lights:
            if slot[2] <= 0:
                continue
            slot[2] -= dt
            k = max(0.0, slot[2] / slot[3])
            c = slot[4] * k
            c.w = 1
            slot[1].setColor(c)


class Effects:
    def __init__(self, base, textures):
        self.base = base
        root = base.render.attachNewNode("effects")
        self.root = root
        self.sparks = ParticleSystem(root, textures.sprite("spark"), True, 260, "sparks")
        self.fire = ParticleSystem(root, textures.sprite("smoke"), True, 90, "fire")
        self.flash = ParticleSystem(root, textures.sprite("flash"), True, 24, "flash")
        self.dust = ParticleSystem(root, textures.sprite("smoke"), False, 200, "dust")
        self.smoke = ParticleSystem(root, textures.sprite("smoke"), False, 260, "smoke")
        self.blood = ParticleSystem(root, textures.sprite("blood"), False, 80, "blood")
        self.tracers = TracerSystem(root, textures.sprite("tracer"))
        self.holes = DecalSystem(root, textures.sprite("hole"), 200, 0.085, "holes")
        self.scorches = DecalSystem(root, textures.sprite("scorch"), 12, 3.2, "scorch")
        self.lights = FlashLights(base.render, 2)
        self.smoke_clouds = {}

    def clear(self):
        for ps in (self.sparks, self.fire, self.flash, self.dust, self.smoke, self.blood):
            ps.clear()
        self.holes.clear()
        self.scorches.clear()
        self.smoke_clouds.clear()

    # ------------------------------------------------------------------ events
    def muzzle(self, pos, direction, big=False, light=True):
        d = Vec3(*direction)
        s = 0.55 if big else 0.38
        self.flash.emit(Point3(*pos) + d * 0.08, life=0.05, size0=s, size1=s * 1.3, c0=(1, 0.9, 0.7, 1),
                        c1=(1, 0.6, 0.2, 0))
        for _ in range(2):
            self.dust.emit(Point3(*pos) + d * 0.2, d * random.uniform(0.5, 1.5) + Vec3(0, 0, 0.4), 0.6,
                           0.12, 0.6, (0.7, 0.7, 0.7, 0.25), (0.7, 0.7, 0.7, 0.0), drag=2.0)
        if light:
            self.lights.flash(Point3(*pos) + d * 0.3, (3.2, 2.2, 1.0) if not big else (5, 3.4, 1.5), 0.05)

    def impact(self, point, normal, mat):
        p = Point3(*point)
        n = Vec3(*normal)
        metal = mat is not None and (mat.startswith("container") or mat in ("steel_plate", "deck", "hull",
                                                                          "bulwark", "parapet", "mast",
                                                                          "superstructure", "hatch"))
        if metal:
            for _ in range(random.randint(5, 9)):
                v = (n * random.uniform(1.5, 4.0) + Vec3(random.uniform(-2, 2), random.uniform(-2, 2),
                                                         random.uniform(0, 3)))
                self.sparks.emit(p, v, random.uniform(0.15, 0.4), 0.05, 0.02, (1, 0.85, 0.5, 1), (1, 0.4, 0.1, 0),
                                 grav=9.0)
        dust_c = (0.55, 0.45, 0.32) if (mat or "").startswith("crate") else (0.6, 0.6, 0.58)
        for _ in range(3):
            v = n * random.uniform(0.4, 1.2) + Vec3(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3),
                                                    random.uniform(0.0, 0.5))
            self.dust.emit(p + n * 0.05, v, random.uniform(0.5, 0.9), 0.08, random.uniform(0.35, 0.6),
                           (*dust_c, 0.55), (*dust_c, 0.0), drag=2.5)
        self.holes.place(p, n)

    def blood_hit(self, point, direction, head=False):
        p = Point3(*point)
        d = Vec3(*direction)
        for _ in range(4 if not head else 7):
            v = d * random.uniform(0.5, 2.0) + Vec3(random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6),
                                                    random.uniform(-0.2, 0.8))
            self.blood.emit(p, v, random.uniform(0.25, 0.5), 0.12, random.uniform(0.3, 0.5),
                            (0.75, 0.05, 0.03, 0.9), (0.4, 0.02, 0.02, 0.0), grav=4.0, drag=2.0)

    def tracer(self, start, end):
        self.tracers.fire(start, end)

    def explosion(self, pos):
        p = Point3(*pos)
        self.lights.flash(p + Vec3(0, 0, 0.6), (14, 8, 3), 0.35, atten=0.03)
        self.flash.emit(p + Vec3(0, 0, 0.4), life=0.12, size0=2.5, size1=6.0, c0=(1, 0.9, 0.7, 1),
                        c1=(1, 0.5, 0.1, 0))
        for _ in range(26):
            v = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.1, 1.2))
            v.normalize()
            self.fire.emit(p + Vec3(0, 0, 0.3), v * random.uniform(2, 7), random.uniform(0.3, 0.6),
                           random.uniform(0.6, 1.2), random.uniform(2.0, 3.2), (1, 0.62, 0.22, 0.9),
                           (0.6, 0.2, 0.05, 0.0), drag=4.0, spin=random.uniform(-60, 60))
        for _ in range(40):
            v = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.2, 1.5))
            v.normalize()
            self.sparks.emit(p + Vec3(0, 0, 0.2), v * random.uniform(6, 16), random.uniform(0.4, 1.0), 0.08, 0.03,
                             (1, 0.8, 0.4, 1), (1, 0.3, 0.05, 0), grav=12.0, drag=0.8)
        for _ in range(26):
            v = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.3, 1.0))
            v.normalize()
            self.dust.emit(p + Vec3(0, 0, 0.4), v * random.uniform(1.0, 4.0), random.uniform(1.8, 3.2),
                           random.uniform(1.0, 1.6), random.uniform(3.5, 5.5), (0.22, 0.2, 0.18, 0.75),
                           (0.35, 0.33, 0.3, 0.0), grav=-0.5, drag=1.8)
        self.scorches.place(Point3(p.x, p.y, p.z - 0.05), Vec3(0, 0, 1))

    def smoke_grenade(self, sid, pos, radius, duration):
        self.smoke_clouds[sid] = {"pos": Point3(*pos), "radius": radius, "t": 0.0, "dur": duration,
                                  "emitted": 0}

    def update(self, dt):
        for sid, c in list(self.smoke_clouds.items()):
            c["t"] += dt
            target = int(min(1.0, c["t"] / 1.8) * 46)
            while c["emitted"] < target:
                c["emitted"] += 1
                r = c["radius"] * 0.75
                off = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.15, 0.9))
                if off.length() > 1:
                    off.normalize()
                pos = c["pos"] + Vec3(off.x * r, off.y * r, max(0.6, off.z * r * 0.8 + 0.8))
                remaining = max(1.0, c["dur"] - c["t"] + random.uniform(-1.0, 0.5))
                shade = random.uniform(0.72, 0.9)
                self.smoke.emit(pos, Vec3(random.uniform(-0.15, 0.15), random.uniform(-0.15, 0.15), 0.05),
                                remaining, random.uniform(1.4, 2.0), random.uniform(3.6, 4.8),
                                (shade, shade, shade * 1.02, 0.95), (shade, shade, shade, 0.0),
                                drag=0.3, spin=random.uniform(-6, 6), hold=0.85)
            if c["t"] > c["dur"]:
                del self.smoke_clouds[sid]
        for ps in (self.sparks, self.fire, self.flash, self.dust, self.smoke, self.blood):
            ps.update(dt)
        self.tracers.update(dt)
        self.lights.update(dt)

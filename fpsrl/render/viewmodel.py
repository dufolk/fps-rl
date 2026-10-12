"""First-person weapon view (separate display region + camera)."""
from __future__ import annotations

import math
import random

from panda3d.core import (AmbientLight, AntialiasAttrib, Camera, DirectionalLight, NodePath, PerspectiveLens, Point3,
                          PointLight, Vec3, Vec4)

from ..weapons import WEAPONS
from .effects import _additive, _card
from .meshes import MeshBuilder
from .models import GLOVE, TEAM_STYLE, PartSet

HIP = {
    "ak47": (Point3(0.15, 0.62, -0.2), Vec3(2.0, 0, 0)),
    "awm": (Point3(0.15, 0.66, -0.205), Vec3(2.0, 0, 0)),
    "deagle": (Point3(0.13, 0.52, -0.17), Vec3(2.5, 0, 0)),
    "knife": (Point3(0.13, 0.4, -0.1), Vec3(40, 25, -15)),
    "frag": (Point3(0.13, 0.46, -0.105), Vec3(-15, 10, 0)),
    "smoke": (Point3(0.13, 0.46, -0.105), Vec3(-15, 10, 0)),
}
ADS_Y = {"ak47": 0.38, "awm": 0.3, "deagle": 0.42}

# inspect keyframes: (time, position offset, hpr offset)
INSPECT = {
    "gun": [(0.0, (0, 0, 0), (0, 0, 0)), (0.45, (-0.09, -0.05, 0.05), (48, 10, -32)),
            (1.5, (-0.1, -0.04, 0.06), (55, 14, -38)), (2.05, (-0.04, -0.07, 0.04), (-22, 26, 42)),
            (2.75, (-0.04, -0.07, 0.04), (-26, 30, 46)), (3.25, (0, 0, 0), (0, 0, 0))],
    "knife": [(0.0, (0, 0, 0), (0, 0, 0)), (0.4, (-0.06, 0.03, 0.05), (-30, 0, 70)),
              (1.2, (-0.06, 0.03, 0.06), (-36, 6, 80)), (1.65, (-0.05, 0.04, 0.08), (-30, 0, 440)),
              (2.45, (-0.06, 0.03, 0.06), (-36, 6, 446)), (3.0, (0, 0, 0), (0, 0, 360))],
    "grenade": [(0.0, (0, 0, 0), (0, 0, 0)), (0.3, (-0.04, 0, 0.02), (0, 0, -30)),
                (0.7, (-0.04, 0, 0.13), (180, 40, -30)), (1.05, (-0.04, 0, 0.02), (360, 0, -30)),
                (1.75, (-0.04, 0, 0.03), (370, 6, -34)), (2.2, (0, 0, 0), (360, 0, 0))],
}


def _inspect_pose(kind: str, t: float):
    keys = INSPECT[kind]
    for (t0, p0, h0), (t1, p1, h1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            k = (t - t0) / max(t1 - t0, 1e-6)
            k = k * k * (3 - 2 * k)
            return (Vec3(*(a + (b - a) * k for a, b in zip(p0, p1))),
                    Vec3(*(a + (b - a) * k for a, b in zip(h0, h1))))
    return Vec3(0, 0, 0), Vec3(0, 0, 0)


def _arm(ps, hand, elbow_dir, sleeve, length=0.42):
    """Glove at `hand` and a sleeve extending in `elbow_dir` (weapon frame)."""
    hand = Point3(*hand)
    d = Vec3(*elbow_dir)
    d.normalize()
    ps.b("cloth").cylinder(hand + d * 0.04, hand + d * 0.12, 0.028, GLOVE, 10, radius1=0.03)
    ps.b("cloth").cbox(hand + Vec3(0, 0.005, 0.0), (0.045, 0.06, 0.05), GLOVE)
    ps.b("cloth").cylinder(hand + d * 0.11, hand + d * length, 0.036, sleeve, 10, radius1=0.045)


class ViewModel:
    def __init__(self, base, models, team: int, fov: float = 56.0, skins: dict | None = None):
        self.base = base
        self.models = models
        self.team = team
        self.skins = dict(skins or {})
        self.guns = {}
        self.root = NodePath("vm_root")
        lens = PerspectiveLens()
        lens.setNearFar(0.01, 20.0)
        lens.setFov(fov)
        self.lens = lens
        cam_node = Camera("vm_cam", lens)
        self.cam = self.root.attachNewNode(cam_node)
        self.dr = base.win.makeDisplayRegion()
        self.dr.setSort(5)
        self.dr.setClearDepthActive(True)
        self.dr.setCamera(self.cam)
        sun = DirectionalLight("vm_sun")
        sun.setColor(Vec4(1.25, 1.1, 0.9, 1))
        self.sun_np = self.root.attachNewNode(sun)
        self.root.setLight(self.sun_np)
        amb = AmbientLight("vm_amb")
        amb.setColor(Vec4(0.42, 0.45, 0.52, 1))
        self.root.setLight(self.root.attachNewNode(amb))
        pl = PointLight("vm_flash")
        pl.setColor(Vec4(0, 0, 0, 1))
        pl.setAttenuation(Vec3(1, 0, 8))
        self.flash_light = pl
        self.flash_np = self.root.attachNewNode(pl)
        self.flash_np.setPos(0.1, 0.9, -0.05)
        self.root.setLight(self.flash_np)
        self.root.setShaderAuto()
        self.root.setAntialias(AntialiasAttrib.M_multisample)

        self.pivot = self.root.attachNewNode("pivot")
        self.holders = {}
        self.infos = {}
        self.flashes = {}
        sleeve = TEAM_STYLE[team]["shirt"]
        flash_tex = base_textures(base).sprite("flash")
        side_tex = base_textures(base).sprite("flash_side")
        for name in ("ak47", "awm", "deagle", "knife", "frag", "smoke"):
            holder = self.pivot.attachNewNode(name)
            gun, info = models.weapon(name, self.skins.get(name))
            gun.reparentTo(holder)
            self.guns[name] = gun
            ps = PartSet("arms_" + name)
            grip = info.get("grip", Point3(0, 0, 0))
            if name in ("ak47", "awm"):
                _arm(ps, grip + Vec3(0.012, 0, -0.01), (0.25, -0.75, -0.5), sleeve)
                _arm(ps, info["fore"] + Vec3(-0.018, 0, -0.02), (-0.45, -0.6, -0.55), sleeve)
            elif name == "deagle":
                _arm(ps, grip + Vec3(0.01, 0, -0.01), (0.2, -0.75, -0.5), sleeve)
                _arm(ps, grip + Vec3(-0.03, 0.01, -0.025), (-0.55, -0.6, -0.5), sleeve)
            elif name == "knife":
                _arm(ps, grip + Vec3(0.0, 0, 0.0), (0.2, -0.85, -0.45), sleeve)
            else:
                _arm(ps, (0.0, -0.01, -0.035), (0.25, -0.75, -0.55), sleeve)
            arms = ps.node()
            arms.reparentTo(holder)
            holder.hide()
            self.holders[name] = holder
            self.infos[name] = info
            if name in ("ak47", "awm", "deagle"):
                fl = holder.attachNewNode("flash")
                fl.setPos(info["muzzle"])
                fl.setLightOff(1)
                fl.setShaderOff(1)
                _additive(fl)
                front = _card("flash_front")
                front.reparentTo(fl)
                front.setTexture(flash_tex, 1)
                front.setScale(0.11 if name != "deagle" else 0.08)
                front.setBillboardPointEye()
                for r in (0, 90):
                    side = MeshBuilder("flash_side")
                    side.quad((0, 0, -0.5), (0, 1, -0.5), (0, 1, 0.5), (0, 0, 0.5), uvs=((0, 0), (1, 0), (1, 1), (0, 1)))
                    side.quad((0, 1, -0.5), (0, 0, -0.5), (0, 0, 0.5), (0, 1, 0.5), uvs=((1, 0), (0, 0), (0, 1), (1, 1)))
                    s = side.node()
                    s.reparentTo(fl)
                    s.setTexture(side_tex, 1)
                    s.setScale(0.07, 0.22 if name != "deagle" else 0.14, 0.07)
                    s.setR(r)
                fl.hide()
                self.flashes[name] = fl
        # shell casings
        self.casings = []
        brass = MeshBuilder("casing")
        brass.cylinder((0, -0.012, 0), (0, 0.012, 0), 0.0045, (0.85, 0.65, 0.25, 1), 8)
        proto = brass.node()
        for _ in range(14):
            c = proto.copyTo(self.root)
            c.hide()
            self.casings.append([c, Vec3(), Vec3(), 0.0])
        self.casing_idx = 0

        self.current = None
        self.ads_blend = 0.0
        self.kick = 0.0
        self.kick_roll = 0.0
        self.sway = Vec3()
        self.bob_phase = 0.0
        self.land = 0.0
        self.flash_t = 0.0
        self.anim = None  # (kind, t0, dur)
        self.anim_t = 0.0
        self._anim_ref = None
        self.inspect_kind = "gun"
        self.hidden = False

    def set_aspect(self, aspect: float):
        self.lens.setAspectRatio(aspect)

    def set_skins(self, skins: dict):
        for name, gun in list(self.guns.items()):
            if skins.get(name) == self.skins.get(name):
                continue
            new, _ = self.models.weapon(name, skins.get(name))
            new.reparentTo(self.holders[name])
            gun.removeNode()
            self.guns[name] = new
        self.skins = dict(skins)

    def inspect(self) -> bool:
        """Start the inspect animation for the current weapon (F key)."""
        if self.current is None or (self.anim and self.anim[0] not in ("inspect",)):
            return False
        kind = WEAPONS[self.current].kind
        kind = "gun" if kind == "gun" else "knife" if kind == "melee" else "grenade"
        self.inspect_kind = kind
        self._play("inspect", INSPECT[kind][-1][0])
        return True

    def _play(self, kind: str, dur: float):
        self.anim = [kind, 0.0, dur]

    @property
    def inspecting(self) -> bool:
        return bool(self.anim and self.anim[0] == "inspect")

    def cancel_inspect(self):
        if self.inspecting:
            self.anim = None
            self.anim_t = 0.0

    def set_visible(self, visible: bool):
        self.dr.setActive(visible)

    # ------------------------------------------------------------------ events
    def on_event(self, e, agent):
        t = e["type"]
        if t == "shot":
            spec = WEAPONS[e["weapon"]]
            self.kick += {"ak47": 1.0, "awm": 2.6, "deagle": 1.8}.get(e["weapon"], 1.0)
            self.kick = min(self.kick, 3.2)
            self.kick_roll = random.uniform(-1, 1)
            self.flash_t = 0.045
            fl = self.flashes.get(e["weapon"])
            if fl is not None:
                fl.setR(random.uniform(0, 360))
                fl.setScale(random.uniform(0.85, 1.25))
            self.flash_light.setColor(Vec4(3.0, 2.0, 0.9, 1))
            self._eject(e["weapon"])
            self.cancel_inspect()
            if spec.unscope_after_shot:
                self._play("bolt", spec.interval * 0.9)
        elif t == "melee":
            self._play("stab" if e["heavy"] else "slash", 0.5 if e["heavy"] else 0.32)
        elif t == "throw_start":
            self._play("throw", 0.55)
        elif t == "switch":
            self._play("draw", WEAPONS[e["weapon"]].draw_time)
        elif t == "reload":
            self.cancel_inspect()
        elif t == "land":
            self.land = min(1.0, e["speed"] / 10.0)
        elif t == "spawn":
            self._play("draw", 0.4)

    def _eject(self, weapon):
        holder = self.holders.get(weapon)
        info = self.infos.get(weapon)
        if holder is None or "eject" not in info:
            return
        slot = self.casings[self.casing_idx]
        self.casing_idx = (self.casing_idx + 1) % len(self.casings)
        p = self.root.getRelativePoint(holder, info["eject"])
        slot[0].setPos(p)
        slot[0].show()
        slot[1] = Vec3(random.uniform(0.9, 1.4), random.uniform(-0.2, 0.3), random.uniform(0.6, 1.1))
        slot[2] = Vec3(random.uniform(300, 900), random.uniform(300, 900), 0)
        slot[3] = 0.6

    # ------------------------------------------------------------------ update
    def update(self, dt, world, agent, mouse_dx, mouse_dy, sun_quat=None):
        if sun_quat is not None:
            self.sun_np.setQuat(sun_quat)
        name = agent.weapon.name if agent.alive else None
        if name != self.current:
            if self.current:
                self.holders[self.current].hide()
            self.current = name
            if name:
                self.holders[name].show()
        if name is None:
            return
        spec = WEAPONS[name]
        now = world.time
        target_ads = 1.0 if agent.ads else 0.0
        self.ads_blend += (target_ads - self.ads_blend) * min(1.0, dt * 14.0)
        scoped = spec.scope_overlay and self.ads_blend > 0.85 and agent.ads
        holder = self.holders[name]
        if scoped:
            holder.hide()
        else:
            holder.show()

        self.kick *= math.exp(-dt * 16.0)
        self.land *= math.exp(-dt * 8.0)
        target_sway = Vec3(max(-4, min(4, -mouse_dx * 0.06)), max(-4, min(4, mouse_dy * 0.06)), 0)
        self.sway += (target_sway - self.sway) * min(1.0, dt * 10.0)
        speed = math.hypot(agent.vel[0], agent.vel[1])
        amp = min(1.0, speed / 5.0) * (1.0 if agent.on_ground else 0.2) * (1 - 0.85 * self.ads_blend)
        self.bob_phase += dt * speed * 1.9

        hip_pos, hip_hpr = HIP[name]
        info = self.infos[name]
        if name in ADS_Y:
            ads_pos = Point3(0.0, ADS_Y[name], -info["sight_z"])
        else:
            ads_pos = hip_pos
        a = self.ads_blend
        pos = hip_pos * (1 - a) + ads_pos * a
        hpr = Vec3(hip_hpr) * (1 - a)
        pos += Vec3(math.sin(self.bob_phase) * 0.008 * amp, 0, -abs(math.cos(self.bob_phase)) * 0.01 * amp)
        hpr += Vec3(self.sway.x * (1 - 0.7 * a), self.sway.y * (1 - 0.7 * a), self.sway.x * 0.8)
        pos += Vec3(0, -self.kick * (0.022 if a < 0.5 else 0.012), -self.land * 0.03)
        hpr += Vec3(0, self.kick * (2.4 if a < 0.5 else 1.0), self.kick * self.kick_roll * 1.5)
        if agent.crouching:
            pos += Vec3(0.005, 0, -0.01)

        if agent.reloading:
            p = 1.0 - (agent.reload_until - now) / spec.reload_time
            s = math.sin(math.pi * max(0.0, min(1.0, p)))
            pos += Vec3(-0.02 * s, 0, -0.09 * s)
            hpr += Vec3(8 * s, -18 * s, 28 * s)
        if agent.ads or agent.reloading:
            self.cancel_inspect()
        if self.anim is not self._anim_ref:
            self._anim_ref = self.anim
            self.anim_t = 0.0
        if self.anim:
            kind, t0, dur = self.anim
            self.anim_t += dt
            u = self.anim_t / max(dur, 1e-3)
            if u >= 1.0:
                self.anim = None
                self.anim_t = 0.0
            else:
                if kind == "draw":
                    k = (1 - u) ** 2
                    pos += Vec3(0, 0, -0.22 * k)
                    hpr += Vec3(0, -50 * k, 10 * k)
                elif kind == "slash":
                    s = math.sin(math.pi * u)
                    pos += Vec3(-0.18 * u + 0.06, 0.06 * s, 0.03 * s)
                    hpr += Vec3(60 * s, -10 * s, -55 * s)
                elif kind == "stab":
                    s = math.sin(math.pi * min(1.0, u * 1.4))
                    pos += Vec3(-0.06 * s, 0.17 * s, 0.05 * s)
                    hpr += Vec3(10 * s, -55 * s, 0)
                elif kind == "throw":
                    if u < 0.4:
                        k = u / 0.4
                        pos += Vec3(0.04 * k, -0.08 * k, 0.1 * k)
                        hpr += Vec3(0, 35 * k, 0)
                    else:
                        k = (u - 0.4) / 0.6
                        pos += Vec3(0.04 - 0.1 * k, -0.08 + 0.3 * k, 0.1 - 0.25 * k)
                        hpr += Vec3(0, 35 - 80 * k, 0)
                elif kind == "bolt":
                    s = math.sin(math.pi * u)
                    pos += Vec3(0.0, -0.03 * s, -0.04 * s)
                    hpr += Vec3(0, 8 * s, 20 * s)
                elif kind == "inspect":
                    dp, dh = _inspect_pose(self.inspect_kind, self.anim_t)
                    pos += dp
                    hpr += dh
        if spec.kind == "grenade" and agent.weapon.mag <= 0:
            holder.hide()
        holder.setPos(pos)
        holder.setHpr(hpr)

        if self.flash_t > 0:
            self.flash_t -= dt
            fl = self.flashes.get(name)
            if fl is not None:
                if self.flash_t > 0 and not scoped:
                    fl.show()
                else:
                    fl.hide()
            if self.flash_t <= 0:
                self.flash_light.setColor(Vec4(0, 0, 0, 1))
        for slot in self.casings:
            if slot[3] <= 0:
                continue
            slot[3] -= dt
            slot[1].z -= 4.0 * dt
            n = slot[0]
            n.setPos(n.getPos() + slot[1] * dt)
            n.setHpr(n.getHpr() + slot[2] * dt)
            if slot[3] <= 0:
                n.hide()


def base_textures(base):
    return base.textures

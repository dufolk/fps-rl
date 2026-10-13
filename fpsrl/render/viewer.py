"""Panda3D viewer: renders a `World` from an agent's point of view.

Used directly by the RL env (`render_mode="human"`, RGB observations) and as
the base class of the human-playable `GameApp`.
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
from panda3d.core import (AntialiasAttrib, FrameBufferProperties, GraphicsOutput, NodePath, PerspectiveLens,
                          Point3, TextNode, Texture, Vec3, WindowProperties, loadPrcFileData)

from .. import config as C
from ..geometry import raycast_boxes
from ..sim import LEAN_DIST, LEAN_ROLL, World, dir_from_angles
from ..weapons import WEAPONS
from .effects import Effects
from .models import ModelCache, SoldierModel
from .textures import TextureLibrary
from .dynamic_view import DestructView, GadgetView, point_colors
from .world_scene import MAIN_MASK, WorldScene

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
BASE_HFOV = 96.0


def configure(onscreen=True, size=(1600, 900), vsync=True, msaa=4, fullscreen=False, title="FPS-RL"):
    loadPrcFileData("", f"""
window-title {title}
win-size {size[0]} {size[1]}
fullscreen {'#t' if fullscreen else '#f'}
framebuffer-multisample {1 if msaa else 0}
multisamples {msaa}
sync-video {'#t' if vsync else '#f'}
textures-power-2 none
show-frame-rate-meter #f
audio-library-name p3openal_audio
notify-level-display error
notify-level-glgsg error
gl-coordinate-system default
""")
    if not onscreen:
        loadPrcFileData("", "audio-library-name null\nwin-size 64 64\n")


class SoldierView:
    def __init__(self, viewer, agent):
        self.model = SoldierModel(agent.team, viewer.models)
        self.model.root.reparentTo(viewer.render)
        self.agent_id = agent.id
        self.dead_since = None
        tn = TextNode("tag")
        tn.setText(agent.name)
        tn.setAlign(TextNode.ACenter)
        tn.setTextColor(0.5, 1.0, 0.55, 1)
        tn.setShadow(0.05, 0.05)
        tn.setShadowColor(0, 0, 0, 0.8)
        font = viewer.hud.font_cjk if viewer.hud is not None and viewer.hud.font_cjk else viewer.hud_font
        if font is not None:
            tn.setFont(font)
        self.tag_node = tn
        self.tag = self.model.root.attachNewNode(tn)
        self.tag.setScale(0.22)
        self.tag.setZ(2.1)
        self.tag.setBillboardPointEye()
        self.tag.setLightOff(1)
        self.tag.setShaderOff(1)
        self.tag.setDepthTest(False)
        self.tag.setDepthWrite(False)
        self.tag.setBin("fixed", 50)
        self.tag.hide()


class Viewer:
    _shared: "Viewer | None" = None

    @classmethod
    def shared(cls, world: World, pov: int = 0, onscreen: bool = True) -> "Viewer":
        if cls._shared is None:
            cls._shared = Viewer(world, pov=pov, onscreen=onscreen, hud=onscreen, viewmodel=onscreen,
                                 audio=onscreen)
        elif cls._shared.world is not world:
            cls._shared.bind(world, pov)
        return cls._shared

    @classmethod
    def destroy_shared(cls):
        if cls._shared is not None:
            cls._shared.base.destroy()
            cls._shared = None

    def __init__(self, world: World, pov: int = 0, onscreen: bool = True, size=(1600, 900),
                 shadows: bool = True, vsync: bool = False, msaa: int = 4, fullscreen: bool = False,
                 hud: bool = True, viewmodel: bool = True, audio: bool = True, title: str = "FPS-RL"):
        from direct.showbase.ShowBase import ShowBase

        configure(onscreen, size, vsync, msaa, fullscreen, title)
        self.base = base = ShowBase(windowType="onscreen" if onscreen else "offscreen")
        self.onscreen = onscreen
        base.disableMouse()
        base.textures = self.textures = TextureLibrary()
        self.models = ModelCache()
        base.render.setShaderAuto()
        base.render.setAntialias(AntialiasAttrib.M_multisample)
        base.cam.node().setCameraMask(MAIN_MASK)
        base.camLens.setNearFar(0.04, 3000.0)
        base.camLens.setFov(BASE_HFOV)
        base.setBackgroundColor(0.78, 0.74, 0.7, 1)
        self.render = base.render
        self.world = world
        self.shadows = shadows
        self.scene = WorldScene(base, world.map, self.textures, shadows=shadows)
        self.destruct_view = None
        self.gadget_view = None
        self.bomb_beep_at = 0.0
        self.effects = Effects(base, self.textures)
        self.hud_font = None
        self.hud = None
        self.vm = None
        self.audio = None
        self.local_skins: dict = {}            # weapon -> skin key for the first-person player
        self.agent_skins: dict[int, dict] = {}  # agent id -> {weapon: skin} (third-person models)
        self.cinematic = False
        self.cine_t = 0.0
        if hud:
            from .hud import HUD, load_font
            self.hud = HUD(base, self.textures, pov)
            self.hud_font = self.hud.font
        if viewmodel:
            from .viewmodel import ViewModel
            self.vm = ViewModel(base, self.models, world.agents[pov].team)
        if audio:
            from .audio import Audio
            self.audio = Audio(base, os.path.join(ASSET_DIR, "sfx"))
        self.soldiers: dict[int, SoldierView] = {}
        self.proj_nodes: dict[int, NodePath] = {}
        self.pov = pov
        self.third_person = False
        self.tp_ads = 0.0
        self.first_person_now = True  # camera currently at the pov agent's eye
        self.event_seq = 0
        self.shake = 0.0
        self.t = 0.0
        self.fov = BASE_HFOV
        self.pending_sounds = []
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.closed = False
        self.obs_buffer = None
        self.show_board = False
        self.death_cam = None
        if onscreen:
            base.win.setCloseRequestEvent("viewer-close")
            base.accept("viewer-close", self._on_close)
            base.accept("window-event", self._on_window)
            base.accept("v", self.cycle_pov)
            base.accept("c", self.toggle_third_person)
            self._on_window(base.win)
        self.bind(world, pov)

    # ------------------------------------------------------------------ binding
    def bind(self, world: World, pov: int):
        for sv in self.soldiers.values():
            sv.model.root.removeNode()
        for n in self.proj_nodes.values():
            n.removeNode()
        if world.map.name != self.scene.map.name:
            self.scene.destroy()
            self.scene = WorldScene(self.base, world.map, self.textures, shadows=self.shadows)
        self.world = world
        self.pov = pov
        if self.hud is not None:
            self.hud.local_id = pov
        if self.destruct_view is not None:
            self.destruct_view.root.removeNode()
            self.gadget_view.root.removeNode()
        self.destruct_view = DestructView(self.scene.root, world, self.textures, self.scene.ts_normal)
        self.gadget_view = GadgetView(self.scene.root, world, self.textures, self.scene.ts_normal, self.models)
        self.bomb_beep_at = 0.0
        self.soldiers = {a.id: SoldierView(self, a) for a in world.agents}
        self.proj_nodes = {}
        self.event_seq = world.event_log[-1]["seq"] if world.event_log else 0
        self.effects.clear()
        self.pending_sounds = []
        self.death_cam = None
        self.agent_skins = {pov: self.local_skins}
        self.set_pov(pov)

    def _on_close(self):
        self.closed = True
        self.base.userExit()

    def _on_window(self, win):
        if win is None or win != self.base.win:
            return
        props = win.getProperties()
        if props.getXSize() > 0 and props.getYSize() > 0:
            aspect = props.getXSize() / props.getYSize()
            self.base.camLens.setAspectRatio(aspect)
            self.base.camLens.setFov(self.fov)
            if self.vm is not None:
                self.vm.set_aspect(aspect)

    def cycle_pov(self):
        n = len(self.world.agents)
        self.set_pov((self.pov + 1) % n)

    def set_pov(self, aid: int):
        self.pov = aid
        if self.hud is not None:
            self.hud.local_id = aid
        if self.vm is not None and self.vm.team != self.world.agents[aid].team:
            self.vm.root.removeNode()
            self.vm.dr.setActive(False)
            self.base.win.removeDisplayRegion(self.vm.dr)
            from .viewmodel import ViewModel
            self.vm = ViewModel(self.base, self.models, self.world.agents[aid].team, skins=self.local_skins)
            self.vm.set_visible(not self.cinematic)
            self._on_window(self.base.win)

    def toggle_third_person(self):
        self.third_person = not self.third_person

    def set_local_skins(self, skins: dict):
        self.local_skins = dict(skins)
        self.agent_skins[self.pov] = self.local_skins
        if self.vm is not None:
            self.vm.set_skins(self.local_skins)

    def inspect(self):
        """F key: play the inspect animation and show the skin name."""
        a = self.world.agents[self.pov]
        if self.vm is None or not a.alive or self.third_person or not self.vm.inspect():
            return
        if self.hud is not None:
            from .skins import RARITY, get_skin
            name = a.weapon.name
            sk = get_skin(name, self.local_skins.get(name))
            zh = self.hud.cjk
            rz, ren, col = RARITY[sk.rarity]
            title = f"{WEAPONS[name].display}  |  {sk.zh if zh else sk.en}"
            self.hud.show_inspect(title, f"{rz if zh else ren}  ·  {sk.en if zh else ''}".strip(" ·"), col,
                                  self.vm.anim[2])
        self._sound("draw", None, 0.35, 0.8)

    def set_cinematic(self, on: bool):
        """Menu background mode: slow fly-over camera, no HUD / first-person view."""
        self.cinematic = on
        if self.audio is not None:
            self.audio.muted = on
        if self.hud is not None:
            self.hud.set_visible(not on)
        if self.vm is not None:
            self.vm.set_visible(not on)
        self.third_person = False

    def _cinematic_camera(self, dt):
        self.cine_t += dt
        t = self.cine_t * 0.045
        cam = self.base.camera
        c = self.scene.center
        r = self.scene.extent * 0.42
        if self.world.map.theme == "ship":
            x = math.sin(t) * 34.0
            y = -24.0 + math.cos(t * 0.7) * 6.0
            z = 9.0 + math.sin(t * 1.3) * 2.5
            look = Point3(x * 0.55, 0.0, 1.0)
        else:
            x = c.x + math.sin(t) * r
            y = c.y - math.cos(t) * r
            z = 14.0 + self.scene.extent * 0.08 + math.sin(t * 1.3) * 2.5
            look = Point3(c.x + (x - c.x) * 0.2, c.y + (y - c.y) * 0.2, 1.0)
        cam.setPos(x, y, z)
        cam.lookAt(look)
        if abs(self.fov - 70.0) > 0.01:
            self.fov = 70.0
            self.base.camLens.setFov(self.fov)

    # ------------------------------------------------------------------ events
    def listener(self):
        a = self.world.agents[self.pov]
        return (self.base.camera.getPos(self.render), a.yaw)

    def _sound(self, name, pos=None, volume=1.0, rate=1.0, ref=6.0):
        if self.audio is not None:
            self.audio.play(name, pos, self.listener() if pos is not None else None, volume, rate, ref)

    def _muzzle_world(self, agent):
        sv = self.soldiers.get(agent.id)
        if sv is not None and sv.model.weapon_np is not None:
            info = self.models.weapon(agent.weapon.name)[1]
            m = info.get("muzzle")
            if m is not None:
                return self.render.getRelativePoint(sv.model.weapon_np, m)
        e = agent.eye()
        return Point3(e[0], e[1], e[2] - 0.2)

    def _process_events(self):
        w = self.world
        evs = w.events_since(self.event_seq)
        if not evs:
            return
        self.event_seq = evs[-1]["seq"]
        me = self.pov
        fp = self.first_person_now and not self.cinematic
        for e in evs:
            t = e["type"]
            if self.hud is not None:
                self.hud.on_event(e, w)
            if self.vm is not None and e.get("agent") == me:
                self.vm.on_event(e, w.agents[me])
            if t == "shot":
                shooter = w.agents[e["agent"]]
                origin = Point3(*e["origin"])
                end = Point3(*e["end"])
                own = e["agent"] == me and fp
                if own:
                    cam = self.base.camera
                    start = self.render.getRelativePoint(cam, Point3(0.1, 0.7, -0.09))
                else:
                    start = self._muzzle_world(shooter)
                    d = end - origin
                    if d.length() > 0:
                        d.normalize()
                    if WEAPONS[e["weapon"]].silenced:
                        self.effects.muzzle(start, d, light=False, small=True)
                    else:
                        self.effects.muzzle(start, d, big=e["weapon"] in ("awm", "scar20", "m249"))
                if not own or random.random() < 0.6:
                    self.effects.tracer(start, end)
                if e["hit_agent"] is not None:
                    d = end - origin
                    d.normalize()
                    self.effects.blood_hit(end, d, e["group"] == 0)
                    self._sound("flesh", end, 0.6)
                elif e["mat"] is not None:
                    self.effects.impact(end, e["normal"], e["mat"])
                    if random.random() < 0.5:
                        soft = e["mat"].startswith(("crate", "wood", "drywall", "plaster", "carpet"))
                        self._sound("impact_metal" if not soft else "impact_wood",
                                    end, 0.35, random.uniform(0.85, 1.2), 4.0)
                vol = 0.75 if own else 1.0
                self._sound(e["weapon"], origin, vol, random.uniform(0.95, 1.05), 9.0 if not own else 30.0)
                if own and e["weapon"] == "awm":
                    self.pending_sounds.append((w.time + 0.55, "bolt", None, 0.7))
            elif t == "damage":
                if e["attacker"] == me and e["victim"] != me:
                    self._sound("headshot" if e["group"] == 0 else "hit", None, 0.6 if e["group"] else 0.8)
                if e["victim"] == me:
                    self._sound("hurt", None, 0.6)
            elif t == "kill":
                fx = self.hud.kill_fx if self.hud is not None else None
                if fx is not None:
                    self.hud.kill_fx = None
                    streak, kind = fx
                    self.pending_sounds.append((w.time + 0.05, f"kill_{min(streak, 6)}", None, 0.75))
                    accent = {"headshot": "kill_hs", "knife": "kill_knife", "grenade": "kill_nade"}.get(kind)
                    if accent:
                        self.pending_sounds.append((w.time + 0.05, accent, None, 0.7))
            elif t == "explode":
                p = Point3(*e["pos"])
                self.effects.explosion(p)
                self._sound("explosion", p, 1.0, random.uniform(0.92, 1.05), 18.0)
                cam = self.base.camera.getPos(self.render)
                d = (cam - p).length()
                self.shake = max(self.shake, max(0.0, 1.0 - d / 22.0) * 1.5)
            elif t == "smoke":
                self.effects.smoke_grenade(e["proj"], e["pos"], e["radius"], e["duration"])
                self._sound("smoke_hiss", Point3(*e["pos"]), 0.8, 1.0, 10.0)
            elif t == "bounce":
                self._sound("bounce", Point3(*e["pos"]), min(1.0, e["speed"] / 10.0), random.uniform(0.9, 1.1), 6.0)
            elif t == "throw_start":
                a = w.agents[e["agent"]]
                self._sound("pin", Point3(*a.eye()), 0.7)
            elif t == "throw":
                self._sound("knife_swing", Point3(*e["pos"]), 0.5, 0.7)
            elif t == "melee":
                a = w.agents[e["agent"]]
                self._sound("knife_swing", Point3(*a.eye()), 0.8, 0.9 if e["heavy"] else 1.1)
            elif t == "melee_hit":
                p = Point3(*e["point"])
                if e["victim"] >= 0:
                    self._sound("knife_hit", p, 1.0)
                    self.effects.blood_hit(p, Vec3(*e["normal"]) * -1, False)
                else:
                    self._sound("knife_wall", p, 0.8)
                    self.effects.impact(p, e["normal"], "steel_plate")
            elif t == "reload":
                a = w.agents[e["agent"]]
                pos = Point3(*a.eye())
                vol = 0.8 if e["agent"] == me else 0.6
                self.pending_sounds.append((w.time + e["duration"] * 0.2, "mag_out", e["agent"], vol))
                self.pending_sounds.append((w.time + e["duration"] * 0.7, "mag_in", e["agent"], vol))
                if e["weapon"] == "awm":
                    self.pending_sounds.append((w.time + e["duration"] * 0.88, "bolt", e["agent"], vol))
            elif t == "switch":
                if e["agent"] == me:
                    self._sound("draw", None, 0.5)
            elif t == "footstep":
                a = w.agents[e["agent"]]
                vol = 0.25 if e["agent"] == me else 0.75
                self._sound(f"step{random.randint(0, 3)}", Point3(*e["pos"]), vol, random.uniform(0.9, 1.1), 5.0)
            elif t == "land":
                a = w.agents[e["agent"]]
                self._sound("land", Point3(*a.pos), 0.5 if e["agent"] == me else 0.6, 1.0, 5.0)
            elif t == "jump":
                if e["agent"] == me:
                    self._sound("jump", None, 0.25)
            elif t == "dry_fire":
                if e["agent"] == me:
                    self._sound("dry", None, 0.7)
            elif t == "flash":
                p = Point3(*e["pos"])
                if e.get("shield"):
                    self._sound("flash_bang", p, 0.5, 1.4, 6.0)
                else:
                    self.effects.flashbang(p)
                    self._sound("flash_bang", p, 1.0, 1.0, 12.0)
            elif t == "wall_break":
                p = Point3(*e["pos"])
                self.effects.debris(p, e.get("mat", ""))
                self._sound("wall_break", p, 0.7, random.uniform(0.9, 1.1), 8.0)
            elif t == "reinforce":
                self._sound("reinforce", Point3(*e["pos"]), 0.8, 1.0, 8.0)
            elif t in ("buy", "armor"):
                if e["agent"] == me:
                    self._sound("buy", None, 0.6)
            elif t == "busy":
                a = w.agents[e["agent"]]
                self._sound("defuse_tick" if e["kind"] == "defuse" else "beep", Point3(*a.eye()), 0.6, 1.0, 6.0)
            elif t == "bomb_planted":
                self._sound("plant_done", Point3(*e["pos"]), 1.0, 1.0, 14.0)
                self.bomb_beep_at = w.time + 1.0
            elif t == "bomb_defused":
                self._sound("plant_done", None, 0.8, 0.8)
            elif t == "bomb_explode":
                p = Point3(*e["pos"])
                self.effects.explosion(p)
                self.effects.explosion(p + Vec3(0, 0, 1.5))
                self._sound("bomb_explode", p, 1.0, 1.0, 40.0)
                d = (self.base.camera.getPos(self.render) - p).length()
                self.shake = max(self.shake, max(0.0, 1.0 - d / 45.0) * 2.5)
            elif t == "point_captured":
                mine = w.agents[me].team
                self._sound("capture" if e["team"] == mine else "lost", None, 0.7)
            elif t == "round_end":
                self._sound("round_win" if e["team"] == w.agents[me].team else "round_lose", None, 0.7)
            elif t == "drone_deploy":
                self._sound("drone", Point3(*w.agents[e["agent"]].eye()), 0.6)
            elif t in ("gadget_placed", "launch"):
                a = w.agents[e["agent"]]
                self._sound("pin" if t == "gadget_placed" else "knife_swing", Point3(*a.eye()), 0.6)
            elif t == "gadget_destroyed":
                p = Point3(*e["pos"])
                self.effects.impact(p, (0, 0, 1), "steel_plate")
                self._sound("impact_metal", p, 0.7)
            elif t == "spawn":
                pass
        # deferred sounds
        keep = []
        for item in self.pending_sounds:
            at, name, aid, vol = item
            if w.time >= at:
                pos = None
                if aid is not None and aid != me:
                    pos = Point3(*w.agents[aid].eye())
                self._sound(name, pos, vol)
            else:
                keep.append(item)
        self.pending_sounds = keep

    # ------------------------------------------------------------------ sync
    def _sync_soldiers(self, dt, alpha):
        w = self.world
        for a in w.agents:
            sv = self.soldiers[a.id]
            m = sv.model
            pos = a.prev_pos + (a.pos - a.prev_pos) * alpha
            m.root.setPos(pos[0], pos[1], pos[2])
            if a.alive:
                if sv.dead_since is not None:
                    sv.dead_since = None
                    m.root.setP(0)
                    m.root.setR(0)
                    m.root.setColorScale(1, 1, 1, 1)
                    m.root.clearTransparency()
                m.root.setH(a.yaw - 90.0)
                m.set_weapon(a.weapon.name, self.agent_skins.get(a.id, {}).get(a.weapon.name))
                if sv.tag_node.getText() != a.name:
                    sv.tag_node.setText(a.name)
                crouch = (C.STAND_HEIGHT - a.height) / (C.STAND_HEIGHT - C.CROUCH_HEIGHT)
                speed = math.hypot(a.vel[0], a.vel[1])
                m.pose(dt, speed, crouch, a.pitch + a.punch[0], a.on_ground,
                       a.lean_off / LEAN_DIST if LEAN_DIST else 0.0)
                hidden = a.id == self.pov and self.first_person_now and not self.cinematic
                if hidden:
                    m.root.hide()
                else:
                    m.root.show()
                mine = w.agents[self.pov].team
                if a.team == mine and a.id != self.pov and not self.cinematic:
                    sv.tag.show()
                else:
                    sv.tag.hide()
            else:
                sv.tag.hide()
                if sv.dead_since is None:
                    sv.dead_since = w.time
                    sv.fall_dir = random.choice((-1, 1))
                m.root.show()
                td = w.time - sv.dead_since
                k = min(1.0, td / 0.45)
                k = k * k * (3 - 2 * k)
                m.root.setP(-88.0 * k)
                m.root.setR(sv.fall_dir * 8 * k)
                m.pose(dt, 0.0, 0.0, 0.0, True)
                if td > 2.2:
                    m.root.setZ(pos[2] - (td - 2.2) * 0.4)

    def _sync_projectiles(self, dt):
        alive = set()
        for p in self.world.projectiles:
            alive.add(p.id)
            n = self.proj_nodes.get(p.id)
            if n is None:
                n, _ = self.models.weapon(p.kind)
                n.reparentTo(self.render)
                n.setScale(1.3)
                self.proj_nodes[p.id] = n
            n.setPos(*p.pos)
            speed = float(np.linalg.norm(p.vel))
            n.setHpr(n.getH() + speed * 40 * dt, n.getP() + speed * 60 * dt, 0)
        for pid in list(self.proj_nodes):
            if pid not in alive:
                self.proj_nodes.pop(pid).removeNode()

    def _camera(self, dt, alpha):
        w = self.world
        a = w.agents[self.pov]
        cam = self.base.camera
        self.shake *= math.exp(-dt * 4.0)
        sh = self.shake
        jitter = (random.uniform(-1, 1) * sh * 1.6, random.uniform(-1, 1) * sh * 1.6, random.uniform(-1, 1) * sh)
        spec = a.spec if a.slot in a.inv else None
        scoped_fp = bool(a.alive and a.ads and spec is not None and spec.scope_overlay)
        want_ads = bool(a.alive and a.ads and spec is not None and spec.ads_fov)
        self.tp_ads += ((1.0 if want_ads else 0.0) - self.tp_ads) * min(1.0, dt * 14.0)
        pose = w.gadgets.view_pose(a) if a.view_mode else None
        if pose is not None:
            p, yaw, pitch = pose
            cam.setPos(float(p[0]), float(p[1]), float(p[2]))
            cam.setHpr(yaw - 90.0, pitch, 0.0)
        self.first_person_now = pose is None and a.alive and (not self.third_person or scoped_fp)
        if pose is not None:
            pass
        elif self.first_person_now:
            self.death_cam = None
            pos = a.prev_pos + (a.pos - a.prev_pos) * alpha
            punch = a.prev_punch + (a.punch - a.prev_punch) * alpha
            off = a.lean_vec()
            cam.setPos(pos[0] + off[0], pos[1] + off[1],
                       pos[2] + a.height - C.EYE_FROM_TOP - abs(a.lean_off) * 0.12)
            roll = (a.lean_off / LEAN_DIST) * LEAN_ROLL if LEAN_DIST else 0.0
            cam.setHpr(a.yaw + punch[1] - 90.0 + jitter[0], a.pitch + punch[0] + jitter[1], jitter[2] + roll)
        elif a.alive:
            self._third_person_camera(a, alpha)
        else:
            if self.death_cam is None:
                e = a.eye()
                self.death_cam = Point3(e[0], e[1], e[2])
            killer = w.agents[a.killer] if a.killer >= 0 and a.killer != a.id else None
            look = Point3(*(killer.eye() if killer is not None and killer.alive else a.pos))
            td = w.time - a.death_time
            body = Point3(*a.pos)
            up = min(1.0, td / 0.8)
            p = self.death_cam + Vec3(0, 0, 1.5 * up)
            to_body = p - body
            to_body.z = 0
            if to_body.length() < 0.5:
                to_body = Vec3(-1, 0, 0)
            to_body.normalize()
            cam.setPos(p + to_body * 1.5 * up)
            cam.lookAt(look)
        # FOV
        target_fov = BASE_HFOV
        if pose is not None:
            target_fov = 84.0 if a.view_mode == 1 else 76.0
        elif want_ads:
            target_fov = spec.ads_fov
            if self.third_person and not scoped_fp:
                target_fov = min(BASE_HFOV, spec.ads_fov * 1.12)
        if abs(target_fov - self.fov) > 0.01:
            rate = 20.0 if target_fov < self.fov else 14.0
            self.fov += (target_fov - self.fov) * min(1.0, dt * rate)
            if scoped_fp or pose is not None:
                self.fov = target_fov
            self.base.camLens.setFov(self.fov)

    def _third_person_camera(self, a, alpha):
        """Over-the-shoulder camera aimed at the point the gun actually hits, so the screen-centre
        crosshair is truthful. Aiming pulls the camera in to the shoulder."""
        w = self.world
        cam = self.base.camera
        pos = a.prev_pos + (a.pos - a.prev_pos) * alpha
        punch = a.prev_punch + (a.punch - a.prev_punch) * alpha
        k = self.tp_ads
        yr = math.radians(a.yaw)
        fwd = Vec3(math.cos(yr), math.sin(yr), 0)
        right = Vec3(math.sin(yr), -math.cos(yr), 0)
        side = -1.0 if a.lean_off < -0.05 else 1.0
        eye = a.eye()
        eye = Point3(float(eye[0]), float(eye[1]), float(eye[2]))
        # aim point along the real bullet path
        d = dir_from_angles(a.yaw + punch[1], a.pitch + punch[0])
        t, _, _ = raycast_boxes(np.array([eye]), d[None], 200.0, w.bmin, w.bmax)
        aim = eye + Vec3(*d) * float(min(t[0], 200.0))
        # shoulder anchor
        anchor = Point3(pos[0], pos[1], pos[2] + a.height - 0.2)
        back = 3.2 + (1.6 - 3.2) * k
        lat = (0.6 + (0.78 - 0.6) * k) * side
        up = 0.5 + (0.3 - 0.5) * k
        pitch = math.radians(a.pitch)
        want = anchor - fwd * back * math.cos(pitch) + right * lat + Vec3(0, 0, up - math.sin(pitch) * back * 0.5)
        dv = want - anchor
        dist = dv.length()
        dv.normalize()
        tt, _, _ = raycast_boxes(np.array([anchor]), np.array([dv]), dist, w.bmin, w.bmax)
        cam_pos = anchor + dv * max(0.15, min(dist, float(tt[0]) - 0.2))
        cam.setPos(cam_pos)
        if (aim - cam_pos).length() < 0.6:
            aim = cam_pos + Vec3(*d) * 4.0
        cam.lookAt(aim)
        roll = (a.lean_off / LEAN_DIST) * LEAN_ROLL * 0.5 if LEAN_DIST else 0.0
        cam.setR(cam.getR() + roll)

    def vfov(self) -> float:
        return float(self.base.camLens.getFov()[1])

    def _sync_dynamic(self):
        w = self.world
        self.destruct_view.update()
        self.gadget_view.update(w.time)
        cols = point_colors(w)
        if cols:
            self.scene.set_point_colors(cols)
        bomb = getattr(w.mode, "bomb", None)
        if bomb is not None and bomb.get("state") == "planted" and w.time >= self.bomb_beep_at > 0:
            left = max(0.0, w.mode.phase_end - w.time)
            self._sound("beep", Point3(*bomb["pos"]), 0.9, 1.0, 12.0)
            self.bomb_beep_at = w.time + max(0.12, min(1.0, left / 12.0))

    def sync(self, dt: float, alpha: float = 1.0):
        self.t += dt
        if self.cinematic:
            self._process_events()
            self.pending_sounds = []
            self._sync_soldiers(dt, alpha)
            self._sync_projectiles(dt)
            self._sync_dynamic()
            self._cinematic_camera(dt)
            self.scene.update(self.t)
            self.effects.update(dt)
            return
        self._process_events()
        self._sync_soldiers(dt, alpha)
        self._sync_projectiles(dt)
        self._sync_dynamic()
        self._camera(dt, alpha)
        self.scene.update(self.t)
        self.effects.update(dt)
        a = self.world.agents[self.pov]
        fp = self.first_person_now
        scope = bool(fp and a.ads and a.slot in a.inv and a.spec.scope_overlay
                     and (self.vm is None or self.vm.ads_blend > 0.85 or self.third_person))
        if self.vm is not None:
            self.vm.set_visible(fp)
            self.vm.update(dt, self.world, a, self.mouse_dx, self.mouse_dy,
                           self.scene.sun_np.getQuat(self.base.camera))
            if self.hud is not None and self.hud.inspect_t > 0 and not self.vm.inspecting:
                self.hud.hide_inspect()
        if self.hud is not None:
            self.hud.update(dt, self.world, a, self.vfov(), self.show_board, scope, first_person=fp)

    def draw(self, dt: float, alpha: float = 1.0):
        """Sync and render one frame (for externally driven loops such as the RL env)."""
        if self.closed:
            return
        self.sync(dt, alpha)
        self.base.taskMgr.step()

    # ------------------------------------------------------------------ RGB observations
    def enable_obs_buffer(self, w: int, h: int):
        if self.obs_buffer is not None:
            return
        tex = Texture("obs")
        fb = FrameBufferProperties()
        fb.setRgbColor(True)
        fb.setRgbaBits(8, 8, 8, 0)
        fb.setDepthBits(24)
        buf = self.base.win.makeTextureBuffer("obs", w, h, tex, True, fb)
        lens = PerspectiveLens()
        lens.setFov(BASE_HFOV)
        lens.setAspectRatio(w / h)
        lens.setNearFar(0.04, 3000.0)
        cam = self.base.makeCamera(buf, lens=lens)
        cam.node().setCameraMask(MAIN_MASK)
        cam.reparentTo(self.render)
        buf.setClearColor((0.78, 0.74, 0.7, 1))
        self.obs_buffer = buf
        self.obs_tex = tex
        self.obs_cam = cam
        self.obs_size = (w, h)

    def capture_obs(self, aid: int) -> np.ndarray:
        a = self.world.agents[aid]
        e = a.eye()
        self.obs_cam.setPos(e[0], e[1], e[2])
        yaw, pitch = a.view_angles()
        self.obs_cam.setHpr(yaw - 90.0, pitch, 0)
        self._sync_soldiers(0.0, 1.0)
        own = self.soldiers[aid].model.root
        own.hide()
        self._sync_projectiles(0.0)
        self.effects.update(self.world.dt)
        self.base.graphicsEngine.renderFrame()
        self.base.graphicsEngine.renderFrame()
        own.show()
        w, h = self.obs_size
        data = self.obs_tex.getRamImageAs("RGB")
        if not data:
            return np.zeros((h, w, 3), dtype=np.uint8)
        arr = np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(arr[::-1])

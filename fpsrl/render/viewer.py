"""Panda3D viewer: renders a `World` from an agent's point of view.

Used directly by the RL env (`render_mode="human"`, RGB observations) and by the
playable / spectator session in `game.py`.
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
from panda3d.core import (AntialiasAttrib, CardMaker, FrameBufferProperties, PerspectiveLens, Point3, TextNode,
                          Texture, TransparencyAttrib, Vec3, WindowProperties, loadPrcFileData)

from .. import config as C
from ..sim import LEAN_DIST, LEAN_ROLL, World
from .effects import Effects
from .models import ModelCache, body_model
from .textures import TextureLibrary
from .world_scene import MAIN_MASK, WorldScene

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
BASE_HFOV = 96.0
GOD_FOV = 86.0
# overhead plate tints, same pair as the scoreboard
_TEAM_TINT = ((1.0, 0.55, 0.3, 1.0), (0.45, 0.75, 1.0, 1.0))


def fly_velocity(yaw_deg, forward, strafe, vertical):
    """Metres per second before the speed scale. Yaw 0 looks along +X; strafe > 0 is the agent's right."""
    yr = math.radians(yaw_deg)
    fx, fy = math.cos(yr), math.sin(yr)
    rx, ry = math.sin(yr), -math.cos(yr)
    return (fx * forward + rx * strafe, fy * forward + ry * strafe, float(vertical))


def agent_markers(world, agent):
    """(flashed, burning) for the plate above a body. Burning is the molotov volume."""
    if not agent.alive:
        return False, False
    flashed = world.time < agent.flash_until
    burning = False
    for f in world.fires:
        if world.time >= f.t1:
            continue
        if (math.hypot(agent.pos[0] - f.pos[0], agent.pos[1] - f.pos[1]) <= f.radius
                and abs(agent.pos[2] - f.pos[2]) <= 1.5):
            burning = True
            break
    return flashed, burning


def configure(onscreen=True, size=(1600, 900), vsync=True, msaa=4, fullscreen=False, title="FPS-RL"):
    loadPrcFileData("", f"""
window-title {title}
win-size {size[0]} {size[1]}
fullscreen #f
framebuffer-multisample {1 if msaa else 0}
multisamples {msaa}
sync-video {'#t' if vsync else '#f'}
textures-power-2 none
show-frame-rate-meter #f
audio-library-name p3openal_audio
notify-level-display error
notify-level-glgsg error
gl-coordinate-system default
text-encoding utf8
""")
    if not onscreen:
        loadPrcFileData("", "audio-library-name null\nwin-size 64 64\n")


class SoldierView:
    def __init__(self, viewer, agent):
        self.viewer = viewer
        self.model = body_model(agent.team, viewer.models)
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
        self.plate = self.model.root.attachNewNode("plate")
        self.plate.setBillboardPointEye()
        self.plate.setLightOff(1)
        self.plate.setShaderOff(1)
        self.plate.setDepthTest(False)
        self.plate.setDepthWrite(False)
        self.plate.setBin("fixed", 48)
        self.plate.hide()
        bg = CardMaker("hpbg")
        bg.setFrame(-0.55, 0.55, -0.05, 0.05)
        self.hp_bg = self.plate.attachNewNode(bg.generate())
        self.hp_bg.setColor(0, 0, 0, 0.72)
        self.hp_bg.setTransparency(TransparencyAttrib.M_alpha)
        self.hp_bg.setTwoSided(True)
        fill = CardMaker("hpfill")
        fill.setFrame(0.0, 1.06, -0.034, 0.034)
        self.hp_fill = self.plate.attachNewNode(fill.generate())
        self.hp_fill.setPos(-0.53, 0, 0)
        self.hp_fill.setTransparency(TransparencyAttrib.M_alpha)
        self.hp_fill.setTwoSided(True)
        self.blind_np = self._plate_text("blind", "致盲", (1, 0.96, 0.75, 1), font)
        self.burn_np = self._plate_text("burn", "燃烧", (1, 0.42, 0.1, 1), font)

    def _plate_text(self, name, label, color, font):
        tn = TextNode(name)
        tn.setText(label)
        tn.setAlign(TextNode.ACenter)
        tn.setTextColor(*color)
        tn.setShadow(0.06, 0.06)
        tn.setShadowColor(0, 0, 0, 0.9)
        if font is not None:
            tn.setFont(font)
        np = self.plate.attachNewNode(tn)
        np.setScale(0.2)
        np.setZ(-0.32)
        np.hide()
        return np

    def set_overhead(self, z, frac, color, flashed, burning):
        self.plate.setZ(z)
        self.plate.show()
        self.hp_fill.setSx(max(0.001, frac))
        self.hp_fill.setColor(*color)
        shown = []
        for node, on in ((self.blind_np, flashed), (self.burn_np, burning)):
            if on:
                shown.append(node)
            else:
                node.hide()
        for i, node in enumerate(shown):
            node.setX(0.0 if len(shown) == 1 else (-0.28 if i == 0 else 0.28))
            node.show()


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
                 hud: bool = True, viewmodel: bool = True, audio: bool = True, title: str = "FPS-RL",
                 shadow_size: int = 2048):
        from direct.showbase.ShowBase import ShowBase

        configure(onscreen, size, vsync, msaa, fullscreen, title)
        self.base = base = ShowBase(windowType="onscreen" if onscreen else "offscreen")
        if onscreen and fullscreen:  # desktop-sized borderless window, never exclusive fullscreen
            di = base.pipe.getDisplayInformation()
            dw, dh = di.getDisplayModeWidth(0), di.getDisplayModeHeight(0)
            props = WindowProperties()
            props.setUndecorated(True)
            props.setOrigin(0, 0)
            props.setSize(dw if dw > 0 else size[0], dh if dh > 0 else size[1])
            base.win.requestProperties(props)
        self.onscreen = onscreen
        base.disableMouse()
        base.textures = self.textures = TextureLibrary()
        self.models = ModelCache()
        base.render.setShaderAuto()
        if msaa:
            base.render.setAntialias(AntialiasAttrib.M_multisample)
        base.cam.node().setCameraMask(MAIN_MASK)
        base.camLens.setNearFar(0.04, 3000.0)
        if size[0] > 0 and size[1] > 0:
            base.camLens.setAspectRatio(size[0] / size[1])
        base.camLens.setFov(BASE_HFOV)
        base.setBackgroundColor(0.78, 0.74, 0.7, 1)
        self.render = base.render
        self.world = world
        self.shadows = shadows
        self.shadow_size = int(shadow_size)
        self.scene = WorldScene(base, world.map, self.textures, shadows=shadows,
                                shadow_size=self.shadow_size)
        self.effects = Effects(base, self.textures)
        self.hud_font = None
        self.hud = None
        self.vm = None
        self.audio = None
        if hud:
            from .hud import HUD
            self.hud = HUD(base, self.textures, pov)
            self.hud_font = self.hud.font
        if viewmodel:
            from .viewmodel import ViewModel
            self.vm = ViewModel(base, self.models, world.agents[pov].team)
        if audio:
            from .audio import Audio
            self.audio = Audio(base, os.path.join(ASSET_DIR, "sfx"))
        self.soldiers: dict[int, SoldierView] = {}
        self.proj_nodes: dict = {}
        self.pov = pov
        self.owner = pov  # human being played; pov may follow a teammate while spectating
        self.spectate_id = None
        self.third_person = False
        self.watch = False          # free spectator: every agent is a bot
        self.god = True             # god view; False follows one bot in first person
        self.fly_pos = np.zeros(3)
        self.fly_yaw = 90.0
        self.fly_pitch = -32.0
        self.follow_id = None
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
        self.on_reshape = None
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
            self.scene = WorldScene(self.base, world.map, self.textures, shadows=self.shadows,
                                    shadow_size=self.shadow_size)
        self.world = world
        self.pov = pov
        if self.hud is not None:
            self.hud.local_id = pov
            self.hud.clear_transient()
        self.soldiers = {a.id: SoldierView(self, a) for a in world.agents}
        self.proj_nodes = {}
        self.event_seq = world.event_log[-1]["seq"] if world.event_log else 0
        self.effects.clear()
        self.pending_sounds = []
        self.death_cam = None
        self.watch = False
        self.god = True
        if self.hud is not None:
            self.hud.watch = False
            self.hud.god_view = False
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
        if self.on_reshape is not None:
            self.on_reshape(win)

    def cycle_pov(self):
        if self.watch:
            self.cycle_follow(1)
            return
        if self.spectate_active():
            self.cycle_spectate(1)
            return
        n = len(self.world.agents)
        self.set_pov((self.pov + 1) % n)

    def set_pov(self, aid: int):
        """Possess `aid`. Spectating a teammate does not come through here."""
        self.owner = aid
        self.spectate_id = None
        self._apply_pov(aid, own=True)

    def _apply_pov(self, aid: int, own: bool):
        self.pov = aid
        if own and self.hud is not None:
            self.hud.local_id = aid
        if self.vm is not None and self.vm.team != self.world.agents[aid].team:
            self.vm.root.removeNode()
            self.vm.dr.setActive(False)
            self.base.win.removeDisplayRegion(self.vm.dr)
            from .viewmodel import ViewModel
            self.vm = ViewModel(self.base, self.models, self.world.agents[aid].team)
            self._on_window(self.base.win)

    def spectate_active(self) -> bool:
        w = self.world
        return not w.done and not w.agents[self.owner].alive

    def _spectate_candidates(self):
        w = self.world
        me = w.agents[self.owner]
        mates = [a for a in w.agents if a.alive and a.team == me.team and a.id != me.id]
        if mates:
            return mates
        return [a for a in w.agents if a.alive and a.id != me.id]

    def _resolve_spectate(self):
        """Dead, no respawn: follow a living teammate in third person."""
        if not self.spectate_active():
            self.spectate_id = None
            if self.pov != self.owner:
                self._apply_pov(self.owner, own=True)
            return
        ids = [a.id for a in self._spectate_candidates()]
        if not ids:
            self.spectate_id = None
            if self.pov != self.owner:
                self._apply_pov(self.owner, own=False)
            return
        if self.spectate_id not in ids:
            self.spectate_id = ids[0]
        if self.pov != self.spectate_id:
            self._apply_pov(self.spectate_id, own=False)
        watched = self.world.agents[self.pov]
        if watched.alive:
            self.first_person_now = False

    def cycle_spectate(self, direction: int):
        ids = [a.id for a in self._spectate_candidates()]
        if not ids:
            return
        if self.spectate_id in ids:
            i = (ids.index(self.spectate_id) + direction) % len(ids)
        else:
            i = 0
        self.spectate_id = ids[i]
        self._apply_pov(self.spectate_id, own=False)

    def toggle_third_person(self):
        self.third_person = not self.third_person
        if 0 <= self.owner < len(self.world.agents):
            self.world.agents[self.owner].third_person = self.third_person and not self.watch

    def inspect(self):
        """F key: weapon inspect animation."""
        a = self.world.agents[self.pov]
        if self.vm is None or not a.alive or self.third_person or not self.vm.inspect():
            return
        self._sound("draw", None, 0.35, 0.8)

    # ------------------------------------------------------------------ events
    def start_watch(self):
        """God-view camera over the current map. F drops into a bot's third-person camera."""
        self.watch = True
        self.god = True
        self.third_person = True
        self.follow_id = None
        c = self.scene.center
        ext = float(self.scene.extent)
        self.fly_yaw = 90.0
        self.fly_pitch = -48.0
        self.fly_pos = np.array([float(c.x), float(c.y) - ext * 0.12, 8.0 + ext * 0.18], dtype=np.float64)
        if self.hud is not None:
            self.hud.watch = True
            self.hud.god_view = True
        if self.vm is not None:
            self.vm.set_visible(False)

    def toggle_god(self):
        if not self.watch:
            return
        if self.god:
            aid = self._pick_follow()
            if aid is None:
                return
            self.god = False
            self.follow_id = aid
            self.third_person = True
            self._apply_pov(aid, own=False)
            return
        a = self.world.agents[self.pov]
        yr = math.radians(a.yaw)
        self.fly_yaw = a.yaw
        self.fly_pitch = -42.0
        self.fly_pos = np.array([a.pos[0] - math.cos(yr) * 14.0,
                                 a.pos[1] - math.sin(yr) * 14.0,
                                 a.pos[2] + 14.0], dtype=np.float64)
        self.god = True

    def cycle_follow(self, direction: int):
        ids = [a.id for a in self.world.agents if a.alive]
        if not ids:
            return
        cur = self.follow_id if self.follow_id in ids else self.pov
        if cur in ids:
            i = (ids.index(cur) + direction) % len(ids)
        else:
            i = 0
        self.follow_id = ids[i]
        self._apply_pov(self.follow_id, own=False)

    def _pick_follow(self):
        alive = [a for a in self.world.agents if a.alive]
        if not alive:
            return None
        yr = math.radians(self.fly_yaw)
        fwd = (math.cos(yr), math.sin(yr))
        best, score = alive[0].id, -1e9
        for a in alive:
            dx, dy = a.pos[0] - self.fly_pos[0], a.pos[1] - self.fly_pos[1]
            dist = math.hypot(dx, dy) + 0.75
            facing = (dx * fwd[0] + dy * fwd[1]) / dist
            s = facing * 8.0 - dist
            if s > score:
                best, score = a.id, s
        return best

    def _watch_follow(self):
        if self.god:
            return
        a = self.world.agents[self.follow_id] if self.follow_id is not None else None
        if a is not None and a.alive:
            if self.pov != a.id:
                self._apply_pov(a.id, own=False)
            return
        ids = [x.id for x in self.world.agents if x.alive]
        if not ids:
            self.god = True
            return
        self.follow_id = ids[0]
        self._apply_pov(self.follow_id, own=False)

    def fly_look(self, dx, dy, sensitivity):
        k = 0.022 * sensitivity
        self.fly_yaw = (self.fly_yaw - dx * k) % 360.0
        self.fly_pitch = max(-89.0, min(89.0, self.fly_pitch - dy * k))

    def fly_move(self, dt, forward, strafe, vertical, fast):
        speed = 52.0 if fast else 18.0
        vx, vy, vz = fly_velocity(self.fly_yaw, forward, strafe, vertical)
        self.fly_pos[0] += vx * speed * dt
        self.fly_pos[1] += vy * speed * dt
        self.fly_pos[2] = max(-40.0, min(160.0, self.fly_pos[2] + vz * speed * dt))

    def _god_camera(self):
        cam = self.base.camera
        p = self.fly_pos
        cam.setPos(float(p[0]), float(p[1]), float(p[2]))
        cam.setHpr(self.fly_yaw - 90.0, self.fly_pitch, 0.0)
        if abs(self.fov - GOD_FOV) > 0.01:
            self.fov = GOD_FOV
            self.base.camLens.setFov(self.fov)

    def listener(self):
        cam = self.base.camera
        q = cam.getQuat(self.render)
        return cam.getPos(self.render), q.getForward(), q.getRight(), q.getUp()

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
        fp = self.first_person_now
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
                    self.effects.muzzle(start, d, big=e["weapon"] in ("awm", "nova"))
                if not own or random.random() < 0.6:
                    self.effects.tracer(start, end)
                for p_end, p_normal, p_mat, p_agent in e.get("pellets", ()):
                    p_end = Point3(*p_end)
                    if random.random() < 0.5:
                        self.effects.tracer(start, p_end)
                    if p_agent is not None:
                        d = p_end - origin
                        d.normalize()
                        self.effects.blood_hit(p_end, d, False)
                    elif p_mat is not None:
                        self.effects.impact(p_end, p_normal, p_mat)
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
                elif own and e["weapon"] == "nova":
                    self.pending_sounds.append((w.time + 0.3, "pump", None, 0.7))
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
            elif t == "fire":
                p = Point3(*e["pos"])
                self.effects.fire_zone(e["proj"], e["pos"], e["radius"], e["duration"])
                self._sound("explosion", p, 0.45, random.uniform(1.15, 1.35), 14.0)
                self._sound("smoke_hiss", p, 0.5, 0.55, 10.0)
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
                vol = 0.95 if e["agent"] == me else 0.7
                # timed to the viewmodel's reload keyframes (mag yank, slam, bolt rack)
                self.pending_sounds.append((w.time + e["duration"] * 0.2, "mag_out", e["agent"], vol))
                self.pending_sounds.append((w.time + e["duration"] * 0.55, "mag_in", e["agent"], vol))
                self.pending_sounds.append((w.time + e["duration"] * 0.76, "bolt", e["agent"], vol * 0.85))
            elif t == "switch":
                if e["agent"] == me:
                    self._sound("draw", None, 0.5)
            elif t == "footstep":
                a = w.agents[e["agent"]]
                # Real hard-surface samples are quieter / shorter than the old synth clank.
                vol = 0.38 if e["agent"] == me else 0.85
                self._sound(f"step{random.randint(0, 3)}", Point3(*e["pos"]), vol, random.uniform(0.92, 1.08), 5.0)
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
                self.effects.flashbang(p)
                self._sound("flash_bang", p, 1.0, 1.0, 12.0)
            elif t == "round_end" and e["team"] >= 0:
                self._sound("round_win" if e["team"] == w.agents[me].team else "round_lose", None, 0.7)
            elif t == "round_start":
                self.effects.clear()
                self.pending_sounds.clear()
                self._sound("round_start", None, 0.6)
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
                m.set_weapon(a.weapon.name)
                if sv.tag_node.getText() != a.name:
                    sv.tag_node.setText(a.name)
                crouch = (C.STAND_HEIGHT - a.height) / (C.STAND_HEIGHT - C.CROUCH_HEIGHT)
                speed = math.hypot(a.vel[0], a.vel[1])
                m.pose(dt, speed, crouch, a.pitch + a.punch[0], a.on_ground,
                       a.lean_off / LEAN_DIST if LEAN_DIST else 0.0)
                hidden = a.id == self.pov and self.first_person_now and not (self.watch and self.god)
                if hidden:
                    m.root.hide()
                    sv.tag.hide()
                    sv.plate.hide()
                else:
                    m.root.show()
                    if self.watch:
                        if a.id == self.pov and not self.god:
                            sv.tag.hide()
                            sv.plate.hide()
                        else:
                            sv.tag.show()
                            sv.tag.setZ(a.height + 0.58)
                            tint = _TEAM_TINT[a.team]
                            sv.tag_node.setTextColor(*tint)
                            flashed, burning = agent_markers(w, a)
                            frac = max(0.0, min(1.0, a.hp / C.MAX_HP))
                            if frac < 0.35:
                                fill = (1.0, 0.28, 0.2, 1.0)
                            elif frac < 0.6:
                                fill = (1.0, 0.78, 0.25, 1.0)
                            else:
                                fill = tint
                            sv.set_overhead(a.height + 0.32, frac, fill, flashed, burning)
                    else:
                        sv.plate.hide()
                        sv.tag.setZ(2.1)
                        sv.tag_node.setTextColor(0.5, 1.0, 0.55, 1)
                        mine = w.agents[self.pov].team
                        if a.team == mine and a.id != self.pov:
                            sv.tag.show()
                        else:
                            sv.tag.hide()
            else:
                sv.tag.hide()
                sv.plate.hide()
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
        spec_view = self.watch or self.spectate_active()
        scoped_fp = bool(a.alive and a.ads and spec is not None and spec.scope_overlay and not spec_view)
        want_ads = bool(a.alive and a.ads and spec is not None and spec.ads_fov)
        self.tp_ads += ((1.0 if want_ads else 0.0) - self.tp_ads) * min(1.0, dt * 14.0)
        self.first_person_now = a.alive and not spec_view and (not self.third_person or scoped_fp)
        if self.first_person_now:
            self.death_cam = None
            pos = a.prev_pos + (a.pos - a.prev_pos) * alpha
            punch = a.prev_punch + (a.punch - a.prev_punch) * alpha
            off = a.lean_vec()
            cam.setPos(pos[0] + off[0], pos[1] + off[1],
                       pos[2] + a.height - C.EYE_FROM_TOP - abs(a.lean_off) * 0.12)
            # through a scope the world stays upright; the reticle carries the lean angle
            roll = 0.0 if scoped_fp else ((a.lean_off / LEAN_DIST) * LEAN_ROLL if LEAN_DIST else 0.0)
            cam.setHpr(a.yaw + punch[1] - 90.0 + jitter[0], a.pitch + punch[0] + jitter[1], jitter[2] + roll)
        elif a.alive:
            self._third_person_camera(a, alpha, jitter)
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
        if want_ads:
            target_fov = spec.ads_fov
            if self.third_person and not scoped_fp:
                target_fov = min(BASE_HFOV, spec.ads_fov * 1.12)
        if abs(target_fov - self.fov) > 0.01:
            rate = 20.0 if target_fov < self.fov else 14.0
            self.fov += (target_fov - self.fov) * min(1.0, dt * rate)
            if scoped_fp:
                self.fov = target_fov
            self.base.camLens.setFov(self.fov)

    def _third_person_camera(self, a, alpha, jitter):
        """Over-the-shoulder camera framing the upper body. Camera orientation follows the
        agent's smooth look angles directly so the crosshair never jumps at geometry edges;
        bullets raycast from `a.eye()` toward the point under the crosshair (`World.aim_angles`)."""
        pos = a.prev_pos + (a.pos - a.prev_pos) * alpha
        punch = a.prev_punch + (a.punch - a.prev_punch) * alpha
        cam_pos, _ = self.world.third_person_camera(a, pos, punch, self.tp_ads)
        cam = self.base.camera
        cam.setPos(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        roll = (a.lean_off / LEAN_DIST) * LEAN_ROLL * 0.5 if LEAN_DIST else 0.0
        cam.setHpr(a.yaw + punch[1] - 90.0 + jitter[0], a.pitch + punch[0] + jitter[1], jitter[2] + roll)

    def vfov(self) -> float:
        return float(self.base.camLens.getFov()[1])

    def sync(self, dt: float, alpha: float = 1.0):
        self.t += dt
        if 0 <= self.owner < len(self.world.agents):
            self.world.agents[self.owner].third_person = self.third_person and not self.watch
        if self.watch:
            self._watch_follow()
            if self.god:
                self.first_person_now = False
        else:
            self._resolve_spectate()
        if self.hud is not None:
            self.hud.watch = self.watch
            self.hud.god_view = self.watch and self.god
        self._process_events()
        if self.watch and self.god:
            self._god_camera()
            self.first_person_now = False
        else:
            self._camera(dt, alpha)
        self._sync_soldiers(dt, alpha)
        self._sync_projectiles(dt)
        self.scene.update(self.t)
        self.effects.update(dt)
        a = self.world.agents[self.pov]
        fp = self.first_person_now
        scope = bool(fp and a.ads and a.slot in a.inv and a.spec.scope_overlay
                     and (self.vm is None or self.vm.ads_blend > 0.85 or self.third_person))
        if self.vm is not None:
            self.vm.set_visible(fp)
            # The local mouse belongs to the dead observer, not the watched player.
            # Passing it through made a teammate's first-person gun sway while spectating.
            mdx, mdy = (0.0, 0.0) if self.watch or self.spectate_active() else (self.mouse_dx, self.mouse_dy)
            self.vm.update(dt, self.world, a, mdx, mdy,
                           self.scene.sun_np.getQuat(self.base.camera))
        self.base.camLens.setFilmOffset(0.0, 0.0)
        if self.hud is not None:
            self.hud.watch = self.watch
            self.hud.god_view = self.watch and self.god
            watching = None if self.watch else (a if self.spectate_active() and a.id != self.owner else None)
            self.hud.update(dt, self.world, a, self.vfov(), self.show_board, scope,
                            spectating=watching, first_person=fp and not (self.watch and self.god))

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

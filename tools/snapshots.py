"""Render a set of staged screenshots (visual regression / docs).

    py tools/snapshots.py [out_dir]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename, Point3  # noqa: E402

from fpsrl import Action, Bot, MatchConfig, World  # noqa: E402
from fpsrl.render.viewer import Viewer  # noqa: E402


def main(out="shots"):
    os.makedirs(out, exist_ok=True)
    world = World(MatchConfig(team_size=4, seed=5, primaries=["ak47", "awm"] + ["ak47"] * 6))
    v = Viewer(world, pov=0, onscreen=True, size=(1280, 720), vsync=False)
    bots = [Bot(world, a.id, "normal", seed=a.id) for a in world.agents if a.id != 0]
    me = world.agents[0]

    def run(seconds, act=None, watch=True):
        n = int(seconds * 60)
        for _ in range(n):
            acts = {b.aid: b.act() for b in bots}
            if act is not None:
                acts[0] = act
            world.step(acts)
            if watch:
                v.sync(world.dt)
                v.base.graphicsEngine.renderFrame()

    def snap(name, frames=3):
        for _ in range(frames):
            v.sync(1 / 60)
            v.base.graphicsEngine.renderFrame()
        path = os.path.join(out, name + ".png")
        v.base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path)

    def place(x, y, z, yaw, pitch=0.0):
        me.pos[:] = (x, y, z)
        me.prev_pos[:] = me.pos
        me.yaw, me.pitch = yaw, pitch
        me.vel[:] = 0

    # freeze the opponents for staged shots
    for a in world.agents[1:]:
        a.protect_until = 1e9
    run(0.5)
    place(-30.0, -1.0, 0.4, 5.0, -2.0)
    run(0.6, Action())
    snap("01_hip_ak")
    run(0.4, Action(alt=True))
    snap("02_ads_ak")
    run(0.1, Action(fire=True, alt=True))
    snap("03_fire_ads")
    run(0.12, Action(fire=True))
    snap("04_fire_hip")
    run(0.6, Action())
    world.set_primary(me, "awm")
    me.spawn_time = world.time
    run(1.0, Action())
    snap("05_awm_hip")
    run(0.5, Action(alt=True))
    snap("06_awm_scope")
    run(0.1, Action(switch=3))
    run(0.5, Action())
    snap("07_knife")
    run(0.05, Action(fire=True))
    run(0.08, Action())
    snap("08_knife_slash")
    run(0.1, Action(switch=4))
    run(0.6, Action())
    snap("09_frag")
    place(-26.0, 0.0, 0.0, 0.0, 18.0)
    run(0.1, Action(fire=True))
    run(0.3, Action())
    place(-34.0, -4.0, 1.2, 0.0, 0.0)
    from fpsrl.sim import angles_to
    for _ in range(300):
        if world.projectiles:
            me.yaw, me.pitch = angles_to(me.eye(), world.projectiles[0].pos)
        evs = world.step({b.aid: b.act() for b in bots} | {0: Action()})
        v.sync(world.dt)
        v.base.graphicsEngine.renderFrame()
        if any(e["type"] == "explode" for e in evs):
            break
    run(0.1, Action())
    snap("10_explosion", 1)
    run(0.1, Action(switch=5))
    run(0.6, Action())
    place(-26.0, 0.0, 0.0, 0.0, 15.0)
    run(0.1, Action(fire=True))
    run(4.0, Action())
    place(-34.0, -4.0, 1.2, 10.0, -5.0)
    run(0.5, Action())
    snap("11_smoke")

    # third person / overview shots with a free camera
    v.third_person = True
    run(3.0, Action())
    snap("12_third_person")
    v.third_person = False
    for a in world.agents[1:]:
        a.protect_until = 0
    run(6.0, None)
    cam = v.base.camera

    def free_cam(name, pos, look):
        v.sync(1 / 60)
        cam.setPos(*pos)
        cam.lookAt(Point3(*look))
        v.vm.set_visible(False)
        v.hud.root.hide()
        v.base.graphicsEngine.renderFrame()
        v.base.graphicsEngine.renderFrame()
        path = os.path.join(out, name + ".png")
        v.base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path)
        v.hud.root.show()

    free_cam("13_overview", (-62, -38, 30), (0, 0, 0))
    free_cam("14_soldiers", (world.agents[4].pos[0] - 3, world.agents[4].pos[1] - 3, 2.4),
             tuple(world.agents[4].pos + [0, 0, 1.0]))
    free_cam("15_side", (5, -40, 4), (0, 0, 2))
    v.base.destroy()


if __name__ == "__main__":
    main(*sys.argv[1:])

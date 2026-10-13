"""Staged screenshots of one map / mode (overview, first person, lean, third-person ADS).

    py tools/map_shots.py <map> <mode> [out_dir] [--bots]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename, Point3  # noqa: E402

from fpsrl import Action, MatchConfig, World  # noqa: E402
from fpsrl.render.viewer import Viewer  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    map_name = args[0] if args else "transport_ship"
    mode = args[1] if len(args) > 1 else "tdm"
    out = args[2] if len(args) > 2 else "shots"
    use_bots = "--bots" in sys.argv
    os.makedirs(out, exist_ok=True)
    world = World(MatchConfig(team_size=4, seed=5, map_name=map_name, mode=mode))
    v = Viewer(world, pov=0, onscreen=True, size=(1280, 720), vsync=False)
    bots = []
    if use_bots:
        from fpsrl import Bot
        bots = [Bot(world, a.id, "normal", seed=a.id) for a in world.agents if a.id != 0]
    me = world.agents[0]
    prefix = f"{map_name}_{mode}"

    def run(seconds, act=None):
        for _ in range(int(seconds * 60)):
            acts = {b.aid: b.act() for b in bots}
            if act is not None:
                acts[0] = act
            world.step(acts)
            v.sync(world.dt)
            v.base.graphicsEngine.renderFrame()

    def snap(name, frames=3):
        for _ in range(frames):
            v.sync(1 / 60)
            v.base.graphicsEngine.renderFrame()
        path = os.path.join(out, f"{prefix}_{name}.png")
        v.base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path)

    run(0.5, Action())
    snap("01_spawn")
    run(0.4, Action(lean=-1))
    snap("02_lean_left")
    run(0.4, Action(lean=1, alt=True))
    snap("03_lean_right_ads")
    v.third_person = True
    run(0.5, Action(alt=True))
    snap("04_tp_ads")
    run(0.5, Action())
    snap("05_tp_hip")
    v.third_person = False
    run(2.0 if use_bots else 0.2, None if use_bots else Action())
    cam = v.base.camera
    c = v.scene.center
    e = v.scene.extent

    def free_cam(name, pos, look):
        v.sync(1 / 60)
        cam.setPos(*pos)
        cam.lookAt(Point3(*look))
        if v.vm is not None:
            v.vm.set_visible(False)
        v.hud.root.hide()
        for _ in range(2):
            v.base.graphicsEngine.renderFrame()
        path = os.path.join(out, f"{prefix}_{name}.png")
        v.base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path)
        v.hud.root.show()

    free_cam("10_overview", (c.x - e * 0.55, c.y - e * 0.55, e * 0.45), (c.x, c.y, 0))
    free_cam("11_top", (c.x, c.y - 1, e * 0.9), (c.x, c.y, 0))
    free_cam("12_overview2", (c.x + e * 0.55, c.y + e * 0.4, e * 0.35), (c.x, c.y, 0))
    v.base.destroy()


if __name__ == "__main__":
    main()

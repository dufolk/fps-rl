"""First-person shots at fixed poses inside a map (indoor maps / destruction check).

    py tools/interior_shots.py [map] [mode] [out_dir]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl import Action, MatchConfig, World  # noqa: E402
from fpsrl.render.viewer import Viewer  # noqa: E402

POSES = {
    "bank": [
        ("lobby", (0.0, -14.0, 0.01, 90.0, 5.0)),
        ("stairs", (-2.5, -1.5, 0.01, 60.0, 15.0)),
        ("ceo", (10.0, -14.5, 4.01, 45.0, -3.0)),
        ("basement", (-3.0, -9.5, -3.99, 180.0, 0.0)),
        ("roof", (-10.0, -12.0, 8.01, 30.0, -5.0)),
    ],
    "desert2": [
        ("mid", (-4.0, -8.0, 0.01, 90.0, 0.0)),
        ("long", (34.0, -10.0, 0.81, 90.0, 0.0)),
        ("a_site", (26.0, 22.0, 1.61, 30.0, 0.0)),
    ],
}


def main():
    map_name = sys.argv[1] if len(sys.argv) > 1 else "bank"
    mode = sys.argv[2] if len(sys.argv) > 2 else "tdm"
    out = sys.argv[3] if len(sys.argv) > 3 else "shots"
    os.makedirs(out, exist_ok=True)
    w = World(MatchConfig(team_size=4, seed=5, map_name=map_name, mode=mode))
    v = Viewer(w, pov=0, onscreen=True, size=(1280, 720))
    me = w.agents[0]

    def run(sec):
        for _ in range(int(sec * 60)):
            for a in w.agents:
                a.frozen = False
            w.step({0: Action()})
            v.sync(w.dt)
            v.base.graphicsEngine.renderFrame()

    def snap(name):
        for _ in range(3):
            v.sync(1 / 60)
            v.base.graphicsEngine.renderFrame()
        path = os.path.join(out, f"{map_name}_in_{name}.png")
        v.base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path)

    run(0.3)
    for name, (x, y, z, yaw, pitch) in POSES.get(map_name, []):
        me.pos[:] = (x, y, z)
        me.prev_pos[:] = me.pos
        me.yaw, me.pitch = yaw, pitch
        me.vel[:] = 0
        run(0.3)
        snap(name)
    if len(w.destruct.panels):
        p = w.destruct.panels[len(w.destruct.panels) // 3]
        w.destruct.explode(p.center, 1.1, "soft", floors=False)
        c = p.center
        me.pos[:] = (c[0] + (2.5 if p.axis == 0 else 0.0), c[1] + (2.5 if p.axis == 1 else 0.0), c[2] - 1.2)
        me.prev_pos[:] = me.pos
        me.yaw = 180.0 if p.axis == 0 else 270.0
        me.pitch = 0.0
        run(0.3)
        snap("breach")
    v.base.destroy()


if __name__ == "__main__":
    main()

"""Screenshots of the Siege HUD: site markers, drone feed (attack) and camera feed (defense).

    py tools/siege_shots.py [out_dir] [att|def]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl.render.game import GameApp  # noqa: E402


def shot(app, path, frames=6):
    for _ in range(frames):
        app.session.update(1 / 60)
        app.base.graphicsEngine.renderFrame()
    app.base.win.saveScreenshot(Filename.fromOsSpecific(path))
    print("saved", path)


def main(out, side):
    os.makedirs(out, exist_ok=True)
    team = 0 if side == "att" else 1
    app = GameApp(team_size=3, map_name="bank", mode="r6", seed=4, size=(1280, 720), team=team)
    s = app.session
    w = s.world
    me = w.agents[s.player]
    for _ in range(30):
        s.update(1 / 60)
    shot(app, os.path.join(out, f"siege_{side}_markers.png"))
    for e in w.agents:
        if e.team != me.team:
            e.spotted_until = w.time + 10.0
    w.gadgets.toggle_view(me)
    print("view", me.view_mode, me.view_ref)
    shot(app, os.path.join(out, f"siege_{side}_feed.png"), 20)
    app.session.stop()
    app.base.destroy()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shots", sys.argv[2] if len(sys.argv) > 2 else "att")

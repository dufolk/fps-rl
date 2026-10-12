"""Screenshots of in-game features: kill medal, F inspect, pause menu. py tools/feature_shots.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl.render.app import App  # noqa: E402
from fpsrl.render.game import LocalSession  # noqa: E402


def main():
    app = App(size=(1280, 720), fullscreen=False, vsync=False)
    app.menu.hide()
    p = app.profile
    p.skins = dict(p.skins, ak47="neon", knife="doppler")
    app.session = s = LocalSession(app.viewer, p, 4, "easy", 0, "ak47", 0, 0, seed=11, autoplay=True,
                                   on_escape=app.toggle_pause)
    base = app.base
    hud = app.viewer.hud
    state = {"stage": "kill", "t": 0.0, "kills": 0}

    def snap(name):
        base.graphicsEngine.renderFrame()
        path = f"shots/feat_{name}.png"
        base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path, flush=True)

    def watch(task):
        me = s.world.agents[s.player]
        st = state
        st["t"] += globalClock.getDt()  # noqa: F821
        if st["stage"] == "kill":
            if me.kills > st["kills"]:
                st["kills"] = me.kills
                st["stage"] = "kill_wait"
                st["t"] = 0.0
                print("kill_fx", hud.kill_fx, flush=True)
            elif s.world.time > 90:
                print("no kill in 90s", flush=True)
                st["stage"] = "inspect"
        elif st["stage"] == "kill_wait" and st["t"] > 0.35:
            snap("kill")
            st["stage"] = "inspect"
            st["t"] = 0.0
        elif st["stage"] == "inspect" and me.alive:
            s.autoplay = False
            s.bots.clear()
            app.viewer.inspect()
            st["stage"] = "inspect_wait"
            st["t"] = 0.0
        elif st["stage"] == "inspect_wait" and st["t"] > 1.3:
            snap("inspect_ak")
            me.slot = 3
            me.ready_at = s.world.time
            st["stage"] = "knife"
            st["t"] = 0.0
        elif st["stage"] == "knife" and st["t"] > 0.8:
            app.viewer.inspect()
            st["stage"] = "knife_wait"
            st["t"] = 0.0
        elif st["stage"] == "knife_wait" and st["t"] > 1.1:
            snap("inspect_knife")
            app.toggle_pause()
            st["stage"] = "pause"
            st["t"] = 0.0
        elif st["stage"] == "pause" and st["t"] > 0.3:
            snap("pause")
            base.userExit()
        return task.cont

    base.taskMgr.add(watch, "feat-watch", sort=50)
    app.run()


if __name__ == "__main__":
    main()

"""Screenshot every menu page: py tools/menu_shots.py [out_dir]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl.render.app import App  # noqa: E402


def main(out="shots"):
    os.makedirs(out, exist_ok=True)
    app = App(size=(1600, 900), fullscreen=False, vsync=False)
    base = app.base

    def frames(n):
        for _ in range(n):
            base.taskMgr.step()

    def snap(name):
        frames(2)
        path = os.path.join(out, name + ".png")
        base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print("saved", path)

    frames(90)
    for page in ("single", "host", "join", "settings", "armory"):
        app.menu.open(page)
        frames(60 if page == "join" else 20)
        snap(f"menu_{page}")
    app.menu._arm_select("knife")
    app.menu._arm_show("knife", "fade")
    frames(30)
    snap("menu_armory_knife")
    base.userExit()


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Screenshots of the in-game B menu: buy menu (desert2 / bomb) and operator pick (bank / r6).

    py tools/ui_shots.py [out_dir]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl.render.game import GameApp  # noqa: E402


def shot(app, path, frames=4):
    for _ in range(frames):
        app.session.update(1 / 60)
        app.base.graphicsEngine.renderFrame()
    app.base.win.saveScreenshot(Filename.fromOsSpecific(path))
    print("saved", path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "shots"
    os.makedirs(out, exist_ok=True)
    app = GameApp(team_size=5, map_name="desert2", mode="bomb", seed=2, size=(1280, 720), team=1)
    s = app.session
    for _ in range(30):
        s.update(1 / 60)
    s.ctrl._buy_or_primary()
    shot(app, os.path.join(out, "ui_buy_menu.png"))
    s._buy("m4a1")
    s._buy("frag")
    s.ctrl.overlay.refresh()
    shot(app, os.path.join(out, "ui_buy_menu_after.png"))
    me = s.world.agents[s.player]
    print("money", me.money, "inv", {k: v.name for k, v in me.inv.items()})
    s.ctrl._buy_or_primary()
    shot(app, os.path.join(out, "ui_bomb_hud.png"))
    app.session.stop()
    app.base.destroy()


def main_r6(out):
    app = GameApp(team_size=5, map_name="bank", mode="r6", seed=2, size=(1280, 720), team=1)
    s = app.session
    for _ in range(30):
        s.update(1 / 60)
    s.ctrl._buy_or_primary()
    shot(app, os.path.join(out, "ui_operator_menu.png"))
    s.ctrl.overlay._pick("castle", "uzi")
    me = s.world.agents[s.player]
    print("operator", me.operator, "inv", {k: v.name for k, v in me.inv.items()})
    shot(app, os.path.join(out, "ui_operator_after.png"), 30)
    app.session.stop()
    app.base.destroy()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "r6":
        main_r6(sys.argv[1])
    else:
        main()

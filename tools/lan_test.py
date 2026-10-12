"""Loopback LAN smoke test with bots driving both players.

    py tools/lan_test.py host  [seconds]     # terminal 1
    py tools/lan_test.py client [seconds]    # terminal 2 (discovers the room via UDP broadcast)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl.net import NetClient, discover  # noqa: E402
from fpsrl.render.app import App  # noqa: E402
from fpsrl.render.game import ClientSession, LocalSession  # noqa: E402


def main(mode="host", seconds="20"):
    seconds = float(seconds)
    app = App(size=(1280, 720), fullscreen=False, vsync=False)
    app.menu.hide()
    p = app.profile
    if mode == "host":
        p.name = "Host"
        app.session = LocalSession(app.viewer, p, 4, "normal", 0, "ak47", 0, 0, seed=5, autoplay=True,
                                   host={"room": "loopback", "port": 47777})
    else:
        p.name = "Guest"
        p.skins = {"ak47": "fire_serpent", "knife": "fade", "deagle": "gold"}
        rooms = discover(1.5)
        print("discovered:", rooms)
        host, port = (rooms[0]["ip"], rooms[0]["port"]) if rooms else ("127.0.0.1", 47777)
        client = NetClient(host, port, p.name, p.skins, team=1)
        app.session = ClientSession(app.viewer, p, client, autoplay=True, on_disconnect=app._disconnected)
    base = app.base
    t0 = time.perf_counter()
    shots = [seconds * 0.5, seconds - 0.5]

    def watch(task):
        el = time.perf_counter() - t0
        s = app.session
        if shots and el >= shots[0]:
            shots.pop(0)
            path = f"shots/lan_{mode}_{len(shots)}.png"
            base.win.saveScreenshot(Filename.fromOsSpecific(path))
            w = s.world if s else None
            me = w.agents[s.player if mode == "host" else s.aid] if w else None
            extra = (f"clients={len(s.server.players())}" if mode == "host" and s else
                     f"pending={len(s.pending)} seq={s.seq}" if s else "session gone")
            print(f"[{mode}] t={el:.1f}s saved {path} world_t={w.time if w else 0:.1f} "
                  f"score={w.team_kills if w else None} me={me.name if me else None} k/d="
                  f"{me.kills if me else 0}/{me.deaths if me else 0} {extra}", flush=True)
        if el >= seconds:
            base.userExit()
        return task.cont

    base.taskMgr.add(watch, "lan-watch", sort=50)
    app.run()


if __name__ == "__main__":
    main(*sys.argv[1:])

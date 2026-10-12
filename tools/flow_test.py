"""Menu flow smoke test through the real menu code paths.

    py tools/flow_test.py host  [seconds]   # single -> pause -> leave -> host room
    py tools/flow_test.py client            # join page -> discovered room -> play -> leave
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import Filename  # noqa: E402

from fpsrl.render.app import App  # noqa: E402


def main(mode="host", seconds="30"):
    seconds = float(seconds)
    app = App(size=(1280, 720), fullscreen=False, vsync=False)
    base = app.base
    t0 = time.perf_counter()
    steps = []

    def snap(name):
        path = f"shots/flow_{name}.png"
        base.win.saveScreenshot(Filename.fromOsSpecific(path))
        print(f"[{mode}] {time.perf_counter() - t0:.1f}s saved {path}", flush=True)

    def log(msg):
        print(f"[{mode}] {time.perf_counter() - t0:.1f}s {msg}", flush=True)

    if mode == "host":
        steps += [
            (1.0, lambda: app.menu.open("single")),
            (1.5, app.start_single),
            (4.0, app.toggle_pause),
            (4.5, lambda: snap("pause")),
            (5.0, app.leave_game),
            (6.0, lambda: snap("back_to_menu")),
            (6.5, lambda: app.menu.open("host")),
            (7.0, lambda: app.menu.page.getChildren() and app.start_host()),
            (9.0, lambda: log(f"hosting, clients={len(app.session.server.players())}")),
            (seconds * 0.6, lambda: log(f"clients={len(app.session.server.players())} "
                                        f"score={app.session.world.team_kills}")),
            (seconds * 0.6 + 0.2, lambda: snap("host_ingame")),
            (seconds, base.userExit),
        ]
    else:
        def join_first():
            rooms = app.menu.discoverer.rooms if app.menu.discoverer else []
            log(f"rooms: {[(r['room'], r['ip'], r['port']) for r in rooms]}")
            if rooms:
                app.menu._join(rooms[0]["ip"], rooms[0]["port"])
            else:
                app.menu._join("127.0.0.1", 47777)

        def report():
            s = app.session
            log(f"session={type(s).__name__ if s else None} "
                + (f"aid={s.aid} pending={len(s.pending)} me={s.world.agents[s.aid].name}" if s else ""))

        steps += [
            (0.5, lambda: app.menu.open("join")),
            (3.0, lambda: snap("join_page")),
            (3.2, join_first),
            (6.0, report),
            (6.2, lambda: snap("client_ingame")),
            (7.0, app.toggle_pause),
            (7.5, lambda: snap("client_pause")),
            (8.0, app.leave_game),
            (9.0, lambda: snap("client_left")),
            (9.2, report),
            (9.5, base.userExit),
        ]

    def tick(task):
        el = time.perf_counter() - t0
        while steps and el >= steps[0][0]:
            steps.pop(0)[1]()
        return task.cont

    base.taskMgr.add(tick, "flow", sort=50)
    app.run()


if __name__ == "__main__":
    main(*sys.argv[1:])

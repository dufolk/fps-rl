"""Play FPS-RL.

    py play.py                       # main menu: single player, LAN host / join, armory, settings
    py play.py --quick               # jump straight into a 4v4 bot match (old behaviour)
    py play.py --quick --primary awm --difficulty hard --team-size 5
    py play.py --quick --map desert2 --mode bomb
    py play.py --quick --map bank --mode r6 --team-size 5
    py play.py --autoplay            # watch a bot play from first person
    py play.py --spectate --map bank --mode r6
"""
import argparse

from fpsrl.maps import MAPS, MODES
from fpsrl.profile import Profile
from fpsrl.render.app import App
from fpsrl.render.game import GameApp
from fpsrl.weapons import PRIMARY_CHOICES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="skip the menu and start a bot match")
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--team", type=int, default=0, choices=[0, 1])
    p.add_argument("--difficulty", default="normal", help="easy | normal | hard | insane | 0..1")
    p.add_argument("--map", default="transport_ship", choices=sorted(MAPS))
    p.add_argument("--mode", default=None, choices=sorted(MODES), help="default: the map's first mode")
    p.add_argument("--primary", default="ak47", choices=PRIMARY_CHOICES)
    p.add_argument("--kill-limit", type=int, default=60)
    p.add_argument("--time-limit", type=float, default=420.0)
    p.add_argument("--size", default="1600x900")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--no-shadows", action="store_true", help="disable shadows (overrides --graphics)")
    p.add_argument("--msaa", type=int, default=None, help="0/2/4/8; default follows --graphics / settings")
    p.add_argument("--graphics", default=None, choices=["low", "medium", "high"],
                   help="quality preset (default: profile setting, usually medium)")
    p.add_argument("--sensitivity", type=float, default=1.6)
    p.add_argument("--autoplay", action="store_true", help="the player is controlled by a bot (implies --quick)")
    p.add_argument("--spectate", action="store_true",
                   help="god-view a bot match; WASD/mouse fly, Q/E up/down, F toggles first person")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--screenshot", default=None, help="save a screenshot after --shot-time seconds and exit")
    p.add_argument("--shot-time", type=float, default=6.0)
    a = p.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    gfx = a.graphics or Profile.load().graphics
    shadows = False if a.no_shadows else None
    if a.quick or a.autoplay or a.spectate or a.screenshot:
        diff = a.difficulty
        try:
            diff = float(diff)
        except ValueError:
            pass
        app = GameApp(team_size=a.team_size, difficulty=diff, primary=a.primary, kill_limit=a.kill_limit,
                      time_limit=a.time_limit, size=(w, h), fullscreen=a.fullscreen, vsync=not a.no_vsync,
                      shadows=shadows, msaa=a.msaa, graphics=gfx, sensitivity=a.sensitivity,
                      autoplay=a.autoplay, seed=a.seed, screenshot=a.screenshot, shot_time=a.shot_time,
                      team=a.team, map_name=a.map, mode=a.mode, spectate=a.spectate)
    else:
        if a.graphics:
            prof = Profile.load()
            prof.graphics = a.graphics
            prof.save()
        app = App(size=(w, h), fullscreen=True if a.fullscreen else None, vsync=not a.no_vsync,
                  shadows=shadows, msaa=a.msaa)
    app.run()


if __name__ == "__main__":
    main()

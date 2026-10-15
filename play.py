"""Experiment environment: elimination, fixed loadouts (rifle / sniper / shotgun), no main menu.

    py play.py                                  # human play: you vs bots (4v4)
    py play.py --loadout sniper --difficulty hard --team-size 5
    py play.py --spectate                       # watch a bot match (god view / first person)
    py play.py --autoplay                       # a bot drives the player, first-person camera
"""
import argparse

from fpsrl.maps import MAPS
from fpsrl.render.game import GRAPHICS_PRESETS, GameApp
from fpsrl.weapons import LOADOUT_KEYS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spectate", action="store_true",
                   help="spectator mode: every agent is a bot; WASD/mouse fly, Q/E up/down, F toggles first person")
    p.add_argument("--autoplay", action="store_true", help="the player is controlled by a bot")
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--team", type=int, default=0, choices=[0, 1])
    p.add_argument("--difficulty", default="normal", help="easy | normal | hard | insane | 0..1")
    p.add_argument("--map", default="desert2", choices=sorted(MAPS))
    p.add_argument("--loadout", default="random", choices=[*LOADOUT_KEYS, "random"],
                   help="player loadout (bots always draw one at random)")
    p.add_argument("--win-rounds", type=int, default=None, help="rounds needed to win the match (default 13)")
    p.add_argument("--size", default="1600x900")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true")
    p.add_argument("--graphics", default="medium", choices=sorted(GRAPHICS_PRESETS))
    p.add_argument("--sensitivity", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--screenshot", default=None, help="save a screenshot after --shot-time seconds and exit")
    p.add_argument("--shot-time", type=float, default=6.0)
    a = p.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    diff = a.difficulty
    try:
        diff = float(diff)
    except ValueError:
        pass
    params = {}
    if a.win_rounds:
        params = {"win_rounds": a.win_rounds, "half_rounds": max(1, a.win_rounds - 1)}
    GameApp(team_size=a.team_size, difficulty=diff, loadout=a.loadout, size=(w, h), fullscreen=a.fullscreen,
            vsync=not a.no_vsync, graphics=a.graphics, sensitivity=a.sensitivity, autoplay=a.autoplay,
            seed=a.seed, screenshot=a.screenshot, shot_time=a.shot_time, team=a.team, map_name=a.map,
            mode_params=params, spectate=a.spectate).run()


if __name__ == "__main__":
    main()

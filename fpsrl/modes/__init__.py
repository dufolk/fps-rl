from .base import GameMode, TeamDeathmatch
from .bio import Biohazard
from .domination import Domination
from .rounds import ATT, DEF, BombDefusal, Elimination, RoundMode
from .siege import Siege

MODE_CLASSES = {"tdm": TeamDeathmatch, "dom": Domination, "bomb": BombDefusal, "elim": Elimination,
                "r6": Siege, "bio": Biohazard}


def make_mode(world) -> GameMode:
    return MODE_CLASSES[world.cfg.mode](world)


__all__ = ["GameMode", "TeamDeathmatch", "Domination", "BombDefusal", "Elimination", "RoundMode",
           "Siege", "Biohazard", "MODE_CLASSES", "make_mode", "ATT", "DEF"]

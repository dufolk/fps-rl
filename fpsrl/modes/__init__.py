from .base import GameMode, TeamDeathmatch
from .domination import Domination
from .rounds import ATT, DEF, BombDefusal, RoundMode
from .siege import Siege

MODE_CLASSES = {"tdm": TeamDeathmatch, "dom": Domination, "bomb": BombDefusal, "r6": Siege}


def make_mode(world) -> GameMode:
    return MODE_CLASSES[world.cfg.mode](world)


__all__ = ["GameMode", "TeamDeathmatch", "Domination", "BombDefusal", "RoundMode", "Siege",
           "MODE_CLASSES", "make_mode", "ATT", "DEF"]

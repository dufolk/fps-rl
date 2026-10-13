from .base import Box, MapData, Prop

_CACHE = {}

# name -> (zh display, en display, supported modes)
MAPS = {
    "transport_ship": ("运输船", "Transport Ship", ("tdm", "dom")),
    "desert2": ("沙漠灰", "Desert 2", ("bomb", "tdm")),
    "hp_town": ("占点小镇", "HP Town", ("dom", "tdm")),
    "bank": ("银行", "Bank", ("r6", "tdm")),
}

MODES = {
    "tdm": ("团队竞技", "Team Deathmatch"),
    "dom": ("占点模式", "Domination"),
    "bomb": ("爆破模式", "Bomb Defusal"),
    "r6": ("彩六模式", "Siege"),
}


def _builder(name: str):
    if name == "transport_ship":
        from .transport_ship import build
    elif name == "desert2":
        from .desert2 import build
    elif name == "hp_town":
        from .hp_town import build
    elif name == "bank":
        from .bank import build
    else:
        raise KeyError(f"unknown map {name!r}")
    return build


def load_map(name: str = "transport_ship") -> MapData:
    if name not in _CACHE:
        _CACHE[name] = _builder(name)
    return _CACHE[name]()


def default_mode(name: str) -> str:
    return MAPS[name][2][0] if name in MAPS else "tdm"


__all__ = ["Box", "MapData", "Prop", "load_map", "MAPS", "MODES", "default_mode"]

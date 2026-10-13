from .base import Box, MapData, Prop

_CACHE = {}

# name -> (zh display, en display, supported modes)
MAPS = {
    "transport_ship": ("运输船", "Transport Ship", ("tdm", "dom")),
    "desert2": ("沙漠灰", "Desert 2", ("bomb", "elim", "tdm")),
    "hp_town": ("占点小镇", "HP Town", ("dom", "tdm")),
    "bank": ("银行", "Bank", ("r6", "tdm")),
}

MODES = {
    "tdm": ("团队竞技", "Team Deathmatch"),
    "dom": ("占点模式", "Domination"),
    "bomb": ("爆破模式", "Bomb Defusal"),
    "elim": ("歼灭模式", "Elimination"),
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


def resolve_map_mode(map_name, mode) -> tuple[str, str]:
    """A startable pair. A real mode the current map does not support switches to the first map
    that does, so choosing 彩六 lands on the bank and 爆破 / 歼灭 land on desert2."""
    map_name = map_name if map_name in MAPS else "transport_ship"
    if mode in MAPS[map_name][2]:
        return map_name, mode
    if mode in MODES:
        for name, info in MAPS.items():
            if mode in info[2]:
                return name, mode
    return map_name, default_mode(map_name)


__all__ = ["Box", "MapData", "Prop", "load_map", "MAPS", "MODES", "default_mode", "resolve_map_mode"]

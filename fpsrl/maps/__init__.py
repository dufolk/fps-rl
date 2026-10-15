from .base import Box, MapData, Prop

_CACHE = {}

# name -> (zh display, en display)
MAPS = {
    "desert2": ("沙漠灰", "Desert 2"),
}


def _builder(name: str):
    if name == "desert2":
        from .desert2 import build
    else:
        raise KeyError(f"unknown map {name!r}")
    return build


def load_map(name: str = "desert2") -> MapData:
    if name not in _CACHE:
        _CACHE[name] = _builder(name)
    return _CACHE[name]()


__all__ = ["Box", "MapData", "Prop", "load_map", "MAPS"]

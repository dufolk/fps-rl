from .base import Box, MapData, Prop

_CACHE = {}

# name -> (zh display, en display)
MAPS = {
    "garage": ("假车库", "Garage"),
}


def _builder(name: str):
    if name == "garage":
        from .garage import build
    else:
        raise KeyError(f"unknown map {name!r}")
    return build


def load_map(name: str = "garage") -> MapData:
    if name not in _CACHE:
        _CACHE[name] = _builder(name)
    return _CACHE[name]()


__all__ = ["Box", "MapData", "Prop", "load_map", "MAPS"]

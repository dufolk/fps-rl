from .base import Box, MapData, Prop

_REGISTRY = {}


def load_map(name: str = "transport_ship") -> MapData:
    if name not in _REGISTRY:
        if name == "transport_ship":
            from .transport_ship import build
        else:
            raise KeyError(f"unknown map {name!r}")
        _REGISTRY[name] = build
    return _REGISTRY[name]()


__all__ = ["Box", "MapData", "Prop", "load_map"]

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Box:
    mn: tuple
    mx: tuple
    mat: str
    solid: bool = True
    blocks_bullets: bool = True
    visible: bool = True
    tag: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class Prop:
    """Visual-only object drawn by the renderer (collision, if any, is a separate Box)."""
    kind: str
    pos: tuple
    params: dict = field(default_factory=dict)


@dataclass
class MapData:
    name: str
    boxes: list
    props: list
    spawns: dict  # side -> list of (x, y, z, yaw)
    bounds: tuple  # ((xmin, ymin), (xmax, ymax)) of the playable area
    strategic_points: list
    water_z: float = -5.0
    sun_dir: tuple = (-0.55, 0.45, -0.55)
    theme: str = "garage"  # renderer branch
    # zone: {"center": (cx, cy), "radius": float, "dps": float}  (poison zone outside the OBJ edge)
    objectives: dict = field(default_factory=dict)
    nav_max_floor: float = 1.3  # floors above this are not walkable for the nav grid
    nav_bounds: tuple | None = None  # optional tighter bounds for the bot NavGrid
    mesh: str = ""  # imported render mesh under maps/data/<mesh>/ (collision boxes are then invisible)
    ground_z: float = -0.03  # height of the renderer's far ground plane

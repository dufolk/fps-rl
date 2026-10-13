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
    spawns: dict  # team -> list of (x, y, z, yaw)
    bounds: tuple  # ((xmin, ymin), (xmax, ymax)) of the playable area
    strategic_points: list
    water_z: float = -5.0
    sun_dir: tuple = (-0.55, 0.45, -0.55)
    theme: str = "ship"  # renderer branch: ship | desert | town | bank
    modes: tuple = ("tdm",)
    # objectives:
    #   points: [{"name", "poly": [[x, y]...], "zmin", "zmax"}]            (domination)
    #   sites:  [{"name", "poly", "zmin", "zmax"}]                        (bomb / R6 sites)
    #   site_sets: [[site index, ...], ...]                               (R6: sites defended together)
    #   defender_spawns: {set index: [(x, y, z, yaw), ...]}               (R6)
    objectives: dict = field(default_factory=dict)
    buy_zones: dict = field(default_factory=dict)  # side -> list of polygons
    cameras: list = field(default_factory=list)  # [(x, y, z, yaw, pitch)] security cameras (R6)
    nav_max_floor: float = 1.3  # floors above this are not walkable for the nav grid (ship bridge roof)
    nav_levels: tuple = ()  # extra z heights for multi-storey maps (handled by the nav grid)
    mesh: str = ""  # imported render mesh under maps/data/<mesh>/ (collision boxes are then invisible)

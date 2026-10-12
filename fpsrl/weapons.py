"""Weapon definitions. All angles in degrees, times in seconds."""
from __future__ import annotations

from dataclasses import dataclass, field

SLOT_PRIMARY, SLOT_PISTOL, SLOT_MELEE, SLOT_FRAG, SLOT_SMOKE = 1, 2, 3, 4, 5
NUM_SLOTS = 5


def _ak_recoil():
    pitch = [0.75, 0.85, 0.95, 1.0, 1.0, 0.95, 0.85, 0.75, 0.65, 0.55] + [0.28] * 20
    yaw = ([0.05, -0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.1, -0.1, -0.3]
           + [-0.45] * 5 + [0.55] * 6 + [-0.4] * 9)
    return list(zip(pitch, yaw))


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    display: str
    slot: int
    kind: str  # "gun" | "melee" | "grenade"
    damage: float = 0.0
    interval: float = 0.1
    auto: bool = False
    mag: int = 0
    reserve: int = 0
    reload_time: float = 2.0
    draw_time: float = 0.5
    move_mult: float = 1.0
    ads_move_mult: float = 1.0
    ads_fov: float | None = None  # horizontal FOV while aiming, None = cannot aim
    scope_overlay: bool = False
    unscope_after_shot: bool = False
    spread_base: float = 0.0
    spread_ads: float = 0.0
    spread_move: float = 0.0
    spread_air: float = 0.0
    spread_per_shot: float = 0.0
    spread_max: float = 0.0
    spread_recover: float = 8.0
    crouch_mult: float = 0.75
    recoil: tuple = ((0.0, 0.0),)
    recoil_jitter: float = 0.15
    punch_recover: float = 7.0
    head_mult: float = 4.0
    leg_mult: float = 0.75
    falloff: float = 0.97  # damage multiplier per 10 m
    range: float = 200.0
    pellets: int = 1
    # melee
    heavy_damage: float = 0.0
    heavy_interval: float = 1.0
    melee_range: float = 1.7
    heavy_range: float = 1.45
    hit_delay: float = 0.1
    heavy_hit_delay: float = 0.3
    # grenade
    fuse: float = 2.8
    throw_speed: float = 16.0
    windup: float = 0.22
    blast_radius: float = 0.0
    smoke_radius: float = 0.0
    smoke_duration: float = 0.0
    extra: dict = field(default_factory=dict)


WEAPONS: dict[str, WeaponSpec] = {
    "ak47": WeaponSpec(
        name="ak47", display="AK-47", slot=SLOT_PRIMARY, kind="gun",
        damage=36, interval=0.1, auto=True, mag=30, reserve=90,
        reload_time=2.4, draw_time=0.6, move_mult=0.92, ads_move_mult=0.8, ads_fov=62,
        spread_base=0.18, spread_ads=0.07, spread_move=3.2, spread_air=7.0,
        spread_per_shot=0.38, spread_max=4.5, spread_recover=9.0,
        recoil=tuple(_ak_recoil()), recoil_jitter=0.2, punch_recover=7.5,
        head_mult=4.0, leg_mult=0.75, falloff=0.98,
    ),
    "awm": WeaponSpec(
        name="awm", display="AWM", slot=SLOT_PRIMARY, kind="gun",
        damage=115, interval=1.45, auto=False, mag=5, reserve=20,
        reload_time=3.4, draw_time=0.9, move_mult=0.85, ads_move_mult=0.5, ads_fov=24,
        scope_overlay=True, unscope_after_shot=True,
        spread_base=6.5, spread_ads=0.02, spread_move=7.0, spread_air=14.0,
        spread_per_shot=0.0, spread_max=0.0, spread_recover=10.0,
        recoil=((3.2, 0.4),), recoil_jitter=0.3, punch_recover=4.5,
        head_mult=3.0, leg_mult=0.85, falloff=1.0,
    ),
    "deagle": WeaponSpec(
        name="deagle", display="Desert Eagle", slot=SLOT_PISTOL, kind="gun",
        damage=50, interval=0.24, auto=False, mag=7, reserve=35,
        reload_time=2.0, draw_time=0.4, move_mult=1.0,
        spread_base=0.3, spread_move=1.8, spread_air=5.0,
        spread_per_shot=1.3, spread_max=5.0, spread_recover=6.0,
        recoil=((2.0, 0.25), (2.2, -0.3), (2.4, 0.35)), recoil_jitter=0.3, punch_recover=6.0,
        head_mult=3.0, leg_mult=0.75, falloff=0.95,
    ),
    "knife": WeaponSpec(
        name="knife", display="M9 Knife", slot=SLOT_MELEE, kind="melee",
        damage=40, interval=0.45, draw_time=0.3, move_mult=1.08,
        heavy_damage=90, heavy_interval=1.05, melee_range=1.75, heavy_range=1.5,
        hit_delay=0.1, heavy_hit_delay=0.32,
    ),
    "frag": WeaponSpec(
        name="frag", display="HE Grenade", slot=SLOT_FRAG, kind="grenade",
        damage=115, mag=1, draw_time=0.4, move_mult=1.04,
        fuse=2.6, throw_speed=17.0, windup=0.22, blast_radius=7.0,
    ),
    "smoke": WeaponSpec(
        name="smoke", display="Smoke Grenade", slot=SLOT_SMOKE, kind="grenade",
        mag=1, draw_time=0.4, move_mult=1.04,
        fuse=1.6, throw_speed=15.0, windup=0.22, smoke_radius=4.6, smoke_duration=16.0,
    ),
}

PRIMARY_CHOICES = ("ak47", "awm")
SECONDARY = "deagle"
MELEE = "knife"

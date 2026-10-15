"""Weapon / item definitions. All angles in degrees, times in seconds.

The experiment environment uses three fixed loadouts (`LOADOUTS`): rifle, sniper and shotgun.
Every loadout also carries the pistol, the knife and one of each grenade.
"""
from __future__ import annotations

from dataclasses import dataclass

SLOT_PRIMARY, SLOT_PISTOL, SLOT_MELEE, SLOT_FRAG, SLOT_SMOKE, SLOT_FLASH, SLOT_FIRE = 1, 2, 3, 4, 5, 6, 7
NUM_SLOTS = 7


def _pattern(rise, yaw, tail_rise=0.3, tail_yaw=(-0.4, 0.5, -0.35), n=30):
    """Recoil pattern: explicit first shots, then a slow rise with alternating side drift."""
    pitch = list(rise)
    yaws = list(yaw)
    k = 0
    while len(pitch) < n:
        pitch.append(tail_rise)
        yaws.append(tail_yaw[(k // 5) % len(tail_yaw)])
        k += 1
    return tuple(zip(pitch[:n], yaws[:n]))


AK = _pattern([0.75, 0.85, 0.95, 1.0, 1.0, 0.95, 0.85, 0.75, 0.65, 0.55],
              [0.05, -0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.1, -0.1, -0.3], 0.28, (-0.45, 0.55, -0.4))


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    display: str
    slot: int
    kind: str  # "gun" | "melee" | "grenade"
    category: str = "misc"  # pistol | rifle | sniper | heavy | grenade | misc
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
    flash_radius: float = 0.0
    fire_radius: float = 0.0
    fire_duration: float = 0.0
    fire_dps: float = 0.0


WEAPONS: dict[str, WeaponSpec] = {
    "ak47": WeaponSpec(
        name="ak47", display="AK-47", slot=SLOT_PRIMARY, kind="gun", category="rifle",
        damage=36, interval=0.1, auto=True, mag=30, reserve=90, reload_time=1.9, draw_time=0.6,
        move_mult=0.92, ads_move_mult=0.8, ads_fov=62,
        spread_base=0.18, spread_ads=0.07, spread_move=3.2, spread_air=7.0, spread_per_shot=0.38,
        spread_max=4.5, spread_recover=9.0, recoil=AK, recoil_jitter=0.2, punch_recover=7.5, falloff=0.98,
    ),
    "awm": WeaponSpec(
        name="awm", display="AWM", slot=SLOT_PRIMARY, kind="gun", category="sniper",
        damage=115, interval=1.45, auto=False, mag=5, reserve=20,
        reload_time=2.8, draw_time=0.9, move_mult=0.85, ads_move_mult=0.5, ads_fov=24,
        scope_overlay=True, unscope_after_shot=True,
        spread_base=6.5, spread_ads=0.02, spread_move=7.0, spread_air=14.0,
        spread_per_shot=0.0, spread_max=0.0, spread_recover=10.0,
        recoil=((3.2, 0.4),), recoil_jitter=0.3, punch_recover=4.5,
        head_mult=3.0, leg_mult=0.85, falloff=1.0,
    ),
    "nova": WeaponSpec(
        name="nova", display="Nova", slot=SLOT_PRIMARY, kind="gun", category="heavy",
        damage=24, interval=0.88, auto=False, mag=8, reserve=32, reload_time=2.6, draw_time=0.6,
        move_mult=0.95, ads_move_mult=0.85, ads_fov=70, pellets=9,
        spread_base=3.6, spread_ads=2.9, spread_move=0.8, spread_air=2.5, spread_per_shot=0.0,
        spread_max=0.0, spread_recover=10.0, crouch_mult=0.9,
        recoil=((3.4, 0.3),), recoil_jitter=0.3, punch_recover=5.0,
        head_mult=2.0, leg_mult=0.8, falloff=0.55, range=60.0,
    ),
    "glock": WeaponSpec(
        name="glock", display="Glock", slot=SLOT_PISTOL, kind="gun", category="pistol",
        damage=28, interval=0.15, auto=False, mag=20, reserve=120,
        reload_time=1.7, draw_time=0.35, move_mult=1.0,
        spread_base=0.45, spread_move=1.4, spread_air=4.5,
        spread_per_shot=0.55, spread_max=3.2, spread_recover=8.0,
        recoil=((0.7, 0.12), (0.75, -0.14), (0.8, 0.16)), recoil_jitter=0.25, punch_recover=9.0,
        head_mult=4.0, leg_mult=0.75, falloff=0.9,
    ),
    "knife": WeaponSpec(
        name="knife", display="M9 Knife", slot=SLOT_MELEE, kind="melee",
        damage=40, interval=0.45, draw_time=0.3, move_mult=1.08,
        heavy_damage=90, heavy_interval=1.05, melee_range=1.75, heavy_range=1.5,
        hit_delay=0.1, heavy_hit_delay=0.32,
    ),
    "frag": WeaponSpec(
        name="frag", display="HE Grenade", slot=SLOT_FRAG, kind="grenade", category="grenade",
        damage=115, mag=1, draw_time=0.4, move_mult=1.04,
        fuse=2.6, throw_speed=25.0, windup=0.22, blast_radius=7.0,
    ),
    "smoke": WeaponSpec(
        name="smoke", display="Smoke Grenade", slot=SLOT_SMOKE, kind="grenade", category="grenade",
        mag=1, draw_time=0.4, move_mult=1.04,
        fuse=1.6, throw_speed=22.0, windup=0.22, smoke_radius=4.6, smoke_duration=16.0,
    ),
    "flash": WeaponSpec(
        name="flash", display="Flashbang", slot=SLOT_FLASH, kind="grenade", category="grenade",
        mag=1, draw_time=0.4, move_mult=1.04, fuse=1.5, throw_speed=25.0, windup=0.2, flash_radius=14.0,
    ),
    "molotov": WeaponSpec(
        name="molotov", display="Molotov", slot=SLOT_FIRE, kind="grenade", category="grenade",
        mag=1, draw_time=0.4, move_mult=1.04,
        fuse=3.2, throw_speed=22.0, windup=0.24, fire_radius=3.0, fire_duration=7.0, fire_dps=42.0,
    ),
}

ITEM_ZH = {"glock": "格洛克", "knife": "M9 军刀", "frag": "高爆手雷", "smoke": "烟雾弹", "flash": "闪光弹",
           "molotov": "燃烧瓶", "zone": "毒圈"}


def item_name(name: str, zh: bool) -> str:
    """Display name of a weapon in the UI language (guns keep their model names)."""
    if zh and name in ITEM_ZH:
        return ITEM_ZH[name]
    if name == "zone":
        return "Zone"
    return WEAPONS[name].display if name in WEAPONS else name


SECONDARY = "glock"
MELEE = "knife"
GRENADES = ("frag", "smoke", "flash", "molotov")
# fixed loadouts: key, zh, en, primary, secondary
LOADOUTS = (
    ("rifle", "步枪", "Rifleman", "ak47", SECONDARY),
    ("sniper", "狙击枪", "Sniper", "awm", SECONDARY),
    ("shotgun", "散弹枪", "Shotgun", "nova", SECONDARY),
)
LOADOUT_BY_KEY = {lo[0]: lo for lo in LOADOUTS}
LOADOUT_KEYS = tuple(LOADOUT_BY_KEY)
PRIMARY_CHOICES = tuple(lo[3] for lo in LOADOUTS)
SCOPED = tuple(n for n, s in WEAPONS.items() if s.scope_overlay)


def random_loadout(rng) -> str:
    """Uniformly random loadout key, from a `random.Random` or numpy Generator."""
    return LOADOUT_KEYS[int(rng.random() * len(LOADOUT_KEYS)) % len(LOADOUT_KEYS)]

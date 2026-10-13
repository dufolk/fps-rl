"""Weapon / item definitions. All angles in degrees, times in seconds, prices in $."""
from __future__ import annotations

from dataclasses import dataclass, field

SLOT_PRIMARY, SLOT_PISTOL, SLOT_MELEE, SLOT_FRAG, SLOT_SMOKE, SLOT_GADGET, SLOT_BOMB = 1, 2, 3, 4, 5, 6, 7
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
M4 = _pattern([0.55, 0.62, 0.7, 0.74, 0.74, 0.7, 0.62, 0.55, 0.48, 0.42],
              [0.02, -0.04, 0.06, 0.1, 0.14, 0.16, 0.1, 0.0, -0.12, -0.2], 0.22, (-0.32, 0.38, -0.28))
M4S = _pattern([0.48, 0.55, 0.6, 0.64, 0.62, 0.58, 0.52, 0.46, 0.4, 0.36],
               [0.0, -0.03, 0.05, 0.08, 0.1, 0.12, 0.06, -0.04, -0.1, -0.14], 0.2, (-0.26, 0.3, -0.24), 25)
FAMAS = _pattern([0.6, 0.66, 0.7, 0.7, 0.66, 0.6, 0.52], [0.0, 0.06, -0.06, 0.1, -0.1, 0.14, -0.14],
                 0.24, (0.35, -0.35), 25)
SMG = _pattern([0.4, 0.45, 0.48, 0.5, 0.48, 0.44, 0.4], [0.02, -0.05, 0.08, -0.08, 0.1, -0.1, 0.06],
               0.18, (0.3, -0.32, 0.24), 32)
UZI = _pattern([0.36, 0.42, 0.46, 0.48, 0.46], [0.08, -0.1, 0.12, -0.14, 0.16], 0.17, (0.38, -0.38), 32)
LMG = _pattern([0.5, 0.58, 0.64, 0.66, 0.64, 0.6, 0.55, 0.5], [0.0, 0.08, 0.12, 0.08, -0.06, -0.14, -0.2, -0.1],
               0.24, (-0.4, 0.45, -0.3, 0.35), 100)


@dataclass(frozen=True)
class WeaponSpec:
    name: str
    display: str
    slot: int
    kind: str  # "gun" | "melee" | "grenade" | "bomb" | "gadget" | "shield"
    category: str = "misc"  # buy-menu category: pistol | smg | rifle | sniper | heavy | grenade | misc
    price: int = 0
    kill_reward: int = 300
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
    penetration: float = 0.5  # fraction of damage kept after passing a soft (destructible) wall
    wall_damage: float = 10.0  # damage dealt to destructible panels per hit
    silenced: bool = False
    dual: bool = False
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
    # placeable gadgets / bomb
    place: str = ""  # "wall" | "ground" | "slot" | "floor" | "site"
    place_time: float = 0.0
    extra: dict = field(default_factory=dict)


def _rifle(name, display, price, **kw):
    base = dict(slot=SLOT_PRIMARY, kind="gun", category="rifle", price=price, auto=True, mag=30, reserve=90,
                reload_time=2.5, draw_time=0.6, move_mult=0.93, ads_move_mult=0.8, ads_fov=62,
                spread_base=0.17, spread_ads=0.07, spread_move=3.0, spread_air=7.0, spread_per_shot=0.34,
                spread_max=4.2, spread_recover=9.0, recoil_jitter=0.2, punch_recover=7.5, falloff=0.98,
                penetration=0.55, wall_damage=12.0)
    base.update(kw)
    return WeaponSpec(name=name, display=display, **base)


def _smg(name, display, price, **kw):
    base = dict(slot=SLOT_PRIMARY, kind="gun", category="smg", price=price, kill_reward=600, auto=True,
                mag=30, reserve=120, reload_time=2.2, draw_time=0.45, move_mult=1.0, ads_move_mult=0.85,
                ads_fov=68, spread_base=0.3, spread_ads=0.16, spread_move=1.2, spread_air=4.5,
                spread_per_shot=0.26, spread_max=3.6, spread_recover=10.0, recoil=SMG, recoil_jitter=0.25,
                punch_recover=9.0, head_mult=3.6, falloff=0.93, penetration=0.4, wall_damage=8.0)
    base.update(kw)
    return WeaponSpec(name=name, display=display, **base)


WEAPONS: dict[str, WeaponSpec] = {
    # ---------------------------------------------------------------- rifles
    "ak47": _rifle("ak47", "AK-47", 2700, damage=36, interval=0.1, reload_time=2.4,
                   move_mult=0.92, spread_base=0.18, spread_per_shot=0.38, spread_max=4.5, spread_move=3.2,
                   recoil=AK),
    "m4a4": _rifle("m4a4", "M4A4", 3100, damage=33, interval=0.09, reload_time=2.6, ads_fov=64,
                   spread_base=0.16, spread_ads=0.06, spread_per_shot=0.3, spread_max=3.9, recoil=M4),
    "m4a1": _rifle("m4a1", "M4A1", 2900, damage=34, interval=0.1, mag=25, reserve=75, reload_time=2.7,
                   ads_fov=64, spread_base=0.13, spread_ads=0.05, spread_per_shot=0.28, spread_max=3.6,
                   recoil=M4S, silenced=True),
    "famas": _rifle("famas", "FAMAS", 2050, damage=30, interval=0.09, mag=25, reserve=90, reload_time=2.9,
                    move_mult=0.92, ads_fov=66, spread_base=0.2, spread_per_shot=0.36, spread_max=4.4,
                    recoil=FAMAS),
    # ---------------------------------------------------------------- smgs
    "mp5": _smg("mp5", "MP5", 1500, damage=27, interval=0.075),
    "mp9": _smg("mp9", "MP9", 1250, damage=26, interval=0.07, draw_time=0.4, spread_base=0.32),
    "uzi": _smg("uzi", "UZI", 1200, damage=25, interval=0.065, mag=32, reserve=128, spread_base=0.36,
                spread_per_shot=0.3, spread_max=4.0, recoil=UZI),
    # ---------------------------------------------------------------- snipers
    "awm": WeaponSpec(
        name="awm", display="AWM", slot=SLOT_PRIMARY, kind="gun", category="sniper", price=4750, kill_reward=100,
        damage=115, interval=1.45, auto=False, mag=5, reserve=20,
        reload_time=3.4, draw_time=0.9, move_mult=0.85, ads_move_mult=0.5, ads_fov=24,
        scope_overlay=True, unscope_after_shot=True,
        spread_base=6.5, spread_ads=0.02, spread_move=7.0, spread_air=14.0,
        spread_per_shot=0.0, spread_max=0.0, spread_recover=10.0,
        recoil=((3.2, 0.4),), recoil_jitter=0.3, punch_recover=4.5,
        head_mult=3.0, leg_mult=0.85, falloff=1.0, penetration=0.75, wall_damage=40.0,
    ),
    "scar20": WeaponSpec(
        name="scar20", display="SCAR-20", slot=SLOT_PRIMARY, kind="gun", category="sniper", price=5000,
        kill_reward=300, damage=80, interval=0.25, auto=False, mag=20, reserve=60,
        reload_time=3.1, draw_time=0.8, move_mult=0.82, ads_move_mult=0.55, ads_fov=30,
        scope_overlay=True, unscope_after_shot=False,
        spread_base=5.0, spread_ads=0.05, spread_move=6.0, spread_air=12.0,
        spread_per_shot=0.35, spread_max=1.6, spread_recover=6.0,
        recoil=((1.5, 0.2), (1.6, -0.25), (1.7, 0.3)), recoil_jitter=0.25, punch_recover=5.5,
        head_mult=3.2, leg_mult=0.85, falloff=0.99, penetration=0.65, wall_damage=25.0,
    ),
    # ---------------------------------------------------------------- heavy
    "m249": WeaponSpec(
        name="m249", display="M249", slot=SLOT_PRIMARY, kind="gun", category="heavy", price=5200, kill_reward=300,
        damage=32, interval=0.08, auto=True, mag=100, reserve=200, reload_time=5.7, draw_time=1.0,
        move_mult=0.78, ads_move_mult=0.75, ads_fov=64,
        spread_base=0.34, spread_ads=0.14, spread_move=4.0, spread_air=8.0, spread_per_shot=0.22,
        spread_max=4.6, spread_recover=6.0, recoil=LMG, recoil_jitter=0.25, punch_recover=6.0,
        head_mult=3.6, falloff=0.97, penetration=0.6, wall_damage=14.0,
    ),
    # ---------------------------------------------------------------- pistols
    "deagle": WeaponSpec(
        name="deagle", display="Desert Eagle", slot=SLOT_PISTOL, kind="gun", category="pistol", price=700,
        damage=50, interval=0.24, auto=False, mag=7, reserve=35,
        reload_time=2.0, draw_time=0.4, move_mult=1.0,
        spread_base=0.3, spread_move=1.8, spread_air=5.0,
        spread_per_shot=1.3, spread_max=5.0, spread_recover=6.0,
        recoil=((2.0, 0.25), (2.2, -0.3), (2.4, 0.35)), recoil_jitter=0.3, punch_recover=6.0,
        head_mult=3.0, leg_mult=0.75, falloff=0.95, penetration=0.5, wall_damage=14.0,
    ),
    "dualies": WeaponSpec(
        name="dualies", display="Dual Berettas", slot=SLOT_PISTOL, kind="gun", category="pistol", price=300,
        damage=34, interval=0.12, auto=False, mag=30, reserve=120, reload_time=3.6, draw_time=0.6,
        move_mult=1.0, spread_base=0.42, spread_move=1.5, spread_air=5.0, spread_per_shot=0.5,
        spread_max=3.6, spread_recover=7.0, recoil=((0.9, 0.3), (1.0, -0.35)), recoil_jitter=0.3,
        punch_recover=8.0, head_mult=3.5, falloff=0.93, penetration=0.35, wall_damage=6.0, dual=True,
    ),
    # ---------------------------------------------------------------- melee
    "knife": WeaponSpec(
        name="knife", display="M9 Knife", slot=SLOT_MELEE, kind="melee", kill_reward=1500,
        damage=40, interval=0.45, draw_time=0.3, move_mult=1.08,
        heavy_damage=90, heavy_interval=1.05, melee_range=1.75, heavy_range=1.5,
        hit_delay=0.1, heavy_hit_delay=0.32, wall_damage=50.0,
    ),
    # ---------------------------------------------------------------- grenades
    "frag": WeaponSpec(
        name="frag", display="HE Grenade", slot=SLOT_FRAG, kind="grenade", category="grenade", price=300,
        damage=115, mag=1, draw_time=0.4, move_mult=1.04,
        fuse=2.6, throw_speed=17.0, windup=0.22, blast_radius=7.0, wall_damage=120.0,
    ),
    "smoke": WeaponSpec(
        name="smoke", display="Smoke Grenade", slot=SLOT_SMOKE, kind="grenade", category="grenade", price=300,
        mag=1, draw_time=0.4, move_mult=1.04,
        fuse=1.6, throw_speed=15.0, windup=0.22, smoke_radius=4.6, smoke_duration=16.0,
    ),
    "flash": WeaponSpec(
        name="flash", display="Flashbang", slot=SLOT_FRAG, kind="grenade", category="grenade", price=200,
        mag=2, draw_time=0.4, move_mult=1.04, fuse=1.5, throw_speed=17.0, windup=0.2, flash_radius=14.0,
    ),
    # ---------------------------------------------------------------- objectives
    "c4": WeaponSpec(name="c4", display="C4", slot=SLOT_BOMB, kind="bomb", mag=1, draw_time=0.5, move_mult=1.0,
                     place="site", place_time=3.2),
    "defuser": WeaponSpec(name="defuser", display="Defuser", slot=SLOT_BOMB, kind="bomb", mag=1, draw_time=0.5,
                          move_mult=1.0, place="site", place_time=7.0),
    # ---------------------------------------------------------------- R6 generic gadgets
    "breach": WeaponSpec(name="breach", display="Breach Charge", slot=SLOT_SMOKE, kind="gadget", mag=2,
                         draw_time=0.5, place="wall", place_time=1.0, fuse=2.0, extra={"radius": 1.1}),
    "barbed": WeaponSpec(name="barbed", display="Barbed Wire", slot=SLOT_FRAG, kind="gadget", mag=2,
                         draw_time=0.5, place="ground", place_time=1.0),
    "dshield": WeaponSpec(name="dshield", display="Deployable Shield", slot=SLOT_SMOKE, kind="gadget", mag=1,
                          draw_time=0.6, place="ground", place_time=1.5),
    # ---------------------------------------------------------------- R6 operator gadgets
    "thermite": WeaponSpec(name="thermite", display="Exothermic Charge", slot=SLOT_GADGET, kind="gadget", mag=2,
                           draw_time=0.6, place="wall", place_time=1.5, fuse=4.0, extra={"hard": True}),
    "ash_round": WeaponSpec(name="ash_round", display="Breaching Round", slot=SLOT_GADGET, kind="gadget", mag=2,
                            draw_time=0.6, place="launch", place_time=0.0, fuse=1.2,
                            extra={"radius": 0.9, "speed": 45.0}),
    "hammer": WeaponSpec(name="hammer", display="Breaching Hammer", slot=SLOT_GADGET, kind="melee", mag=25,
                         draw_time=0.7, damage=60, interval=0.9, heavy_damage=60, heavy_interval=0.9,
                         melee_range=1.9, heavy_range=1.9, hit_delay=0.35, heavy_hit_delay=0.35,
                         wall_damage=10000.0, extra={"breach_radius": 0.85, "floors": True}),
    "flash_shield": WeaponSpec(name="flash_shield", display="Flash Shield", slot=SLOT_GADGET, kind="shield",
                               mag=5, draw_time=0.8, interval=0.9, move_mult=0.88, flash_radius=6.0),
    "castle": WeaponSpec(name="castle", display="Armor Panel", slot=SLOT_GADGET, kind="gadget", mag=3,
                         draw_time=0.6, place="slot", place_time=2.0),
    "jammer": WeaponSpec(name="jammer", display="Signal Disruptor", slot=SLOT_GADGET, kind="gadget", mag=4,
                         draw_time=0.5, place="ground", place_time=0.8, extra={"radius": 3.5}),
    "edd": WeaponSpec(name="edd", display="Entry Denial Device", slot=SLOT_GADGET, kind="gadget", mag=4,
                      draw_time=0.5, place="slot", place_time=1.0, damage=65),
    "armor_bag": WeaponSpec(name="armor_bag", display="Armor Pack", slot=SLOT_GADGET, kind="gadget", mag=1,
                            draw_time=0.5, place="ground", place_time=1.0),
}

PRIMARY_CHOICES = ("ak47", "m4a4", "m4a1", "famas", "mp5", "mp9", "uzi", "awm", "scar20", "m249")
SECONDARY_CHOICES = ("deagle", "dualies")
SECONDARY = "deagle"
MELEE = "knife"
BUY_CATEGORIES = (
    ("pistol", "手枪", "Pistols", ("deagle", "dualies")),
    ("smg", "冲锋枪", "SMGs", ("mp9", "uzi", "mp5")),
    ("rifle", "步枪", "Rifles", ("famas", "m4a1", "m4a4", "ak47")),
    ("sniper", "狙击", "Snipers", ("awm", "scar20")),
    ("heavy", "重型", "Heavy", ("m249",)),
    ("grenade", "投掷物", "Grenades", ("frag", "smoke")),
)
SCOPED = tuple(n for n, s in WEAPONS.items() if s.scope_overlay)
# primary weapon mix for bots in free-weapon modes (TDM / domination)
BOT_PRIMARY_MIX = (("ak47", 0.22), ("m4a4", 0.14), ("m4a1", 0.12), ("famas", 0.08), ("mp5", 0.08),
                   ("mp9", 0.06), ("uzi", 0.05), ("awm", 0.15), ("scar20", 0.05), ("m249", 0.05))


def random_primary(rng) -> str:
    """Draw from BOT_PRIMARY_MIX with a `random.Random` or numpy Generator."""
    names, probs = zip(*BOT_PRIMARY_MIX)
    x = rng.random() * sum(probs)
    for n, p in zip(names, probs):
        x -= p
        if x <= 0:
            return n
    return names[-1]

"""Siege operators: two primaries from the shared arsenal, a secondary, a unique gadget
(slot 6) and generic gadgets (each in its own slot)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    name: str
    zh: str
    side: int  # 0 attack, 1 defend
    primaries: tuple
    secondary: str
    gadget: str
    generics: tuple  # weapon names placed into their own slots
    desc_zh: str
    desc_en: str


OPERATORS = {
    "thermite": Operator("thermite", "热切", 0, ("m4a4", "famas"), "deagle", "thermite", ("flash", "breach"),
                         "放置热切割炸药，可炸开加固墙", "Exothermic charge breaches reinforced walls"),
    "ash": Operator("ash", "灰烬", 0, ("m4a1", "mp9"), "deagle", "ash_round", ("flash", "breach"),
                    "发射破墙弹，远程炸开软墙", "Breaching rounds blow soft walls from range"),
    "sledge": Operator("sledge", "大锤", 0, ("m4a4", "mp5"), "dualies", "hammer", ("frag", "breach"),
                       "破墙锤，砸开软墙与地板", "Hammer smashes soft walls, floors and barricades"),
    "blitz": Operator("blitz", "闪盾", 0, (), "deagle", "flash_shield", ("frag", "smoke"),
                      "手持闪光盾，正面防弹并致盲", "Ballistic shield with a blinding flash"),
    "castle": Operator("castle", "城堡", 1, ("mp5", "uzi"), "deagle", "castle", ("barbed", "dshield"),
                       "在门窗安装防弹板", "Armor panels block doors and windows"),
    "mute": Operator("mute", "禁言", 1, ("mp5", "famas"), "dualies", "jammer", ("barbed", "dshield"),
                     "信号干扰器，屏蔽无人机与破墙炸药", "Jammers disable drones and breach charges"),
    "kapkan": Operator("kapkan", "卡普坎", 1, ("ak47", "mp9"), "deagle", "edd", ("barbed", "dshield"),
                       "门窗绊雷", "Trip mines on doors and windows"),
    "rook": Operator("rook", "战车", 1, ("mp5", "famas"), "dualies", "armor_bag", ("barbed", "dshield"),
                     "护甲包，队友拾取获得减伤", "Armor pack reduces team damage"),
}

ATTACKERS = tuple(n for n, o in OPERATORS.items() if o.side == 0)
DEFENDERS = tuple(n for n, o in OPERATORS.items() if o.side == 1)

"""Procedural models for the extended arsenal, objective items and Siege gadgets.

Same frame as `models.py`: +Y = barrel, +Z = up, origin near the trigger. Every builder returns
(node, info) with muzzle / sight_z / grip / fore / eject anchor points.
"""
from __future__ import annotations

import math

from panda3d.core import Point3, Vec3

from .models import PartSet, _trigger_group, obox

BLACK = (0.07, 0.07, 0.075, 1)
DARK = (0.12, 0.12, 0.13, 1)
STEEL = (0.2, 0.2, 0.21, 1)
TAN = (0.55, 0.47, 0.33, 1)
OLIVE = (0.3, 0.33, 0.22, 1)
GREY = (0.32, 0.33, 0.34, 1)


def _info(muzzle, sight_z, grip=(0, -0.035, -0.03), fore=(0, 0.25, 0.0), eject=(0.03, 0.05, 0.06)):
    return {"muzzle": Point3(*muzzle), "sight_z": sight_z, "grip": Point3(*grip), "fore": Point3(*fore),
            "eject": Point3(*eject)}


def _pistol_grip(m, color, y=-0.035, angle=-18, h=0.11):
    obox(m, (0, y, -0.05), (0.03, 0.038, h), (0, angle, 0), color)


def _rail(m, y0, y1, z, w=0.022, color=BLACK):
    m.box((-w / 2, y0, z), (w / 2, y1, z + 0.008), color)
    n = int((y1 - y0) / 0.012)
    for k in range(n):
        yy = y0 + k * 0.012
        m.box((-w / 2 - 0.001, yy, z + 0.008), (w / 2 + 0.001, yy + 0.006, z + 0.013), color)


def _straight_mag(m, y, z, length, color, w=0.024, d=0.06, curve=0.0):
    t = Vec3(0, y, z)
    segs = 3
    for k in range(segs):
        p = curve * k
        dv = Vec3(0, math.sin(math.radians(p)), -math.cos(math.radians(p)))
        seg = length / segs
        obox(m, t + dv * (seg / 2), (w, d, seg + 0.004), (0, p, 0), color)
        t = t + dv * seg


def _ar15(name, silencer=False, mag_len=0.17, mag_curve=4.0, color=BLACK):
    ps = PartSet(name)
    m, p = ps.b("metal"), ps.b("poly")
    m.box((-0.022, -0.1, 0.0), (0.022, 0.16, 0.06), color)           # lower / upper receiver
    m.box((-0.02, -0.1, 0.06), (0.02, 0.17, 0.082), DARK)
    _rail(m, -0.09, 0.16, 0.082)
    m.box((-0.008, 0.3, 0.082), (0.008, 0.33, 0.12), DARK)            # front sight post
    m.box((-0.01, -0.08, 0.09), (0.01, -0.05, 0.118), DARK)           # rear sight
    p.box((-0.027, 0.16, 0.012), (0.027, 0.4, 0.075), DARK)           # handguard
    for k in range(5):
        y = 0.18 + k * 0.045
        m.box((-0.028, y, 0.03), (0.028, y + 0.012, 0.055), BLACK)
    if silencer:
        m.cylinder((0, 0.4, 0.045), (0, 0.48, 0.045), 0.009, BLACK, 12)
        m.cylinder((0, 0.47, 0.045), (0, 0.69, 0.045), 0.019, (0.1, 0.1, 0.11, 1), 16)
        muzzle = (0, 0.7, 0.045)
    else:
        m.cylinder((0, 0.4, 0.045), (0, 0.6, 0.045), 0.0095, BLACK, 12)
        m.cylinder((0, 0.585, 0.045), (0, 0.64, 0.045), 0.0135, DARK, 10)   # flash hider
        muzzle = (0, 0.645, 0.045)
    _straight_mag(m, 0.07, 0.002, mag_len, (0.16, 0.16, 0.17, 1), curve=mag_curve)
    m.box((-0.012, 0.045, -0.01), (0.012, 0.11, 0.004), color)        # mag well
    _pistol_grip(p, BLACK)
    _trigger_group(m, BLACK)
    m.cylinder((0, -0.1, 0.045), (0, -0.28, 0.045), 0.013, BLACK, 10)  # buffer tube
    p.tapered(-0.4, -0.25, 0.04, 0.115, 0.036, 0.07, -0.07, -0.01, color)  # stock
    m.box((0.022, 0.02, 0.035), (0.03, 0.08, 0.05), STEEL)            # ejection port cover
    return ps.node(), _info(muzzle, 0.12, fore=(0, 0.28, 0.01))


def build_m4a4():
    return _ar15("m4a4", mag_len=0.18, mag_curve=5.0)


def build_m4a1():
    return _ar15("m4a1", silencer=True, mag_len=0.15, mag_curve=3.0, color=(0.1, 0.1, 0.105, 1))


def build_famas():
    ps = PartSet("famas")
    m, p = ps.b("metal"), ps.b("poly")
    body = (0.36, 0.36, 0.33, 1)
    p.box((-0.028, -0.32, -0.01), (0.028, 0.18, 0.065), body)        # bullpup body
    p.tapered(-0.42, -0.32, 0.04, 0.11, 0.05, 0.075, -0.06, -0.01, body)
    p.box((-0.012, -0.22, 0.065), (0.012, 0.15, 0.08), DARK)         # long carry handle
    p.box((-0.01, -0.22, 0.08), (0.01, -0.19, 0.125), DARK)
    p.box((-0.01, 0.12, 0.08), (0.01, 0.15, 0.125), DARK)
    p.box((-0.015, -0.21, 0.12), (0.015, 0.14, 0.135), DARK)
    m.cylinder((0, 0.18, 0.035), (0, 0.46, 0.035), 0.0095, BLACK, 12)
    m.cylinder((0, 0.44, 0.035), (0, 0.5, 0.035), 0.013, DARK, 10)
    m.box((-0.004, 0.25, 0.015), (0.004, 0.33, 0.035), DARK)         # bipod folded
    _straight_mag(m, -0.2, -0.005, 0.13, (0.18, 0.18, 0.19, 1), curve=2.0)
    obox(p, (0, 0.03, -0.055), (0.03, 0.036, 0.1), (0, -10, 0), BLACK)
    m.box((-0.012, 0.0, -0.1), (0.012, 0.1, -0.09), BLACK)           # large trigger guard
    _trigger_group(m, BLACK, 0.0, 0.07)
    node = ps.node()
    return node, _info((0, 0.505, 0.035), 0.135, grip=(0, 0.03, -0.035), fore=(0, 0.15, -0.0),
                       eject=(0.03, -0.16, 0.05))


def build_mp5():
    ps = PartSet("mp5")
    m, p = ps.b("metal"), ps.b("poly")
    m.cylinder((0, -0.08, 0.04), (0, 0.18, 0.04), 0.024, BLACK, 14)  # tubular receiver
    m.box((-0.018, -0.08, 0.0), (0.018, 0.12, 0.03), DARK)
    p.cylinder((0, 0.18, 0.035), (0, 0.32, 0.035), 0.026, (0.08, 0.08, 0.085, 1), 14)  # tropical handguard
    m.cylinder((0, 0.3, 0.04), (0, 0.4, 0.04), 0.009, BLACK, 10)
    m.cylinder((0, 0.38, 0.04), (0, 0.405, 0.04), 0.015, DARK, 10)
    m.cylinder((0, 0.32, 0.075), (0, 0.32, 0.098), 0.012, DARK, 10)  # front sight hood
    m.cylinder((0, -0.06, 0.075), (0, -0.06, 0.098), 0.012, DARK, 10)
    m.box((-0.01, -0.08, 0.064), (0.01, 0.33, 0.07), DARK)
    _straight_mag(m, 0.1, 0.0, 0.16, (0.14, 0.14, 0.15, 1), w=0.02, d=0.05, curve=9.0)
    _pistol_grip(p, BLACK)
    _trigger_group(m, BLACK)
    m.box((-0.012, -0.32, -0.06), (0.012, -0.3, 0.05), BLACK)       # retractable stock
    m.box((-0.005, -0.31, 0.03), (0.005, -0.08, 0.04), BLACK)
    m.box((-0.005, -0.31, -0.02), (0.005, -0.08, -0.01), BLACK)
    m.cylinder((-0.025, 0.12, 0.06), (-0.025, 0.19, 0.06), 0.006, STEEL, 8)  # cocking tube handle
    node = ps.node()
    return node, _info((0, 0.41, 0.04), 0.098, fore=(0, 0.26, 0.0), eject=(0.03, 0.06, 0.05))


def build_mp9():
    ps = PartSet("mp9")
    m, p = ps.b("metal"), ps.b("poly")
    p.box((-0.022, -0.06, 0.0), (0.022, 0.2, 0.06), BLACK)
    _rail(m, -0.05, 0.19, 0.06, 0.02)
    m.cylinder((0, 0.2, 0.035), (0, 0.27, 0.035), 0.009, BLACK, 10)
    p.box((-0.012, 0.16, -0.08), (0.012, 0.19, 0.0), BLACK)         # vertical fore grip
    p.box((-0.013, 0.0, -0.13), (0.013, 0.04, 0.0), BLACK)          # mag in grip
    m.box((-0.011, 0.002, -0.2), (0.011, 0.036, -0.13), (0.16, 0.16, 0.17, 1))
    _trigger_group(m, BLACK, 0.04, 0.1)
    m.box((-0.004, -0.27, 0.01), (0.004, -0.06, 0.02), BLACK)       # folding stock
    m.box((-0.012, -0.29, -0.05), (0.012, -0.27, 0.05), BLACK)
    node = ps.node()
    return node, _info((0, 0.275, 0.035), 0.075, grip=(0, 0.02, -0.04), fore=(0, 0.175, -0.04),
                       eject=(0.025, 0.08, 0.05))


def build_uzi():
    ps = PartSet("uzi")
    m, p = ps.b("metal"), ps.b("poly")
    m.box((-0.026, -0.12, 0.0), (0.026, 0.16, 0.065), DARK)
    m.box((-0.022, -0.12, 0.065), (0.022, 0.15, 0.075), BLACK)
    m.cylinder((0, 0.16, 0.035), (0, 0.24, 0.035), 0.009, BLACK, 10)
    m.box((-0.004, 0.13, 0.075), (0.004, 0.15, 0.095), BLACK)
    m.box((-0.008, -0.1, 0.075), (0.008, -0.085, 0.09), BLACK)
    p.box((-0.016, 0.0, -0.12), (0.016, 0.045, 0.0), (0.1, 0.1, 0.1, 1))
    m.box((-0.012, 0.004, -0.2), (0.012, 0.04, -0.12), (0.16, 0.16, 0.17, 1))
    _trigger_group(m, BLACK, 0.045, 0.1)
    m.box((-0.004, -0.3, 0.02), (0.004, -0.12, 0.03), DARK)        # wire stock
    m.box((-0.015, -0.31, -0.04), (0.015, -0.3, 0.04), DARK)
    m.box((0.026, 0.02, 0.06), (0.032, 0.06, 0.07), STEEL)
    node = ps.node()
    return node, _info((0, 0.245, 0.035), 0.095, grip=(0, 0.022, -0.04), fore=(0, 0.12, -0.0),
                       eject=(0.03, 0.06, 0.06))


def build_glock():
    ps = PartSet("glock")
    m, p = ps.b("metal"), ps.b("poly")
    poly = (0.13, 0.13, 0.14, 1)
    slide = (0.18, 0.18, 0.19, 1)
    m.box((-0.013, -0.025, 0.03), (0.013, 0.15, 0.056), slide)          # squared slide
    for i in range(4):
        y = -0.022 + i * 0.006
        m.box((-0.0136, y, 0.034), (0.0136, y + 0.002, 0.052), BLACK)   # rear serrations
    m.cylinder((0, 0.15, 0.043), (0, 0.172, 0.043), 0.006, BLACK, 10)   # barrel
    p.box((-0.014, -0.01, 0.0), (0.014, 0.135, 0.032), poly)            # frame
    p.box((-0.01, 0.07, -0.008), (0.01, 0.125, 0.0), BLACK)             # accessory rail
    obox(p, (0, -0.012, -0.048), (0.028, 0.042, 0.1), (0, -20, 0), poly)
    p.box((-0.015, -0.028, -0.105), (0.015, 0.012, -0.092), BLACK)      # magazine floorplate
    _trigger_group(m, BLACK, 0.03, 0.075)
    m.box((-0.003, 0.155, 0.056), (0.003, 0.163, 0.066), BLACK)         # front sight
    m.box((-0.011, -0.012, 0.056), (-0.004, -0.002, 0.064), BLACK)      # rear sight notch
    m.box((0.004, -0.012, 0.056), (0.011, -0.002, 0.064), BLACK)
    node = ps.node()
    return node, _info((0, 0.175, 0.043), 0.066, grip=(0, -0.012, -0.03), fore=(-0.01, 0.04, -0.04),
                       eject=(0.016, 0.05, 0.05))


def build_beretta(name="dualies"):
    ps = PartSet(name)
    m, c = ps.b("metal"), ps.b("chrome")
    c.box((-0.012, -0.03, 0.028), (0.012, 0.17, 0.058), (0.42, 0.42, 0.44, 1))   # slide
    m.box((-0.0125, 0.06, 0.04), (0.0125, 0.12, 0.058), BLACK)                    # open-top slide cut
    m.cylinder((0, 0.1, 0.045), (0, 0.18, 0.045), 0.0065, BLACK, 10)
    m.box((-0.011, -0.025, 0.0), (0.011, 0.15, 0.028), BLACK)
    obox(m, (0, -0.02, -0.05), (0.027, 0.045, 0.105), (0, -12, 0), (0.15, 0.1, 0.07, 1))
    _trigger_group(m, BLACK, 0.0, 0.05)
    m.box((-0.003, 0.16, 0.058), (0.003, 0.168, 0.066), BLACK)
    node = ps.node()
    return node, _info((0, 0.185, 0.045), 0.066, grip=(0, -0.02, -0.03), fore=(-0.01, -0.01, -0.05),
                       eject=(0.02, 0.07, 0.055))


def build_dualies():
    return build_beretta("dualies")


def build_scar20():
    ps = PartSet("scar20")
    m, p = ps.b("metal"), ps.b("poly")
    p.box((-0.026, -0.12, 0.0), (0.026, 0.2, 0.07), TAN)
    p.box((-0.024, 0.2, 0.01), (0.024, 0.42, 0.072), TAN)
    _rail(m, -0.11, 0.42, 0.07, 0.022)
    m.cylinder((0, 0.42, 0.04), (0, 0.72, 0.04), 0.011, BLACK, 12)
    m.cylinder((0, 0.7, 0.04), (0, 0.76, 0.04), 0.016, BLACK, 12)
    _straight_mag(m, 0.07, 0.004, 0.11, BLACK, w=0.026, d=0.065)
    _pistol_grip(p, BLACK)
    _trigger_group(m, BLACK)
    p.tapered(-0.45, -0.12, 0.04, 0.13, 0.045, 0.075, -0.075, -0.01, TAN)
    p.box((-0.022, -0.36, 0.06), (0.022, -0.2, 0.09), BLACK)          # cheek riser
    # scope
    m.cylinder((0, -0.08, 0.118), (0, 0.18, 0.118), 0.016, BLACK, 16)
    m.cylinder((0, 0.16, 0.118), (0, 0.25, 0.118), 0.016, BLACK, 16, radius1=0.025)
    m.cylinder((0, -0.14, 0.118), (0, -0.08, 0.118), 0.021, BLACK, 16, radius1=0.016)
    m.box((-0.018, -0.03, 0.078), (0.018, -0.01, 0.12), BLACK)
    m.box((-0.018, 0.1, 0.078), (0.018, 0.12, 0.12), BLACK)
    gl = ps.b("glass")
    gl.cylinder((0, 0.25, 0.118), (0, 0.252, 0.118), 0.022, (0.15, 0.32, 0.5, 1), 16)
    node = ps.node()
    return node, _info((0, 0.765, 0.04), 0.118, fore=(0, 0.33, 0.0), eject=(0.03, 0.05, 0.06))


def build_m249():
    ps = PartSet("m249")
    m, p = ps.b("metal"), ps.b("poly")
    m.box((-0.03, -0.12, 0.0), (0.03, 0.2, 0.075), DARK)
    m.box((-0.032, -0.06, 0.075), (0.032, 0.14, 0.1), BLACK)           # feed cover
    m.cylinder((0, 0.2, 0.04), (0, 0.66, 0.04), 0.012, BLACK, 12)
    m.cylinder((0, 0.64, 0.04), (0, 0.7, 0.04), 0.017, DARK, 10)
    p.box((-0.03, 0.2, 0.0), (0.03, 0.36, 0.07), BLACK)                 # handguard
    m.cylinder((0, 0.25, 0.08), (0, 0.31, 0.11), 0.007, BLACK, 8)       # carry handle
    m.cylinder((-0.012, 0.5, 0.01), (-0.06, 0.56, -0.18), 0.006, BLACK, 6)   # bipod legs (folded-ish)
    m.cylinder((0.012, 0.5, 0.01), (0.06, 0.56, -0.18), 0.006, BLACK, 6)
    p.box((-0.07, 0.0, -0.14), (-0.03, 0.12, -0.02), OLIVE)            # ammo box
    p.box((-0.072, 0.02, -0.02), (-0.028, 0.1, -0.012), (0.22, 0.25, 0.16, 1))
    for k in range(6):                                                  # belt
        m.box((-0.035 + k * 0.004, 0.03 + k * 0.004, -0.02 + k * 0.012), (-0.03 + k * 0.004, 0.09, -0.012 + k * 0.012),
              (0.75, 0.6, 0.25, 1))
    _pistol_grip(p, BLACK)
    _trigger_group(m, BLACK)
    p.tapered(-0.42, -0.12, 0.045, 0.12, 0.05, 0.07, -0.075, -0.0, BLACK)
    node = ps.node()
    return node, _info((0, 0.705, 0.04), 0.112, fore=(0, 0.3, 0.0), eject=(0.035, 0.05, 0.03))


def build_nova():
    ps = PartSet("nova")
    m, w = ps.b("metal"), ps.b("wood")
    wood = (0.38, 0.22, 0.11, 1)
    m.box((-0.024, -0.1, 0.0), (0.024, 0.14, 0.068), BLACK)              # receiver
    m.box((0.024, -0.02, 0.025), (0.03, 0.08, 0.055), STEEL)             # loading / ejection port
    m.cylinder((0, 0.14, 0.05), (0, 0.66, 0.05), 0.013, BLACK, 14)       # barrel
    m.cylinder((0, 0.14, 0.018), (0, 0.6, 0.018), 0.012, DARK, 12)       # magazine tube
    m.cylinder((0, 0.6, 0.018), (0, 0.615, 0.018), 0.014, BLACK, 12)     # tube cap
    m.box((-0.006, 0.585, 0.024), (0.006, 0.62, 0.05), BLACK)            # barrel clamp
    m.box((-0.003, 0.64, 0.063), (0.003, 0.65, 0.074), (0.75, 0.6, 0.2, 1))  # bead sight
    m.box((-0.01, -0.08, 0.068), (0.01, 0.12, 0.076), DARK)              # top rib
    w.box((-0.026, 0.2, -0.004), (0.026, 0.36, 0.04), wood)              # pump
    for k in range(5):
        y = 0.215 + k * 0.03
        w.box((-0.0275, y, 0.0), (0.0275, y + 0.01, 0.036), (0.3, 0.17, 0.08, 1))
    _pistol_grip(w, wood, angle=-22)
    _trigger_group(m, BLACK)
    w.tapered(-0.42, -0.1, 0.045, 0.125, 0.04, 0.07, -0.08, -0.005, wood)  # stock
    m.box((-0.024, -0.43, -0.08), (0.024, -0.415, 0.05), BLACK)          # recoil pad
    node = ps.node()
    return node, _info((0, 0.665, 0.05), 0.076, fore=(0, 0.28, -0.0), eject=(0.035, 0.03, 0.04))


# ----------------------------------------------------------------------------- objective items
def build_flash():
    ps = PartSet("flash")
    m, p = ps.b("metal"), ps.b("paint")
    p.cylinder((0, 0, -0.055), (0, 0, 0.05), 0.028, (0.25, 0.27, 0.3, 1), 16)
    for z in (-0.03, 0.0, 0.03):
        p.cylinder((0, 0, z - 0.003), (0, 0, z + 0.003), 0.029, (0.12, 0.14, 0.16, 1), 16)
    m.cylinder((0, 0, 0.05), (0, 0, 0.066), 0.015, (0.5, 0.5, 0.48, 1), 12)
    obox(m, (0.028, 0, 0.02), (0.006, 0.016, 0.08), (0, 0, -8), (0.5, 0.5, 0.48, 1))
    return ps.node(), {}


def build_c4():
    ps = PartSet("c4")
    m, p = ps.b("metal"), ps.b("poly")
    for k in range(3):
        p.box((-0.07 + k * 0.047, -0.06, -0.02), (-0.025 + k * 0.047, 0.06, 0.02), (0.75, 0.7, 0.55, 1))
    m.box((-0.05, -0.04, 0.02), (0.05, 0.04, 0.035), (0.08, 0.08, 0.08, 1))   # keypad / timer
    ps.b("glass").box((-0.035, 0.0, 0.035), (0.035, 0.03, 0.038), (0.1, 0.6, 0.15, 1))
    for i in range(3):
        for j in range(2):
            m.box((-0.03 + i * 0.022, -0.035 + j * 0.016, 0.035), (-0.016 + i * 0.022, -0.024 + j * 0.016, 0.04),
                  (0.6, 0.6, 0.6, 1))
    p.box((-0.075, -0.004, -0.024), (0.075, 0.004, 0.024), (0.15, 0.12, 0.08, 1))  # tape
    m.cylinder((0.06, 0.05, 0.03), (0.02, 0.07, 0.035), 0.003, (0.8, 0.1, 0.1, 1), 6)
    return ps.node(), {"grip": Point3(0, -0.04, -0.02)}


def build_defuser():
    ps = PartSet("defuser")
    m, p = ps.b("metal"), ps.b("poly")
    p.box((-0.05, -0.08, -0.02), (0.05, 0.08, 0.03), (0.85, 0.55, 0.1, 1))
    m.box((-0.035, -0.05, 0.03), (0.035, 0.04, 0.04), (0.08, 0.08, 0.08, 1))
    ps.b("glass").box((-0.028, -0.02, 0.04), (0.028, 0.03, 0.043), (0.9, 0.2, 0.15, 1))
    m.cylinder((0, 0.08, 0.0), (0, 0.16, 0.0), 0.008, (0.3, 0.3, 0.3, 1), 8)
    return ps.node(), {"grip": Point3(0, -0.05, -0.02)}


def build_charge(name, color, size=(0.16, 0.05, 0.16)):
    def builder():
        ps = PartSet(name)
        p, m = ps.b("poly"), ps.b("metal")
        p.cbox((0, 0, 0), size, color)
        m.cbox((0, size[1] * 0.5 + 0.005, 0), (size[0] * 0.5, 0.01, size[2] * 0.3), (0.08, 0.08, 0.08, 1))
        ps.b("glass").cbox((0, size[1] * 0.5 + 0.011, 0.0), (0.02, 0.003, 0.02), (1.0, 0.15, 0.1, 1))
        return ps.node(), {"grip": Point3(0, -0.03, -0.04)}
    return builder


def build_barbed():
    ps = PartSet("barbed")
    m = ps.b("metal")
    for k in range(8):
        a = k / 8 * math.tau
        m.cylinder((math.cos(a) * 0.06, 0, math.sin(a) * 0.06), (math.cos(a + 0.8) * 0.06, 0.02, math.sin(a + 0.8) * 0.06),
                   0.003, (0.45, 0.45, 0.45, 1), 4)
    return ps.node(), {"grip": Point3(0, 0, -0.06)}


def build_dshield():
    ps = PartSet("dshield")
    m = ps.b("metal")
    m.box((-0.12, 0.0, -0.1), (0.12, 0.02, 0.12), (0.2, 0.22, 0.24, 1))
    ps.b("glass").box((-0.05, 0.02, 0.06), (0.05, 0.022, 0.1), (0.2, 0.3, 0.4, 1))
    return ps.node(), {"grip": Point3(0, -0.02, -0.08)}


def build_ash_round():
    ps = PartSet("ash_round")
    m, p = ps.b("metal"), ps.b("poly")
    p.box((-0.03, -0.05, -0.0), (0.03, 0.12, 0.05), (0.18, 0.18, 0.16, 1))
    m.cylinder((0, 0.1, 0.025), (0, 0.3, 0.025), 0.022, (0.12, 0.12, 0.12, 1), 12)
    m.cylinder((0, 0.27, 0.025), (0, 0.31, 0.025), 0.026, (0.25, 0.22, 0.12, 1), 12)
    _pistol_grip(p, BLACK)
    return ps.node(), _info((0, 0.31, 0.025), 0.06)


def build_hammer():
    ps = PartSet("hammer")
    m, w = ps.b("metal"), ps.b("poly")
    w.cylinder((0, -0.15, 0), (0, 0.5, 0), 0.016, (0.12, 0.12, 0.12, 1), 10)
    m.box((-0.07, 0.48, -0.045), (0.07, 0.58, 0.045), (0.25, 0.25, 0.27, 1))
    m.box((-0.075, 0.5, -0.05), (-0.06, 0.56, 0.05), (0.4, 0.4, 0.42, 1))
    return ps.node(), {"grip": Point3(0, -0.08, 0), "fore": Point3(0, 0.2, 0), "muzzle": Point3(0, 0.55, 0)}


def build_flash_shield():
    ps = PartSet("flash_shield")
    m = ps.b("metal")
    m.box((-0.25, 0.0, -0.45), (0.25, 0.03, 0.35), (0.12, 0.13, 0.14, 1))
    ps.b("glass").box((-0.12, 0.03, 0.18), (0.12, 0.035, 0.26), (0.2, 0.28, 0.35, 1))
    for i in range(3):
        for j in range(2):
            ps.b("glass").cbox((-0.12 + i * 0.12, 0.034, -0.1 + j * 0.12), (0.06, 0.006, 0.06), (0.9, 0.9, 0.8, 1))
    return ps.node(), {"grip": Point3(0, -0.04, 0.0)}


def build_jammer():
    ps = PartSet("jammer")
    m, p = ps.b("metal"), ps.b("poly")
    p.box((-0.06, -0.05, 0.0), (0.06, 0.05, 0.05), (0.1, 0.12, 0.1, 1))
    m.cylinder((0.04, 0.0, 0.05), (0.04, 0.0, 0.15), 0.004, BLACK, 6)
    ps.b("glass").box((-0.03, 0.05, 0.02), (0.0, 0.052, 0.035), (0.1, 0.9, 0.2, 1))
    return ps.node(), {"grip": Point3(0, -0.04, -0.02)}


def build_armor_bag():
    ps = PartSet("armor_bag")
    p = ps.b("cloth")
    p.box((-0.12, -0.08, 0.0), (0.12, 0.08, 0.16), (0.2, 0.22, 0.18, 1))
    ps.b("paint").box((-0.03, 0.08, 0.06), (0.03, 0.085, 0.12), (0.9, 0.85, 0.2, 1))
    return ps.node(), {"grip": Point3(0, -0.06, 0.0)}


def build_drone():
    ps = PartSet("drone")
    m, p = ps.b("metal"), ps.b("poly")
    p.box((-0.07, -0.06, 0.03), (0.07, 0.06, 0.08), (0.15, 0.15, 0.16, 1))
    for x in (-0.09, 0.09):
        m.cylinder((x, 0, 0.055), (x * 1.25, 0, 0.055), 0.055, (0.08, 0.08, 0.08, 1), 14)
    ps.b("glass").cylinder((0, 0.06, 0.06), (0, 0.068, 0.06), 0.012, (0.2, 0.6, 1.0, 1), 10)
    return ps.node(), {}


def build_camera():
    ps = PartSet("camera")
    m = ps.b("metal")
    m.sphere((0, 0, 0), 0.07, (0.85, 0.85, 0.85, 1), 8, 12, scale=(1, 1, 0.8))
    m.box((-0.02, -0.02, 0.04), (0.02, 0.02, 0.12), (0.6, 0.6, 0.6, 1))
    ps.b("glass").sphere((0, 0.05, -0.01), 0.025, (0.05, 0.05, 0.08, 1), 6, 10)
    return ps.node(), {}


GUN_BUILDERS = {
    "m4a4": build_m4a4, "m4a1": build_m4a1, "famas": build_famas, "mp5": build_mp5, "mp9": build_mp9,
    "uzi": build_uzi, "glock": build_glock, "dualies": build_dualies, "scar20": build_scar20, "m249": build_m249,
    "nova": build_nova, "flash": build_flash, "c4": build_c4, "defuser": build_defuser,
    "breach": build_charge("breach", (0.3, 0.3, 0.25, 1)),
    "thermite": build_charge("thermite", (0.55, 0.35, 0.15, 1), (0.22, 0.06, 0.3)),
    "castle": build_charge("castle", (0.2, 0.22, 0.25, 1), (0.25, 0.03, 0.18)),
    "edd": build_charge("edd", (0.2, 0.2, 0.18, 1), (0.08, 0.04, 0.1)),
    "barbed": build_barbed, "dshield": build_dshield, "ash_round": build_ash_round, "hammer": build_hammer,
    "flash_shield": build_flash_shield, "jammer": build_jammer, "armor_bag": build_armor_bag,
    "drone": build_drone, "camera": build_camera,
}

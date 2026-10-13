"""Procedurally synthesised sound effects + simple positional playback."""
from __future__ import annotations

import math
import os
import wave

import numpy as np
from panda3d.core import Filename

SR = 44100
SYNTH_VERSION = "6"
CUSTOM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "custom_sfx")


def _lowpass(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X *= 1.0 / (1.0 + (f / fc) ** 4)
    return np.fft.irfft(X, len(x))


def _highpass(x, fc):
    return x - _lowpass(x, fc)


def _bandpass(x, lo, hi):
    return _lowpass(_highpass(x, lo), hi)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _norm(x, peak=0.9):
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


def _echo(x, taps):
    out = x.copy()
    for delay, gain, fc in taps:
        d = int(delay * SR)
        if d < len(x):
            out[d:] += _lowpass(x, fc)[:-d] * gain
    return out


def gunshot(dur, crack_fc, body_fc, thump_hz, tail, drive, seed, crack_decay=0.010, body_decay=0.05):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    n = rng.standard_normal(len(t))
    crack = _highpass(_lowpass(n, crack_fc), 400) * np.exp(-t / crack_decay)
    body = _lowpass(n, body_fc) * np.exp(-t / body_decay)
    thump = np.sin(2 * np.pi * thump_hz * t * (1 - 0.8 * t)) * np.exp(-t / 0.045)
    tailn = _lowpass(rng.standard_normal(len(t)), 900) * np.exp(-t / tail) * 0.35
    x = crack * 1.2 + body * 0.9 + thump * 1.1 + tailn
    x = _echo(x, [(0.05, 0.22, 2500), (0.12, 0.16, 1500), (0.21, 0.1, 900), (0.33, 0.06, 600)])
    x = np.tanh(_norm(x, 1.0) * drive)
    return _norm(x * np.minimum(1.0, (dur - t) * 20))


def click(freq=3200, dur=0.05, seed=0, decay=0.006, noise=0.6):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    x = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    x += _highpass(rng.standard_normal(len(t)), 2000) * np.exp(-t / (decay * 0.6)) * noise
    return _norm(x, 0.8)


def compose(dur, parts):
    out = np.zeros(int(dur * SR))
    for at, sig, gain in parts:
        i = int(at * SR)
        j = min(len(out), i + len(sig))
        out[i:j] += sig[: j - i] * gain
    return _norm(out, 0.85)


def whoosh(dur=0.25, lo=600, hi=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    x = _bandpass(rng.standard_normal(len(t)), lo, hi)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return _norm(x * env, 0.7)


def thud(freq=90, dur=0.18, seed=0):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    x = np.sin(2 * np.pi * freq * t * (1 - 0.6 * t / dur)) * np.exp(-t / 0.05)
    x += _lowpass(rng.standard_normal(len(t)), 1500) * np.exp(-t / 0.02) * 0.6
    return _norm(x, 0.85)


def explosion(seed=11):
    rng = np.random.default_rng(seed)
    t = _t(2.4)
    n = rng.standard_normal(len(t))
    boom = _lowpass(n, 180) * np.exp(-t / 0.5) * 3.0
    crack = _lowpass(n, 5000) * np.exp(-t / 0.03)
    rumble = _lowpass(rng.standard_normal(len(t)), 90) * np.exp(-t / 0.9) * 2.0
    debris = _bandpass(rng.standard_normal(len(t)), 1500, 6000) * np.exp(-((t - 0.25) ** 2) / 0.05) * 0.25
    x = boom + crack + rumble + debris
    x = _echo(x, [(0.15, 0.3, 600), (0.4, 0.2, 400)])
    return _norm(np.tanh(_norm(x, 1.0) * 2.2))


def hiss(dur=3.0, seed=5):
    rng = np.random.default_rng(seed)
    t = _t(dur)
    x = _bandpass(rng.standard_normal(len(t)), 1500, 9000)
    env = np.minimum(1, t / 0.08) * np.exp(-t / 1.6)
    return _norm(x * env, 0.6)


def footstep(seed):
    rng = np.random.default_rng(seed)
    t = _t(0.16)
    clank = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, d in
                ((rng.uniform(380, 460), 0.03), (rng.uniform(900, 1100), 0.015), (rng.uniform(2200, 2600), 0.008)))
    scuff = _bandpass(rng.standard_normal(len(t)), 300, 3000) * np.exp(-t / 0.02)
    return _norm(clank * 0.6 + scuff, 0.7)


def tone(freqs, dur, decay):
    t = _t(dur)
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) * np.exp(-t / decay)
    return _norm(x * np.minimum(1, t / 0.002), 0.7)


def bounce(seed=3):
    t = _t(0.12)
    x = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / 0.03) for f in (1300, 2900, 4100))
    return _norm(x, 0.7)


def _bell(f0, dur, decay, partials=((1.0, 1.0), (2.76, 0.5), (5.4, 0.25), (8.93, 0.12))):
    t = _t(dur)
    x = sum(g * np.sin(2 * np.pi * f0 * r * t) * np.exp(-t / (decay / (1 + 0.4 * i)))
            for i, (r, g) in enumerate(partials))
    return x * np.minimum(1, t / 0.002)


def kill_sting(streak: int, seed: int = 0) -> np.ndarray:
    """Impact + metallic ring + rising arpeggio; more notes / higher pitch for longer streaks."""
    rng = np.random.default_rng(seed)
    dur = 1.3
    t = _t(dur)
    semis = [0, 4, 7, 12, 16, 19]
    root = 440.0 * 2 ** ((min(streak, 6) - 1) * 2 / 12)
    out = np.zeros(len(t))
    boom = np.sin(2 * np.pi * 52 * t * (1 - 0.35 * t)) * np.exp(-t / 0.16) * 1.1
    snap = _bandpass(rng.standard_normal(len(t)), 1500, 9000) * np.exp(-t / 0.012) * 0.7
    out += boom + snap
    ring = _bell(root * 2, dur, 0.45)
    out += ring * 0.55
    for k in range(min(streak, 6)):
        i = int((0.035 + k * 0.055) * SR)
        note = _bell(root * 2 ** (semis[k] / 12), dur - i / SR, 0.28, ((1, 1), (2, 0.35), (3, 0.15)))
        out[i:i + len(note)] += note[: len(out) - i] * 0.42
    out = _echo(out, [(0.09, 0.25, 4000), (0.19, 0.15, 3000), (0.31, 0.08, 2000)])
    return _norm(np.tanh(out * 1.3) * np.minimum(1.0, (dur - t) * 8))


def headshot_accent(seed=81):
    rng = np.random.default_rng(seed)
    t = _t(0.6)
    ping = _bell(3520, 0.6, 0.18, ((1, 1), (1.5, 0.6), (2.7, 0.3)))
    crunch = _bandpass(rng.standard_normal(len(t)), 600, 4000) * np.exp(-t / 0.03)
    return _norm(ping * 0.8 + crunch * 0.6)


def knife_accent(seed=83):
    rng = np.random.default_rng(seed)
    t = _t(0.55)
    sweep = np.zeros(len(t))
    n = rng.standard_normal(len(t))
    for lo, hi, at in ((1500, 3500, 0.0), (3000, 7000, 0.06), (5000, 11000, 0.12)):
        sweep += _bandpass(n, lo, hi) * np.exp(-((t - at - 0.05) ** 2) / 0.003)
    shing = _bell(2600, 0.55, 0.22, ((1, 1), (2.3, 0.5), (4.1, 0.3)))
    return _norm(sweep * 0.6 + shing * 0.7)


def grenade_accent(seed=85):
    rng = np.random.default_rng(seed)
    t = _t(0.9)
    boom = _lowpass(rng.standard_normal(len(t)), 160) * np.exp(-t / 0.25) * 3
    return _norm(np.tanh(boom) + _bell(660, 0.9, 0.4) * 0.4)


def ui_click():
    return _norm(_bell(1800, 0.09, 0.02, ((1, 1), (2.1, 0.4))) + click(4200, 0.09, 91, 0.004, 0.3), 0.6)


def ui_hover():
    return _norm(_bell(2600, 0.06, 0.012, ((1, 1),)), 0.35)


def build_library() -> dict[str, np.ndarray]:
    lib = {
        "ak47": gunshot(0.75, 6000, 1400, 85, 0.22, 2.6, 1),
        "awm": gunshot(1.4, 7000, 1000, 60, 0.45, 3.2, 2, crack_decay=0.014, body_decay=0.09),
        "deagle": gunshot(0.6, 5000, 1700, 110, 0.18, 2.4, 3),
        "dry": click(2600, 0.06, 4),
        "mag_out": compose(0.3, [(0.0, click(1800, 0.08, 5, 0.01), 1.0), (0.05, thud(220, 0.12, 6), 0.4)]),
        "mag_in": compose(0.3, [(0.0, click(1500, 0.08, 7, 0.012), 1.0), (0.04, click(3000, 0.06, 8), 0.8)]),
        "bolt": compose(0.45, [(0.0, click(1200, 0.1, 9, 0.02), 0.9), (0.18, click(2200, 0.08, 10, 0.01), 1.0)]),
        "draw": compose(0.25, [(0.0, whoosh(0.18, 800, 4000, 11), 0.5), (0.12, click(2500, 0.06, 12), 0.7)]),
        "knife_swing": whoosh(0.26, 500, 3500, 13),
        "knife_hit": compose(0.3, [(0.0, thud(120, 0.2, 14), 1.0), (0.0, whoosh(0.08, 1000, 5000, 15), 0.5)]),
        "knife_wall": compose(0.25, [(0.0, click(3500, 0.15, 16, 0.03, 0.9), 1.0)]),
        "explosion": explosion(),
        "bounce": bounce(),
        "pin": compose(0.35, [(0.0, click(4200, 0.08, 17, 0.01), 0.8), (0.12, click(2800, 0.1, 18, 0.02), 0.6)]),
        "smoke_hiss": hiss(),
        "step0": footstep(20), "step1": footstep(21), "step2": footstep(22), "step3": footstep(23),
        "land": thud(70, 0.22, 24),
        "jump": whoosh(0.12, 300, 1500, 25),
        "hit": tone((2100, 3150), 0.08, 0.025),
        "headshot": tone((1800, 3600, 5400), 0.25, 0.08),
        "kill": tone((660, 990, 1320), 0.35, 0.12),
        "hurt": thud(140, 0.2, 26),
        "impact_metal": compose(0.2, [(0.0, click(2800, 0.15, 27, 0.025, 0.4), 1.0)]),
        "impact_wood": thud(300, 0.1, 28),
        "flesh": thud(160, 0.12, 29),
        "kill_hs": headshot_accent(),
        "kill_knife": knife_accent(),
        "kill_nade": grenade_accent(),
        "ui_click": ui_click(),
        "ui_hover": ui_hover(),
        # extended arsenal
        "m4a4": gunshot(0.7, 6500, 1600, 90, 0.2, 2.4, 31),
        "m4a1": _norm(_lowpass(gunshot(0.35, 3500, 1200, 70, 0.06, 1.2, 32, crack_decay=0.004,
                                       body_decay=0.02), 3200), 0.55),
        "famas": gunshot(0.65, 6800, 1800, 95, 0.18, 2.3, 33),
        "mp5": gunshot(0.5, 5200, 1900, 120, 0.12, 1.9, 34, body_decay=0.035),
        "mp9": gunshot(0.45, 5600, 2100, 130, 0.1, 1.8, 35, body_decay=0.03),
        "uzi": gunshot(0.45, 4800, 1800, 125, 0.1, 2.0, 36, body_decay=0.03),
        "dualies": gunshot(0.45, 5800, 2000, 140, 0.1, 1.9, 37, body_decay=0.03),
        "scar20": gunshot(1.0, 7000, 1200, 70, 0.32, 2.9, 38, crack_decay=0.012, body_decay=0.07),
        "m249": gunshot(0.8, 5500, 1200, 75, 0.25, 2.8, 39),
        # objectives / gadgets
        "beep": tone((2600,), 0.09, 0.03),
        "plant_done": compose(0.5, [(0.0, tone((1800, 2700), 0.2, 0.06), 1.0), (0.18, click(2200, 0.1, 40), 0.6)]),
        "defuse_tick": click(3000, 0.05, 41, 0.004),
        "bomb_explode": _norm(explosion(91) + np.pad(explosion(92), (int(0.05 * SR), 0))[:len(explosion(91))], 0.95),
        "capture": tone((880, 1320, 1760), 0.45, 0.15),
        "lost": tone((440, 330), 0.45, 0.15),
        "round_win": tone((660, 880, 1320), 0.7, 0.25),
        "round_lose": tone((440, 370, 294), 0.7, 0.25),
        "flash_bang": compose(1.0, [(0.0, explosion(93), 0.6), (0.0, tone((3800,), 1.0, 0.6), 0.25)]),
        "wall_break": compose(0.45, [(0.0, thud(110, 0.3, 42), 1.0), (0.02, whoosh(0.25, 300, 2500, 43), 0.6),
                                     (0.05, click(1400, 0.2, 44, 0.04, 0.9), 0.5)]),
        "reinforce": compose(0.6, [(0.0, click(900, 0.2, 45, 0.05, 0.3), 1.0), (0.25, thud(80, 0.3, 46), 0.8)]),
        "drone": hiss(0.6, 47) * 0.4,
        "view_static": compose(0.3, [(0.0, hiss(0.22, 50), 0.5), (0.0, click(5200, 0.04, 51, 0.003), 0.7)]),
        "spot": tone((1500, 2250), 0.25, 0.06),
        "round_start": compose(0.5, [(0.0, tone((520,), 0.18, 0.06), 0.8), (0.2, tone((780,), 0.25, 0.08), 0.9)]),
        "buy": compose(0.2, [(0.0, click(2000, 0.08, 48), 0.8), (0.06, click(3000, 0.06, 49), 0.6)]),
    }
    for k in range(1, 7):
        lib[f"kill_{k}"] = kill_sting(k, 70 + k)
    return lib


def _write_wav(path, x):
    data = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def ensure_sounds(cache_dir: str) -> dict[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    stamp = os.path.join(cache_dir, "version.txt")
    current = open(stamp).read().strip() if os.path.exists(stamp) else ""
    paths = {}
    if current != SYNTH_VERSION:
        for name, sig in build_library().items():
            p = os.path.join(cache_dir, name + ".wav")
            _write_wav(p, sig)
        with open(stamp, "w") as f:
            f.write(SYNTH_VERSION)
    for fn in os.listdir(cache_dir):
        if fn.endswith(".wav"):
            paths[fn[:-4]] = os.path.join(cache_dir, fn)
    if os.path.isdir(CUSTOM_DIR):  # user-provided overrides, e.g. custom_sfx/kill_2.wav
        for fn in os.listdir(CUSTOM_DIR):
            name, ext = os.path.splitext(fn)
            if ext.lower() in (".wav", ".ogg", ".mp3"):
                paths[name] = os.path.join(CUSTOM_DIR, fn)
    return paths


POOL = {"ak47": 8, "awm": 4, "deagle": 6, "m4a4": 8, "m4a1": 8, "famas": 8, "mp5": 8, "mp9": 8, "uzi": 8,
        "dualies": 6, "scar20": 6, "m249": 10, "step0": 3, "step1": 3, "step2": 3, "step3": 3, "hit": 4,
        "impact_metal": 4, "flesh": 4, "bounce": 3, "explosion": 3}


class Audio:
    def __init__(self, base, cache_dir: str, master: float = 0.8):
        self.base = base
        self.master = master
        self.muted = False
        self.enabled = base.sfxManagerList and base.sfxManagerList[0] is not None
        self.pools: dict[str, list] = {}
        self.idx: dict[str, int] = {}
        if not self.enabled:
            return
        try:
            paths = ensure_sounds(cache_dir)
        except Exception as exc:  # pragma: no cover
            print("[audio] synthesis failed:", exc)
            self.enabled = False
            return
        for name, p in paths.items():
            n = POOL.get(name, 2)
            fn = Filename.fromOsSpecific(p)
            self.pools[name] = [base.loader.loadSfx(fn) for _ in range(n)]
            self.idx[name] = 0

    def ui(self, name, volume=0.6):
        """Interface sound: ignores `muted` (menu background)."""
        muted, self.muted = self.muted, False
        self.play(name, volume=volume)
        self.muted = muted

    def play(self, name, pos=None, listener=None, volume=1.0, rate=1.0, ref_dist=6.0):
        if not self.enabled or self.muted or name not in self.pools:
            return
        vol = volume * self.master
        bal = 0.0
        if pos is not None and listener is not None:
            lpos, lyaw = listener
            dx, dy, dz = pos[0] - lpos[0], pos[1] - lpos[1], pos[2] - lpos[2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            vol *= 1.0 / (1.0 + (d / ref_dist) ** 1.4)
            if d > 0.5:
                yr = math.radians(lyaw)
                right = dx * math.sin(yr) - dy * math.cos(yr)
                bal = max(-1.0, min(1.0, right / max(d, 1e-3))) * 0.8
        if vol < 0.01:
            return
        pool = self.pools[name]
        i = self.idx[name]
        self.idx[name] = (i + 1) % len(pool)
        s = pool[i]
        s.stop()
        s.setVolume(min(1.0, vol))
        s.setBalance(bal)
        s.setPlayRate(rate)
        s.play()

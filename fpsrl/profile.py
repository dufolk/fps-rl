"""Persistent player settings (~/.fpsrl/profile.json)."""
from __future__ import annotations

import getpass
import json
import os
from dataclasses import asdict, dataclass, field, fields

PROFILE_PATH = os.path.join(os.path.expanduser("~"), ".fpsrl", "profile.json")
SENSITIVITY_MIN = 0.1
SENSITIVITY_MAX = 12.0
SENSITIVITY_STEP = 0.1


def clamp_sensitivity(value) -> float:
    return max(SENSITIVITY_MIN, min(SENSITIVITY_MAX, round(float(value), 2)))


def _default_name():
    try:
        return getpass.getuser()[:12] or "Player"
    except Exception:
        return "Player"


@dataclass
class Profile:
    name: str = field(default_factory=_default_name)
    sensitivity: float = 1.6
    ads_toggle: bool = False
    volume: float = 0.8
    show_fps: bool = False
    fullscreen: bool = False
    primary: str = "ak47"
    skins: dict = field(default_factory=dict)  # weapon -> skin key
    single: dict = field(default_factory=lambda: {"map": "transport_ship", "mode": "tdm", "team_size": 4,
                                                  "difficulty": "normal", "team": 0, "kill_limit": 60,
                                                  "time_limit": 420})
    host: dict = field(default_factory=lambda: {"room": "运输船", "port": 47777, "map": "transport_ship",
                                                "mode": "tdm", "team_size": 4, "difficulty": "normal",
                                                "kill_limit": 60, "time_limit": 600})
    last_ip: str = ""
    last_port: int = 47777
    team_pref: int = -1

    @classmethod
    def load(cls, path: str = PROFILE_PATH) -> "Profile":
        p = cls()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return p
        known = {f.name: f for f in fields(cls)}
        for k, v in data.items():
            if k not in known:
                continue
            cur = getattr(p, k)
            if isinstance(cur, dict) and isinstance(v, dict):
                cur.update(v)
            elif isinstance(cur, bool):
                setattr(p, k, bool(v))
            elif isinstance(cur, (int, float)) and isinstance(v, (int, float)):
                setattr(p, k, type(cur)(v))
            elif isinstance(cur, str) and isinstance(v, str):
                setattr(p, k, v)
        p.sensitivity = clamp_sensitivity(p.sensitivity)
        return p

    def save(self, path: str = PROFILE_PATH):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except OSError as exc:
            print("[profile] save failed:", exc)

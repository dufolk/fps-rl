"""In-game B menu: CS style buy menu (bomb mode), operator selection (Siege) or loadout
presets (TDM / domination)."""
from __future__ import annotations

from direct.gui.DirectGui import DirectFrame
from panda3d.core import TextNode

from ..operators import OPERATORS
from ..weapons import BUY_CATEGORIES, LOADOUT_BY_KEY, LOADOUTS, SLOT_BOMB, WEAPONS, item_name
from .menu import ACCENT, BTN, BTN_HOVER, BTN_OFF, BTN_PRESS, DIM, PANEL, UI, WHITE

DEFUSER_PRICE = 400
GEAR = {
    "kevlar": ("防弹衣", "Kevlar", 650),
    "kevlar_helmet": ("防弹衣 + 头盔", "Kevlar + Helmet", 1000),
    "defuser": ("拆弹器", "Defuse kit", DEFUSER_PRICE),
}
OWNED = (0.2, 0.32, 0.22, 0.95)
POOR = (0.1, 0.1, 0.11, 0.7)


class BuyMenu:
    """Click to buy / pick. `on_buy(name)`, `on_operator(name, primary)` and `on_loadout(key)` send the
    request to the authoritative world (directly when hosting, as a command when connected to a host).
    `on_loadout` returns True when the weapons were swapped at once."""

    def __init__(self, viewer, on_buy, on_operator, on_loadout=None, sound=None):
        self.viewer = viewer
        self.base = viewer.base
        self.on_buy = on_buy
        self.on_operator = on_operator
        self.on_loadout = on_loadout
        self.ui = UI(self.base, sound)
        self.root = self.base.aspect2d.attachNewNode("buy_menu")
        self.root.setBin("gui-popup", 40)
        self.root.hide()
        self.page = None
        self.kind = None
        self.buttons: dict = {}
        self.money = None
        self.visible = False
        self.refresh_in = 0.0

    @property
    def world(self):
        return self.viewer.world

    @property
    def agent(self):
        return self.world.agents[self.viewer.pov]

    def available(self) -> bool:
        m = self.world.mode
        if m.free_weapons:
            return self.on_loadout is not None
        if getattr(m, "economy", False):
            return m.can_buy_now(self.agent)
        return m.name == "r6"

    def toggle(self):
        if self.visible:
            self.close()
        else:
            self.open()

    def open(self):
        L = self.ui.L
        if not self.available():
            hud = self.viewer.hud
            if hud is not None:
                hud.set_info(L("现在不能购买", "Buying is not possible now"))
            return False
        self._build()
        self.root.show()
        self.visible = True
        return True

    def close(self):
        if self.page is not None:
            self.page.destroy()
            self.page = None
        self.buttons = {}
        self.root.hide()
        self.visible = False

    # ------------------------------------------------------------------ pages
    def _build(self):
        if self.page is not None:
            self.page.destroy()
        self.page = DirectFrame(parent=self.root, frameColor=(0, 0, 0, 0))
        DirectFrame(parent=self.page, frameColor=PANEL, frameSize=(-1.25, 1.25, -0.72, 0.72))
        self.buttons = {}
        if self.world.mode.free_weapons:
            self.kind = "loadout"
            self._build_loadout()
        elif getattr(self.world.mode, "economy", False):
            self.kind = "buy"
            self._build_buy()
        else:
            self.kind = "ops"
            self._build_ops()
        self.refresh()

    def _build_buy(self):
        ui, L = self.ui, self.ui.L
        ui.label(self.page, L("购买", "BUY"), (-1.18, 0.6), 0.07, WHITE)
        self.money = ui.label(self.page, "", (1.18, 0.6), 0.06, (0.45, 1.0, 0.5, 1), TextNode.ARight)
        ui.label(self.page, L("点击购买   B / Esc 关闭", "Click to buy   B / ESC close"), (-1.18, -0.68), 0.034, DIM)
        cats = list(BUY_CATEGORIES)
        gear = ["kevlar", "kevlar_helmet"]
        if getattr(self.world.mode, "bomb_item", "") == "c4" and self.world.mode.side(self.agent.team) == 1:
            gear.append("defuser")
        cats.append(("gear", "装备", "Gear", tuple(gear)))
        col_w = 2.36 / len(cats)
        for ci, (_key, zh, en, names) in enumerate(cats):
            x = -1.18 + col_w * (ci + 0.5)
            ui.label(self.page, L(zh, en), (x, 0.45), 0.045, ACCENT, TextNode.ACenter)
            shown = []
            for name in names:
                if name in GEAR:
                    shown.append(name)
                    continue
                spec = WEAPONS[name]
                allowed = getattr(self.world.mode, "SIDE_WEAPONS", None)
                if spec.kind == "gun" and allowed is not None \
                        and name not in allowed[self.world.mode.side(self.agent.team)]:
                    continue
                shown.append(name)
            for ri, name in enumerate(shown):
                if name in GEAR:
                    zh_name, en_name, price = GEAR[name]
                    label = L(zh_name, en_name)
                else:
                    spec = WEAPONS[name]
                    price = 600 if name == "molotov" and self.world.mode.side(self.agent.team) == 1 else spec.price
                    label = L(item_name(name, True), spec.display)
                cost = "$1000 / $350" if name == "kevlar_helmet" else f"${price}"
                b = ui.button(self.page, f"{label}\n{cost}", (x, 0.32 - ri * 0.15), self._buy,
                              (col_w * 0.46, 0.062), 0.034, [name])
                self.buttons[name] = (b, price)

    def _build_loadout(self):
        ui, L = self.ui, self.ui.L
        a = self.agent
        ui.label(self.page, L("选择背包", "LOADOUT"), (-1.18, 0.6), 0.07, WHITE)
        ui.label(self.page, L("出生 6 秒内立即换装，否则下次出生生效", "Applies at once within 6 s of spawning, "
                              "otherwise on the next spawn"), (-1.18, 0.5), 0.034, DIM)
        ui.label(self.page, L("点击或按 1-3 选择   B / Esc 关闭", "Click or press 1-3   B / ESC close"),
                 (-1.18, -0.68), 0.034, DIM)
        cur = (a.loadout.get("primary"), a.loadout.get("secondary"))
        for i, (key, zh, en, primary, secondary) in enumerate(LOADOUTS):
            x = -0.8 + i * 0.8
            p = WEAPONS[primary]
            picked = cur == (primary, secondary)
            text = f"{i + 1}  {L(zh, en)}\n\n{p.display}\n{WEAPONS[secondary].display}"
            b = ui.button(self.page, text, (x, 0.02), self._loadout, (0.36, 0.38), 0.05, [key],
                          color=OWNED if picked else BTN)
            b["text_pos"] = (0, 0.1)
            self.buttons[key] = (b, 0)
            dmg = p.damage * p.pellets
            stats = L(f"伤害 {dmg:.0f}   弹匣 {p.mag}   射速 {60 / p.interval:.0f}/分",
                      f"DMG {dmg:.0f}   MAG {p.mag}   RPM {60 / p.interval:.0f}")
            ui.label(self.page, stats, (x, -0.3), 0.032, DIM, TextNode.ACenter)

    def pick_index(self, i: int) -> bool:
        """Number-key shortcut while the loadout page is open."""
        if self.kind != "loadout" or not 0 <= i < len(LOADOUTS):
            return False
        self._loadout(LOADOUTS[i][0])
        return True

    def _build_ops(self):
        ui, L = self.ui, self.ui.L
        m = self.world.mode
        side = m.side(self.agent.team)
        ui.label(self.page, L("选择干员", "OPERATOR"), (-1.18, 0.6), 0.07, WHITE)
        note = L("准备阶段内立即生效，否则下回合生效", "Applies now during prep, otherwise next round") \
            if m.phase != "prep" else L("准备阶段：选择后立即换装", "Prep phase: loadout changes at once")
        ui.label(self.page, note, (-1.18, 0.5), 0.034, DIM)
        ui.label(self.page, L("B / Esc 关闭", "B / ESC close"), (-1.18, -0.68), 0.034, DIM)
        ops = [o for o in OPERATORS.values() if o.side == side]
        cur = m.ops.get(self.agent.id, {}).get("attack" if side == 0 else "defend")
        for i, o in enumerate(ops):
            y = 0.34 - i * 0.24
            name = L(o.zh, o.name.capitalize())
            b = ui.button(self.page, name, (-0.82, y), self._pick, (0.3, 0.07), 0.05, [o.name, None],
                          color=OWNED if o.name == cur else BTN)
            self.buttons[o.name] = (b, 0)
            ui.label(self.page, L(o.desc_zh, o.desc_en), (-0.44, y + 0.03), 0.034, WHITE)
            gear = [o.gadget, *o.generics, o.secondary]
            ui.label(self.page, "  ·  ".join(L(item_name(g, True), WEAPONS[g].display) for g in gear),
                     (-0.44, y - 0.04), 0.03, DIM)
            for k, p in enumerate(o.primaries):
                ui.button(self.page, WEAPONS[p].display, (0.62 + k * 0.32, y), self._pick, (0.15, 0.05), 0.034,
                          [o.name, p])

    # ------------------------------------------------------------------ actions
    def _buy(self, name):
        self.on_buy(name)
        self.refresh_in = 0.1

    def _loadout(self, key):
        now = self.on_loadout(key)
        hud = self.viewer.hud
        if hud is not None:
            _key, zh, en, _p, _s = LOADOUT_BY_KEY[key]
            hud.set_info(self.ui.L(f"背包: {zh}" + ("" if now else " (下次出生生效)"),
                                   f"Loadout: {en}" + ("" if now else " (next spawn)")))
        self.close()

    def _pick(self, name, primary):
        self.on_operator(name, primary)
        hud = self.viewer.hud
        if hud is not None:
            o = OPERATORS[name]
            hud.set_info(self.ui.L(f"干员: {o.zh}", f"Operator: {o.name.capitalize()}"))
        self.close()

    def refresh(self):
        if self.kind != "buy":
            return
        a = self.agent
        self.money["text"] = f"${a.money}"
        for name, (b, price) in self.buttons.items():
            if name == "kevlar":
                owned = a.armor > 0
            elif name == "kevlar_helmet":
                owned = a.helmet
            elif name == "defuser":
                owned = SLOT_BOMB in a.inv
            else:
                spec = WEAPONS[name]
                cur = a.inv.get(spec.slot)
                max_carry = int(spec.extra.get("max_carry", 1))
                owned = cur is not None and cur.name == name \
                    and (spec.kind != "grenade" or cur.mag >= max_carry)
            actual = getattr(self.world.mode, "gear_price", lambda _a, _n: price)(a, name) \
                if name in GEAR else price
            poor = not owned and (actual is None or a.money < actual)
            col = OWNED if owned else (POOR if poor else BTN)
            b["frameColor"] = (col, BTN_PRESS, BTN_HOVER if not owned else col, BTN_OFF)
            b["text_fg"] = (0.5, 0.5, 0.52, 1) if poor else WHITE

    def update(self, dt):
        if not self.visible:
            return
        if not self.available():
            self.close()
            return
        self.refresh_in -= dt
        if self.refresh_in <= 0:
            self.refresh_in = 0.25
            self.refresh()

    def destroy(self):
        self.close()
        self.root.removeNode()

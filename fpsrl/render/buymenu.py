"""In-game B menu: CS style buy menu (bomb mode) or operator selection (Siege)."""
from __future__ import annotations

from direct.gui.DirectGui import DirectFrame
from panda3d.core import TextNode

from ..operators import OPERATORS
from ..weapons import BUY_CATEGORIES, SLOT_BOMB, WEAPONS
from .menu import ACCENT, BTN, BTN_HOVER, BTN_OFF, BTN_PRESS, DIM, PANEL, UI, WHITE

DEFUSER_PRICE = 400
OWNED = (0.2, 0.32, 0.22, 0.95)
POOR = (0.1, 0.1, 0.11, 0.7)


class BuyMenu:
    """Click to buy / pick. `on_buy(name)` and `on_operator(name, primary)` send the request to the
    authoritative world (directly when hosting, as a command when connected to a host)."""

    def __init__(self, viewer, on_buy, on_operator, sound=None):
        self.viewer = viewer
        self.base = viewer.base
        self.on_buy = on_buy
        self.on_operator = on_operator
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
        if m.name == "bomb":
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
        if self.world.mode.name == "bomb":
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
        if self.world.mode.side(self.agent.team) == 1:
            cats.append(("gear", "装备", "Gear", ("defuser",)))
        col_w = 2.36 / len(cats)
        for ci, (_key, zh, en, names) in enumerate(cats):
            x = -1.18 + col_w * (ci + 0.5)
            ui.label(self.page, L(zh, en), (x, 0.45), 0.045, ACCENT, TextNode.ACenter)
            for ri, name in enumerate(names):
                spec = WEAPONS[name]
                price = DEFUSER_PRICE if name == "defuser" else spec.price
                label = L("拆弹器", "Defuse kit") if name == "defuser" else spec.display
                b = ui.button(self.page, f"{label}\n${price}", (x, 0.32 - ri * 0.15), self._buy,
                              (col_w * 0.46, 0.062), 0.034, [name])
                self.buttons[name] = (b, price)

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
            gadget = WEAPONS[o.gadget].display
            ui.label(self.page, f"{gadget}  ·  {WEAPONS[o.secondary].display}", (-0.44, y - 0.04), 0.03, DIM)
            for k, p in enumerate(o.primaries):
                ui.button(self.page, WEAPONS[p].display, (0.62 + k * 0.32, y), self._pick, (0.15, 0.05), 0.034,
                          [o.name, p])

    # ------------------------------------------------------------------ actions
    def _buy(self, name):
        self.on_buy(name)
        self.refresh_in = 0.1

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
            spec = WEAPONS[name]
            cur = a.inv.get(spec.slot)
            owned = cur is not None and cur.name == name
            if name == "defuser":
                owned = SLOT_BOMB in a.inv
            poor = not owned and a.money < price
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

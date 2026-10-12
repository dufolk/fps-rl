"""Render every weapon skin into one contact sheet: py tools/skin_preview.py [out.png]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import (AmbientLight, DirectionalLight, Filename, Point3, TextNode,  # noqa: E402
                          Vec4, loadPrcFileData)

from fpsrl.render.models import ModelCache  # noqa: E402
from fpsrl.render.skins import SKIN_WEAPONS, SKINS  # noqa: E402


def main(out="shots/skins.png"):
    loadPrcFileData("", "win-size 1600 900\nframebuffer-multisample 1\nmultisamples 4\nsync-video #f")
    from direct.showbase.ShowBase import ShowBase
    base = ShowBase()
    base.disableMouse()
    base.setBackgroundColor(0.16, 0.17, 0.2, 1)
    base.render.setShaderAuto()
    sun = DirectionalLight("sun")
    sun.setColor(Vec4(1.3, 1.2, 1.05, 1))
    sn = base.render.attachNewNode(sun)
    sn.setHpr(30, -35, 0)
    base.render.setLight(sn)
    amb = AmbientLight("amb")
    amb.setColor(Vec4(0.45, 0.47, 0.55, 1))
    base.render.setLight(base.render.attachNewNode(amb))
    models = ModelCache()
    span = {"ak47": 1.15, "awm": 1.55, "deagle": 0.45, "knife": 0.4, "frag": 0.16}
    for row, w in enumerate(SKIN_WEAPONS):
        for col, skin in enumerate(SKINS[w]):
            node, _ = models.weapon(w, skin.key)
            node.reparentTo(base.render)
            k = 1.9 / span[w]
            node.setScale(k)
            node.setPos(-4.4 + col * 2.2, 0, 2.6 - row * 1.3)
            node.setHpr(90, 8, 0)
            tn = TextNode("l")
            tn.setText(f"{w} / {skin.en}")
            tn.setAlign(TextNode.ACenter)
            t = base.render.attachNewNode(tn)
            t.setScale(0.14)
            t.setPos(-4.4 + col * 2.2, 0, 2.15 - row * 1.3)
    base.camLens.setFov(48)
    base.cam.setPos(0, -12, -0.1)
    base.cam.lookAt(0, 0, 0)
    for _ in range(3):
        base.graphicsEngine.renderFrame()
    base.win.saveScreenshot(Filename.fromOsSpecific(out))
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])

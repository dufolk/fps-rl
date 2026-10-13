"""Render every weapon / gadget model in a grid and save a screenshot (visual check of the
procedural models). Usage: py tools/gun_gallery.py [out.png] [skin]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda3d.core import AmbientLight, DirectionalLight, Filename, TextNode, Vec4, loadPrcFileData  # noqa: E402

loadPrcFileData("", "win-size 1600 900\nwindow-title gallery\naudio-library-name null\n")

from direct.showbase.ShowBase import ShowBase  # noqa: E402

from fpsrl.render.models import ModelCache  # noqa: E402
from fpsrl.render.textures import TextureLibrary  # noqa: E402

NAMES = ["ak47", "m4a4", "m4a1", "famas", "mp5", "mp9", "uzi", "awm", "scar20", "m249", "nova", "glock", "deagle", "dualies",
         "knife", "frag", "smoke", "flash", "molotov", "c4", "defuser", "breach", "thermite", "ash_round", "hammer",
         "flash_shield", "castle", "jammer", "edd", "barbed", "dshield", "armor_bag", "drone", "camera"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "gallery.png"
    skin = sys.argv[2] if len(sys.argv) > 2 else None
    base = ShowBase()
    base.textures = TextureLibrary()
    base.disableMouse()
    base.setBackgroundColor(0.5, 0.52, 0.56, 1)
    base.render.setShaderAuto()
    sun = DirectionalLight("sun")
    sun.setColor(Vec4(1.2, 1.1, 1.0, 1))
    sn = base.render.attachNewNode(sun)
    sn.setHpr(-30, -40, 0)
    base.render.setLight(sn)
    amb = AmbientLight("amb")
    amb.setColor(Vec4(0.45, 0.45, 0.5, 1))
    base.render.setLight(base.render.attachNewNode(amb))
    models = ModelCache()
    cols = 8
    for i, name in enumerate(NAMES):
        node, _ = models.weapon(name, skin)
        node.reparentTo(base.render)
        x = (i % cols) * 1.2 - (cols - 1) * 0.6
        z = -(i // cols) * 0.75 + 1.1
        node.setPos(x, 0, z)
        node.setH(-90)
        node.setScale(1.0 if name not in ("frag", "smoke", "flash", "molotov", "c4", "defuser", "jammer", "edd", "camera",
                                          "drone", "barbed", "breach") else 2.5)
        tn = TextNode(name)
        tn.setText(name)
        tn.setAlign(TextNode.ACenter)
        t = base.render.attachNewNode(tn)
        t.setScale(0.09)
        t.setPos(x, -0.3, z - 0.3)
    base.camera.setPos(0, -7.5, 0)
    base.camera.lookAt(0, 0, 0)
    base.camLens.setFov(70)
    for _ in range(3):
        base.graphicsEngine.renderFrame()
    base.win.saveScreenshot(Filename.fromOsSpecific(os.path.abspath(out)))
    print("saved", out)


if __name__ == "__main__":
    main()

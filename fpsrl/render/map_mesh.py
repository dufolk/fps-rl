"""Imported map meshes (maps/data/<name>/mesh.npz + textures), drawn instead of the collision boxes."""
from __future__ import annotations

import os

import numpy as np
from panda3d.core import ColorWriteAttrib, Geom, GeomNode, GeomTriangles, GeomVertexData, NodePath

from . import models as M
from .meshes import vertex_format
from .textures import flat_normal, make_texture

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "data")
WOOD = ("Crt", "Crte", "Door")


def _load_png(path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _tangents(pos, uv, tri):
    p0, p1, p2 = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    u0, u1, u2 = uv[tri[:, 0]], uv[tri[:, 1]], uv[tri[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = u1 - u0, u2 - u0
    n = np.cross(e1, e2)
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
    normal = np.zeros_like(pos)
    tangent = np.zeros_like(pos)
    for k in range(3):
        np.add.at(normal, tri[:, k], n)
        np.add.at(tangent, tri[:, k], t)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    tangent -= normal * np.sum(tangent * normal, axis=1, keepdims=True)
    bad = np.linalg.norm(tangent, axis=1) < 1e-9
    tangent[bad] = np.cross(normal[bad], [0.0, 0.0, 1.0]) + np.cross(normal[bad], [1.0, 0.0, 0.0]) * 1e-3
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    binormal = np.cross(normal, tangent)
    return normal, tangent, binormal


def _geom_node(name, pos, uv, tri):
    normal, tangent, binormal = _tangents(pos, uv, tri)
    rows = np.empty((len(pos), 18), dtype=np.float32)
    rows[:, 0:3] = pos
    rows[:, 3:6] = normal
    rows[:, 6:10] = 1.0
    rows[:, 10:12] = uv
    rows[:, 12:15] = tangent
    rows[:, 15:18] = binormal
    vdata = GeomVertexData(name, vertex_format(), Geom.UH_static)
    vdata.uncleanSetNumRows(len(pos))
    vdata.modifyArray(0).modifyHandle().copyDataFrom(rows.tobytes())
    prim = GeomTriangles(Geom.UH_static)
    prim.setIndexType(Geom.NT_uint32)
    idx = prim.modifyVertices()
    idx.uncleanSetNumRows(len(tri) * 3)
    idx.modifyHandle().copyDataFrom(tri.astype(np.uint32).tobytes())
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def has_sky_shell(name) -> bool:
    """True when the imported map is sealed by a sky shell (nothing outside it should be drawn)."""
    path = os.path.join(DATA, name, "mesh.npz")
    return os.path.exists(path) and "sky_pos" in np.load(path).files


def build(parent, name, ts_normal):
    """Attach the imported mesh `name` under `parent`; returns its root NodePath."""
    folder = os.path.join(DATA, name)
    d = np.load(os.path.join(folder, "mesh.npz"))
    root = parent.attachNewNode(f"mesh_{name}")
    flat = make_texture(flat_normal(), f"{name}_flat_n")
    for k, (mat, label, tex) in enumerate(zip(d["materials"], d["names"], d["textures"])):
        np_ = _geom_node(str(mat), d[f"pos{k}"].astype(np.float64), d[f"uv{k}"].astype(np.float64),
                         d[f"tri{k}"])
        np_.reparentTo(root)
        path = os.path.join(folder, str(tex))
        if tex and os.path.exists(path):
            np_.setTexture(make_texture(_load_png(path), str(tex)), 1)
        np_.setTexture(ts_normal, flat, 1)
        surface = "wood" if any(w in str(label) for w in WOOD) else "cloth"
        np_.setMaterial(M.make_material(surface), 1)
    if "sky_pos" in d.files:
        _sky_mask(root, d["sky_pos"].astype(np.float64), d["sky_tri"])
    return root


def _sky_mask(root, pos, tri):
    """The BSP's sky-shell faces: depth only, drawn right after the sky dome, so whatever lies past the
    map's edge (the procedural ground and backdrop) is hidden behind sky, as in Source."""
    from .world_scene import SHADOW_MASK
    uv = np.zeros((len(pos), 2))
    node = _geom_node("sky_shell", pos, uv, tri)
    node.reparentTo(root)
    node.setAttrib(ColorWriteAttrib.make(ColorWriteAttrib.C_off))
    node.setBin("background", 10)
    node.setDepthWrite(True)
    node.setLightOff(1)
    node.setFogOff(1)
    node.setShaderOff(1)
    node.setTextureOff(1)
    node.hide(SHADOW_MASK)

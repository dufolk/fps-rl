"""Procedural mesh construction (boxes, tapered blocks, cylinders, spheres)."""
from __future__ import annotations

import math

from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexData,
                          GeomVertexFormat, GeomVertexWriter, InternalName, NodePath, Vec3)

_FORMAT = None


def vertex_format():
    global _FORMAT
    if _FORMAT is None:
        arr = GeomVertexArrayFormat()
        arr.addColumn(InternalName.getVertex(), 3, Geom.NT_float32, Geom.C_point)
        arr.addColumn(InternalName.getNormal(), 3, Geom.NT_float32, Geom.C_normal)
        arr.addColumn(InternalName.getColor(), 4, Geom.NT_float32, Geom.C_color)
        arr.addColumn(InternalName.getTexcoord(), 2, Geom.NT_float32, Geom.C_texcoord)
        arr.addColumn(InternalName.getTangent(), 3, Geom.NT_float32, Geom.C_vector)
        arr.addColumn(InternalName.getBinormal(), 3, Geom.NT_float32, Geom.C_vector)
        fmt = GeomVertexFormat()
        fmt.addArray(arr)
        _FORMAT = GeomVertexFormat.registerFormat(fmt)
    return _FORMAT


# Corner order: 0 (x0,y0,z0) 1 (x1,y0,z0) 2 (x1,y1,z0) 3 (x0,y1,z0) 4..7 same at z1.
_FACES = {
    "+x": (1, 2, 6, 5), "-x": (3, 0, 4, 7),
    "+y": (2, 3, 7, 6), "-y": (0, 1, 5, 4),
    "+z": (4, 5, 6, 7), "-z": (1, 0, 3, 2),
}


class MeshBuilder:
    def __init__(self, name: str = "mesh"):
        self.name = name
        self.vdata = GeomVertexData(name, vertex_format(), Geom.UH_static)
        self.vw = GeomVertexWriter(self.vdata, "vertex")
        self.nw = GeomVertexWriter(self.vdata, "normal")
        self.cw = GeomVertexWriter(self.vdata, "color")
        self.tw = GeomVertexWriter(self.vdata, "texcoord")
        self.tanw = GeomVertexWriter(self.vdata, "tangent")
        self.binw = GeomVertexWriter(self.vdata, "binormal")
        self.tris = GeomTriangles(Geom.UH_static)
        self.n = 0

    def _vertex(self, p, n, c, uv, t, b):
        self.vw.addData3(*p)
        self.nw.addData3(*n)
        self.cw.addData4(*c)
        self.tw.addData2(*uv)
        self.tanw.addData3(*t)
        self.binw.addData3(*b)
        self.n += 1

    def quad(self, p0, p1, p2, p3, color=(1, 1, 1, 1), uv_mode="world", uv_scale=1.0, uvs=None):
        """CCW quad (bottom-left, bottom-right, top-right, top-left seen from the front)."""
        p0, p1, p2, p3 = (Vec3(*p) for p in (p0, p1, p2, p3))
        t = p1 - p0
        b = p3 - p0
        if t.length() < 1e-9:
            t = p2 - p3
        if b.length() < 1e-9:
            b = p2 - p1
        n = t.cross(b)
        if n.length() < 1e-12:
            return
        n.normalize()
        t.normalize()
        b = n.cross(t)
        b.normalize()
        base = self.n
        for i, p in enumerate((p0, p1, p2, p3)):
            if uvs is not None:
                uv = uvs[i]
            elif uv_mode == "world":
                uv = (p.dot(t) * uv_scale, p.dot(b) * uv_scale)
            else:
                q = p - p0
                uv = (q.dot(t) * uv_scale, q.dot(b) * uv_scale)
            self._vertex(p, n, color, uv, t, b)
        self.tris.addVertices(base, base + 1, base + 2)
        self.tris.addVertices(base, base + 2, base + 3)

    def tri(self, p0, p1, p2, color=(1, 1, 1, 1)):
        p0, p1, p2 = Vec3(*p0), Vec3(*p1), Vec3(*p2)
        t = p1 - p0
        n = t.cross(p2 - p0)
        if n.length() < 1e-12:
            return
        n.normalize()
        t.normalize()
        b = n.cross(t)
        base = self.n
        for p, uv in ((p0, (0, 0)), (p1, (1, 0)), (p2, (0.5, 1))):
            self._vertex(p, n, color, uv, t, b)
        self.tris.addVertices(base, base + 1, base + 2)

    def hexa(self, corners, color=(1, 1, 1, 1), uv_mode="local", uv_scale=1.0, skip=()):
        for name, idx in _FACES.items():
            if name in skip:
                continue
            self.quad(*(corners[i] for i in idx), color=color, uv_mode=uv_mode, uv_scale=uv_scale)

    def box(self, mn, mx, color=(1, 1, 1, 1), uv_mode="local", uv_scale=1.0, skip=()):
        x0, y0, z0 = mn
        x1, y1, z1 = mx
        corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        self.hexa(corners, color, uv_mode, uv_scale, skip)

    def cbox(self, center, size, color=(1, 1, 1, 1), **kw):
        cx, cy, cz = center
        sx, sy, sz = size
        self.box((cx - sx / 2, cy - sy / 2, cz - sz / 2), (cx + sx / 2, cy + sy / 2, cz + sz / 2), color, **kw)

    def tapered(self, y0, y1, w0, h0, w1, h1, z0=0.0, z1=None, color=(1, 1, 1, 1), x=0.0):
        """Block along +Y whose cross-section changes from (w0,h0) at y0 to (w1,h1) at y1.

        Cross-sections are centred on x and bottom-aligned at z0 (front) / z1 (back)."""
        z1 = z0 if z1 is None else z1
        corners = [(x - w0 / 2, y0, z0), (x + w0 / 2, y0, z0), (x + w1 / 2, y1, z1), (x - w1 / 2, y1, z1),
                   (x - w0 / 2, y0, z0 + h0), (x + w0 / 2, y0, z0 + h0), (x + w1 / 2, y1, z1 + h1),
                   (x - w1 / 2, y1, z1 + h1)]
        self.hexa(corners, color)

    def cylinder(self, p0, p1, radius, color=(1, 1, 1, 1), segments=16, caps=True, radius1=None,
                 uv_scale=1.0):
        """Cylinder (or cone frustum) between points p0 and p1."""
        a = Vec3(*p0)
        b = Vec3(*p1)
        axis = b - a
        length = axis.length()
        if length < 1e-9:
            return
        axis /= length
        ref = Vec3(0, 0, 1) if abs(axis.z) < 0.9 else Vec3(1, 0, 0)
        u = axis.cross(ref)
        u.normalize()
        v = axis.cross(u)
        r0 = radius
        r1 = radius if radius1 is None else radius1
        ring = []
        for i in range(segments + 1):
            ang = 2.0 * math.pi * i / segments
            ring.append(u * math.cos(ang) + v * math.sin(ang))
        slope = (r0 - r1) / length
        for i in range(segments):
            d0, d1 = ring[i], ring[i + 1]
            n0 = d0 + axis * slope
            n1 = d1 + axis * slope
            n0.normalize()
            n1.normalize()
            base = self.n
            pts = (a + d0 * r0, a + d1 * r0, b + d1 * r1, b + d0 * r1)
            nrm = (n0, n1, n1, n0)
            us = (i / segments, (i + 1) / segments)
            uvs = ((us[0] * uv_scale, 0), (us[1] * uv_scale, 0), (us[1] * uv_scale, length * uv_scale),
                   (us[0] * uv_scale, length * uv_scale))
            for p, n, uv in zip(pts, nrm, uvs):
                t = axis.cross(n)
                self._vertex(p, n, color, uv, t, axis)
            self.tris.addVertices(base, base + 1, base + 2)
            self.tris.addVertices(base, base + 2, base + 3)
        if caps:
            for center, r, sign in ((a, r0, -1.0), (b, r1, 1.0)):
                if r <= 0:
                    continue
                n = axis * sign
                base = self.n
                self._vertex(center, n, color, (0.5, 0.5), u, v)
                for i in range(segments + 1):
                    d = ring[i]
                    self._vertex(center + d * r, n, color, (0.5 + 0.5 * d.dot(u), 0.5 + 0.5 * d.dot(v)), u, v)
                for i in range(segments):
                    if sign > 0:
                        self.tris.addVertices(base, base + 1 + i, base + 2 + i)
                    else:
                        self.tris.addVertices(base, base + 2 + i, base + 1 + i)

    def sphere(self, center, radius, color=(1, 1, 1, 1), rings=10, segments=16, scale=(1, 1, 1),
               inward=False, color_fn=None):
        cx, cy, cz = center
        sx, sy, sz = scale
        base = self.n
        for r in range(rings + 1):
            th = math.pi * r / rings
            for s in range(segments + 1):
                ph = 2.0 * math.pi * s / segments
                nx = math.sin(th) * math.cos(ph)
                ny = math.sin(th) * math.sin(ph)
                nz = math.cos(th)
                p = (cx + nx * radius * sx, cy + ny * radius * sy, cz + nz * radius * sz)
                n = (-nx, -ny, -nz) if inward else (nx, ny, nz)
                col = color_fn(nx, ny, nz) if color_fn else color
                self._vertex(p, n, col, (s / segments, 1 - r / rings), (-math.sin(ph), math.cos(ph), 0), (0, 0, 1))
        for r in range(rings):
            for s in range(segments):
                i0 = base + r * (segments + 1) + s
                i1 = i0 + segments + 1
                if inward:
                    self.tris.addVertices(i0, i0 + 1, i1)
                    self.tris.addVertices(i0 + 1, i1 + 1, i1)
                else:
                    self.tris.addVertices(i0, i1, i0 + 1)
                    self.tris.addVertices(i0 + 1, i1, i1 + 1)

    def node(self) -> NodePath:
        geom = Geom(self.vdata)
        geom.addPrimitive(self.tris)
        gn = GeomNode(self.name)
        if self.n:
            gn.addGeom(geom)
        return NodePath(gn)

    def empty(self) -> bool:
        return self.n == 0

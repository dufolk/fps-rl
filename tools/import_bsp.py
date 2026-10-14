"""Import a Source-engine BSP the way desert2 is imported: a render mesh plus box collision.

    py tools/import_bsp.py imports/transport_ship.bsp transport_ship
    py tools/import_bsp.py imports/zm_lostcities.bsp pyramid

Hammer units are scaled by 0.025 (a 72-unit player hull is 1.8 m). The map is re-centred so sea
level (or, without water, the lowest face) sits at z = 0, where the renderer puts its water / ground.

Only the areas that hold a spawn are imported: the 3D skybox and sealed-off rooms are dropped.
World faces, displacements and the faces of visible brush entities (moved to the entity origin)
become the mesh; water surfaces are left to the renderer's own water plane. Collision is one box per
axial player-solid brush. Slanted brushes (ramps, roofs, diagonal walls, pillars) are sampled in
columns and merged into a stack of smaller boxes, so their bounding box does not become an
invisible wall. The skybox shell and triggers are not solid.
Embedded VTF textures are written as PNG; a material with no packed texture gets a flat colour.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from import_dust2 import SCALE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_RENDER = ("TOOLS/TOOLSNODRAW", "TOOLS/TOOLSCLIP", "TOOLS/TOOLSPLAYERCLIP", "TOOLS/TOOLSSKYBOX",
               "TOOLS/TOOLSINVISIBLE", "TOOLS/TOOLSTRIGGER", "TOOLS/TOOLSHINT", "TOOLS/TOOLSSKIP",
               "TOOLS/TOOLSSKYBOX2D", "JBRODY/CONCRETECLIP", "TOOLS/TOOLSAREAPORTAL",
               "TOOLS/TOOLSOCCLUDER", "TOOLS/TOOLSFOG", "TOOLS/TOOLSORIGIN",
               # zm_lostcities weapon-box / supply-box pickup markers: translucent see-through-walls
               # icons on func_rotating brushes; drawn opaque they are floating white boards
               "XMFLS/WEAPONBOX", "XMFLS/SUPPLYBOX")
# brush entities that are drawn / that block players (others: triggers, buttons, dust ...)
RENDER_ENTS = {"func_brush", "func_wall", "func_wall_toggle", "func_breakable", "func_door",
               "func_door_rotating", "func_physbox", "func_movelinear", "func_illusionary",
               "func_rotating", "func_lod", "func_detail_blocker"}
SOLID_ENTS = {"func_brush", "func_wall", "func_wall_toggle", "func_breakable", "func_door",
              "func_door_rotating", "func_physbox", "func_movelinear", "func_lod"}
COL_CELL = 16.0     # column size (units) when a slanted brush is split into boxes
COL_ZQ = 4.0        # box tops / bottoms of a split brush snap to this (units)


def _lumps(data):
    return [struct.unpack_from("<III4s", data, 8 + i * 16)[:3] for i in range(64)]


def _lump(data, lumps, i):
    ofs, ln, _ver = lumps[i]
    return data[ofs:ofs + ln]


def _strings(blob, table):
    out = []
    for off in table:
        end = blob.index(b"\x00", off)
        out.append(blob[off:end].decode("latin1", "replace"))
    return out


def _entities(text):
    ents, cur = [], None
    for line in text.splitlines():
        line = line.strip()
        if line == "{":
            cur = {}
        elif line == "}":
            if cur is not None:
                ents.append(cur)
            cur = None
        elif cur is not None and line.startswith('"'):
            parts = line.split('"')
            if len(parts) >= 4:
                cur[parts[1]] = parts[3]
    return ents


def _rgb565(c):
    return (((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31)


def _dxt_color(block):
    c0, c1, bits = struct.unpack_from("<HHI", block)
    col = [_rgb565(c0), _rgb565(c1)]
    if c0 > c1:
        col.append(tuple((2 * a + b) // 3 for a, b in zip(col[0], col[1])))
        col.append(tuple((a + 2 * b) // 3 for a, b in zip(col[0], col[1])))
    else:
        col.append(tuple((a + b) // 2 for a, b in zip(col[0], col[1])))
        col.append((0, 0, 0))
    rgb = np.empty((4, 4, 3), np.uint8)
    for i in range(16):
        rgb[i // 4, i % 4] = col[(bits >> (2 * i)) & 3]
    return rgb


def _dxt5_block(block):
    a0, a1 = block[0], block[1]
    abits = int.from_bytes(block[2:8], "little")
    alphas = [a0, a1]
    if a0 > a1:
        alphas += [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    else:
        alphas += [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)]
        alphas += [0, 255]
    rgb = _dxt_color(block[8:16])
    # Alpha is unused: the renderer stores RGB.
    return rgb


def _decode_vtf(raw):
    """Largest mip of a VTF as an RGB uint8 array. Supports DXT1, DXT5 and uncompressed BGRA."""
    header = struct.unpack_from("<I", raw, 12)[0]
    width, height = struct.unpack_from("<HH", raw, 16)
    fmt, mips, lowfmt, loww, lowh = struct.unpack_from("<IBIBB", raw, 52)
    pos = header
    if lowfmt != 0xFFFFFFFF and loww and lowh:
        bw = max(1, (loww + 3) // 4) * max(1, (lowh + 3) // 4)
        pos += bw * (8 if lowfmt in (13, 20) else 16)
    bpp = {13: 8, 15: 16, 20: 8}.get(fmt)
    image = None
    for level in range(max(mips, 1)):
        mw = max(1, width >> (mips - 1 - level))
        mh = max(1, height >> (mips - 1 - level))
        if bpp is not None:
            n = max(1, (mw + 3) // 4) * max(1, (mh + 3) // 4)
            block = raw[pos:pos + n * bpp]
            pos += n * bpp
            if level != mips - 1:
                continue
            out = np.zeros((mh, mw, 3), np.uint8)
            bw, bh = max(1, (mw + 3) // 4), max(1, (mh + 3) // 4)
            for by in range(bh):
                for bx in range(bw):
                    i = (by * bw + bx) * bpp
                    rgb = _dxt_color(block[i:i + 8]) if bpp == 8 else _dxt5_block(block[i:i + 16])
                    x, y = bx * 4, by * 4
                    out[y:y + rgb.shape[0], x:x + rgb.shape[1]] = rgb[:mh - y, :mw - x]
            image = out
        elif fmt in (0, 12):  # RGBA8888 / BGRA8888, one mip assumed at the end
            n = mw * mh * 4
            pix = np.frombuffer(raw[pos:pos + n], np.uint8).reshape(mh, mw, 4)
            pos += n
            if level == mips - 1:
                image = pix[:, :, :3] if fmt == 0 else pix[:, :, 2::-1]
        else:
            return None
    return image


def _flat_color(name):
    key = name.upper()
    if "SAND" in key or "DIRT" in key:
        return (176, 148, 96)
    if "WATER" in key or "LIQUID" in key:
        return (70, 120, 150)
    if "GLASS" in key:
        return (160, 190, 200)
    if "WOOD" in key or "CRATE" in key or "BOX" in key:
        return (140, 100, 60)
    if "METAL" in key or "TRAIN" in key or "PIPE" in key:
        return (120, 124, 128)
    if "GRASS" in key:
        return (90, 120, 60)
    if "BRICK" in key:
        return (150, 80, 60)
    h = sum(ord(c) for c in key) % 360
    # a muted tint so unnamed surfaces stay apart
    return (90 + (h * 3) % 80, 90 + (h * 5) % 70, 90 + (h * 7) % 60)


def _pak_textures(data, lumps, wanted):
    """Map a lower-case material path (no extension) to PNG bytes."""
    ofs, ln, _ = lumps[40]
    if ln <= 0:
        return {}
    z = zipfile.ZipFile(io.BytesIO(data[ofs:ofs + ln]))
    index = {n.lower(): n for n in z.namelist()}
    out = {}
    for name in wanted:
        key = "materials/" + name.lower().replace("\\", "/")
        vmt = index.get(key + ".vmt")
        base = name
        if vmt:
            text = z.read(vmt).decode("latin1", "replace")
            for line in text.splitlines():
                low = line.lower()
                if "basetexture" in low and "bump" not in low:
                    q = line.replace("'", '"').split('"')
                    if len(q) >= 4:
                        base = q[-2]
                    break
        vtf = index.get("materials/" + base.lower().replace("\\", "/") + ".vtf")
        if vtf is None:
            continue
        image = _decode_vtf(z.read(vtf))
        if image is None:
            continue
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(image, "RGB").save(buf, "PNG")
        out[name.lower()] = buf.getvalue()
    return out


def _yaw_matrix(ent):
    ang = ent.get("angles", "0 0 0").split() if ent else []
    yaw = np.radians(float(ang[1])) if len(ang) > 1 else 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _angles_matrix(pitch, yaw, roll):
    p, y, r = np.radians([pitch, yaw, roll])
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp, sr * cp, cr * cp],
    ])


def _mdl_tris(mdl, vvd, vtx):
    """Triangles of LOD 0 of a Source MDL/VVD/VTX trio as (pos (N,3,3), uv (N,3,2), mat) tuples."""
    num_tex, tex_ofs = struct.unpack_from("<ii", mdl, 204)
    num_cd, cd_ofs = struct.unpack_from("<ii", mdl, 212)
    cds = []
    for i in range(num_cd):
        cofs = struct.unpack_from("<i", mdl, cd_ofs + i * 4)[0]
        cds.append(mdl[cofs:].split(b"\x00", 1)[0].decode("latin1", "replace").replace("\\", "/").strip("/"))
    prefix = cds[0] + "/" if cds and cds[0] else ""
    mats = []
    for i in range(num_tex):
        tofs = tex_ofs + i * 64
        name_ofs = struct.unpack_from("<i", mdl, tofs)[0]
        tname = mdl[tofs + name_ofs:].split(b"\x00", 1)[0].decode("latin1", "replace").replace("\\", "/").strip("/")
        mats.append((prefix + tname).upper())
    num_verts = struct.unpack_from("<i", vvd, 16)[0]
    num_fixups, fixup_ofs, vert_ofs = struct.unpack_from("<3i", vvd, 48)
    v_arr = np.ndarray(num_verts, dtype=[("w", "<3f"), ("b", "3u1"), ("nb", "u1"),
                                         ("pos", "<3f"), ("norm", "<3f"), ("uv", "<2f")],
                       buffer=vvd[vert_ofs:vert_ofs + num_verts * 48])
    if num_fixups > 0:
        fixed = np.empty_like(v_arr)
        out_i = 0
        for f in range(num_fixups):
            _lod, src_id, cnt = struct.unpack_from("<3i", vvd, fixup_ofs + f * 12)
            fixed[out_i:out_i + cnt] = v_arr[src_id:src_id + cnt]
            out_i += cnt
        v_arr = fixed
    num_bp, bp_ofs = struct.unpack_from("<ii", mdl, 232)
    vtx_num_bps, vtx_bp_ofs = struct.unpack_from("<ii", vtx, 28)
    out, seen = [], set()
    for b in range(min(num_bp, vtx_num_bps)):
        bo = bp_ofs + b * 16
        _name_ofs, num_models, _base, model_ofs = struct.unpack_from("<iiii", mdl, bo)
        vbo = vtx_bp_ofs + b * 8
        v_num_models, v_model_ofs = struct.unpack_from("<ii", vtx, vbo)
        if num_models < 1 or v_num_models < 1:
            continue
        mo = bo + model_ofs
        mname = mdl[mo:mo + 64].split(b"\x00", 1)[0].decode("latin1", "replace")
        if mname in seen:
            continue
        seen.add(mname)
        num_meshes, mesh_ofs, _m_nv, vert_index, _ = struct.unpack_from("<5i", mdl, mo + 72)
        vmo = vbo + v_model_ofs
        v_num_lods, v_lod_ofs = struct.unpack_from("<ii", vtx, vmo)
        if v_num_lods < 1:
            continue
        vlo = vmo + v_lod_ofs
        v_num_meshes, v_mesh_ofs, _ = struct.unpack_from("<iif", vtx, vlo)
        for ms in range(min(num_meshes, v_num_meshes)):
            mso = mo + mesh_ofs + ms * 116
            mat_idx, _midx, _mnv, m_vert_ofs = struct.unpack_from("<4i", mdl, mso)
            mat = mats[mat_idx] if 0 <= mat_idx < len(mats) else "UNKNOWN"
            vert_base = vert_index // 48 + m_vert_ofs
            vmso = vlo + v_mesh_ofs + ms * 9
            num_sg, sg_ofs, _ = struct.unpack_from("<iiB", vtx, vmso)
            for sg in range(num_sg):
                sgo = vmso + sg_ofs + sg * 25
                nv, vo, ni, io_, _nst, _sto, _sgf = struct.unpack_from("<iiiiiiB", vtx, sgo)
                vtx_verts = np.ndarray(nv, dtype=[("bw", "3u1"), ("nb", "u1"), ("vid", "<u2"), ("bid", "3u1")],
                                       buffer=vtx[sgo + vo:sgo + vo + nv * 9])
                vtx_inds = np.frombuffer(vtx[sgo + io_:sgo + io_ + ni * 2], np.uint16).reshape(-1, 3)
                vids = vert_base + vtx_verts["vid"][vtx_inds[:, ::-1]]
                pos = v_arr["pos"][vids].astype(np.float64)
                uv = v_arr["uv"][vids].astype(np.float64).copy()
                uv[..., 1] = 1.0 - uv[..., 1]
                out.append((pos, uv, mat))
    return out


def _static_prop_polys(bsp, plo, phi):
    """Render polygons from packed static props (gamelump 'sprp') inside the play area."""
    ofs, ln, _ = bsp.lumps[40]
    if ln <= 0:
        return []
    z = zipfile.ZipFile(io.BytesIO(bsp.data[ofs:ofs + ln]))
    index = {n.lower(): n for n in z.namelist()}
    gamelump = _lump(bsp.data, bsp.lumps, 35)
    if len(gamelump) < 4:
        return []
    n_gl = struct.unpack_from("<i", gamelump, 0)[0]
    cache, polys = {}, []
    for i in range(n_gl):
        gid, _flags, ver, gofs, gln = struct.unpack_from("<4sHHii", gamelump, 4 + i * 16)
        if gid != b"prps":
            continue
        blob = bsp.data[gofs:gofs + gln]
        ndict = struct.unpack_from("<i", blob, 0)[0]
        pos = 4
        names = [blob[pos + k * 128:pos + (k + 1) * 128].split(b"\x00", 1)[0].decode("latin1", "replace")
                 for k in range(ndict)]
        pos += ndict * 128
        nleaf = struct.unpack_from("<i", blob, pos)[0]
        pos += 4 + nleaf * 2
        nprop = struct.unpack_from("<i", blob, pos)[0]
        pos += 4
        if nprop <= 0:
            continue
        rec_size = (len(blob) - pos) // nprop
        for p in range(nprop):
            r = blob[pos + p * rec_size:pos + (p + 1) * rec_size]
            ox, oy, oz, ax, ay, az, ptype = struct.unpack_from("<6fH", r, 0)
            origin = np.array([ox, oy, oz], np.float64)
            if not _inside(origin, plo, phi, 64.0):
                continue
            key = names[ptype].lower().replace("\\", "/")
            if key not in cache:
                base = key[:-4] if key.endswith(".mdl") else key
                m_f, v_f, x_f = index.get(base + ".mdl"), index.get(base + ".vvd"), index.get(base + ".dx90.vtx")
                cache[key] = _mdl_tris(z.read(m_f), z.read(v_f), z.read(x_f)) if (m_f and v_f and x_f) else []
            if not cache[key]:
                continue
            scale = struct.unpack_from("<f", r, 76)[0] if ver >= 11 and rec_size >= 80 else 1.0
            if not np.isfinite(scale) or scale <= 0.0:
                scale = 1.0
            rot = _angles_matrix(ax, ay, az)
            for tpos, tuv, mat in cache[key]:
                wpos = (tpos * scale) @ rot.T + origin
                for tri_p, tri_uv in zip(wpos, tuv):
                    polys.append((tri_p, tri_uv, mat))
    if polys:
        print(f"static props: {len(polys)} triangles")
    return polys


def _ent_origin(ent):
    if not ent or "origin" not in ent:
        return np.zeros(3)
    return np.array([float(v) for v in ent["origin"].split()[:3]])


class Bsp:
    """Raw lumps plus the tree helpers the importer needs."""

    def __init__(self, path):
        data = open(path, "rb").read()
        if data[:4] != b"VBSP":
            raise SystemExit(f"{path} is not a BSP")
        self.data = data
        self.lumps = lumps = _lumps(data)
        for i, (ofs, ln, _v) in enumerate(lumps):
            if ln >= 4 and data[ofs:ofs + 4] == b"LZMA":
                raise SystemExit(f"lump {i} is LZMA-compressed; decompress the BSP first")
        self.planes = np.frombuffer(_lump(data, lumps, 1), np.float32).reshape(-1, 5).astype(np.float64)
        self.nodes = _lump(data, lumps, 5)
        self.leaves = _lump(data, lumps, 10)
        self.leaf_size = 56 if lumps[10][2] == 0 else 32
        self.leafbrush = np.frombuffer(_lump(data, lumps, 17), np.uint16)
        self.strings = _strings(_lump(data, lumps, 43), np.frombuffer(_lump(data, lumps, 44), np.int32))
        self.texinfo = _lump(data, lumps, 6)
        self.texdata = _lump(data, lumps, 2)
        self.ents = _entities(_lump(data, lumps, 0).decode("latin1", "replace"))
        raw = _lump(data, lumps, 14)
        self.models = [struct.unpack_from("<9f3i", raw, i * 48) for i in range(len(raw) // 48)]
        self.model_ent = {}
        for e in self.ents:
            m = e.get("model", "")
            if m.startswith("*"):
                self.model_ent[int(m[1:])] = e

    def tex_name(self, ti):
        if ti < 0:
            return ""
        di = struct.unpack_from("<i", self.texinfo, ti * 72 + 68)[0]
        nid = struct.unpack_from("<i", self.texdata, di * 32 + 12)[0]
        return self.strings[nid] if 0 <= nid < len(self.strings) else "UNKNOWN"

    def leaf(self, i):
        o = i * self.leaf_size
        area = struct.unpack_from("<h", self.leaves, o + 6)[0] & 0x1FF
        mins = np.array(struct.unpack_from("<3h", self.leaves, o + 8), np.float64)
        maxs = np.array(struct.unpack_from("<3h", self.leaves, o + 14), np.float64)
        fb, nb = struct.unpack_from("<HH", self.leaves, o + 24)
        return area, mins, maxs, fb, nb

    def leaf_at(self, p):
        n = 0
        while n >= 0:
            pl, c0, c1 = struct.unpack_from("<3i", self.nodes, n * 32)
            d = float(self.planes[pl][:3] @ p - self.planes[pl][3])
            n = c0 if d >= 0 else c1
        return -1 - n

    def area_bounds(self, areas):
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for i in range(len(self.leaves) // self.leaf_size):
            a, mn, mx, _fb, _nb = self.leaf(i)
            if a in areas:
                lo, hi = np.minimum(lo, mn), np.maximum(hi, mx)
        return lo, hi

    def model_brushes(self, m):
        """Brush indices under model m's BSP subtree."""
        stack, out = [self.models[m][9]], set()
        while stack:
            n = stack.pop()
            if n < 0:
                _a, _mn, _mx, fb, nb = self.leaf(-1 - n)
                out.update(int(b) for b in self.leafbrush[fb:fb + nb])
            else:
                _pl, c0, c1 = struct.unpack_from("<3i", self.nodes, n * 32)
                stack += [c0, c1]
        return sorted(out)


def _split_slanted(normals, dists, lo, hi):
    """Boxes covering a convex brush whose sides are not all axial: columns of COL_CELL units.

    Each column spans the union of the brush's z ranges at five points in the cell, snapped
    outward to COL_ZQ, so a thin diagonal wall stays closed. Columns with the same z range are
    merged along x, then along y.
    """
    span = hi[:2] - lo[:2]
    nx, ny = (np.maximum(1, np.ceil(span / COL_CELL))).astype(int)
    cx, cy = span[0] / nx, span[1] / ny
    offs = np.array([(0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)])
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    px = lo[0] + (ii[..., None] + offs[:, 0]) * cx          # (nx, ny, 5)
    py = lo[1] + (jj[..., None] + offs[:, 1]) * cy
    zlo = np.full(px.shape, lo[2])
    zhi = np.full(px.shape, hi[2])
    ok = np.ones(px.shape, bool)
    for (a, b, c), d in zip(normals, dists):
        rest = d - a * px - b * py
        if c > 1e-6:
            zhi = np.minimum(zhi, rest / c)
        elif c < -1e-6:
            zlo = np.maximum(zlo, rest / c)
        else:
            ok &= rest >= -0.1
    ok &= zhi - zlo > 0.5
    lo_c = np.where(ok, zlo, np.inf).min(-1)
    hi_c = np.where(ok, zhi, -np.inf).max(-1)
    filled = np.isfinite(lo_c)
    lo_c = np.where(filled, np.floor(lo_c / COL_ZQ) * COL_ZQ, 0)
    hi_c = np.where(filled, np.ceil(hi_c / COL_ZQ) * COL_ZQ, 0)

    def runs(order):
        boxes = []
        open_ = {}
        outer, inner = (ny, nx) if order == "x" else (nx, ny)
        for o in range(outer):
            segs = []
            k = 0
            while k < inner:
                i, j = (k, o) if order == "x" else (o, k)
                if not filled[i, j]:
                    k += 1
                    continue
                key = (lo_c[i, j], hi_c[i, j])
                k1 = k
                while k1 + 1 < inner:
                    i2, j2 = (k1 + 1, o) if order == "x" else (o, k1 + 1)
                    if not filled[i2, j2] or (lo_c[i2, j2], hi_c[i2, j2]) != key:
                        break
                    k1 += 1
                segs.append((k, k1, key))
                k = k1 + 1
            nxt = {}
            for s in segs:
                if s in open_:
                    nxt[s] = open_.pop(s)
                else:
                    nxt[s] = o
            for (k0, k1, key), o0 in open_.items():
                boxes.append((order, k0, k1, o0, o - 1, key))
            open_ = nxt
        for (k0, k1, key), o0 in open_.items():
            boxes.append((order, k0, k1, o0, outer - 1, key))
        return boxes

    best = min((runs("x"), runs("y")), key=len)
    out = []
    for order, k0, k1, o0, o1, (z0, z1) in best:
        (i0, i1), (j0, j1) = ((k0, k1), (o0, o1)) if order == "x" else ((o0, o1), (k0, k1))
        out.append((np.array([lo[0] + i0 * cx, lo[1] + j0 * cy, max(z0, lo[2])]),
                    np.array([lo[0] + (i1 + 1) * cx, lo[1] + (j1 + 1) * cy, min(z1, hi[2])])))
    return out


def _brush_geometry(bsp, b, rot, shift):
    """(boxes, names) for brush b moved by rot/shift, or None when it is not player-solid."""
    brushes = np.frombuffer(_lump(bsp.data, bsp.lumps, 18), np.int32).reshape(-1, 3)
    first, count, contents = (int(v) for v in brushes[b])
    if contents & 0x20 or not (contents & (0x1 | 0x2 | 0x8 | 0x10000)) or count < 4:
        return None
    raw = _lump(bsp.data, bsp.lumps, 19)
    sides = np.ndarray(len(raw) // 8, dtype=[("plane", "<u2"), ("tex", "<i2"), ("disp", "<i2"),
                                             ("bevel", "u1"), ("thin", "u1")], buffer=raw)[first:first + count]
    names = [bsp.tex_name(int(s["tex"])).upper() for s in sides if int(s["tex"]) >= 0]
    if names and all("TRIGGER" in t or "SKYBOX" in t or t.endswith("HINT") or t.endswith("SKIP")
                     or t.endswith("ORIGIN") for t in names):
        return None
    pl = bsp.planes[sides["plane"].astype(int)]
    normals = pl[:, :3] @ rot.T
    dists = pl[:, 3] + normals @ shift
    bevel = sides["bevel"].astype(bool)
    verts = []
    n = len(pl)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                m = normals[[i, j, k]]
                if abs(np.linalg.det(m)) < 1e-6:
                    continue
                x = np.linalg.solve(m, dists[[i, j, k]])
                if np.all(normals @ x <= dists + 0.5):
                    verts.append(x)
    if len(verts) < 4:
        return None
    v = np.array(verts)
    lo, hi = v.min(0), v.max(0)
    if np.any(hi - lo < 1.0):
        return None
    face = normals[~bevel] if (~bevel).any() else normals
    if np.all(np.abs(face).max(1) > 0.999):
        return [(lo, hi)], names
    return _split_slanted(normals, dists, lo, hi), names


def _face_verts(edges, surfedges, first, count):
    idx = []
    for k in range(count):
        se = int(surfedges[first + k])
        idx.append(int(edges[se][0] if se >= 0 else edges[-se][1]))
    return idx


def _orient(pos, normal):
    """Wind the polygon so its normal matches the BSP plane normal."""
    from import_dust2 import newell
    return pos[::-1] if float(np.dot(newell(pos), normal)) < 0 else pos


def _uv(pos, s, t, w, h):
    u = (pos[:, 0] * s[0] + pos[:, 1] * s[1] + pos[:, 2] * s[2] + s[3]) / max(w, 1)
    v = (pos[:, 0] * t[0] + pos[:, 1] * t[1] + pos[:, 2] * t[2] + t[3]) / max(h, 1)
    return np.stack([u, 1.0 - v], 1)


def face_polys(bsp):
    """-> list of (positions (k,3), texinfo, material, face index) for every face (displacements tessellated)."""
    data, lumps = bsp.data, bsp.lumps
    verts = np.frombuffer(_lump(data, lumps, 3), np.float32).reshape(-1, 3).astype(np.float64)
    edges = np.frombuffer(_lump(data, lumps, 12), np.uint16).reshape(-1, 2)
    surf = np.frombuffer(_lump(data, lumps, 13), np.int32)
    faces = _lump(data, lumps, 7)
    disp_blob = _lump(data, lumps, 26)
    dverts = np.frombuffer(_lump(data, lumps, 33), np.float32).reshape(-1, 5)
    out = []
    for i in range(len(faces) // 56):
        plane, _side, _on, first, num, tinfo, disp = struct.unpack_from("<HBBIhhh", faces, i * 56)
        if num < 3 or tinfo < 0:
            continue
        name = bsp.tex_name(tinfo)
        # A face's own plane points out of its front; `side` only relates it to the node plane.
        normal = bsp.planes[plane][:3]
        pos = verts[_face_verts(edges, surf, first, num)]
        if disp < 0:
            out.append((_orient(pos, normal), tinfo, name, i))
            continue
        if num != 4:
            continue
        info = disp_blob[disp * 176:(disp + 1) * 176]
        start = np.array(struct.unpack_from("<3f", info))
        vstart, _tri, power = struct.unpack_from("<iii", info, 12)
        n = 1 << power
        corners = _orient(pos, normal)
        i0 = int(np.argmin(np.linalg.norm(corners - start, axis=1)))
        p = [corners[(i0 + k) % 4] for k in range(4)]
        grid = np.empty((n + 1, n + 1, 3))
        for y in range(n + 1):
            ty = y / n
            for x in range(n + 1):
                tx = x / n
                flat = (1 - tx) * (1 - ty) * p[0] + tx * (1 - ty) * p[1] + tx * ty * p[2] + (1 - tx) * ty * p[3]
                dv = dverts[vstart + y * (n + 1) + x]
                grid[y, x] = flat + dv[:3] * dv[3]
        for y in range(n):
            for x in range(n):
                quad = np.array([grid[y, x], grid[y, x + 1], grid[y + 1, x + 1], grid[y + 1, x]])
                out.append((quad, tinfo, name, i))
    return out


def _spawns(ents, kind):
    out = []
    for e in ents:
        if e.get("classname") != kind:
            continue
        x, y, z = (float(v) for v in e.get("origin", "0 0 0").split())
        ang = e.get("angles", "0 0 0").split()
        yaw = float(ang[1]) if len(ang) > 1 else 0.0
        out.append((x, y, z, yaw))
    return out


def _inside(p, lo, hi, pad=0.0):
    return bool(np.all(p >= lo - pad) and np.all(p <= hi + pad))


# Materials that are adverts / author watermarks, not map art (zm_lostcities: the mapper's Bilibili
# profile card and donation ranking). Their faces take the material of the wall behind them.
AD_MATERIALS = ("LOSTCITIES/SHJZTCAIDAN",)


def _is_ad(name):
    u = name.upper()
    return any(u.startswith(a) for a in AD_MATERIALS)


def _reskin_ads(polys, texinfo, texdata):
    """Re-texture advert faces with the closest parallel face just behind them (the wall they hang
    on), continuing that face's texture mapping; an advert with no wall behind it is dropped."""
    from import_dust2 import newell
    ads = [k for k, p in enumerate(polys) if _is_ad(p[2])]
    if not ads:
        return polys
    info = []
    for pos, _uv0, mat, ti in polys:
        n = newell(pos)
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        info.append((n, float(n @ pos[0]), pos.min(0), pos.max(0)))
    out = list(polys)
    drop = set()
    for k in ads:
        pos = polys[k][0]
        n, _d, _lo, _hi = info[k]
        c = pos.mean(0)
        best, best_key = None, None
        for j, (n2, d2, lo2, hi2) in enumerate(info):
            if j == k or _is_ad(polys[j][2]) or float(n @ n2) < 0.99:
                continue
            behind = float(n2 @ c) - d2  # how far the advert stands in front of that plane
            if not 0.0 <= behind <= 16.0:
                continue
            gap = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo2 - c, c - hi2))))
            if gap > 96.0:
                continue
            key = (round(behind, 1), gap)
            if best_key is None or key < best_key:
                best, best_key = j, key
        if best is None:
            drop.add(k)
            continue
        ti = polys[best][3]
        s = struct.unpack_from("<8f", texinfo, ti * 72)
        di = struct.unpack_from("<i", texinfo, ti * 72 + 68)[0]
        _nid, tw, th = struct.unpack_from("<iii", texdata, di * 32 + 12)
        out[k] = (pos, _uv(pos, s[:4], s[4:8], tw, th), polys[best][2], ti)
    print(f"adverts: {len(ads)} faces re-skinned, {len(drop)} dropped")
    return [p for k, p in enumerate(out) if k not in drop]


def _is_water(name):
    u = name.upper()
    return "WATER" in u and (u.startswith("LIQUIDS/") or u.startswith("DEV/DEV_WATER") or "/WATER" in u)


def main():
    src = sys.argv[1]
    name = sys.argv[2]
    credit = sys.argv[3] if len(sys.argv) > 3 else src
    bsp = Bsp(src)
    ents = bsp.ents
    ct, t = _spawns(ents, "info_player_counterterrorist"), _spawns(ents, "info_player_terrorist")
    spawn_pts = np.array([s[:3] for s in ct + t], np.float64)
    play_areas = {bsp.leaf(bsp.leaf_at(p + [0, 0, 8]))[0] for p in spawn_pts} - {0}
    plo, phi = bsp.area_bounds(play_areas)
    sky = next((e for e in ents if e.get("classname") == "sky_camera"), None)
    sky_lo = sky_hi = None
    if sky is not None:
        so = _ent_origin(sky)
        sa = bsp.leaf(bsp.leaf_at(so))[0]
        if sa and sa not in play_areas:
            sky_lo, sky_hi = bsp.area_bounds({sa})
            sky_scale = float(sky.get("scale", 16))
    print(f"{name}: play areas {sorted(play_areas)} {plo.tolist()} .. {phi.tolist()}"
          + (f", skybox {sky_lo.tolist()} .. {sky_hi.tolist()}" if sky_lo is not None else ""))

    # -------------------------------------------------- faces
    face_model = {}
    for m, mdl in enumerate(bsp.models):
        ff, nf = mdl[10], mdl[11]
        for f in range(ff, ff + nf):
            face_model[f] = m
    polys, water_z = [], []
    sky_polys = 0
    # Sky-shell faces of the play area. Source draws them as the sky, hiding everything past the map's
    # edge; with no 3D skybox to show behind them they are exported as the renderer's sky mask.
    shell = []
    all_faces = face_polys(bsp)
    # brush entities carrying an advert are free-standing billboards (zm_lostcities *15 floats in
    # mid-air): drop the whole entity, there is no wall behind it to show
    ad_models = {face_model[fi] for _p, _t, mat, fi in all_faces if face_model.get(fi, 0) and _is_ad(mat)}
    if ad_models:
        print(f"adverts: dropped brush entities {sorted(ad_models)}")
    for pos, tinfo, mat, fi in all_faces:
        m = face_model.get(fi, 0)
        s = struct.unpack_from("<8f", bsp.texinfo, tinfo * 72)
        di = struct.unpack_from("<i", bsp.texinfo, tinfo * 72 + 68)[0]
        _nid, tw, th = struct.unpack_from("<iii", bsp.texdata, di * 32 + 12)
        uv = _uv(pos, s[:4], s[4:8], tw, th)
        if m:
            if m in ad_models:
                continue
            ent = bsp.model_ent.get(m)
            if ent is None or ent.get("classname") not in RENDER_ENTS or ent.get("StartDisabled") == "1":
                continue
            pos = pos @ _yaw_matrix(ent).T + _ent_origin(ent)
        c = pos.mean(0)
        upper = mat.upper()
        if upper in SKIP_RENDER:
            if (not m and sky_lo is None and upper in ("TOOLS/TOOLSSKYBOX", "TOOLS/TOOLSSKYBOX2D")
                    and _inside(c, plo, phi, 16.0)):
                shell.append(pos)
            continue
        if _inside(c, plo, phi, 16.0):
            if _is_water(mat):
                if upper.startswith("LIQUIDS/") and abs(newell_z(pos)) > 0.9:
                    water_z.append(float(c[2]))
                continue
        elif sky_lo is not None and _inside(c, sky_lo, sky_hi, 16.0):
            if _is_water(mat):
                continue
            pos = (pos - so) * sky_scale
            sky_polys += 1
        else:
            continue
        polys.append((pos, uv, mat, tinfo))
    polys = [p[:3] for p in _reskin_ads(polys, bsp.texinfo, bsp.texdata)]
    if water_z:
        floor_z = float(np.median(water_z))
    else:
        floor_z = min(float(p[0][:, 2].min()) for p in polys if _inside(p[0].mean(0), plo, phi, 16.0))
    polys += _static_prop_polys(bsp, plo, phi)
    origin = np.array([(plo[0] + phi[0]) / 2, (plo[1] + phi[1]) / 2, floor_z])
    water = 0.0 if water_z else None
    print(f"render faces {len(polys)} (skybox {sky_polys}), origin {origin.tolist()}, water {water}")

    # -------------------------------------------------- collision
    boxes = []
    solid_models = [(0, None)] + [(m, e) for m, e in bsp.model_ent.items()
                                  if e.get("classname") in SOLID_ENTS and e.get("StartDisabled") != "1"
                                  and e.get("Solidity", "0") != "1"]
    clo, chi = plo - 64.0, phi + 64.0
    split = 0
    for m, ent in solid_models:
        rot, shift = _yaw_matrix(ent), _ent_origin(ent)
        for b in bsp.model_brushes(m):
            got = _brush_geometry(bsp, b, rot, shift)
            if got is None:
                continue
            parts, _names = got
            split += len(parts) > 1
            for lo, hi in parts:
                lo, hi = np.maximum(lo, clo), np.minimum(hi, chi)
                if np.all(hi - lo > 0.5):
                    boxes.append((lo, hi))
    mn = np.array([b[0] for b in boxes])
    mx = np.array([b[1] for b in boxes])
    print(f"collision: {len(mn)} boxes ({split} slanted brushes split)")
    out = os.path.join(ROOT, "fpsrl", "maps", "data", name)
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(out):
        if fn.endswith(".png"):
            os.remove(os.path.join(out, fn))
    np.savez_compressed(os.path.join(out, "collision.npz"),
                        mn=((mn - origin) * SCALE).astype(np.float32),
                        mx=((mx - origin) * SCALE).astype(np.float32))

    groups = {}
    for pos, uv, mat in polys:
        g = groups.setdefault(mat, {"pos": [], "uv": [], "tri": [], "n": 0})
        base = g["n"]
        g["pos"].append(pos)
        g["uv"].append(uv)
        g["tri"] += [(base, base + k, base + k + 1) for k in range(1, len(pos) - 1)]
        g["n"] += len(pos)
    arrays, mats = {}, []
    for k, (mat, g) in enumerate(sorted(groups.items())):
        mats.append(mat)
        arrays[f"pos{k}"] = ((np.concatenate(g["pos"]) - origin) * SCALE).astype(np.float32)
        arrays[f"uv{k}"] = np.concatenate(g["uv"]).astype(np.float32)
        arrays[f"tri{k}"] = np.array(g["tri"], dtype=np.int32)
    textures = _pak_textures(bsp.data, bsp.lumps, mats)
    tex_names = []
    from PIL import Image
    for mat in mats:
        fn = mat.replace("/", "_").replace("\\", "_") + ".png"
        png = textures.get(mat.lower())
        if png is None:
            Image.new("RGB", (8, 8), _flat_color(mat)).save(os.path.join(out, fn), "PNG")
        else:
            open(os.path.join(out, fn), "wb").write(png)
        tex_names.append(fn)
    arrays["materials"] = np.array(mats)
    arrays["names"] = np.array(mats)
    arrays["textures"] = np.array(tex_names)
    if shell:
        tri, n = [], 0
        for pos in shell:
            tri += [(n, n + k, n + k + 1) for k in range(1, len(pos) - 1)]
            n += len(pos)
        arrays["sky_pos"] = ((np.concatenate(shell) - origin) * SCALE).astype(np.float32)
        arrays["sky_tri"] = np.array(tri, dtype=np.int32)
        print(f"sky shell: {len(shell)} faces")
    np.savez_compressed(os.path.join(out, "mesh.npz"), **arrays)

    def conv(rows):
        if not rows:
            return np.zeros((0, 4), np.float32)
        a = np.array(rows, np.float64)
        a[:, :3] = (a[:, :3] - origin) * SCALE
        return a.astype(np.float32)

    np.savez_compressed(os.path.join(out, "spawns.npz"), ct=conv(ct), t=conv(t),
                        water=np.array([] if water is None else [water], np.float32))
    credits = os.path.join(out, "CREDITS.txt")
    if len(sys.argv) > 3 or not os.path.exists(credits):
        with open(credits, "w", encoding="utf-8") as f:
            f.write(credit + "\nImported with tools/import_bsp.py. Collision rebuilt as boxes; "
                    "the render mesh is the BSP faces.\n")
    print("wrote", out)


def newell_z(pos):
    from import_dust2 import newell
    return float(newell(pos)[2])


if __name__ == "__main__":
    main()

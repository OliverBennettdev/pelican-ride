#!/usr/bin/env python3
"""Build a textured 2.5D GLB relief from the chroma-keyed pelican image."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SRC = ASSETS / "pelican-source.jpg"
OUT_GLB = ASSETS / "pelican.glb"
OUT_ALPHA = ASSETS / "pelican-alpha.png"

GRID = 192
DEPTH_SCALE = 0.55
XY_SCALE = 1.8


def chroma_rgba(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    corners = np.concatenate(
        [
            rgb[:12, :12].reshape(-1, 3),
            rgb[:12, -12:].reshape(-1, 3),
            rgb[-12:, :12].reshape(-1, 3),
            rgb[-12:, -12:].reshape(-1, 3),
        ]
    )
    key = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - key, axis=2)
    alpha = np.clip((dist - 18.0) / 22.0, 0.0, 1.0)
    alpha = (alpha * 255.0).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(radius=1.2))
    alpha = np.asarray(alpha_img)
    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    return rgba


def chamfer_edt(mask: np.ndarray) -> np.ndarray:
    inf = 1.0e6
    h, w = mask.shape
    d = np.where(mask, inf, 0.0).astype(np.float32)
    for y in range(h):
        row = d[y]
        prev = d[y - 1] if y else None
        for x in range(w):
            v = row[x]
            if v == 0.0:
                continue
            if x:
                v = min(v, row[x - 1] + 1.0)
            if prev is not None:
                v = min(v, prev[x] + 1.0)
                if x:
                    v = min(v, prev[x - 1] + 1.41421356)
                if x + 1 < w:
                    v = min(v, prev[x + 1] + 1.41421356)
            row[x] = v
    for y in range(h - 1, -1, -1):
        row = d[y]
        nxt = d[y + 1] if y + 1 < h else None
        for x in range(w - 1, -1, -1):
            v = row[x]
            if x + 1 < w:
                v = min(v, row[x + 1] + 1.0)
            if nxt is not None:
                v = min(v, nxt[x] + 1.0)
                if x + 1 < w:
                    v = min(v, nxt[x + 1] + 1.41421356)
                if x:
                    v = min(v, nxt[x - 1] + 1.41421356)
            row[x] = v
    d[d > inf * 0.5] = 0.0
    return d


def build_mesh(rgba: np.ndarray):
    img = Image.fromarray(rgba, "RGBA").resize((GRID, GRID), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    alpha = arr[:, :, 3].astype(np.float32) / 255.0
    mask = alpha > 0.18
    dist = chamfer_edt(mask)
    if dist.max() > 0:
        dist = dist / dist.max()
    depth = np.power(np.clip(dist, 0, 1), 0.72) * DEPTH_SCALE

    index_of = np.full((GRID, GRID), -1, dtype=np.int32)
    positions = []
    uvs = []
    for y in range(GRID):
        v = y / (GRID - 1)
        for x in range(GRID):
            if not mask[y, x]:
                continue
            u = x / (GRID - 1)
            a = alpha[y, x]
            px = (u - 0.5) * XY_SCALE
            py = (0.5 - v) * XY_SCALE
            pz = depth[y, x] * a
            index_of[y, x] = len(positions)
            positions.append((px, py, pz))
            uvs.append((u, 1.0 - v))

    faces = []
    for y in range(GRID - 1):
        for x in range(GRID - 1):
            i00 = int(index_of[y, x])
            i10 = int(index_of[y, x + 1])
            i01 = int(index_of[y + 1, x])
            i11 = int(index_of[y + 1, x + 1])
            if i00 >= 0 and i10 >= 0 and i01 >= 0:
                faces.append((i00, i01, i10))
            if i10 >= 0 and i01 >= 0 and i11 >= 0:
                faces.append((i10, i01, i11))

    pos = np.asarray(positions, dtype=np.float32)
    uv = np.asarray(uvs, dtype=np.float32)
    idx = np.asarray(faces, dtype=np.uint32)
    normals = np.zeros_like(pos)
    p0 = pos[idx[:, 0]]
    p1 = pos[idx[:, 1]]
    p2 = pos[idx[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    for i in range(3):
        np.add.at(normals, idx[:, i], n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens[lens < 1e-8] = 1.0
    normals = (normals / lens).astype(np.float32)
    return pos, normals, uv, idx, img


def pad4(data: bytes, fill: bytes) -> bytes:
    extra = (4 - (len(data) % 4)) % 4
    return data + fill * extra


def write_glb(path: Path, pos, normals, uv, idx, texture: Image.Image) -> None:
    png_buf = __import__("io").BytesIO()
    texture.save(png_buf, format="PNG", optimize=True)
    png = png_buf.getvalue()

    pos_b = pos.tobytes()
    nrm_b = normals.tobytes()
    uv_b = uv.tobytes()
    idx_b = idx.astype(np.uint32).tobytes()

    views = []
    offset = 0

    def add_view(blob: bytes, target: int) -> int:
        nonlocal offset
        extra = (4 - (offset % 4)) % 4
        offset += extra
        start = offset
        views.append(
            {
                "buffer": 0,
                "byteOffset": start,
                "byteLength": len(blob),
                "target": target,
            }
        )
        offset += len(blob)
        return start

    chunks = []

    def emit(blob: bytes, target: int) -> None:
        extra = (4 - (len(b"".join(chunks)) % 4)) % 4
        if extra:
            chunks.append(b"\x00" * extra)
        add_view(blob, target)
        chunks.append(blob)

    emit(pos_b, 34962)
    emit(nrm_b, 34962)
    emit(uv_b, 34962)
    emit(idx_b, 34963)

    extra = (4 - (len(b"".join(chunks)) % 4)) % 4
    if extra:
        chunks.append(b"\x00" * extra)
    png_view = len(views)
    views.append(
        {
            "buffer": 0,
            "byteOffset": len(b"".join(chunks)),
            "byteLength": len(png),
        }
    )
    chunks.append(png)
    bin_blob = pad4(b"".join(chunks), b"\x00")

    mins = pos.min(axis=0).tolist()
    maxs = pos.max(axis=0).tolist()
    gltf = {
        "asset": {"version": "2.0", "generator": "pelican-relief"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "PelicanRelief"}],
        "meshes": [
            {
                "name": "PelicanRelief",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "PelicanPaint",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.05,
                    "roughnessFactor": 0.72,
                },
                "doubleSided": True,
            }
        ],
        "textures": [{"source": 0}],
        "images": [{"bufferView": png_view, "mimeType": "image/png"}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": int(pos.shape[0]),
                "type": "VEC3",
                "min": mins,
                "max": maxs,
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": int(uv.shape[0]),
                "type": "VEC2",
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": int(idx.size),
                "type": "SCALAR",
            },
        ],
        "bufferViews": views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    json_blob = pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    header = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_blob) + 8 + len(bin_blob))
    json_chunk = struct.pack("<I4s", len(json_blob), b"JSON") + json_blob
    bin_chunk = struct.pack("<I4s", len(bin_blob), b"BIN\x00") + bin_blob
    path.write_bytes(header + json_chunk + bin_chunk)


def main() -> None:
    src = Image.open(SRC)
    rgba = chroma_rgba(src)
    Image.fromarray(rgba, "RGBA").save(OUT_ALPHA)
    pos, normals, uv, idx, tex = build_mesh(rgba)
    write_glb(OUT_GLB, pos, normals, uv, idx, tex)
    print(
        json.dumps(
            {
                "glb": str(OUT_GLB),
                "bytes": OUT_GLB.stat().st_size,
                "verts": int(pos.shape[0]),
                "tris": int(idx.shape[0]),
                "alpha": str(OUT_ALPHA),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

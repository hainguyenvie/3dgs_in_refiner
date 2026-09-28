"""M3 test-time: apply the pose-interpolated per-view colour affine (learned on train views, saved as
color_affine.json) to the test renders -> test/ours_30000_m3/renders (+ gt symlink) so metrics.py scores both.
Source-free: only 12 numbers per train view and camera centres are used.

    python m3_apply_test.py <model_dir> <scene_dir> [k=3]
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

model, scene = Path(sys.argv[1]), Path(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
aff = json.load(open(model / "color_affine.json"))
names_tr = list(aff.keys()); C_tr = np.array([aff[n]["center"] for n in names_tr])


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


# test camera centres from COLMAP (sorted names, every 8th = test), same order as render indices
centers = {}
with open(scene / "sparse" / "0" / "images.bin", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    for _ in range(n):
        f.read(4); q = struct.unpack("<4d", f.read(32)); t = np.array(struct.unpack("<3d", f.read(24))); f.read(4)
        s = b""
        while True:
            c = f.read(1)
            if c == b"\x00":
                break
            s += c
        k = struct.unpack("<Q", f.read(8))[0]; f.read(24 * k)
        centers[s.decode()] = (-qvec2rot(q).T @ t)
names = sorted(centers); test_names = [nm for i, nm in enumerate(names) if i % 8 == 0]
src = model / "test" / "ours_30000"; dst = model / "test" / "ours_30000_m3"
(dst / "renders").mkdir(parents=True, exist_ok=True)
if not (dst / "gt").exists():
    os.symlink((src / "gt").resolve(), dst / "gt")
for i, nm in enumerate(test_names):
    p = src / "renders" / f"{i:05d}.png"
    if not p.exists():
        continue
    c = centers[nm]; d = np.linalg.norm(C_tr - c, axis=1); idx = np.argsort(d)[:K]; w = 1 / (d[idx] + 1e-6); w = w / w.sum()
    A = sum(wi * np.array(aff[names_tr[j]]["A"]) for wi, j in zip(w, idx)); b = sum(wi * np.array(aff[names_tr[j]]["b"]) for wi, j in zip(w, idx))
    img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
    out = np.clip(img @ A.T + b, 0, 1)
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(dst / "renders" / f"{i:05d}.png")
print(f"m3: wrote {len(test_names)} test renders with k={K} pose-interpolated affine -> {dst}")

"""E10 — pure resampling floor on the synthetic world. Render the test views from the same model on a pixel grid
shifted by (dx, dy) px (principal point moved in cameras.bin), resample back to the original grid with the same
bilinear interpolation the warp uses, compare with the unshifted render. This is the error a *perfect* warp
would still make on this content just by resampling. Also reports the spectral share to compare with E8.

Usage (two steps, run from src/irgs with the IBGS venv):
  python3 scripts/analysis/e10_resampling_floor.py build  bicycle 0.5 0.5     -> data/e10/bicycle_shift_0.5_0.5 (+ model dir)
  <render test views of outputs/e10/bicycle_shift_0.5_0.5 with render.py -r 1 --skip_train>
  python3 scripts/analysis/e10_resampling_floor.py eval   bicycle 0.5 0.5
"""
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build(scene, dx, dy):
    src = ROOT / "data" / "e8" / f"{scene}_ibgs_syn"                  # rescaled cameras (eval resolution), synthetic images
    dst = ROOT / "data" / "e10" / f"{scene}_shift_{dx}_{dy}"
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in ("images.bin", "points3D.bin"):
        shutil.copy(src / "sparse" / "0" / f, dst / "sparse" / "0" / f)
    if not (dst / "images").exists():
        os.symlink((src / "images").resolve(), dst / "images")
    with open(src / "sparse" / "0" / "cameras.bin", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; cams = []
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24)); npar = {0: 3, 1: 4}[model]
            cams.append((cid, model, w, h, list(struct.unpack("<" + "d" * npar, f.read(8 * npar)))))
    with open(dst / "sparse" / "0" / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", n))
        for cid, model, w, h, p in cams:
            if model == 1: p[2] += dx; p[3] += dy
            else: p[1] += dx; p[2] += dy
            f.write(struct.pack("<iiQQ", cid, model, w, h)); f.write(struct.pack("<" + "d" * len(p), *p))
    m = ROOT / "outputs" / "e10" / f"{scene}_shift_{dx}_{dy}"; m.mkdir(parents=True, exist_ok=True)
    pre = ROOT / "outputs" / "protocolR" / "ibgs" / f"{scene}_pre"
    for f in ("cfg_args", "config.json", "multi_view.json", "multi_view_test.json", "input.ply", "point_cloud", "app_model", "color_aggregate_checkpoint"):
        if (pre / f).exists() and not (m / f).exists():
            os.symlink((pre / f).resolve(), m / f)
    print("built", dst, m)


def evaluate(scene, dx, dy):
    import torch
    import torch.nn.functional as F
    from e1_misalignment import load, dev
    from e4_frequency_color import bands
    sh = ROOT / "outputs" / "e10" / f"{scene}_shift_{dx}_{dy}" / "test" / "ours_30000" / "renders"
    ref = ROOT / "outputs" / "e8" / f"{scene}_ibgs_syn" / "test" / "ours_30000" / "renders"
    mses, bnd, mses_ident = [], [], []
    for nm in sorted(os.listdir(ref)):
        g = load(ref / nm)[None].to(dev); s = load(sh / nm)[None].to(dev)
        _, _, H, W = g.shape
        # shifted render: pixel (u,v) of s shows the point at (u-dx, v-dy) of the original grid, i.e. s(u) = g(u - dx).
        # to recover g on the original grid sample s at (u + dx, v + dy): pure bilinear resampling at a sub-pixel offset.
        yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
        grid = torch.stack([(xx + dx) / (W - 1) * 2 - 1, (yy + dy) / (H - 1) * 2 - 1], -1)[None].float()
        back = F.grid_sample(s, grid, mode="bilinear", padding_mode="border", align_corners=True)
        m = torch.ones_like(g[:, :1]); m[..., :2, :] = 0; m[..., -2:, :] = 0; m[..., :, :2] = 0; m[..., :, -2:] = 0
        e = ((back - g) ** 2).mean(1, keepdim=True)
        mses.append(float((e * m).sum() / m.sum())); bnd.append(bands((back - g) * m))
    mse = float(np.mean(mses)); b = np.mean(bnd, 0)
    r = {"scene": scene, "shift": [dx, dy], "psnr_resample_floor": -10 * np.log10(mse), "band_share_err": (b / b.sum()).round(3).tolist(), "views": len(mses)}
    print(json.dumps(r))
    p = ROOT / "reports" / "e10_resampling_floor.json"
    allr = json.load(open(p)) if p.exists() else {}
    allr[f"{scene}_{dx}_{dy}"] = r; json.dump(allr, open(p, "w"), indent=1)


if __name__ == "__main__":
    cmd, scene, dx, dy = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    (build if cmd == "build" else evaluate)(scene, dx, dy)

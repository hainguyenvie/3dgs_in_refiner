"""B1 — does warping from OVERSAMPLED source images remove the resampling floor (E10) — on REAL photos, no model.
Exact reference: box-downsample(full-res image shifted by an INTEGER number of full-res px) = the target-resolution
image translated by an exact fractional amount. Compare two ways of producing it from source images:
  A  same resolution as the target: interpolate the target-res image at the fractional shift (what IBR warps do today)
  B  4x oversampled source: interpolate a 4x-res image at the (fractional there too) shift, then box-integrate to target
Reports PSNR vs exact and the high-frequency share of the error, per interpolation mode.

    .venv_ibgs/bin/python scripts/analysis/b1_supersampled_resampling.py data/mipnerf360/bicycle [n_images]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(16)


def box(x, k):
    return F.avg_pool2d(x, k) if k > 1 else x


def shift(x, dx, dy, mode):
    """sample x at p + (dx, dy) (pixels of x's own grid)."""
    _, _, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    g = torch.stack([(xx + dx) / (W - 1) * 2 - 1, (yy + dy) / (H - 1) * 2 - 1], -1)[None].to(x.dtype)
    return F.grid_sample(x, g, mode=mode, padding_mode="border", align_corners=True)


def psnr(a, b, m):
    return float(-10 * torch.log10(((a - b) ** 2 * m).sum() / (m.sum() * a.shape[1])))


def hf_share(err):
    e = err.mean(1)[0].numpy(); Fq = np.abs(np.fft.fft2(e)) ** 2
    fy = np.fft.fftfreq(e.shape[0])[:, None]; fx = np.fft.fftfreq(e.shape[1])[None]; r = np.sqrt(fx ** 2 + fy ** 2)
    return float(Fq[r > 0.25].sum() / Fq.sum())


def main():
    scene = Path(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    names = sorted(os.listdir(scene / "images"))[::max(1, len(os.listdir(scene / "images")) // n)][:n]
    T, S = int(os.environ.get("B1_T", 8)), int(os.environ.get("B1_S", 2))   # target = full/T, oversampled source = full/S
    k_full = int(os.environ.get("B1_K", 3))   # exact shift in full-res px (default 3 = 0.375 target px = 1.5 source px)
    out = {m: {"A": [], "B": [], "A_hf": [], "B_hf": []} for m in ("bilinear", "bicubic")}
    for nm in names:
        im = torch.from_numpy(np.asarray(Image.open(scene / "images" / nm).convert("RGB"), dtype=np.float64) / 255.0).permute(2, 0, 1)[None]
        H, W = (im.shape[2] // (T * 8)) * T * 8, (im.shape[3] // (T * 8)) * T * 8; im = im[:, :, :H, :W]
        ref = box(im[:, :, :, k_full:], T)[:, :, :, :-1]                              # exact: content shifted by 0.375 target px
        tgt_src = box(im, T)[:, :, :, :ref.shape[3]]                                  # target-res source image
        ov_src = box(im, S)                                                           # 4x oversampled source
        m = torch.ones_like(ref[:, :1]); m[..., :4, :] = m[..., -4:, :] = m[..., :, :4] = m[..., :, -4:] = 0
        for mode in out:
            A = shift(box(im, T), k_full / T, 0, mode)[:, :, :, :ref.shape[3]]
            B = box(shift(ov_src, k_full / S, 0, mode), T // S)[:, :, :, :ref.shape[3]]
            out[mode]["A"].append(psnr(A, ref, m)); out[mode]["B"].append(psnr(B, ref, m))
            out[mode]["A_hf"].append(hf_share((A - ref) * m)); out[mode]["B_hf"].append(hf_share((B - ref) * m))
        _ = tgt_src
    res = {"scene": scene.name, "target": f"full/{T}", "source_oversampled": f"full/{S}", "shift_target_px": k_full / T, "images": len(names)}
    for mode, d in out.items():
        res[mode] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"{scene.name:9s} {mode:8s} shift {k_full / T:.3f} target px | A same-res {res[mode]['A']:.2f} dB (HF {res[mode]['A_hf']*100:.0f}%) | "
              f"B 4x-oversampled source {res[mode]['B']:.2f} dB (HF {res[mode]['B_hf']*100:.0f}%) | gain {res[mode]['B'] - res[mode]['A']:+.2f}", flush=True)
    p = ROOT / "reports" / "b1_supersampled_resampling.json"; prev = json.load(open(p)) if p.exists() else {}
    prev[f"{scene.name}_T{T}_S{S}_K{k_full}"] = res; json.dump(prev, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()

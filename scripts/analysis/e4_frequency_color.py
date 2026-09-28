"""E4 — what does the IBR residual actually fix? Decompose raw / final error into frequency bands and test
whether a per-view GLOBAL colour affine (3x4, fitted on the view itself = oracle upper bound) explains the
raw error. If indoor gains are mostly low-frequency colour, the warp branch is doing appearance correction
there, not texture transfer. Protocol R IBGS test renders; no training.

    .venv_ibgs/bin/python scripts/analysis/e4_frequency_color.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PR = ROOT / "outputs" / "protocolR" / "ibgs"
SCENES = {"outdoor": ["bicycle", "flowers", "garden", "stump", "treehill"],
          "indoor": ["bonsai", "counter", "kitchen", "room"], "tnt": ["train", "truck"], "db": ["drjohnson", "playroom"]}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMAS = [1.0, 4.0, 16.0]   # band edges: HF (<σ1), MF (σ1..σ4), LF-mid (σ4..σ16), LF (>σ16 incl. DC)


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)


def gauss(x, s):
    k = int(6 * s) | 1
    t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def bands(e):
    """split error image into 4 bands; returns list of MSE per band (sums to total MSE)."""
    out, prev = [], e
    for s in SIGMAS:
        low = gauss(e, s); out.append(float(((prev - low) ** 2).mean())); prev = low
    out.append(float((prev ** 2).mean()))
    return out


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def color_affine(src, gt):
    """oracle per-view 3x4 colour affine mapping src->gt (least squares on all pixels)."""
    X = torch.cat([src[0].reshape(3, -1), torch.ones(1, src[0].numel() // 3, device=dev)], 0).T
    Y = gt[0].reshape(3, -1).T
    A = torch.linalg.lstsq(X, Y).solution
    return (X @ A).T.reshape(1, 3, *src.shape[-2:]).clamp(0, 1)


def main():
    res = {}
    for grp, scenes in SCENES.items():
        for s in scenes:
            d = PR / f"{s}_r1" / "test" / "ours_30000"
            if not (d / "renders_aggregate").is_dir():
                continue
            acc = {"psnr_raw": [], "psnr_fin": [], "psnr_raw_ca": [], "psnr_fin_ca": [],
                   "b_raw": [], "b_fin": [], "b_res": []}
            for nm in sorted(os.listdir(d / "gt")):
                gt, raw, fin = load(d / "gt" / nm), load(d / "renders" / nm), load(d / "renders_aggregate" / nm)
                acc["psnr_raw"].append(psnr(raw, gt)); acc["psnr_fin"].append(psnr(fin, gt))
                acc["psnr_raw_ca"].append(psnr(color_affine(raw, gt), gt)); acc["psnr_fin_ca"].append(psnr(color_affine(fin, gt), gt))
                acc["b_raw"].append(bands(raw - gt)); acc["b_fin"].append(bands(fin - gt)); acc["b_res"].append(bands(fin - raw))
            r = {k: float(np.mean(v)) for k, v in acc.items() if k.startswith("psnr")}
            br, bf, bs = np.mean(acc["b_raw"], 0), np.mean(acc["b_fin"], 0), np.mean(acc["b_res"], 0)
            r["band_share_raw_err"] = (br / br.sum()).round(3).tolist()
            r["band_share_residual"] = (bs / bs.sum()).round(3).tolist()
            r["band_err_reduction"] = (1 - bf / br).round(3).tolist()          # per band: 1 - MSE_fin/MSE_raw
            r["band_abs_reduction_share"] = ((br - bf) / (br.sum() - bf.sum())).round(3).tolist()  # where the gain comes from
            res[s] = {"group": grp, **r}
            print(s, json.dumps(r), flush=True)
    rep = ROOT / "reports"
    json.dump(res, open(rep / "e4_frequency_color.json", "w"), indent=1)
    md = ["# E4 — frequency bands & per-view colour affine (IBGS Protocol R test views)", "",
          "Bands (σ px): HF <1 | MF 1–4 | LFm 4–16 | LF >16 (incl. DC). `share of gain` = which band the raw→final "
          "MSE reduction comes from. `+CA` = oracle per-view 3x4 colour affine fitted on the view itself.", "",
          "| scene | grp | PSNR raw | final | raw+CA | final+CA | raw err share HF/MF/LFm/LF | gain share HF/MF/LFm/LF | "
          "per-band err reduction HF/MF/LFm/LF |", "|---|---|---|---|---|---|---|---|---|"]
    for grp, scenes in SCENES.items():
        for s in scenes:
            if s not in res:
                continue
            r = res[s]; f = lambda v: "/".join(f"{x*100:.0f}" for x in v)
            md.append(f"| {s} | {grp} | {r['psnr_raw']:.2f} | {r['psnr_fin']:.2f} | {r['psnr_raw_ca']:.2f} | {r['psnr_fin_ca']:.2f} | "
                      f"{f(r['band_share_raw_err'])} | {f(r['band_abs_reduction_share'])} | {f(r['band_err_reduction'])} |")
    (rep / "e4_frequency_color.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

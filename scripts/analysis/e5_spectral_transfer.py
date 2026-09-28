"""E5 — spectral transfer T(f) = Re<F(R) F(G)*> / <|F(G)|^2> between render R and GT G (radially averaged over
windowed patches), and the effective Gaussian blur sigma_eff that best fits T(f) = exp(-2 pi^2 sigma^2 f^2).

Under sub-pixel multi-view inconsistency the loss-optimal render is GT blurred by the inconsistency PDF, so
T(f) drops as a Gaussian in f with sigma ~ inconsistency; a capacity limit shows up differently (floor, not
matching COLMAP reprojection sigma). Works on any {gt, renders} folder pair. No training.

    .venv_ibgs/bin/python scripts/analysis/e5_spectral_transfer.py \
        --pairs mcmc:outputs/protocolR/mcmc/bicycle_r1/test/ours_30000 ibgs_raw:... --out reports/e5_x.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P = 256  # patch size


def load_gray(p):
    x = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).to(dev)


def transfer(render_dir, gt_dir, names, stride=P // 2):
    win = torch.hann_window(P, device=dev)[:, None] * torch.hann_window(P, device=dev)[None, :]
    num = torch.zeros(P, P, device=dev); den = torch.zeros(P, P, device=dev); n = 0
    for nm in names:
        base = os.path.splitext(nm)[0]
        rp = [f for f in os.listdir(render_dir) if os.path.splitext(f)[0] == base]
        if not rp:
            continue
        g, r = load_gray(Path(gt_dir) / nm), load_gray(Path(render_dir) / rp[0])
        H, W = g.shape
        for y in range(0, H - P + 1, stride):
            for x in range(0, W - P + 1, stride):
                gp, rp_ = g[y:y + P, x:x + P], r[y:y + P, x:x + P]
                gp, rp_ = (gp - gp.mean()) * win, (rp_ - rp_.mean()) * win
                Fg, Fr = torch.fft.fft2(gp), torch.fft.fft2(rp_)
                num += (Fr * Fg.conj()).real; den += (Fg.abs() ** 2); n += 1
    T = (num / den.clamp_min(1e-12)).cpu().numpy()
    fy = np.fft.fftfreq(P)[:, None]; fx = np.fft.fftfreq(P)[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    edges = np.linspace(0, 0.5, 26)
    prof = [float(T[(fr >= a) & (fr < b)].mean()) for a, b in zip(edges[:-1], edges[1:])]
    fc = 0.5 * (edges[:-1] + edges[1:])
    # fit sigma on 0.05 < f < 0.4 where T is positive
    m = (fc > 0.05) & (fc < 0.4) & (np.array(prof) > 0.02)
    sig = float(np.sqrt(max(np.polyfit(fc[m] ** 2, -np.log(np.array(prof)[m]), 1)[0], 0) / (2 * np.pi ** 2))) if m.sum() > 3 else float("nan")
    return {"f": fc.round(3).tolist(), "T": np.round(prof, 4).tolist(), "sigma_eff_px": sig,
            "T_at_0.25": float(np.interp(0.25, fc, prof)), "T_at_0.4": float(np.interp(0.4, fc, prof)), "patches": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="label:dir  (dir has gt/ and renders/ [or given render subdir via label:dir:sub])")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_views", type=int, default=40)
    a = ap.parse_args()
    out = {}
    for spec in a.pairs:
        parts = spec.split(":"); label, d = parts[0], parts[1]; sub = parts[2] if len(parts) > 2 else "renders"
        names = sorted(os.listdir(Path(d) / "gt"))
        names = names[:: max(1, len(names) // a.max_views)]
        r = transfer(Path(d) / sub, Path(d) / "gt", names)
        out[label] = r
        print(f"{label:28s} sigma_eff={r['sigma_eff_px']:.3f}px  T(0.25)={r['T_at_0.25']:.3f}  T(0.4)={r['T_at_0.4']:.3f}  patches={r['patches']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

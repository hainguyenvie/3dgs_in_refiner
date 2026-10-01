"""Non-learned baselines that the band-wise gate must beat (all from the hybrid dumps, no GT of the evaluated scene used
to pick anything except where marked LOSO):
  fixed-split-k[E]  : Laplacian bands 0..k-1 (fine) from MCMC, bands >= k (coarse) from E (BlendedMVS / Baumberg style);
                      E = IBGS final or IBGS-residual-on-MCMC; k chosen leave-one-scene-out
  support-split-k[E]: same, but only where >= 1 source is valid (MCMC elsewhere)
  mcmc+affine       : MCMC corrected by a 3x4 colour affine fitted to the nearest source's valid warp (IBGS's exposure
                      model, applied to MCMC) — "is the gain just exposure?"
  mcmc+affine-avg   : same, fitted to the mean of all valid warps

    .venv_ibgs/bin/python src/route/baselines_band.py --scenes bonsai counter ... [--L 5]
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lap_pyr(x, L):
    G = [x]
    for _ in range(L - 1):
        G.append(F.avg_pool2d(G[-1], 2, ceil_mode=True))
    up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return [G[l] - up(G[l + 1], G[l]) for l in range(L - 1)] + [G[-1]]


def collapse(P):
    out = P[-1]
    for b in reversed(P[:-1]):
        out = b + F.interpolate(out, size=b.shape[-2:], mode="bilinear", align_corners=False)
    return out


def affine_fit(src, tgt, m):
    """least-squares 3x4 affine mapping src colours to tgt colours on mask m; returns corrected src."""
    X = src[:, m].T; Y = tgt[:, m].T
    if X.shape[0] < 100:
        return src
    X1 = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    A = torch.linalg.lstsq(X1, Y).solution                                               # 4 x 3
    return (torch.cat([src.flatten(1).T, torch.ones_like(src.flatten(1).T[:, :1])], 1) @ A).T.view_as(src)


def psnr(a, b):
    a = (a.clamp(0, 1) * 255 + 0.5).floor() / 255
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", nargs="+", required=True); ap.add_argument("--tag", default="r1"); ap.add_argument("--L", type=int, default=5)
    a = ap.parse_args(); L = a.L
    S = {}
    for s in a.scenes:
        fs = [f for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_{a.tag}", "*.npz"))) if not f.endswith(".geo.npz")]
        if not fs:
            continue
        rows = []
        for f in fs:
            z = np.load(f); t = lambda k: torch.from_numpy(z[k].astype(np.float32)).cuda()
            gt, I = t("gt"), t("mcmc"); r = {"mcmc": psnr(I, gt), "ibgs_final": psnr(t("ibgs_final"), gt), "mcmc_res": psnr(t("mcmc_res"), gt)}
            nv = t("feats")[0]; sup = (nv > 0.5).float()[None, None]
            PI = lap_pyr(I[None], L)
            ms = [sup]
            for _ in range(L - 1):
                ms.append(F.avg_pool2d(ms[-1], 2, ceil_mode=True))
            for E in ("ibgs_final", "mcmc_res"):
                PE = lap_pyr(t(E)[None], L)
                for k in range(1, L + 1):
                    r[f"fixed-split-{k}[{E}]"] = psnr(collapse([PI[l] if l < k else PE[l] for l in range(L)])[0], gt)
                    r[f"support-split-{k}[{E}]"] = psnr(collapse([PI[l] if l < k else PI[l] + ms[l] * (PE[l] - PI[l]) for l in range(L)])[0], gt)
            W = t("warps"); V = torch.from_numpy(z["valid"]).cuda()
            if V.shape[0] > 0 and V[0].any():
                r["mcmc+affine"] = psnr(affine_fit(I, W[0], V[0]), gt)
                nvv = V.float().sum(0); mu = (W * V[:, None]).sum(0) / nvv.clamp_min(1)
                r["mcmc+affine-avg"] = psnr(affine_fit(I, mu, nvv > 0), gt)
            else:
                r["mcmc+affine"] = r["mcmc+affine-avg"] = r["mcmc"]
            rows.append(r)
        S[s] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out = {}
    for s in S:   # LOSO choice of the split level k for each fixed-split family
        others = [o for o in S if o != s] or [s]
        o = {"mcmc": S[s]["mcmc"], "ibgs_final": S[s]["ibgs_final"], "mcmc_res": S[s]["mcmc_res"], "mcmc+affine": S[s]["mcmc+affine"], "mcmc+affine-avg": S[s]["mcmc+affine-avg"]}
        for fam in ("fixed-split", "support-split"):
            for E in ("ibgs_final", "mcmc_res"):
                keys = [f"{fam}-{k}[{E}]" for k in range(1, L + 1)]
                best = max(keys, key=lambda kk: np.mean([S[x][kk] for x in others]))
                o[f"{fam}[{E}] (LOSO k)"] = S[s][best]; o[f"{fam}[{E}] k"] = best
        out[s] = o
        print(f"[{s}] " + " | ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in o.items()), flush=True)
    json.dump({"per_scene": S, "loso": out}, open(os.path.join(ROOT, "outputs", "route", "baselines_band.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

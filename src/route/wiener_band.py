"""Training-free band-wise Wiener fusion of an explicit render I (3DGS-MCMC) and an image-based estimate E (IBGS).

Model: E = T(x + delta) + appearance error; misregistration delta (sigma) makes E's error spectrum grow with frequency,
S_E(w) ~ 2|T(w)|^2 (1 - exp(-w^2 sigma^2 / 2)), while I's error S_I is whatever the Gaussians cannot represent.
Per Laplacian band l and pixel x, with independent errors:
    D_l = local energy of (E - I)_l              ~ S_I + S_E
    V_l = local energy of the spread of the individual source warps in band l   ~ S_E / c   (warps share T, differ by delta)
    w_l = clamp(1 - c V_l / D_l, 0, 1)            (= S_I / (S_I + S_E), the Wiener weight on E)
    F_l = I_l + w_l (E_l - I_l);  w = 0 where no source sees the pixel.
c and the energy window are chosen leave-one-scene-out on a small grid (no GT of the evaluated scene).

    .venv_ibgs/bin/python src/route/wiener_band.py --scenes bonsai counter ... [--E ibgs_final|mcmc_res]
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


def smooth(e, k):
    return F.avg_pool2d(e, 2 * k + 1, stride=1, padding=k, count_include_pad=False) if k > 0 else e


def fuse(I, E, Wp, valid, c, k, L):
    """I, E: 1x3xHxW; Wp: Sx3xHxW warps (0 where invalid); valid: SxHxW bool."""
    nv = valid.float().sum(0, keepdim=True)[None]                                       # 1x1xHxW
    mu = (Wp * valid[:, None]).sum(0, keepdim=True) / nv.clamp_min(1)
    Wf = torch.where(valid[:, None], Wp, mu.expand_as(Wp))                               # invalid slots -> mean (no spread)
    PI, PE = lap_pyr(I, L), lap_pyr(E, L)
    PW = lap_pyr(Wf, L)
    m = [nv]
    for _ in range(L - 1):
        m.append(F.avg_pool2d(m[-1], 2, ceil_mode=True))
    out = []
    for l in range(L):
        d = PE[l] - PI[l]
        D = smooth((d ** 2).mean(1, keepdim=True), k)
        bw = PW[l]; bmu = bw.mean(0, keepdim=True)
        ns = m[l].clamp_min(1)
        V = smooth((((bw - bmu) ** 2).sum(0, keepdim=True) / ns).mean(1, keepdim=True), k)
        w = (1 - c * V / D.clamp_min(1e-8)).clamp(0, 1)
        w = torch.where(m[l] > 0.5, w, torch.zeros_like(w))                             # no evidence -> explicit render
        out.append(PI[l] + w * d)
    return collapse(out).clamp(0, 1)


def psnr(a, b):
    a = (a.clamp(0, 1) * 255 + 0.5).floor() / 255
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", nargs="+", required=True); ap.add_argument("--tag", default="r1")
    ap.add_argument("--E", default="ibgs_final"); ap.add_argument("--L", type=int, default=6)
    a = ap.parse_args()
    grid = [(c, k) for c in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0) for k in (0, 2, 5)]
    S = {}
    for s in a.scenes:
        fs = sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_{a.tag}", "*[0-9a-zA-Z].npz")))
        fs = [f for f in fs if not f.endswith(".geo.npz")]
        if not fs:
            continue
        rows = []
        for f in fs:
            z = np.load(f)
            t = lambda k: torch.from_numpy(z[k].astype(np.float32)).cuda()
            gt, I, E = t("gt"), t("mcmc")[None], t(a.E)[None]
            Wp, valid = t("warps"), torch.from_numpy(z["valid"]).cuda()
            r = {"I": psnr(I[0], gt), "E": psnr(E[0], gt)}
            for c, k in grid:
                r[f"c{c}_k{k}"] = psnr(fuse(I, E, Wp, valid, c, k, a.L)[0], gt)
            rows.append(r)
        S[s] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        print(f"[{s}] I {S[s]['I']:.2f} E {S[s]['E']:.2f} | best grid {max(((k, v) for k, v in S[s].items() if k[0] == 'c'), key=lambda kv: kv[1])}", flush=True)
    # leave-one-scene-out choice of (c, k)
    keys = [f"c{c}_k{k}" for c, k in grid]
    print("\nLOSO:")
    res = {}
    for s in S:
        others = [o for o in S if o != s] or [s]
        best = max(keys, key=lambda kk: np.mean([S[o][kk] - max(S[o]["I"], S[o]["E"]) for o in others]))
        res[s] = {"I": S[s]["I"], "E": S[s]["E"], "fused": S[s][best], "param": best}
        print(f"  {s:10s} I {S[s]['I']:.2f}  E {S[s]['E']:.2f}  best-of-two {max(S[s]['I'], S[s]['E']):.2f}  wiener {S[s][best]:.2f} ({best})")
    json.dump({"per_scene_grid": S, "loso": res}, open(os.path.join(ROOT, "outputs", "route", f"wiener_band_{a.E}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

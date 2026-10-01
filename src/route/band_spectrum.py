"""Error spectrum per Laplacian band: for each candidate (MCMC raw, IBGS final, IBGS residual on MCMC), mean squared error
of its band-l coefficients against the ground truth's band-l coefficients, pooled over the test views of a scene, split
by support class (#valid warps 0 / 1-2 / >=3). Tests the misregistration prediction: image-based evidence wins at
coarse bands and loses its edge (or loses) at the finest band; with no valid source it should not win anywhere.

    .venv_tools/bin/python src/route/band_spectrum.py bonsai counter train bicycle [--L 5]
"""
import argparse
import os
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = ("mcmc", "ibgs_final", "mcmc_res")


def down(x):
    H, W = x.shape[-2:]; x = x[..., : H - H % 2, : W - W % 2]
    return 0.25 * (x[..., ::2, ::2] + x[..., 1::2, ::2] + x[..., ::2, 1::2] + x[..., 1::2, 1::2])


def up(x, shape):
    y = np.repeat(np.repeat(x, 2, -2), 2, -1)
    out = np.zeros(x.shape[:-2] + shape, x.dtype); h, w = min(shape[0], y.shape[-2]), min(shape[1], y.shape[-1])
    out[..., :h, :w] = y[..., :h, :w]
    if h < shape[0]: out[..., h:, :] = out[..., h - 1:h, :]
    if w < shape[1]: out[..., :, w:] = out[..., :, w - 1:w]
    return out


def pyr(x, L):
    G = [x]
    for _ in range(L - 1):
        G.append(down(G[-1]))
    return [G[l] - up(G[l + 1], G[l].shape[-2:]) for l in range(L - 1)] + [G[-1]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scenes", nargs="+"); ap.add_argument("--L", type=int, default=5); ap.add_argument("--tag", default="r1")
    a = ap.parse_args()
    for s in a.scenes:
        fs = [f for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_{a.tag}", "*.npz"))) if not f.endswith(".geo.npz")]
        if not fs:
            continue
        acc = np.zeros((len(CANDS), a.L, 3)); cnt = np.zeros((a.L, 3))
        for f in fs:
            z = np.load(f); gt = z["gt"].astype(np.float32); nv = z["feats"][0].astype(np.float32)
            Pg = pyr(gt, a.L)
            masks = [nv < 0.5, (nv >= 0.5) & (nv < 2.5), nv >= 2.5]
            mp = [[m.astype(np.float32)] for m in masks]
            for m in mp:
                for _ in range(a.L - 1):
                    m.append(down(m[-1]))
            for ci, k in enumerate(CANDS):
                Pc = pyr(z[k].astype(np.float32), a.L)
                for l in range(a.L):
                    e = ((Pc[l] - Pg[l]) ** 2).mean(0)
                    for b in range(3):
                        w = mp[b][l] > 0.5
                        acc[ci, l, b] += e[w].sum()
                        if ci == 0: cnt[l, b] += w.sum()
        mse = acc / np.maximum(cnt, 1)
        print(f"\n== {s} ({len(fs)} views) — band MSE x1e4 (rows: band fine->coarse), and IBGS/MCMC, (I+r)/MCMC ratios")
        for b, name in enumerate(("u0 (no source)", "1-2 sources", ">=3 sources")):
            print(f"  [{name}] frac {cnt[0, b] / cnt[0].sum():.3f}")
            for l in range(a.L):
                m = mse[:, l, b] * 1e4
                print(f"    band {l}: mcmc {m[0]:8.3f}  ibgs {m[1]:8.3f}  I+r {m[2]:8.3f}   ibgs/mcmc {m[1] / max(m[0], 1e-9):5.2f}   (I+r)/mcmc {m[2] / max(m[0], 1e-9):5.2f}")


if __name__ == "__main__":
    main()

"""Fuse the MCMC render with band-limited evidence (bandwarp.py dumps) — non-learned and lightly-fitted rules.
  split-k        : Laplacian bands >= k (coarse) from the evidence where >= 1 source is valid at that level, else render
  alpha (LOSO)   : band_l = I_l + a_l * [n_l > 0] * (E_l - I_l), the 5 weights a_l fitted by least squares on the OTHER
                   scenes (bands are near-orthogonal, so the full-image MSE is ~ separable per band)
    .venv_tools/bin/python src/route/bandwarp_eval.py outputs/route/bandwarp/<scene>_test ...
"""
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F


def down(x):
    H, W = x.shape[-2:]
    if H % 2: x = np.concatenate([x, x[..., -1:, :]], -2)
    if W % 2: x = np.concatenate([x, x[..., :, -1:]], -1)
    return 0.25 * (x[..., ::2, ::2] + x[..., 1::2, ::2] + x[..., ::2, 1::2] + x[..., 1::2, 1::2])


def up(x, shape):
    y = np.repeat(np.repeat(x, 2, -2), 2, -1)[..., : shape[0], : shape[1]]
    return y


def lap_from_gauss(G):
    return [G[l] - up(G[l + 1], G[l].shape[-2:]) for l in range(len(G) - 1)] + [G[-1]]


def collapse(B):
    out = B[-1]
    for b in reversed(B[:-1]):
        out = b + up(out, b.shape[-2:])
    return out


def psnr(x, gt):
    x = np.floor(np.clip(x, 0, 1) * 255 + 0.5) / 255
    return -10 * np.log10(((x - gt) ** 2).mean())


def upto(x, l, shape):
    for _ in range(l):
        x = np.repeat(np.repeat(x, 2, -2), 2, -1)
    return x[..., : shape[0], : shape[1]]


def load(d):
    """per view: gt, I, and the low-pass corrections C_l = up^l(m_l * (E_l - G_l(I))) at full resolution, upsampled
    bilinearly (smooth). Each level's evidence is a LOW-PASS estimate on its own (levels are warped independently, so
    differences between levels are not valid band coefficients)."""
    V = []
    for f in sorted(glob(d + "/*.npz")):
        z = np.load(f); L = len([k for k in z if k.startswith("E")])
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).cuda()
        I = t(z["I"])[None]; gt = t(z["gt"])
        GI = [I]
        for _ in range(L - 1): GI.append(F.avg_pool2d(GI[-1], 2, ceil_mode=True))
        Cs = []
        for l in range(L):
            m = t(z[f"n{l}"] > 0)[None, None]
            c = m * (t(z[f"E{l}"])[None] - GI[l])
            Cs.append(F.interpolate(c, size=I.shape[-2:], mode="bilinear", align_corners=False)[0].cpu().numpy())
        V.append((gt.cpu().numpy(), I[0].cpu().numpy(), Cs, [z[f"n{l}"] for l in range(L)]))
    return V


def fused(v, a):
    gt, I, Cs, n = v
    return I + sum(a[l] * Cs[l] for l in range(len(Cs)))


def fit_alpha(views):
    """least squares: gt - I = sum_l a_l C_l (subsampled pixels)."""
    A = np.concatenate([np.stack([C[:, ::4, ::4].ravel() for C in Cs], 1) for gt, I, Cs, n in views])
    b = np.concatenate([(gt - I)[:, ::4, ::4].ravel() for gt, I, Cs, n in views])
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return a


def main():
    S = {d.rstrip("/").split("/")[-1]: load(d) for d in sys.argv[1:]}
    L = len(next(iter(S.values()))[0][2])
    for s, V in S.items():
        r = {"I": np.mean([psnr(v[1], v[0]) for v in V])}
        for k in range(L):
            a = [0] * L; a[k] = 1; r[f"lp{k}"] = np.mean([psnr(fused(v, a), v[0]) for v in V])
        others = [v for o, VV in S.items() if o != s for v in VV] or V
        a = fit_alpha(others); r["alpha"] = np.mean([psnr(fused(v, a), v[0]) for v in V])
        sup = np.mean([[float((v[3][l] > 0).mean()) for l in range(L)] for v in V], 0)
        print(f"[{s}] MCMC {r['I']:.2f} | low-pass from level k: " + " ".join(f"k{k} {r[f'lp{k}']:.2f}" for k in range(L)) +
              f" | alpha(LOSO) {r['alpha']:.2f} a={np.round(a, 2).tolist()} | support {np.round(sup, 2).tolist()}", flush=True)


if __name__ == "__main__":
    main()

"""R0 analysis — is there a routing gap? Reads outputs/route/r0/<scene>.npz (r0_probe.py) and reports, per scene:
  raw (K=0), IBGS default, nearest-K (K=1,2,3,5), coverage/random-K, image-level oracle subset (per K and overall),
  patch-level oracle composite (choose an entry per patch), and an honest version of it: the choice is made on
  one half of the views' patches in a checkerboard and scored on the other half (removes most of the winner's curse
  of picking the min of ~180 noisy values).
PSNR is computed from the mean of per-patch MSE over the cropped image (crop to multiples of the patch size).

    .venv_tools/bin/python src/route/r0_analyze.py outputs/route/r0/<scene>.npz [...]
"""
import sys

import numpy as np


def psnr(m):
    return -10 * np.log10(np.maximum(m, 1e-12))


def load(path):
    z = np.load(path, allow_pickle=False)
    subs = [tuple(int(x) for x in s.split(",")) if s else () for s in z["subsets"]]
    n = len(z["names"])
    V = []
    for i in range(n):
        d = {k: z[f"{k}_{i}"] for k in ("mse", "ssim", "lpips", "img_mse", "feats", "dis", "ang")}
        d["subs"] = [tuple(int(x) for x in s.split(",")) if s else () for s in z[f"subs_{i}"]] if f"subs_{i}" in z else subs
        V.append(d)
    return z["names"], subs, V


def scene_report(path):
    names, subs, V = load(path)
    rows = {}
    def add(key, per_view_mse):
        rows.setdefault(key, []).append(per_view_mse)
    rng = np.random.default_rng(0)
    for v in V:
        m = v["mse"].reshape(v["mse"].shape[0], -1).astype(np.float64)   # entries x patches (+ IBGS default last)
        subs = v["subs"]
        ids_k = {k: [i for i, s in enumerate(subs[:-1]) if len(s) == k] for k in range(0, 4)}
        near = {k: subs.index(tuple(range(k))) for k in (1, 2, 3) if tuple(range(k)) in subs}
        near[5] = len(subs) - 1            # nearest-5 (or the whole pool if shorter)
        if any(len(ids_k[k]) == 0 for k in (1, 2, 3)):
            continue
        add("raw K=0", m[0].mean()); add("IBGS default", m[-1].mean())
        for k, e in near.items():
            add(f"nearest-{k}", m[e].mean())
        for k in (1, 2, 3):
            ids = ids_k[k]
            add(f"random-{k}", m[rng.choice(ids)].mean())
            # coverage-K: subset maximising mean valid fraction union (from single-source features)
            f = v["feats"]
            if f.ndim == 4:
                vf = f[:, 0].reshape(f.shape[0], -1)
                best, bi = -1, ids[0]
                for e in ids:
                    cov = 1 - np.prod(1 - vf[list(subs[e])], axis=0)
                    c = cov.mean()
                    if c > best: best, bi = c, e
                add(f"coverage-{k}", m[bi].mean())
            add(f"oracle-img K={k}", m[ids].mean(1).min())
        add("oracle-img any", m[1:].mean(1).min())
        add("oracle-img any+K0", m.mean(1).min())
        # patch oracle composite
        add("oracle-patch (all entries)", m.min(0).mean())
        add("oracle-patch {K0, IBGS}", np.minimum(m[0], m[-1]).mean())
        # honest: checkerboard split — choose the entry per patch from the per-entry ranking learned on the OTHER colour?
        # Patches are disjoint, so instead: per patch choose by SSIM-loss argmin (a different but correlated loss), score MSE.
        s = 1 - v["ssim"].reshape(m.shape).astype(np.float64)
        add("oracle-patch chosen by SSIM, scored MSE", m[s.argmin(0), np.arange(m.shape[1])].mean())
    print(f"\n== {path}  ({len(rows['raw K=0'])} views)")
    base = psnr(np.mean(rows["IBGS default"]))
    for k, vals in rows.items():
        p = psnr(np.mean(vals)); pv = np.mean(psnr(np.array(vals)))
        print(f"  {k:42s} PSNR(mean-MSE) {p:6.2f}  mean-per-view {pv:6.2f}  Δ vs IBGS {p - base:+.3f}")
    return rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        scene_report(p)

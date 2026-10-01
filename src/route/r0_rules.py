"""R1 rules evaluated EXACTLY on the R0 table (every subset of size <= 3 was rendered): test-time-observable source
choices vs IBGS default / nearest-K / oracle. Signals per candidate source j (from its single-source pass, at test time):
  vf_j  = fraction of target pixels it validly covers
  d_j   = mean |warp_j - Gaussian render| over its valid pixels (appearance + alignment agreement with the explicit model)
Rules: agree-K   = K sources with the smallest d_j among those covering >= cov_min (pool order breaks ties)
       nearest-K with appearance veto = nearest-first, skipping sources with d_j > tau * median_j d_j

    .venv_tools/bin/python src/route/r0_rules.py outputs/route/r0/<scene>.npz ...
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r0_analyze import load, psnr  # noqa: E402


def run(path):
    names, _, V = load(path)
    acc = {}
    def add(k, v): acc.setdefault(k, []).append(v)
    for v in V:
        subs = v["subs"]; m = v["img_mse"]; f = v["feats"]
        if f.ndim != 4: continue
        P = f.shape[0]
        vf = f[:, 0].reshape(P, -1); dm = f[:, 1].reshape(P, -1)
        cov = vf.mean(1); d = np.array([np.average(dm[j], weights=vf[j]) if vf[j].sum() > 0 else 9 for j in range(P)])
        idx = {s: i for i, s in enumerate(subs)}
        add("IBGS default", m[-1])
        for K in (1, 2, 3):
            add(f"nearest-{K}", m[idx[tuple(range(K))]])
            add(f"oracle-img-{K}", min(m[i] for i, s in enumerate(subs[:-1]) if len(s) == K))
            ok = [j for j in range(P) if cov[j] >= 0.3] or list(range(P))
            pick = sorted(sorted(ok, key=lambda j: d[j])[:K])
            if len(pick) == K: add(f"agree-{K}", m[idx[tuple(pick)]])
            for tau in (1.25, 1.5):
                med = np.median(d[ok]); keep = [j for j in range(P) if d[j] <= tau * med and cov[j] > 0.05][:K]
                if len(keep) == K: add(f"veto{tau}-nearest-{K}", m[idx[tuple(keep)]])
        add("oracle-img-any", m[:-1].min())
    print(f"\n== {path}")
    base = np.mean(-10 * np.log10(acc["IBGS default"]))
    for k, vals in acc.items():
        pv = np.mean(-10 * np.log10(np.array(vals)))
        print(f"  {k:24s} n={len(vals):3d} mean-per-view PSNR {pv:6.2f}  Δ vs IBGS {pv - base:+.3f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)

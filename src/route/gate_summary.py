"""Dataset-level table from gate_loso.py result JSONs: mean PSNR / SSIM / LPIPS over the scenes of Mip-NeRF 360 (9),
Tanks&Temples (2), Deep Blending (2), for MCMC, IBGS final, IBGS residual on MCMC, and the gate; GADA = paper numbers.

    .venv_tools/bin/python src/route/gate_summary.py outputs/route/gate13_loso_band.json [...]
"""
import json
import sys

import numpy as np

GROUPS = {"Mip-360": ["bicycle", "flowers", "garden", "stump", "treehill", "bonsai", "counter", "kitchen", "room"],
          "T&T": ["train", "truck"], "DB": ["drjohnson", "playroom"]}
GADA = {"Mip-360": 28.63, "T&T": 24.93, "DB": 30.22}                       # paper (dataset means)
GADA_SCENE = {"bicycle": 26.16, "flowers": 22.29, "garden": 27.74, "stump": 27.33, "treehill": 23.16, "bonsai": 35.37,
              "counter": 30.84, "kitchen": 32.09, "room": 32.67, "train": 23.67, "truck": 26.19}
KEYS = ["mcmc", "ibgs_final", "mcmc_res", "gate"]


def main():
    for path in sys.argv[1:]:
        R = json.load(open(path))
        print(f"\n== {path}")
        print(f"  {'scene':10s} " + " ".join(f"{k:>18s}" for k in KEYS) + "   GADA(paper)  Δgate-IBGS  Δgate-GADA")
        for s, r in R.items():
            g = GADA_SCENE.get(s)
            print(f"  {s:10s} " + " ".join(f"{r[k][0]:6.2f}/{r[k][1]:.3f}/{r[k][2]:.3f}" for k in KEYS) +
                  f"   {g if g else '  -  ':>6}     {r['gate'][0] - r['ibgs_final'][0]:+.2f}      " + (f"{r['gate'][0] - g:+.2f}" if g else "  -"))
        for grp, scenes in GROUPS.items():
            have = [s for s in scenes if s in R]
            if not have:
                continue
            m = {k: np.mean([R[s][k] for s in have], 0) for k in KEYS}
            tag = "" if len(have) == len(scenes) else f" (only {len(have)}/{len(scenes)})"
            print(f"  {grp + tag:10s} " + " ".join(f"{m[k][0]:6.2f}/{m[k][1]:.3f}/{m[k][2]:.3f}" for k in KEYS) +
                  f"   {GADA[grp]:6.2f}     {m['gate'][0] - m['ibgs_final'][0]:+.2f}      {m['gate'][0] - GADA[grp]:+.2f}")


if __name__ == "__main__":
    main()

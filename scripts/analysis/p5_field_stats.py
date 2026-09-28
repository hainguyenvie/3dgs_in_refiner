"""Learned per-view phase fields (P5): magnitude and agreement with the RAFT-measured fields.
    .venv_mcmc/bin/python scripts/analysis/p5_field_stats.py bicycle_phase_learnZ garden_phase_learnZ
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for t in sys.argv[1:]:
    scene = t.split("_")[0]
    d = torch.load(ROOT / "outputs" / "p4" / t / "phase_fields.pt")
    j = json.load(open(ROOT / "data" / "p4" / f"{scene}_fields.json")); terms = [tuple(x) for x in j["terms"]]
    H, W = 64, 96
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij"); xn, yn = xx / W - .5, yy / H - .5
    B = torch.stack([xn ** i * yn ** j for i, j in terms], -1)
    mags = [float((B @ a).norm(dim=-1).mean()) for a in d.values()]
    cos, ratio = [], []
    for n, a in d.items():
        v = j["views"].get(n)
        if v is None:
            continue
        f1 = (B @ a).flatten(); f2 = (B @ torch.tensor(v["A"])).flatten()
        cos.append(float((f1 @ f2) / (f1.norm() * f2.norm() + 1e-9))); ratio.append(float(f1.norm() / (f2.norm() + 1e-9)))
    print(f"{t}: learned |f| median {np.median(mags):.3f} px (p90 {np.percentile(mags, 90):.3f}); vs measured: cos median {np.median(cos):.2f}, "
          f"norm ratio median {np.median(ratio):.2f}, views {len(d)}")

"""Per-view learned field magnitudes (px) for a phase run; lists the largest ones.
    .venv_mcmc/bin/python scripts/analysis/field_outliers.py outputs/p4/kitchen_p6_m1n [top]
"""
import json, sys
from pathlib import Path
import numpy as np, torch
d = Path(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
P = torch.load(d / "phase_fields.pt"); summ = json.load(open(d / "phase_summary.json"))
deg = 3; terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
H, W = 64, 96
yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij"); xn, yn = xx / W - .5, yy / H - .5
B = torch.stack([xn ** i * yn ** j for i, j in terms], -1)
mags = {k: float((B @ v).norm(dim=-1).mean()) for k, v in P.items()}
vals = np.array(list(mags.values()))
print(f"{d.name}: views {len(mags)} | median {np.median(vals):.3f} p90 {np.percentile(vals,90):.3f} max {vals.max():.3f} px | >0.3px: {(vals>0.3).sum()} views")
for k, m in sorted(mags.items(), key=lambda x: -x[1])[:top]:
    a = P[k]; print(f"  {k}: {m:.3f} px  (const {float(a[0].norm()):.3f}, linear {float(a[1:3].norm()+a[4].norm()):.3f})")

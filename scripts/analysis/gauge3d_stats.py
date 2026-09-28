"""Per-view pose corrections implied by the learned phase fields (no GT): rotation (deg), translation (relative to
the median camera-to-scene distance) and the image-space rms before/after the per-view 6-DoF fit.
    .venv_mcmc/bin/python scripts/analysis/gauge3d_stats.py outputs/p4/kitchen_p6_m1n ...
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gauge3d_fit import build  # noqa: E402

for model in sys.argv[1:]:
    rows, X = build(model, use_shared=False)
    cams = {c["img_name"]: c for c in json.load(open(os.path.join(model, "cameras.json")))}
    ctr = np.median(X, 0); dist = np.median([np.linalg.norm(np.array(cams[r[2]]["position"]) - ctr) for r in rows if r[2] in cams])
    rot, tr, before, after = [], [], [], []
    for J, y, n, Wt in rows:
        Jp = J[:, :, 1:].reshape(-1, 6); th = np.linalg.lstsq(Jp, y.reshape(-1), rcond=None)[0]
        rot.append(np.degrees(np.linalg.norm(th[:3]))); tr.append(np.linalg.norm(th[3:]) / dist)
        before.append(np.sqrt((y ** 2).mean()) * Wt); after.append(np.sqrt(((y.reshape(-1) - Jp @ th) ** 2).mean()) * Wt)
    rot, tr, before, after = map(np.array, (rot, tr, before, after))
    print(f"{os.path.basename(model)}: {len(rows)} views | per-view pose correction: rotation median {np.median(rot):.4f}° (p90 {np.percentile(rot,90):.4f}°), "
          f"translation median {np.median(tr)*100:.3f}% of scene distance | field rms px median {np.median(before):.3f} -> residual {np.median(after):.3f}")

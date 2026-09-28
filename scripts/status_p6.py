#!/usr/bin/env python3
"""All P6-family runs (outputs/p4/*_p6_*) vs the MCMC baseline of the same scene."""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = {"bicycle": 26.13, "garden": 28.19, "stump": 27.69, "counter": 29.43, "bonsai": 32.78, "kitchen": 32.21, "room": 32.48,
        "train": 22.61, "truck": 26.31, "drjohnson": 29.50, "playroom": 30.03, "flowers": 22.41, "treehill": 23.33}
for d in sorted(glob.glob(str(ROOT / "outputs" / "p4" / "*_p6_*"))):
    t = os.path.basename(d); s = t.split("_")[0]
    if not os.path.exists(d + "/results.json"):
        continue
    r = json.load(open(d + "/results.json"))
    for k, v in r.items():
        f = json.load(open(d + "/phase_summary.json"))["field_px_median"] if os.path.exists(d + "/phase_summary.json") else float("nan")
        tag = t + ("" if k == "ours_30000" else " [" + k.replace("ours_30000_", "") + "]")
        print(f"{tag:34s} {v['PSNR']:.2f} ({v['PSNR']-BASE[s]:+.2f})  SSIM {v['SSIM']:.3f}  LPIPS {v['LPIPS']:.3f}  field {f:.2f}px")

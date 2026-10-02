"""Sector (contiguous-azimuth) holdout of a scene, following the published rule of "Mind the Gap" (arXiv 2607.01556):
sort cameras by azimuth around the scene centre, hold out K contiguous views (K = N // 8, the same count as the
standard every-8th split, so training sets have equal size), start chosen with seed 42. Writes data/sector/<scene>
(images symlinked, sparse model unchanged) with split.json {"train": [...], "test": [...]} (names without extension,
as the IBGS / patched 3DGS loaders expect), plus the matching every-8th split as split_interp.json for comparison.

    .venv_tools/bin/python scripts/analysis/make_sector_split.py data/mipnerf360/garden data/sector/garden
"""
import json
import os
import sys

import numpy as np
import pycolmap


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    C = np.array([im.projection_center() for im in ims]); names = [os.path.splitext(im.name)[0] for im in ims]
    c0 = C.mean(0); X = C - c0
    _, _, Vt = np.linalg.svd(X, full_matrices=False)                  # plane of the camera ring: first two PCs
    az = np.arctan2(X @ Vt[1], X @ Vt[0])
    order = np.argsort(az); N = len(ims); K = N // 8
    start = np.random.default_rng(42).integers(N)
    test_idx = set(order[(start + np.arange(K)) % N].tolist())
    split = {"train": [n for i, n in enumerate(names) if i not in test_idx], "test": [n for i, n in enumerate(names) if i in test_idx]}
    interp = {"train": [n for i, n in enumerate(names) if i % 8 != 0], "test": [n for i, n in enumerate(names) if i % 8 == 0]}
    os.makedirs(dst, exist_ok=True)
    for sub in os.listdir(src):
        if (sub.startswith("images") or sub == "sparse") and not os.path.exists(os.path.join(dst, sub)):
            os.symlink(os.path.realpath(os.path.join(src, sub)), os.path.join(dst, sub))
    json.dump(split, open(os.path.join(dst, "split.json"), "w"), indent=0)
    json.dump(interp, open(os.path.join(dst, "split_interp.json"), "w"), indent=0)
    span = np.degrees(np.ptp(np.unwrap(az[sorted(test_idx)]))) if K else 0
    print(f"{os.path.basename(src.rstrip('/'))}: N={N}, sector K={K} (start {start}, azimuth span ~{span:.0f} deg) -> {dst}")


if __name__ == "__main__":
    main()

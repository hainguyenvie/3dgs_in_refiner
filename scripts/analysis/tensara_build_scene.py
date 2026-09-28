"""Build a Tensara-format scene (train/{images,sparse/0} + test/test_poses.csv) for a Mip-NeRF 360 scene at the
IBGS/MCMC evaluation resolution, so that the Tensara post-hoc refiner can be scored against the same test GT.

Images = the loader's downsampled GT saved by the IBGS fork (outputs/protocolR/ibgs/<scene>_pre/{train,test}/ours_30000/gt);
cameras.bin rescaled (data/e8/<scene>_ibgs_syn); images.bin filtered to TRAIN views; test poses -> CSV.

    python3 scripts/analysis/tensara_build_scene.py bicycle
"""
import csv
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e7_build_synthetic import read_images_bin, write_images_bin  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def read_cam(p):
    with open(p, "rb") as f:
        f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24)); npar = {0: 3, 1: 4}[model]
        par = struct.unpack("<" + "d" * npar, f.read(8 * npar))
    if model == 0:
        return par[0], par[0], par[1], par[2], w, h
    return par[0], par[1], par[2], par[3], w, h


def main():
    scene = sys.argv[1]
    src = ROOT / "data" / "mipnerf360" / scene
    cams_scaled = ROOT / "data" / "e8" / f"{scene}_ibgs_syn" / "sparse" / "0" / "cameras.bin"
    pre = ROOT / "outputs" / "protocolR" / "ibgs" / f"{scene}_pre"
    dst = ROOT / "data" / "tensara" / scene
    (dst / "train" / "sparse" / "0").mkdir(parents=True, exist_ok=True); (dst / "train" / "images").mkdir(exist_ok=True)
    (dst / "test" / "gt").mkdir(parents=True, exist_ok=True)
    shutil.copy(cams_scaled, dst / "train" / "sparse" / "0" / "cameras.bin")
    shutil.copy(src / "sparse" / "0" / "points3D.bin", dst / "train" / "sparse" / "0" / "points3D.bin")
    fx, fy, cx, cy, W, H = read_cam(cams_scaled)
    ims = read_images_bin(src / "sparse" / "0" / "images.bin")
    names = sorted(im[4] for im in ims); test = {n for i, n in enumerate(names) if i % 8 == 0}
    train_ims, rows = [], []
    for iid, q, t, cid, name, np2, pts in ims:
        stem = os.path.splitext(name)[0]
        if name in test:
            g = pre / "test" / "ours_30000" / "gt" / f"{stem}.png"
            if not (dst / "test" / "gt" / f"{stem}.png").exists():
                os.symlink(g.resolve(), dst / "test" / "gt" / f"{stem}.png")
            rows.append({"image_name": stem, "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3], "tx": t[0], "ty": t[1], "tz": t[2],
                         "fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": W, "height": H})
        else:
            g = pre / "train" / "ours_30000" / "gt" / f"{stem}.png"
            assert g.exists(), g
            link = dst / "train" / "images" / f"{stem}.png"
            if not link.exists():
                os.symlink(g.resolve(), link)
            train_ims.append([iid, q, t, cid, f"{stem}.png", np2, pts])
    write_images_bin(dst / "train" / "sparse" / "0" / "images.bin", train_ims)
    with open(dst / "test" / "test_poses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{dst}: {len(train_ims)} train images, {len(rows)} test poses, {W}x{H}, fx={fx:.1f}")


if __name__ == "__main__":
    main()

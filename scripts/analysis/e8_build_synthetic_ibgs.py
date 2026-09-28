"""E8 — synthetic consistent world for the WARP test. All views (train+test) rendered from the IBGS released model
become the images of a new scene (same poses; cameras.bin rescaled to the render resolution because the IBGS
loader takes image size from COLMAP intrinsics). Warping these sources with the model's own depth isolates the
errors inherent to depth-warping (visibility, mixed pixels, resampling) from real-data inconsistency
(scene motion, exposure, view dependence).

    python3 scripts/analysis/e8_build_synthetic_ibgs.py --scene bicycle --res 4
"""
import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e7_build_synthetic import read_images_bin, write_images_bin  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def rescale_cameras_bin(src, dst, res):
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cams = []
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}[model]
            p = list(struct.unpack("<" + "d" * npar, f.read(8 * npar)))
            cams.append((cid, model, w, h, p))
    with open(dst, "wb") as f:
        f.write(struct.pack("<Q", n))
        for cid, model, w, h, p in cams:
            W, H = round(w / res), round(h / res)          # IBGS loadCam: round(orig/res)
            assert model in (0, 1), "only SIMPLE_PINHOLE/PINHOLE handled"
            p = [x / res for x in p]                        # f, cx, cy (and fy) scale linearly
            f.write(struct.pack("<iiQQ", cid, model, W, H)); f.write(struct.pack("<" + "d" * len(p), *p))
    return W, H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--res", type=int, default=4)
    a = ap.parse_args()
    src = ROOT / "data" / "mipnerf360" / a.scene
    model = ROOT / "outputs" / "protocolR" / "ibgs" / f"{a.scene}_pre" / "test" / "ours_30000"
    trd = ROOT / "outputs" / "protocolR" / "ibgs" / f"{a.scene}_pre" / "train" / "ours_30000" / "renders"
    dst = ROOT / "data" / "e8" / f"{a.scene}_ibgs_syn"
    (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True); (dst / "images").mkdir(exist_ok=True)
    shutil.copy(src / "sparse" / "0" / "points3D.bin", dst / "sparse" / "0" / "points3D.bin")
    W, H = rescale_cameras_bin(src / "sparse" / "0" / "cameras.bin", dst / "sparse" / "0" / "cameras.bin", a.res)
    ims = read_images_bin(src / "sparse" / "0" / "images.bin")
    new, missing = [], []
    for iid, q, t, cid, name, np2, pts in ims:
        stem = os.path.splitext(name)[0]
        cands = [model / "renders" / f"{stem}.png", trd / f"{stem}.jpg", trd / f"{stem}.png"]   # IBGS names by image name
        rf = next((c for c in cands if c.exists()), None)
        if rf is None:
            missing.append(name); continue
        link = dst / "images" / (stem + rf.suffix)
        if not link.exists():
            os.symlink(rf.resolve(), link)
        new.append([iid, q, t, cid, stem + rf.suffix, np2, pts])
    write_images_bin(dst / "sparse" / "0" / "images.bin", new)
    print(f"{dst}: {len(new)} images at {W}x{H}, missing {len(missing)} {missing[:3]}")


if __name__ == "__main__":
    main()

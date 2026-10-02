"""Prepare one Nerfbusters scene for 3DGS / IBGS loaders: undistort the OPENCV COLMAP model to PINHOLE (pycolmap), and
write split.json from the official naming (train video: frame_XXXXX, eval video: frame_1_XXXXX).

    .venv_tools/bin/python scripts/analysis/prep_nerfbusters.py data/nerfbusters/nerfbusters-dataset/aloe data/nb/aloe
"""
import json
import os
import sys

import pycolmap


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    pycolmap.undistort_images(output_path=dst, input_path=os.path.join(src, "colmap", "sparse", "0"), image_path=os.path.join(src, "images"))
    sp = os.path.join(dst, "sparse")
    if os.path.isdir(sp) and not os.path.isdir(os.path.join(sp, "0")):   # undistorter writes sparse/*.bin
        os.makedirs(os.path.join(sp, "0"), exist_ok=True)
        for f in os.listdir(sp):
            if f.endswith(".bin") or f.endswith(".txt"):
                os.replace(os.path.join(sp, f), os.path.join(sp, "0", f))
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(dst, "images")))
    test = [n for n in names if n.startswith("frame_1_")]; train = [n for n in names if n not in set(test)]
    json.dump({"train": train, "test": test}, open(os.path.join(dst, "split.json"), "w"), indent=0)
    rec = pycolmap.Reconstruction(os.path.join(sp, "0")); cam = next(iter(rec.cameras.values()))
    print(f"{os.path.basename(src.rstrip('/'))}: {len(train)} train / {len(test)} eval, camera {cam.model.name} {cam.width}x{cam.height} -> {dst}")


if __name__ == "__main__":
    main()

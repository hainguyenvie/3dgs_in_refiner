"""Dev-target (cross-fitting) copy of a scene: the official TEST images (every 8th of the sorted names) are removed from
the sparse model entirely, so the standard LLFF hold (every 8th of the remaining sorted names) turns 1/8 of the TRAIN
images into "dev" views. A model trained on the copy never sees the dev views (test-like render errors there) nor the
official test views. Images are symlinked; cameras/poses are byte-identical to the release (no re-BA).

    .venv_tools/bin/python scripts/analysis/make_dev_split.py data/mipnerf360/bonsai data/dev/bonsai
"""
import os
import shutil
import sys

import pycolmap


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))
    names = sorted(im.name for im in rec.images.values())
    test = set(names[::8])
    os.makedirs(dst, exist_ok=True)
    for sub in os.listdir(src):
        if sub.startswith("images") and not os.path.exists(os.path.join(dst, sub)):
            os.symlink(os.path.realpath(os.path.join(src, sub)), os.path.join(dst, sub))
    sp = os.path.join(dst, "sparse", "0"); shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp)
    tmp = sp + "_txt"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp); rec.write_text(tmp)
    # text model, test images dropped from images.txt (2 lines per image); 3DGS loaders ignore point tracks
    lines = open(os.path.join(tmp, "images.txt")).read().splitlines()
    head = [l for l in lines if l.startswith("#")]; body = [l for l in lines if not l.startswith("#")]
    keep, kept = [], []
    for i in range(0, len(body) - 1, 2):
        name = body[i].split()[-1]
        if name not in test:
            keep += [body[i], body[i + 1]]; kept.append(name)
    open(os.path.join(sp, "images.txt"), "w").write("\n".join(head + keep) + "\n")
    for f in ("cameras.txt", "points3D.txt"):
        shutil.copy(os.path.join(tmp, f), os.path.join(sp, f))
    shutil.rmtree(tmp)
    kept = sorted(kept)
    dev = kept[::8]
    assert not (set(kept) & test), "official test image leaked into the dev copy"
    print(f"{os.path.basename(src.rstrip('/'))}: {len(names)} images, removed {len(test)} test, kept {len(kept)} -> dev {len(dev)} (e.g. {dev[:3]}), train {len(kept) - len(dev)}")


if __name__ == "__main__":
    main()

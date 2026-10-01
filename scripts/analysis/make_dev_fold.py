"""Cross-fitting fold k of a scene: like make_dev_split.py (official test images removed), but the held-out "dev" views
are kept[k::8] of the sorted train names. The 3DGS/MCMC/IBGS loaders hold out index % 8 == 0 of the sorted names, so
the images are symlinked under rotated names "<rank>_<orig>" (rank = (i - k) mod N) that put kept[k] first; the sparse
model's image names are rewritten accordingly. Poses/intrinsics/pixels are identical to the release.

    .venv_tools/bin/python scripts/analysis/make_dev_fold.py data/mipnerf360/bonsai data/devf/bonsai_f2 2 [images_dir]
"""
import os
import shutil
import sys

import pycolmap


def main():
    src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    imdir = sys.argv[4] if len(sys.argv) > 4 else "images"
    rec = pycolmap.Reconstruction(os.path.join(src, "sparse", "0"))
    names = sorted(im.name for im in rec.images.values())
    test = set(names[::8])
    kept = [n for n in names if n not in test]
    N = len(kept)
    new = {n: f"{(i - k) % N:05d}_{n}" for i, n in enumerate(kept)}
    os.makedirs(os.path.join(dst, imdir), exist_ok=True)
    for n, m in new.items():
        p = os.path.join(dst, imdir, m)
        if not os.path.lexists(p):
            os.symlink(os.path.realpath(os.path.join(src, imdir, n)), p)
    sp = os.path.join(dst, "sparse", "0"); shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp)
    tmp = sp + "_txt"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp); rec.write_text(tmp)
    lines = open(os.path.join(tmp, "images.txt")).read().splitlines()
    head = [l for l in lines if l.startswith("#")]; body = [l for l in lines if not l.startswith("#")]
    out = []
    for i in range(0, len(body) - 1, 2):
        f = body[i].split()
        if f[-1] in test:
            continue
        f[-1] = new[f[-1]]
        out += [" ".join(f), body[i + 1]]
    open(os.path.join(sp, "images.txt"), "w").write("\n".join(head + out) + "\n")
    for f in ("cameras.txt", "points3D.txt"):
        shutil.copy(os.path.join(tmp, f), os.path.join(sp, f))
    shutil.rmtree(tmp)
    held = sorted(new.values())[::8]
    extra = sorted(set(h.split("_", 1)[1] for h in held) - set(kept[k::8]))
    assert set(kept[k::8]) <= set(h.split("_", 1)[1] for h in held), "fold rotation mismatch"
    if extra: print(f"  note: rotation wrap adds {len(extra)} view(s) also held out in another fold: {extra}")
    print(f"{os.path.basename(src.rstrip('/'))} fold {k}: kept {N}, dev {len(held)} (orig {kept[k::8][:3]}), train {N - len(held)}")


if __name__ == "__main__":
    main()

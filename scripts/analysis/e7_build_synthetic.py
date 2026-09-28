"""E7 — self-consistency oracle dataset. Take a converged MCMC model, render ALL views (train+test) -> a
perfectly multi-view-consistent 'GT' with known poses. Retraining on it isolates the representation /
optimisation floor. Then inject per-view pose inconsistency of sigma px (rotate each TRAIN camera by
sigma/f rad about a random in-plane axis; test cameras untouched) to calibrate how much blur a given
inconsistency produces. No real image is resampled, so no resampling-blur confound.

    python3 scripts/analysis/e7_build_synthetic.py --scene bicycle --sigmas 0 0.15 0.3 0.6 --seed 0
"""
import argparse
import os
import shutil
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_images_bin(p):
    out = []
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = struct.unpack("<4d", f.read(32)); t = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            np2 = struct.unpack("<Q", f.read(8))[0]
            pts = f.read(24 * np2)
            out.append([iid, np.array(q), np.array(t), cid, name.decode(), np2, pts])
    return out


def write_images_bin(p, ims):
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(ims)))
        for iid, q, t, cid, name, np2, pts in ims:
            f.write(struct.pack("<i", iid)); f.write(struct.pack("<4d", *q)); f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", cid)); f.write(name.encode() + b"\x00"); f.write(struct.pack("<Q", np2)); f.write(pts)


def read_focal(p):
    with open(p, "rb") as f:
        f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
        fx = struct.unpack("<d", f.read(8))[0]
    return fx, w


def qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def rot2qvec(R):
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return q * np.sign(q[0]) if q[0] != 0 else q


def small_rot(axis, theta):
    axis = axis / np.linalg.norm(axis); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--res", type=int, default=4)
    a = ap.parse_args()
    src = ROOT / "data" / "mipnerf360" / a.scene
    model = ROOT / "outputs" / "protocolR" / "mcmc" / f"{a.scene}_r1"
    tr, te = model / "train" / "ours_30000" / "renders", model / "test" / "ours_30000" / "renders"
    ims = read_images_bin(src / "sparse" / "0" / "images.bin")
    names = sorted(im[4] for im in ims)
    test_names = {n for i, n in enumerate(names) if i % 8 == 0}        # 3DGS/MCMC llffhold split
    fx, w = read_focal(src / "sparse" / "0" / "cameras.bin")
    f_eval = fx / a.res
    # 3DGS render.py names outputs by INDEX in the (name-sorted) train / test camera lists
    train_names = [n for n in names if n not in test_names]; test_list = [n for n in names if n in test_names]
    idx_of = {n: i for i, n in enumerate(train_names)}; idx_of.update({n: i for i, n in enumerate(test_list)})
    def find(d, name):
        p = d / f"{idx_of[name]:05d}.png"
        return p if p.exists() else None
    for sig in a.sigmas:
        tag = f"syn_s{sig:g}"
        dst = ROOT / "data" / "e7" / f"{a.scene}_{tag}"
        (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True); (dst / "images").mkdir(exist_ok=True)
        for f in ("cameras.bin", "points3D.bin"):
            shutil.copy(src / "sparse" / "0" / f, dst / "sparse" / "0" / f)
        rng = np.random.default_rng(a.seed)
        new, missing = [], 0
        for iid, q, t, cid, name, np2, pts in ims:
            stem = os.path.splitext(name)[0]
            rf = find(te if name in test_names else tr, name)
            if rf is None:
                missing += 1; continue
            link = dst / "images" / (stem + rf.suffix)
            if not link.exists():
                os.symlink(rf.resolve(), link)
            if sig > 0 and name not in test_names:
                theta = rng.normal(0, sig) / f_eval           # px -> rad, isotropic magnitude sigma px
                ang = rng.uniform(0, 2 * np.pi); axis = np.array([np.cos(ang), np.sin(ang), 0.0])  # in-plane axis -> image shift
                dR = small_rot(axis, theta)
                R = qvec2rot(q); R2 = dR @ R; t2 = dR @ t
                q, t = rot2qvec(R2), t2
            new.append([iid, q, t, cid, stem + rf.suffix, np2, pts])
        write_images_bin(dst / "sparse" / "0" / "images.bin", new)
        print(f"{dst}: {len(new)} images ({len(test_names)} test), missing renders {missing}, f_eval={f_eval:.1f}px, sigma={sig}px -> theta_sigma={sig / f_eval:.2e} rad")


if __name__ == "__main__":
    main()

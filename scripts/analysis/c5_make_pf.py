"""C5 — build the re-calibrated copy of a scene: released SfM tracks (train + test images, as in the original release),
bundle adjustment with ONE FOCAL PER IMAGE (focus breathing), weakly constrained images (|focal/median - 1| > clamp)
reset to the median focal and poses re-adjusted with focal fixed. Images are untouched (symlinked); pp stays centred
unless --pp (per-image principal point too; needs --pp_from_colmap at train/render time).
Writes data/calib/<scene>_<tag>/{images*, sparse/0} and outputs/calib/<scene>/c5_<tag>.json.

    .venv_tools/bin/python scripts/analysis/c5_make_pf.py data/mipnerf360/garden [--pp] [--clamp 0.03]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_holdout_calib import per_image_cameras  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--pp", action="store_true"); ap.add_argument("--clamp", type=float, default=0.03); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    name = os.path.basename(a.scene.rstrip("/")); tag = a.tag or ("pfpp" if a.pp else "pf")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = pycolmap.Reconstruction(os.path.join(a.scene, "sparse", "0"))
    err0 = float(np.mean([p.error for p in base.points3D.values()]))
    rec = per_image_cameras(base)
    o = pycolmap.BundleAdjustmentOptions(); o.refine_focal_length = True; o.refine_principal_point = a.pp
    o.ceres.solver_options.num_threads = 16; o.ceres.solver_options.max_num_iterations = 100
    pycolmap.bundle_adjustment(rec, o)
    P = {iid: np.array(rec.cameras[im.camera_id].params) for iid, im in rec.images.items()}
    fx = np.array([p[0] for p in P.values()]); fy = np.array([p[1] for p in P.values()]); mfx, mfy = np.median(fx), np.median(fy)
    cam0 = next(iter(base.cameras.values())).params
    clamped = []
    for iid, im in rec.images.items():
        c = rec.cameras[im.camera_id]; p = np.array(c.params)
        if abs(p[0] / mfx - 1) > a.clamp:
            p[0], p[1] = mfx, mfy
            if a.pp: p[2], p[3] = cam0[2], cam0[3]
            c.params = p; clamped.append(im.name)
    if clamped:
        o2 = pycolmap.BundleAdjustmentOptions(); o2.refine_focal_length = False; o2.refine_principal_point = False
        o2.ceres.solver_options.num_threads = 16; pycolmap.bundle_adjustment(rec, o2)
    # gauge: BA with free intrinsics can drift the global scale (train pfpp shrank 200x) — reprojection is scale-invariant
    # but 3DGS is not (znear, init scales, MCMC noise). Sim3-align back to the released frame over the shared 3D points.
    sim = pycolmap.align_reconstructions_via_points(rec, base)
    if sim is not None:
        rec.transform(sim)
    err1 = float(np.mean([p.error for p in rec.points3D.values()]))
    fx = np.array([rec.cameras[im.camera_id].params[0] for im in rec.images.values()])
    dst = os.path.join(root, "data", "calib", f"{name}_{tag}"); os.makedirs(os.path.join(dst, "sparse"), exist_ok=True)
    for sub in ("images", "images_2", "images_4", "images_8"):
        s = os.path.join(a.scene, sub)
        if os.path.exists(s) and not os.path.exists(os.path.join(dst, sub)):
            os.symlink(os.path.realpath(s), os.path.join(dst, sub))
    sp = os.path.join(dst, "sparse", "0"); shutil.rmtree(sp, ignore_errors=True); os.makedirs(sp); rec.write(sp)
    C0 = np.array([im.projection_center() for im in base.images.values()]); C1 = np.array([im.projection_center() for im in rec.images.values()])
    scale_ratio = float(np.median(np.linalg.norm(C1 - np.median(C1, 0), axis=1)) / np.median(np.linalg.norm(C0 - np.median(C0, 0), axis=1)))
    info = {"scene": name, "tag": tag, "sim3_scale": float(sim.scale) if sim is not None else None, "camera_radius_ratio": scale_ratio, "mean_reproj_released": err0, "mean_reproj_new": err1, "clamped": clamped,
            "focal_rel_std": float(fx.std() / fx.mean()), "focal_rel_range": float((fx.max() - fx.min()) / fx.mean())}
    os.makedirs(os.path.join(root, "outputs", "calib", name), exist_ok=True)
    json.dump(info, open(os.path.join(root, "outputs", "calib", name, f"c5_{tag}.json"), "w"), indent=1)
    print(f"[{name}] {tag}: mean reproj {err0:.3f} -> {err1:.3f} px | focal rel std {info['focal_rel_std']*1e4:.1f}e-4 range {info['focal_rel_range']*1e4:.1f}e-4 | clamped {len(clamped)} | camera radius ratio vs released {scale_ratio:.3f} -> {dst}", flush=True)


if __name__ == "__main__":
    main()

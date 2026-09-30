"""C4 — HOLD-OUT test of a camera-model change (does it generalise, or does BA overfit keypoint noise?). CPU, no images.
Split the released SfM points by id parity. Run bundle adjustment on the EVEN points only, with
  pose : shared PINHOLE camera (as released)          pf  : one focal per image (focus breathing)
then re-triangulate every ODD point (DLT over its track, never seen by the BA) with the refined cameras and report its
reprojection error. A real camera-model defect lowers the held-out error; overfitting does not.

    .venv_tools/bin/python scripts/analysis/c4_holdout_calib.py data/mipnerf360/flowers [pose,pf,fpp,pfpp]
"""
import json
import os
import sys
import tempfile

import numpy as np
import pycolmap


def per_image_cameras(rec):
    tmp = tempfile.mkdtemp(); rec.write_text(tmp)
    cl = [l for l in open(os.path.join(tmp, "cameras.txt")) if not l.startswith("#")][0].split()
    lines = open(os.path.join(tmp, "images.txt")).read().splitlines(); hdr = [l for l in lines if l.startswith("#")]; body = [l for l in lines if not l.startswith("#")]
    cams, out = [], []
    for i in range(0, len(body), 2):
        f = body[i].split(); cid = int(f[0]); f[8] = str(cid); cams.append(" ".join([str(cid)] + cl[1:])); out += [" ".join(f), body[i + 1]]
    open(os.path.join(tmp, "cameras.txt"), "w").write("\n".join(cams) + "\n"); open(os.path.join(tmp, "images.txt"), "w").write("\n".join(hdr + out) + "\n")
    for f in ("rigs.txt", "frames.txt"):
        if os.path.exists(os.path.join(tmp, f)): os.remove(os.path.join(tmp, f))
    return pycolmap.Reconstruction(tmp)


def P_of(rec, iid):
    im = rec.images[iid]; c = rec.cameras[im.camera_id].params; T = im.cam_from_world()
    K = np.array([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1.0]])
    return K @ np.column_stack([T.rotation.matrix(), T.translation])


def triangulate_eval(rec, tracks):
    Ps = {iid: P_of(rec, iid) for iid in rec.images}
    errs = []
    for obs in tracks:
        A = []
        for iid, (x, y) in obs:
            P = Ps[iid]; A += [x * P[2] - P[0], y * P[2] - P[1]]
        _, _, Vt = np.linalg.svd(np.array(A)); X = Vt[-1]; X = X / X[3]
        for iid, (x, y) in obs:
            p = Ps[iid] @ X
            if p[2] <= 0: continue
            errs.append(np.hypot(p[0] / p[2] - x, p[1] / p[2] - y))
    return np.array(errs)


def main():
    scene = sys.argv[1]; variants = (sys.argv[2] if len(sys.argv) > 2 else "pose,pf").split(",")
    base = pycolmap.Reconstruction(os.path.join(scene, "sparse", "0"))
    odd = [pid for pid in base.points3D if pid % 2 == 1]
    tracks = []
    for pid in odd:
        obs = [(el.image_id, tuple(base.images[el.image_id].points2D[el.point2D_idx].xy)) for el in base.points3D[pid].track.elements]
        if len(obs) >= 3: tracks.append(obs)
    rng = np.random.default_rng(0); tracks = [tracks[i] for i in rng.choice(len(tracks), min(40000, len(tracks)), replace=False)]
    res = {"scene": os.path.basename(scene.rstrip("/")), "heldout_points": len(tracks)}
    for v in variants:
        rec = per_image_cameras(base) if v in ("pf", "pfpp") else pycolmap.Reconstruction(os.path.join(scene, "sparse", "0"))
        for pid in [p for p in rec.points3D if p % 2 == 1]:
            rec.delete_point3D(pid)
        o = pycolmap.BundleAdjustmentOptions(); o.refine_focal_length = v != "pose"; o.refine_principal_point = v in ("fpp", "pfpp")
        o.ceres.solver_options.num_threads = 16; o.ceres.solver_options.max_num_iterations = 100
        pycolmap.bundle_adjustment(rec, o)
        e = triangulate_eval(rec, tracks)
        res[v] = {"heldout_median": float(np.median(e)), "heldout_mean": float(e.mean()), "heldout_p90": float(np.percentile(e, 90))}
        print(f"[{res['scene']}] {v:5s}: held-out reprojection median {np.median(e):.4f} px  mean {e.mean():.4f}  p90 {np.percentile(e, 90):.3f} ({len(tracks)} points)", flush=True)
    out = os.path.join("outputs", "calib", res["scene"]); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "c4_holdout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

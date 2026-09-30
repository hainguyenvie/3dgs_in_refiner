"""D2 — re-fit ONLY the test cameras of a reconstruction from their own keypoints, with the scene (3D points) and all
train cameras fixed: pose (6) + optionally a per-image focal (1). Separates the two effects of re-calibration:
  free_focal   on the RELEASED sparse -> test cameras with per-image focal in the released frame (render the old model)
  fixed_focal  on the RE-CALIBRATED sparse -> test cameras with ONE shared focal in the new frame (render the new model)
Test images = every 8th of the sorted names (3DGS LLFF hold). Output json for render_cams_json.py.

    .venv_tools/bin/python scripts/analysis/d2_refit_test_cams.py <sparse> free_focal|fixed_focal <out.json>
"""
import json
import sys

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    sp, mode, out = sys.argv[1:4]
    rec = pycolmap.Reconstruction(sp)
    names = sorted(im.name for im in rec.images.values()); test = set(names[::8])
    f_all = np.array([rec.cameras[im.camera_id].params[0] for im in rec.images.values() if im.name not in test])
    f_shared = float(np.median(f_all))
    res = {"mode": mode, "images": {}}; moved = []
    for im in rec.images.values():
        c = rec.cameras[im.camera_id]; p = np.array(c.params, dtype=np.float64); T = im.cam_from_world()
        Rcw, tcw = T.rotation.matrix(), np.asarray(T.translation)
        if im.name in test:
            obs = [(pt.xy, rec.points3D[pt.point3D_id].xyz) for pt in im.points2D if pt.has_point3D()]
            xy = np.array([o[0] for o in obs]); X = np.array([o[1] for o in obs])
            ratio = p[1] / p[0]
            f0 = p[0] if mode == "free_focal" else f_shared
            def resid(v):
                R = Rotation.from_rotvec(v[:3]).as_matrix() @ Rcw; t = v[3:6] + tcw; f = v[6] if mode == "free_focal" else f0
                Xc = X @ R.T + t
                u = np.column_stack([f * Xc[:, 0] / Xc[:, 2] + p[2], f * ratio * Xc[:, 1] / Xc[:, 2] + p[3]])
                return (u - xy).ravel()
            v0 = np.zeros(7 if mode == "free_focal" else 6)
            if mode == "free_focal": v0[6] = f0
            r0 = np.median(np.linalg.norm(resid(v0).reshape(-1, 2), axis=1))
            sol = least_squares(resid, v0, loss="huber", f_scale=2.0)
            r1 = np.median(np.linalg.norm(sol.fun.reshape(-1, 2), axis=1))
            v = sol.x; Rcw = Rotation.from_rotvec(v[:3]).as_matrix() @ Rcw; tcw = v[3:6] + tcw
            f = v[6] if mode == "free_focal" else f0; p[0], p[1] = f, f * ratio
            moved.append((r0, r1, f / f_shared - 1))
        res["images"][im.name] = {"R_cw": Rcw.tolist(), "t_cw": tcw.tolist(), "model": c.model.name, "params": p.tolist(), "width": c.width, "height": c.height}
    json.dump(res, open(out, "w"))
    m = np.array(moved)
    print(f"{sp} {mode}: {len(m)} test cams refit | median reproj {np.median(m[:, 0]):.3f} -> {np.median(m[:, 1]):.3f} px | "
          f"test focal rel to shared: std {m[:, 2].std() * 1e4:.1f}e-4")


if __name__ == "__main__":
    main()

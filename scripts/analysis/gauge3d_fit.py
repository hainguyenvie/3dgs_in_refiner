"""Is the learned per-view phase field a 3D GAUGE (the whole scene drifted by a small similarity transform) rather than
per-view image-space misalignment?  Uses NO ground truth: only the learned fields (phase_fields.pt + phase_shared.json),
the trained Gaussians and the train cameras.

Model: the render shows model point X at pi_v(X); the photo shows that content at pi_v(X) - f_v(pi_v(X))  (training
convention gt(p) ~ render(p + f(p))).  A global correction X_true = X + s X + w x X + t (7 params) predicts an image
displacement J_v(X) dX; we solve   J_v(X) dX = -f_v(pi_v(X))   by least squares over all train views (visible points
only) and report the fraction of field energy explained by (a) one global sim3, (b) one 6-DoF pose per view.
With --apply, the sim3 is applied to the Gaussians -> <model>_g3d/point_cloud/iteration_30000/point_cloud.ply.

    .venv_mcmc/bin/python scripts/analysis/gauge3d_fit.py outputs/p4/flowers_p6_m1w -r 4 [--apply] [--no_shared]
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "phase"))


def load_ply(p):
    pl = PlyData.read(p)["vertex"]
    xyz = np.stack([pl["x"], pl["y"], pl["z"]], 1).astype(np.float32)
    op = 1 / (1 + np.exp(-pl["opacity"].astype(np.float32)))
    return pl, xyz, op


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def build(model, use_shared, n_pts=150000, cell=(48, 64)):
    cams = {c["img_name"]: c for c in json.load(open(os.path.join(model, "cameras.json")))}
    fields = torch.load(os.path.join(model, "phase_fields.pt"))
    sh = json.load(open(os.path.join(model, "phase_shared.json"))) if os.path.exists(os.path.join(model, "phase_shared.json")) else None
    terms = [tuple(t) for t in sh["terms"]] if sh else [(i, j) for i in range(4) for j in range(4 - i)]
    A_sh = np.array(sh["A"], dtype=np.float64) if (sh and use_shared) else 0.0
    _, xyz, op = load_ply(os.path.join(model, "point_cloud", "iteration_30000", "point_cloud.ply"))
    rng = np.random.default_rng(0); keep = np.where(op > 0.3)[0]; keep = rng.choice(keep, min(n_pts, len(keep)), replace=False)
    X = xyz[keep].astype(np.float64)
    rows = []   # per view: (J rows [N,2,7], target [N,2], name)
    for name, A in fields.items():
        c = cams.get(name) or cams.get(os.path.splitext(name)[0])
        if c is None:
            continue
        R = np.array(c["rotation"], dtype=np.float64); pos = np.array(c["position"], dtype=np.float64)
        Wt, Ht = c["width"], c["height"]; fxn, fyn = c["fx"] / Wt, c["fy"] / Ht       # cameras.json is at TRAIN resolution
        Xc = (X - pos) @ R                                  # R is C2W rotation -> world->cam is R^T (X-pos) = (X-pos) @ R
        z = Xc[:, 2]; ok = z > 1e-3
        u = fxn * Xc[:, 0] / np.where(ok, z, 1) + 0.5; v = fyn * Xc[:, 1] / np.where(ok, z, 1) + 0.5     # normalised [0,1]
        ok &= (u > 0) & (u < 1) & (v > 0) & (v < 1)
        # coarse z-buffer: keep points close to the front surface of their cell
        ci = np.clip((v * cell[0]).astype(int), 0, cell[0] - 1) * cell[1] + np.clip((u * cell[1]).astype(int), 0, cell[1] - 1)
        zmin = np.full(cell[0] * cell[1], np.inf); np.minimum.at(zmin, ci[ok], z[ok])
        ok &= z < 1.25 * zmin[ci]
        idx = np.where(ok)[0]
        if len(idx) < 200:
            continue
        Xw, Xcc, uu, vv, zz = X[idx], Xc[idx], u[idx], v[idx], z[idx]
        xn, yn = uu - 0.5, vv - 0.5
        B = np.stack([xn ** i * yn ** j for i, j in terms], -1)
        f = B @ (A.numpy().astype(np.float64) + A_sh)          # px at train res
        f = f / np.array([Wt, Ht])                              # normalised units
        # Jacobian of normalised pixel wrt camera point, then wrt world dX (dXc = R^T dX)
        Ju = np.stack([fxn / zz, np.zeros_like(zz), -fxn * Xcc[:, 0] / zz ** 2], -1)
        Jv = np.stack([np.zeros_like(zz), fyn / zz, -fyn * Xcc[:, 1] / zz ** 2], -1)
        Jc = np.stack([Ju, Jv], 1) @ R.T                        # N,2,3 : d(u,v)/dX_world
        M = np.concatenate([Xw[:, :, None], -np.stack([skew(x) for x in Xw]), np.broadcast_to(np.eye(3), (len(Xw), 3, 3))], 2)   # N,3,7 : dX = M theta
        J = Jc @ M                                              # N,2,7
        rows.append((J, -f, name, Wt))
    return rows, X


def solve(rows, which=slice(0, 7)):
    J = np.concatenate([r[0][:, :, which].reshape(-1, r[0][:, :, which].shape[-1]) for r in rows]); y = np.concatenate([r[1].reshape(-1) for r in rows])
    theta, *_ = np.linalg.lstsq(J, y, rcond=None)
    return theta, J, y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("-r", type=float, default=1); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no_shared", action="store_true"); ap.add_argument("--anchor_every", type=int, default=8)
    a = ap.parse_args()
    rows, X = build(a.model, not a.no_shared)
    names = sorted(r[2] for r in rows); anchors = set(names[a.anchor_every // 2::a.anchor_every])
    W_med = np.median([r[3] for r in rows])
    def energy(rows_, theta=None, per_view=False):
        tot, res = 0.0, 0.0
        for J, y, n, Wt in rows_:
            th = theta if theta is not None else np.linalg.lstsq(J[:, :, 1:].reshape(-1, 6), y.reshape(-1), rcond=None)[0]
            pred = (J[:, :, 1:] @ th) if (theta is None) else (J @ th)
            tot += (y ** 2).sum() * Wt ** 2; res += ((y - pred) ** 2).sum() * Wt ** 2
        return tot, res
    out = {"model": a.model, "views": len(rows), "anchors": len(anchors & set(names))}
    for tag, sel in [("all", rows), ("non_anchor", [r for r in rows if r[2] not in anchors])]:
        theta, J, y = solve(sel)
        tot, res = energy(sel, theta)
        tot_a, res_a = energy([r for r in rows if r[2] in anchors], theta) if anchors else (0, 0)
        _, res_pv = energy(sel, None)
        out[tag] = {"sim3": {"s": float(theta[0]), "omega": theta[1:4].tolist(), "t": theta[4:].tolist()},
                    "field_rms_px": float(np.sqrt(tot / sum(len(r[1]) for r in sel) / 2)),
                    "explained_sim3": 1 - res / tot, "explained_perview_pose": 1 - res_pv / tot,
                    "anchor_rms_px_after": float(np.sqrt(res_a / max(1, sum(len(r[1]) for r in rows if r[2] in anchors)) / 2)) if anchors else None}
        print(f"[{tag}] field rms {out[tag]['field_rms_px']:.3f} px | explained by ONE sim3 {out[tag]['explained_sim3']*100:.1f}% | "
              f"by per-view 6-DoF pose {out[tag]['explained_perview_pose']*100:.1f}% | s={theta[0]:+.2e} |w|={np.linalg.norm(theta[1:4]):.2e} |t|={np.linalg.norm(theta[4:]):.2e}"
              + (f" | anchors would move {out[tag]['anchor_rms_px_after']:.3f} px" if anchors else ""), flush=True)
    json.dump(out, open(os.path.join(a.model, "gauge3d.json"), "w"), indent=1)
    if a.apply:
        theta = np.array(solve(rows)[0]); s, w, t = theta[0], theta[1:4], theta[4:]
        src = os.path.join(a.model, "point_cloud", "iteration_30000", "point_cloud.ply")
        pl, xyz, _ = load_ply(src)
        Rw = np.eye(3) + skew(w)                                # first order rotation (|w| ~ 1e-3)
        Uw, _, Vt = np.linalg.svd(Rw); Rw = Uw @ Vt
        new = (1 + s) * (xyz.astype(np.float64) @ Rw.T) + t
        v = pl.data.copy()
        v["x"], v["y"], v["z"] = new[:, 0].astype(np.float32), new[:, 1].astype(np.float32), new[:, 2].astype(np.float32)
        for k in ("scale_0", "scale_1", "scale_2"):
            v[k] = (v[k] + np.log1p(s)).astype(np.float32)
        # rotate orientations: q' = q_w * q
        q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)          # w x y z
        ang = np.linalg.norm(w); ax = w / ang if ang > 0 else np.array([1, 0, 0.])
        qw = np.array([np.cos(ang / 2), *(np.sin(ang / 2) * ax)])
        w1, x1, y1, z1 = qw; w2, x2, y2, z2 = q.T
        qn = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)
        for i, k in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
            v[k] = qn[:, i].astype(np.float32)
        dst_model = a.model.rstrip("/") + "_g3d"
        os.makedirs(os.path.join(dst_model, "point_cloud", "iteration_30000"), exist_ok=True)
        PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(dst_model, "point_cloud", "iteration_30000", "point_cloud.ply"))
        for f in ("cfg_args", "cameras.json", "phase_shared.json"):
            if os.path.exists(os.path.join(a.model, f)):
                shutil.copy(os.path.join(a.model, f), dst_model)
        disp = np.abs(new - xyz).mean()
        print(f"[apply] wrote {dst_model} (mean |dX| {disp:.4f} world units, s={s:+.2e})")


if __name__ == "__main__":
    main()

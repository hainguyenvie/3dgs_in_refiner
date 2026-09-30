"""C1 — Tensara-style audit of the RELEASED COLMAP calibration (no model, no GT images; CPU).
The released reconstructions (Mip-360, T&T/DB) use ONE PINHOLE camera per scene with the principal point pinned at
the image centre (COLMAP does not refine it by default) — any real principal-point / focal / distortion error is then
absorbed by bundle adjustment into poses and points, which makes the views sub-pixel inconsistent (our E-chain).
This script measures it from the SfM's own tracks (train AND test images, as in the original release):
  1. reprojection residual per observation; radial profile; residual field pooled on a 12x9 grid (train images) and
     fitted by translation (principal point) + linear (focal/aspect/skew) + radial (distortion) terms;
  2. bundle adjustment variants on the same tracks: poses+points only / + focal / + focal + principal point /
     OPENCV (+ k1 k2 p1 p2) — reprojection before/after, fitted intrinsics, and how far TEST cameras move (px).
Writes outputs/calib/<scene>/c1.json and, for variant `fpp` (PINHOLE focal + pp), the refined sparse model.

    .venv_tools/bin/python scripts/analysis/c1_calib_audit.py data/mipnerf360/flowers [--no_ba]
"""
import argparse
import json
import os
import time

import numpy as np
import pycolmap


def project(cam, Xc):
    """pinhole (+OPENCV distortion) projection of camera-frame points, numpy (matches COLMAP's models)."""
    p = cam.params; x = Xc[:, 0] / Xc[:, 2]; y = Xc[:, 1] / Xc[:, 2]
    if cam.model.name == "OPENCV":
        k1, k2, p1, p2 = p[4:8]; r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
        x, y = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    elif cam.model.name not in ("PINHOLE",):
        raise ValueError(cam.model.name)
    return np.column_stack([p[0] * x + p[2], p[1] * y + p[3]])


def observations(rec, test_ids):
    rows = []
    for iid, im in rec.images.items():
        cam = rec.cameras[im.camera_id]; T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        pts = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
        if not pts:
            continue
        xy = np.array([p[0] for p in pts]); X = np.array([rec.points3D[p[1]].xyz for p in pts])
        Xc = (T.rotation.matrix() @ X.T).T + T.translation
        uv = project(cam, Xc)
        tl = np.array([rec.points3D[p[1]].track.length() for p in pts])
        rows.append(np.column_stack([xy, uv - xy, np.full(len(xy), iid in test_ids), tl]))
    return np.concatenate(rows)          # x, y, rx, ry, is_test, track_len


def field_fit(O, W, H):
    tr = O[O[:, 4] == 0]
    c = np.array([W / 2, H / 2]); s = max(W, H) / 2
    p = (tr[:, :2] - c) / s; r2 = (p ** 2).sum(1, keepdims=True)
    # rx,ry = t + L p + k p |p|^2   (per axis, shared radial k)
    A = np.zeros((2 * len(p), 7)); y = np.concatenate([tr[:, 2], tr[:, 3]])
    A[:len(p), 0] = 1; A[:len(p), 2] = p[:, 0]; A[:len(p), 3] = p[:, 1]; A[:len(p), 6] = p[:, 0] * r2[:, 0]
    A[len(p):, 1] = 1; A[len(p):, 4] = p[:, 0]; A[len(p):, 5] = p[:, 1]; A[len(p):, 6] = p[:, 1] * r2[:, 0]
    th, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ th
    return {"translation_px": th[:2].tolist(), "linear_px_at_edge": th[2:6].tolist(), "radial_px_at_edge": float(th[6]),
            "explained": float(1 - (res ** 2).sum() / (y ** 2).sum())}


def grid_field(O, W, H, gx=12, gy=9):
    tr = O[O[:, 4] == 0]
    ix = np.clip((tr[:, 0] / W * gx).astype(int), 0, gx - 1); iy = np.clip((tr[:, 1] / H * gy).astype(int), 0, gy - 1)
    g = np.zeros((gy, gx, 2)); n = np.zeros((gy, gx))
    np.add.at(g, (iy, ix), tr[:, 2:4]); np.add.at(n, (iy, ix), 1)
    g = g / np.maximum(n, 1)[..., None]
    return {"mean_cell_vector_px": float(np.linalg.norm(g, axis=-1).mean()), "max_cell_vector_px": float(np.linalg.norm(g, axis=-1).max())}


def stats(O, W, H):
    r = np.linalg.norm(O[:, 2:4], axis=1); rad = np.linalg.norm(O[:, :2] - [W / 2, H / 2], axis=1) / (np.hypot(W, H) / 2)
    prof = []
    for a, b in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = (rad >= a) & (rad < b) & (O[:, 4] == 0)
        prof.append(float(np.median(r[m])) if m.any() else None)
    return {"median_train": float(np.median(r[O[:, 4] == 0])), "median_test": float(np.median(r[O[:, 4] == 1])) if (O[:, 4] == 1).any() else None,
            "mean_train": float(r[O[:, 4] == 0].mean()), "radial_profile_median": prof,
            "median_by_tracklen": {k: float(np.median(r[(O[:, 5] >= a) & (O[:, 5] < b)])) for k, (a, b) in
                                   {"2": (2, 3), "3-4": (3, 5), "5-9": (5, 10), "10+": (10, 1e9)}.items() if ((O[:, 5] >= a) & (O[:, 5] < b)).any()}}


def run_ba(path, variant, threads):
    rec = pycolmap.Reconstruction(path)
    if variant in ("pf", "pfpp"):      # one camera PER IMAGE (focus breathing / per-shot intrinsics), same initial values
        import tempfile
        tmp = tempfile.mkdtemp(); rec.write_text(tmp)
        cl = [l for l in open(os.path.join(tmp, "cameras.txt")) if not l.startswith("#")][0].split()
        lines = open(os.path.join(tmp, "images.txt")).read().splitlines(); hdr = [l for l in lines if l.startswith("#")]; body = [l for l in lines if not l.startswith("#")]
        cams, out = [], []
        for i in range(0, len(body), 2):
            f = body[i].split(); cid = int(f[0]); f[8] = str(cid); cams.append(" ".join([str(cid)] + cl[1:])); out += [" ".join(f), body[i + 1]]
        open(os.path.join(tmp, "cameras.txt"), "w").write("\n".join(cams) + "\n"); open(os.path.join(tmp, "images.txt"), "w").write("\n".join(hdr + out) + "\n")
        for f in ("rigs.txt", "frames.txt"):          # COLMAP 3.12+/pycolmap 4: drop rig/frame files -> one trivial rig per camera on read
            if os.path.exists(os.path.join(tmp, f)): os.remove(os.path.join(tmp, f))
        rec = pycolmap.Reconstruction(tmp)
    if variant == "opencv":
        for cid, cam in list(rec.cameras.items()):
            fx, fy, cx, cy = cam.params[:4]
            rec.cameras[cid] = pycolmap.Camera(model="OPENCV", width=cam.width, height=cam.height, params=[fx, fy, cx, cy, 0, 0, 0, 0], camera_id=cid)
    o = pycolmap.BundleAdjustmentOptions()
    o.refine_focal_length = variant in ("f", "fpp", "opencv", "pf", "pfpp"); o.refine_principal_point = variant in ("fpp", "opencv", "pfpp")
    o.refine_extra_params = variant == "opencv"
    try:
        o.ceres.solver_options.num_threads = threads; o.ceres.solver_options.max_num_iterations = 100
    except AttributeError:
        pass
    t0 = time.time(); pycolmap.bundle_adjustment(rec, o)
    return rec, time.time() - t0


def cam_centre_rot(rec):
    out = {}
    for im in rec.images.values():
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        out[im.name] = (T.rotation.matrix(), T.translation)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--no_ba", action="store_true"); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--variants", default="pose,f,fpp,opencv"); a = ap.parse_args()
    sp = os.path.join(a.scene, "sparse", "0"); name = os.path.basename(a.scene.rstrip("/"))
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "outputs", "calib", name); out_dir = os.path.normpath(out_dir); os.makedirs(out_dir, exist_ok=True)
    rec = pycolmap.Reconstruction(sp); cam = next(iter(rec.cameras.values())); W, H = cam.width, cam.height
    names = sorted(im.name for im in rec.images.values()); test_names = set(names[::8])          # 3DGS LLFF hold: idx % 8 == 0 on sorted names
    test_ids = {iid for iid, im in rec.images.items() if im.name in test_names}
    O = observations(rec, test_ids)
    res = {"scene": name, "camera": [cam.model.name, W, H, [float(x) for x in cam.params]], "images": len(names), "test_images": len(test_ids),
           "observations": int(len(O)), "points3D": rec.num_points3D(), "released": {**stats(O, W, H), "field_fit": field_fit(O, W, H), "grid": grid_field(O, W, H)}}
    print(f"[{name}] {cam.model.name} {W}x{H} pp=({cam.params[2]:.1f},{cam.params[3]:.1f}) | {len(names)} img ({len(test_ids)} test) {len(O)} obs | "
          f"reproj median train {res['released']['median_train']:.3f} test {res['released']['median_test']:.3f} px | radial {['%.3f' % v for v in res['released']['radial_profile_median']]} | "
          f"field fit t=({res['released']['field_fit']['translation_px'][0]:+.3f},{res['released']['field_fit']['translation_px'][1]:+.3f}) radial {res['released']['field_fit']['radial_px_at_edge']:+.3f} "
          f"expl {res['released']['field_fit']['explained']*100:.1f}% | grid mean {res['released']['grid']['mean_cell_vector_px']:.3f}", flush=True)
    if not a.no_ba:
        base_cr = cam_centre_rot(rec); res["ba"] = {}
        for v in a.variants.split(","):
            r2, dt = run_ba(sp, v, a.threads)
            c2 = next(iter(r2.cameras.values())); O2 = observations(r2, test_ids); s2 = stats(O2, W, H)
            cr = cam_centre_rot(r2)
            # test-camera motion in px: rotation change * focal (orientation) — the part that moves the image
            f = cam.params[0]; dtest = [np.degrees(np.arccos(np.clip((np.trace(base_cr[n][0].T @ cr[n][0]) - 1) / 2, -1, 1))) * np.pi / 180 * f for n in test_names if n in cr]
            res["ba"][v] = {"seconds": dt, "camera": [c2.model.name, [float(x) for x in c2.params]], **s2, "field_fit": field_fit(O2, W, H),
                            "test_rotation_change_px_median": float(np.median(dtest)) if dtest else None}
            p = c2.params
            if v in ("pf", "pfpp"):
                fs = np.array([r2.cameras[im.camera_id].params[0] for im in r2.images.values()])
                res["ba"][v]["per_image_focal_rel_std"] = float(fs.std() / fs.mean()); res["ba"][v]["per_image_focal_rel_range"] = float((fs.max() - fs.min()) / fs.mean())
                print(f"[{name}] BA {v}: per-image focal rel std {fs.std() / fs.mean() * 1e4:.1f}e-4, range {(fs.max() - fs.min()) / fs.mean() * 1e4:.1f}e-4 "
                      f"(= {fs.std() / fs.mean() * W / 2:.2f} px at the image edge, full res)", flush=True)
            print(f"[{name}] BA {v:6s} {dt:5.0f}s | reproj median train {s2['median_train']:.3f} test {s2['median_test']:.3f} px | radial {['%.3f' % x for x in s2['radial_profile_median']]} | "
                  f"f=({p[0]:.1f},{p[1]:.1f}) pp=({p[2]:.2f},{p[3]:.2f}) d_pp=({p[2]-cam.params[2]:+.2f},{p[3]-cam.params[3]:+.2f}) px"
                  + (f" k=({p[4]:+.4f},{p[5]:+.4f}) p=({p[6]:+.5f},{p[7]:+.5f})" if len(p) > 4 else "")
                  + f" | test rot change {res['ba'][v]['test_rotation_change_px_median']:.2f} px", flush=True)
            if v == "fpp":
                d = os.path.join(out_dir, "sparse_fpp", "0"); os.makedirs(d, exist_ok=True); r2.write(d)
    json.dump(res, open(os.path.join(out_dir, "c1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""P1 — per-view sub-pixel correction estimated from neighbour warps (M1 probe), frozen Gaussians.

For each TRAIN view i (needs `render.py --skip_test --dump_warps` output in <model>/train/ours_30000/{gt,renders,warps,warpmask}):
  measurement  u_i(p): RAFT flow GT_i -> warp_k (fwd-bwd consistent, warp-valid), robustly fused over the K sources
               (signal = "warp"), or RAFT flow GT_i -> model render (signal = "render", CamP-like control);
  model        u(p) ~ J(p; depth) [dw; dt] + k1 * r^2 (p - c)   (6-DoF pose perturbation in the camera frame,
               optional radial term), solved by IRLS-Huber on a pixel subsample;
  output       per-view stats + a corrected COLMAP images.bin (pose only; test views untouched) in
               data/p1/<scene>_<signal>[_k1]/ for retraining with third_party/3dgs-mcmc (-r 1 on the eval-res images is
               NOT needed: we keep the original full-res images and only replace sparse/0/images.bin).

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/p1_estimate_view_corrections.py bicycle --signal warp [--k1]
"""
import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_misalignment import dev, flow_fb, load  # noqa: E402
from e7_build_synthetic import qvec2rot, read_images_bin, rot2qvec, write_images_bin  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def read_cam(p):
    with open(p, "rb") as f:
        f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
        npar = {0: 3, 1: 4}[model]; par = struct.unpack("<" + "d" * npar, f.read(8 * npar))
    if model == 0:
        fx = fy = par[0]; cx, cy = par[1], par[2]
    else:
        fx, fy, cx, cy = par
    return fx, fy, cx, cy, w, h


def fuse_flows(flows, oks):
    """robust per-pixel fusion over sources: median of the consistent ones; mask = at least 2 sources agree within 0.5 px."""
    F_ = torch.stack(flows)                         # S,1,2,H,W
    M = torch.stack([o.float() for o in oks])       # S,H,W
    big = torch.where(M[:, None, None] > 0, F_, torch.full_like(F_, float("nan")))
    med = torch.nanmedian(big, dim=0).values         # 1,2,H,W
    n = M.sum(0)
    agree = ((F_ - med).norm(dim=2)[:, 0] < 0.5).float() * M
    ok = (n >= 1) & (agree.sum(0) >= torch.clamp(n, max=2))
    return med, ok


def fit_view(flow, ok, depth, fx, fy, cx, cy, use_k1, iters=8, step=3):
    """IRLS-Huber fit of u = J [dw;dt] (+ k1 radial). Camera frame: x right, y down, z forward (COLMAP)."""
    _, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    m = ok & (depth > 1e-3) & torch.isfinite(depth)
    sel = torch.zeros_like(m); sel[::step, ::step] = True; m = m & sel
    x = (xx[m] - cx) / fx; y = (yy[m] - cy) / fy; Z = depth[m]
    u = flow[0, 0][m]; v = flow[0, 1][m]
    # image Jacobian of a point (x,y,1)*Z under small camera motion (dw rotation, dt translation), in pixels
    Jx = torch.stack([-fx * x * y, fx * (1 + x * x), -fx * y, fx / Z, torch.zeros_like(Z), -fx * x / Z], 1)
    Jy = torch.stack([-fy * (1 + y * y), fy * x * y, fy * x, torch.zeros_like(Z), fy / Z, -fy * y / Z], 1)
    if use_k1:
        r2 = x * x + y * y
        Jx = torch.cat([Jx, (fx * x * r2)[:, None]], 1); Jy = torch.cat([Jy, (fy * y * r2)[:, None]], 1)
    A = torch.cat([Jx, Jy], 0); b = torch.cat([u, v], 0)
    w = torch.ones_like(b)
    for _ in range(iters):
        Aw = A * w[:, None]; theta = torch.linalg.lstsq(Aw, b * w).solution
        r = (A @ theta - b).view(2, -1).norm(dim=0); r = torch.cat([r, r])
        c = 0.5; w = torch.where(r < c, torch.ones_like(r), c / r).sqrt()
    resid = (A @ theta - b).view(2, -1).norm(dim=0)
    rms_in = b.view(2, -1).norm(dim=0).pow(2).mean().sqrt()
    return theta.cpu().numpy(), float(rms_in), float(resid.pow(2).mean().sqrt()), int(m.sum())


def apply_pose(q, t, dw, dt):
    """new camera = small motion applied in the camera frame: X_c' = R(dw) X_c + dt  =>  R' = R(dw) R,  t' = R(dw) t + dt.
    The fitted flow says GT content sits at p + u relative to the model; moving the camera by (dw,dt) moves the
    rendered content by +J[dw;dt], so applying (dw,dt) makes the render follow the GT."""
    R = qvec2rot(q); dR = rot_exp(dw)
    return rot2qvec(dR @ R), dR @ t + dt


def rot_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--signal", choices=["warp", "render"], default="warp"); ap.add_argument("--k1", action="store_true")
    ap.add_argument("--model", default=None, help="model dir with train/ours_30000 dumps (default outputs/protocolR/ibgs/<scene>_pre)")
    ap.add_argument("--res", type=int, default=None)
    a = ap.parse_args()
    scene = a.scene
    src = ROOT / "data" / "mipnerf360" / scene
    model = Path(a.model) if a.model else ROOT / "outputs" / "protocolR" / "ibgs" / f"{scene}_pre"
    d = model / "train" / "ours_30000"
    res = a.res or (2 if scene in ("bonsai", "counter", "kitchen", "room") else 4)
    fx, fy, cx, cy, W0, H0 = read_cam(src / "sparse" / "0" / "cameras.bin")
    fx, fy, cx, cy = fx / res, fy / res, cx / res, cy / res
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    ims = read_images_bin(src / "sparse" / "0" / "images.bin")
    names = sorted(im[4] for im in ims); test = {n for i, n in enumerate(names) if i % 8 == 0}
    stats, corr = [], {}
    for iid, q, t, cid, name, np2, pts in ims:
        stem = os.path.splitext(name)[0]
        if name in test or not (d / "gt" / f"{stem}.png").exists():
            continue
        gt = load(d / "gt" / f"{stem}.png")
        depth = torch.from_numpy(np.load(d / "warpmask" / f"{stem}_depth.npy")).to(dev)
        if a.signal == "warp":
            flows, oks = [], []
            for k in range(4):
                wp = d / "warps" / f"{stem}_s{k}.png"
                if not wp.exists():
                    continue
                w = load(wp); mk = (load(d / "warpmask" / f"{stem}_s{k}.png")[:1] > 0.5)[0].to(dev)
                f, ok = flow_fb(raft, gt, w); flows.append(f); oks.append(ok & mk)
            if not flows:
                continue
            flow, ok = fuse_flows(flows, oks)
        else:
            rp = [p for p in os.listdir(d / "renders") if os.path.splitext(p)[0] == stem][0]
            flow, ok = flow_fb(raft, gt, load(d / "renders" / rp))
        theta, rms_in, rms_out, npx = fit_view(flow, ok, depth, fx, fy, cx, cy, a.k1)
        dw, dt = theta[:3], theta[3:6]
        shift_px = float(np.linalg.norm([fx * dw[1], fy * dw[0]]))   # dominant image shift from rotation
        stats.append({"name": name, "px": npx, "flow_rms_px": rms_in, "resid_rms_px": rms_out, "rot_shift_px": shift_px,
                      "dt_norm": float(np.linalg.norm(dt)), "k1": float(theta[6]) if a.k1 else None})
        corr[name] = (dw, dt)
        print(f"{name} px={npx} flow_rms={rms_in:.3f} -> resid={rms_out:.3f} rot_shift={shift_px:.3f}px |dt|={np.linalg.norm(dt):.2e}", flush=True)
    tag = f"{scene}_{a.signal}{'_k1' if a.k1 else ''}"
    summ = {k: float(np.mean([s[k] for s in stats])) for k in ("flow_rms_px", "resid_rms_px", "rot_shift_px", "dt_norm")}
    summ["explained_var"] = 1 - np.mean([s["resid_rms_px"] ** 2 for s in stats]) / np.mean([s["flow_rms_px"] ** 2 for s in stats])
    summ["views"] = len(stats)
    out = ROOT / "reports" / "p1"; out.mkdir(exist_ok=True)
    json.dump({"summary": summ, "views": stats}, open(out / f"{tag}.json", "w"), indent=1)
    print("SUMMARY", tag, json.dumps(summ))
    # corrected scene for retraining: original images + sparse with corrected train poses
    dst = ROOT / "data" / "p1" / tag; (dst / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for f in ("cameras.bin", "points3D.bin"):
        shutil.copy(src / "sparse" / "0" / f, dst / "sparse" / "0" / f)
    for sub in ("images", "images_2", "images_4", "images_8"):
        if (src / sub).exists() and not (dst / sub).exists():
            os.symlink((src / sub).resolve(), dst / sub)
    new = []
    for iid, q, t, cid, name, np2, pts in ims:
        if name in corr:
            dw, dt = corr[name]; q, t = apply_pose(q, t, dw, dt)
        new.append([iid, q, t, cid, name, np2, pts])
    write_images_bin(dst / "sparse" / "0" / "images.bin", new)
    print("wrote", dst)


if __name__ == "__main__":
    main()

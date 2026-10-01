"""Single-model feasibility: is the EXPECTED depth of 3DGS-MCMC (alpha-blended camera z, via override colours) consistent
enough across views for depth-tested warping? The IBGS kernel's plane-intersection depth assumes flat (PGSR) Gaussians;
on MCMC it left 12% of bonsai pixels without any valid source (3% with IBGS geometry).
For each test view: depth of the target and of its 4 nearest train cameras (IBGS rule: distance-sorted, angle filter),
backward reprojection, relative depth test |D_s(p') - z'| / z' < tau; reports the fraction of target pixels (with
alpha > 0.5) that no source validates, and the mean number of valid sources.

    cd third_party/3dgs-mcmc && python ../../src/route/mcmc_depth_check.py -s <scene> -m <mcmc model> -r <res> --eval [--max_angle 30 --max_dis 1.5]
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "3dgs-mcmc"))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402


def depth(cam, g, pipe, bg):
    Rt = cam.world_view_transform                                                       # 4x4 (transposed w2c)
    z = (torch.cat([g.get_xyz, torch.ones_like(g.get_xyz[:, :1])], 1) @ Rt)[:, 2:3]
    col = torch.cat([z, torch.ones_like(z), torch.zeros_like(z)], 1)
    o = render(cam, g, pipe, bg, override_color=col)["render"]
    return o[0] / o[1].clamp_min(1e-6), o[1]


def K_of(cam):
    W, H = cam.image_width, cam.image_height
    fx = W / (2 * math.tan(cam.FoVx / 2)); fy = H / (2 * math.tan(cam.FoVy / 2))
    return fx, fy, W / 2, H / 2


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--max_angle", type=float, default=30)
    parser.add_argument("--max_dis", type=float, default=1.5); parser.add_argument("--n_src", type=int, default=4)
    parser.add_argument("--max_views", type=int, default=12); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    ds = model.extract(args); g = GaussianModel(ds.sh_degree); pipe = pipeline.extract(args)
    with torch.no_grad():
        scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda")
        tr = scene.getTrainCameras(); te = scene.getTestCameras()[: args.max_views]
        C = torch.stack([c.camera_center for c in tr]); fw = torch.stack([torch.tensor(c.R, dtype=torch.float32)[:, 2] for c in tr]).cuda()
        cache = {}
        res = {tau: [] for tau in (0.01, 0.02, 0.05)}; nvs = {tau: [] for tau in res}
        for cam in te:
            Dt, At = depth(cam, g, pipe, bg); H, W = Dt.shape
            fx, fy, cx, cy = K_of(cam)
            dis = (C - cam.camera_center).norm(dim=1); ang = torch.rad2deg(torch.arccos((fw @ torch.tensor(cam.R, dtype=torch.float32)[:, 2].cuda()).clamp(-1, 1)))
            order = torch.argsort(dis); order = [int(i) for i in order if ang[i] < args.max_angle and 0.01 < dis[i] < args.max_dis][: args.n_src]
            v, u = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
            Xc = torch.stack([(u + 0.5 - cx) / fx * Dt, (v + 0.5 - cy) / fy * Dt, Dt, torch.ones_like(Dt)], -1).view(-1, 4)
            Xw = Xc @ torch.inverse(cam.world_view_transform)                               # row vectors
            cnt = {tau: torch.zeros(H * W, device="cuda") for tau in res}
            for si in order:
                s = tr[si]
                if si not in cache: cache[si] = depth(s, g, pipe, bg)[0]
                Ds = cache[si]; sfx, sfy, scx, scy = K_of(s)
                Xs = Xw @ s.world_view_transform; zs = Xs[:, 2]
                us = Xs[:, 0] / zs * sfx + scx - 0.5; vs = Xs[:, 1] / zs * sfy + scy - 0.5
                inb = (zs > 0.01) & (us >= 0) & (us <= Ds.shape[1] - 1) & (vs >= 0) & (vs <= Ds.shape[0] - 1)
                grid = torch.stack([us / (Ds.shape[1] - 1) * 2 - 1, vs / (Ds.shape[0] - 1) * 2 - 1], -1).view(1, 1, -1, 2)
                d = F.grid_sample(Ds[None, None], grid, align_corners=True).view(-1)
                for tau in res:
                    cnt[tau] += (inb & (d > 0) & ((d - zs).abs() / zs < tau)).float()
            m = (At > 0.5).view(-1)
            for tau in res:
                res[tau].append(float((cnt[tau][m] == 0).float().mean())); nvs[tau].append(float(cnt[tau][m].mean()))
        for tau in res:
            print(f"[mcmc expected depth] tau {tau}: unsupported frac {np.mean(res[tau]):.3f}  mean valid sources {np.mean(nvs[tau]):.2f} / {args.n_src}  ({len(te)} views)")


if __name__ == "__main__":
    main()

"""Render TEST views applying the camera-common phase field EXACTLY through the camera (no resampling of the
render): the field's diagonal-affine part (scale -> focal/FoV, translation -> off-centre principal point) becomes a
modified projection matrix; only the small residual (rotation/shear/higher order) is applied by bicubic resampling.

Convention (as in training): output(p) = render(p + f(p)),  f(p) = c + L (p - p0)/S  (affine fit in normalised coords).
=> output pixel p shows the old camera's pixel G p + g with G = I + diag(L)/S, g = c - diag(L) p0/S,
   i.e. new intrinsics fx' = fx/Gxx, cx' = (cx - gx)/Gxx  (same for y).

    cd third_party/3dgs-mcmc && python ../../src/phase/render_shared_exact.py -s <scene> -m <model> -r <res> [--residual]
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "3dgs-mcmc"))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def proj_offcentre(znear, zfar, fovX, fovY, dx_ndc, dy_ndc):
    t = math.tan(fovY / 2) * znear; r = math.tan(fovX / 2) * znear
    P = torch.zeros(4, 4)
    P[0, 0] = znear / r; P[1, 1] = znear / t
    P[0, 2] = dx_ndc; P[1, 2] = dy_ndc
    P[3, 2] = 1.0; P[2, 2] = zfar / (zfar - znear); P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--residual", action="store_true")
    parser.add_argument("--out_sub", default="ours_30000_ex"); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser); safe_state(True)
    j = json.load(open(os.path.join(args.model_path, "phase_shared.json"))); terms = [tuple(t) for t in j["terms"]]; A = torch.tensor(j["A"], dtype=torch.float32)
    dataset = model.extract(args); gaussians = GaussianModel(dataset.sh_degree)
    with torch.no_grad():
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        out_dir = os.path.join(args.model_path, "test", args.out_sub, "renders"); os.makedirs(out_dir, exist_ok=True)
        gt_link = os.path.join(args.model_path, "test", args.out_sub, "gt")
        if not os.path.exists(gt_link):
            os.symlink(os.path.abspath(os.path.join(args.model_path, "test", "ours_30000", "gt")), gt_link)
        cams = scene.getTestCameras()
        W, H = cams[0].image_width, cams[0].image_height
        # affine fit of the shared field in normalised coords (same basis as training)
        yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij"); xn, yn = xx / W - .5, yy / H - .5
        B = torch.stack([xn ** i * yn ** j for i, j in terms], -1); f = B @ A                        # H W 2 (px)
        X = torch.stack([xn, yn, torch.ones_like(xn)], -1).reshape(-1, 3); Aff = torch.linalg.lstsq(X, f.reshape(-1, 2)).solution
        Lxx, Lyy = float(Aff[0, 0]), float(Aff[1, 1]); cx0, cy0 = float(Aff[2, 0]), float(Aff[2, 1])   # u = cx0 + Lxx*xn (+Lxy*yn), v = cy0 (+...) + Lyy*yn
        Gxx, Gyy = 1 + Lxx / W, 1 + Lyy / H
        gx, gy = cx0 - Lxx * 0.5, cy0 - Lyy * 0.5                      # xn = px/W - 0.5  ->  L*xn = (L/W) px - L/2
        fdiag = torch.stack([cx0 + Lxx * xn, cy0 + Lyy * yn], -1)       # the part rendered exactly
        resid = f - fdiag
        print(f"[exact] G=({Gxx:.5f},{Gyy:.5f}) g=({gx:+.3f},{gy:+.3f}) px | residual field mean {float(resid.norm(dim=-1).mean()):.3f} px "
              f"(of {float(f.norm(dim=-1).mean()):.3f})", flush=True)
        gxg = (xx + resid[..., 0]) / (W - 1) * 2 - 1; gyg = (yy + resid[..., 1]) / (H - 1) * 2 - 1; grid = torch.stack([gxg, gyg], -1)[None].cuda()
        for idx, cam in enumerate(cams):
            fx = W / (2 * math.tan(cam.FoVx / 2)); fy = H / (2 * math.tan(cam.FoVy / 2))
            fx2, fy2 = fx / Gxx, fy / Gyy
            cam.FoVx = 2 * math.atan(W / (2 * fx2)); cam.FoVy = 2 * math.atan(H / (2 * fy2))
            # principal point: default sits at (W-1)/2 (index space). new pp index = (pp - g)/G  ->  offset from default
            dpx = ((W - 1) / 2 - gx) / Gxx - (W - 1) / 2; dpy = ((H - 1) / 2 - gy) / Gyy - (H - 1) / 2
            P = proj_offcentre(cam.znear, cam.zfar, cam.FoVx, cam.FoVy, 2 * dpx / W, 2 * dpy / H).transpose(0, 1).cuda()
            cam.projection_matrix = P
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(P.unsqueeze(0))).squeeze(0)
            img = render(cam, gaussians, pipeline.extract(args), bg)["render"][None]
            if args.residual:
                img = F.grid_sample(img, grid, mode="bicubic", padding_mode="border", align_corners=True)
            torchvision.utils.save_image(img[0].clamp(0, 1), os.path.join(out_dir, f"{idx:05d}.png"))
        print(f"[exact] wrote {len(cams)} renders -> {out_dir}")


if __name__ == "__main__":
    main()

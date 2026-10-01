"""Band-limited warping — image evidence per pyramid level with a level-dependent geometry tolerance (own method,
single explicit model: 3DGS-MCMC gives both the colour render and the depth; no IBR checkpoint involved).

Theory: a depth error dz misregisters a warp by delta ~ f*b*dz/z^2 px, which only destroys content at wavelengths
<= delta. Level l of a Gaussian pyramid has wavelength ~ 2^(l+1) px, so the depth-consistency test may be relaxed as
tau_l = tau0 * 2^l: coarse levels obtain dense, valid evidence from geometry that fails a full-resolution test.

For every target view (test views of the model's data dir, i.e. official test or cross-fitting dev views):
  render MCMC colour I and expected depth D (alpha-normalised z via override colours) of the target and of its K
  nearest train cameras (angle filter); for every level l: warp the source's Gaussian level l with the target depth
  at level l, depth test at tau_l, average the valid sources -> evidence G_l(E) and support count n_l.
Writes <out>/<name>.npz: gt, I (8-bit like the metric), E_l (l=0..L-1, fp16), n_l (uint8), alpha.

    cd third_party/3dgs-mcmc && python ../../src/route/bandwarp.py -s <data> -m <mcmc model> -r <res> --eval --out <dir>
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


def q8(x):
    return (x.clamp(0, 1) * 255 + 0.5).floor() / 255


def render_cd(cam, g, pipe, bg):
    """colour, expected depth, alpha."""
    col = render(cam, g, pipe, bg)["render"]
    z = (torch.cat([g.get_xyz, torch.ones_like(g.get_xyz[:, :1])], 1) @ cam.world_view_transform)[:, 2:3]
    o = render(cam, g, pipe, bg, override_color=torch.cat([z, torch.ones_like(z), torch.zeros_like(z)], 1))["render"]
    return col, o[0] / o[1].clamp_min(1e-6), o[1]


def K_of(cam):
    W, H = cam.image_width, cam.image_height
    return W / (2 * math.tan(cam.FoVx / 2)), H / (2 * math.tan(cam.FoVy / 2)), W / 2, H / 2


def gpyr(x, L):
    P = [x]
    for _ in range(L - 1):
        P.append(F.avg_pool2d(P[-1], 2, ceil_mode=True))
    return P


def depth_pyr(D, A, L):
    """alpha-weighted depth pyramid (avoids mixing background into edge pixels)."""
    DA, AA = gpyr((D * A)[None, None], L), gpyr(A[None, None], L)
    return [(da / aa.clamp_min(1e-6))[0, 0] for da, aa in zip(DA, AA)], [aa[0, 0] for aa in AA]


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_src", type=int, default=6); parser.add_argument("--max_angle", type=float, default=45)
    parser.add_argument("--L", type=int, default=5); parser.add_argument("--tau0", type=float, default=0.01)
    parser.add_argument("--out", required=True); parser.add_argument("--relax", type=float, default=2.0, help="tau_l = tau0 * relax^l (1 = fixed)")
    parser.add_argument("--max_views", type=int, default=0)
    args = get_combined_args(parser)
    ds = model.extract(args); g = GaussianModel(ds.sh_degree); pipe = pipeline.extract(args)
    os.makedirs(args.out, exist_ok=True)
    L = args.L
    with torch.no_grad():
        scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda")
        tr = scene.getTrainCameras(); te = scene.getTestCameras()
        C = torch.stack([c.camera_center for c in tr]); fw = torch.stack([torch.tensor(c.R, dtype=torch.float32)[:, 2] for c in tr]).cuda()
        cache = {}
        if args.max_views: te = te[: args.max_views]
        for vi, cam in enumerate(te):
            I, D, A = render_cd(cam, g, pipe, bg); H, W = D.shape
            fx, fy, cx, cy = K_of(cam)
            dis = (C - cam.camera_center).norm(dim=1)
            ang = torch.rad2deg(torch.arccos((fw @ torch.tensor(cam.R, dtype=torch.float32)[:, 2].cuda()).clamp(-1, 1)))
            order = [int(i) for i in torch.argsort(dis) if ang[i] < args.max_angle and dis[i] > 1e-4][: args.n_src]
            Dp, Ap = depth_pyr(D, A, L)
            acc = [torch.zeros(3, *Dp[l].shape, device="cuda") for l in range(L)]
            cnt = [torch.zeros(*Dp[l].shape, device="cuda") for l in range(L)]
            winv = torch.inverse(cam.world_view_transform)
            for si in order:
                s = tr[si]
                if si not in cache:
                    _, Ds, As = render_cd(s, g, pipe, bg)
                    cache[si] = (gpyr(s.original_image.cuda()[None], L), depth_pyr(Ds, As, L))
                Sp, (Dsp, Asp) = cache[si]
                sfx, sfy, scx, scy = K_of(s)
                for l in range(L):
                    h, w = Dp[l].shape; sc = 2 ** l
                    v, u = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
                    z = Dp[l]
                    Xc = torch.stack([(u + 0.5 - cx / sc) / (fx / sc) * z, (v + 0.5 - cy / sc) / (fy / sc) * z, z, torch.ones_like(z)], -1).view(-1, 4)
                    Xs = (Xc @ winv) @ s.world_view_transform; zs = Xs[:, 2]
                    hs, ws = Dsp[l].shape
                    us = Xs[:, 0] / zs * (sfx / sc) + scx / sc - 0.5; vs = Xs[:, 1] / zs * (sfy / sc) + scy / sc - 0.5
                    inb = (zs > 1e-3) & (us >= 0) & (us <= ws - 1) & (vs >= 0) & (vs <= hs - 1)
                    grid = torch.stack([us / max(ws - 1, 1) * 2 - 1, vs / max(hs - 1, 1) * 2 - 1], -1).view(1, 1, -1, 2)
                    ds_ = F.grid_sample(Dsp[l][None, None], grid, align_corners=True).view(-1)
                    as_ = F.grid_sample(Asp[l][None, None], grid, align_corners=True).view(-1)
                    col = F.grid_sample(Sp[l], grid, align_corners=True).view(3, -1)
                    tau = min(args.tau0 * args.relax ** l, 0.25)
                    ok = inb & (as_ > 0.5) & ((ds_ - zs).abs() / zs.clamp_min(1e-6) < tau) & (Ap[l].view(-1) > 0.5)
                    acc[l] += (col * ok).view(3, h, w); cnt[l] += ok.view(h, w).float()
            Ip = gpyr(I[None], L)
            E = [torch.where(cnt[l] > 0, acc[l] / cnt[l].clamp_min(1), Ip[l][0]) for l in range(L)]
            gt = q8(cam.original_image.cuda())
            rec = {"gt": gt.half().cpu().numpy(), "I": q8(I).half().cpu().numpy(), "alpha": A.half().cpu().numpy()}
            for l in range(L):
                rec[f"E{l}"] = E[l].half().cpu().numpy(); rec[f"n{l}"] = cnt[l].to(torch.uint8).cpu().numpy()
            np.savez_compressed(os.path.join(args.out, f"{cam.image_name}.npz"), **rec)
            sup = [float((cnt[l] > 0).float().mean()) for l in range(L)]
            print(f"[bw] {vi + 1}/{len(te)} {cam.image_name} support per level (fine->coarse) " + " ".join(f"{x:.3f}" for x in sup), flush=True)


if __name__ == "__main__":
    main()

"""R0 — oracle probe for reference-view routing on the IBGS released checkpoints (week-2 plan §4.2).

Same Gaussians, same in-kernel warper (per pixel: the first valid sources in the given order, depth test), same
aggregation network; only the SOURCE SET handed to the kernel changes. For every target view:
  pool    = IBGS neighbour ordering (distance-sorted, angle/distance filter) widened to --pool candidates
  subsets = every subset of the pool of size 1..--kmax (kept in pool order), + IBGS default (its own first 4 of 8),
            + nearest-5 (the kernel's MAX_M), + K=0 (raw Gaussian render, no residual)
and records per-patch squared error / SSIM / LPIPS(vgg, spatial) for each, plus per-source patch features
(valid fraction, |warp - raw|, baseline angle/distance). Images go through the same 8-bit save/load as IBGS render.py
+ metrics.py, and source images through the same JPEG round trip ("test_time_data").

    cd src/irgs && ../../.venv_ibgs/bin/python ../route/r0_probe.py -s <scene> -m <ibgs model> <ibgs flags> --out <npz>
"""
import itertools
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src", "irgs"))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args  # noqa: E402
from color_aggregation_network import ColorFusionResidualNet, fuse_color  # noqa: E402
from gaussian_renderer import render, render_depth  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.loss_utils import create_window  # noqa: E402

DEFAULT_DEPTH_TYPE = "median_intersected_depth"


def q8(x):  # torchvision.save_image + reload: clamp, round to 8 bit
    return (x.clamp(0, 1) * 255 + 0.5).floor().clamp(0, 255) / 255


def ssim_map(a, b, window):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu1 = F.conv2d(a, window, padding=5, groups=3); mu2 = F.conv2d(b, window, padding=5, groups=3)
    s11 = F.conv2d(a * a, window, padding=5, groups=3) - mu1 ** 2
    s22 = F.conv2d(b * b, window, padding=5, groups=3) - mu2 ** 2
    s12 = F.conv2d(a * b, window, padding=5, groups=3) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return m[0].mean(0)


def candidate_pool(view, scene, args, n):
    """IBGS gaussian_renderer.render neighbour rule, returning the first n candidates (exposure reorder included)."""
    wvt = view.world_view_transform.T
    R = torch.tensor(view.R).float()
    center_ray = torch.tensor([0.0, 0.0, 1.0], device="cuda").float() @ R.cuda().transpose(-1, -2)
    dis = torch.norm(view.camera_center.unsqueeze(0) - scene.camera_centers, dim=-1).cpu().numpy()
    ang = (torch.arccos(torch.sum(center_ray.unsqueeze(0) * scene.center_rays, dim=-1).clamp(-1, 1)) * 180 / np.pi).cpu().numpy()
    order = np.lexsort((ang, dis))
    keep = (ang[order] < args.multi_view_max_angle) & (dis[order] > args.multi_view_min_dis) & (dis[order] < args.multi_view_max_dis)
    order = order[keep][:n].tolist()
    if args.enable_exposure_correction and order:
        rel = torch.matmul(wvt.unsqueeze(0), torch.inverse(scene.world_view_transforms))
        diff = torch.mean(torch.abs(rel - torch.eye(4, device="cuda").unsqueeze(0)), dim=[1, 2]).cpu().numpy()
        first = order[int(np.argmin(diff[order]))]
        order.remove(first); order = [first] + order
    return order, dis, ang


def main():
    parser = ArgumentParser(); model = ModelParams(parser); op = OptimizationParams(parser); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--src_image_ext", default="jpg"); parser.add_argument("--render_geo", action="store_true")
    parser.add_argument("--pool", type=int, default=10); parser.add_argument("--kmax", type=int, default=3)
    parser.add_argument("--patch", type=int, default=32); parser.add_argument("--lpips", type=int, default=1)
    parser.add_argument("--max_views", type=int, default=0); parser.add_argument("--out", required=True)
    args = get_combined_args(parser)
    seed = 678 if "kitchen" in args.model_path else 22
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    dataset, opt, pipe = model.extract(args), op.extract(args), pipeline.extract(args)
    args.shuffle_source_frame = False
    lp = None
    if args.lpips:
        import lpips
        lp = lpips.LPIPS(net="vgg", spatial=True).cuda().eval()
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, args, load_iteration=args.iteration, shuffle=False)
        train = scene.getTrainCameras(); tests = scene.getTestCameras()
        H0, W0 = train[0].image_height, train[0].image_width
        net = ColorFusionResidualNet(height=int(H0 * opt.residual_resolution_scale), width=int(W0 * opt.residual_resolution_scale),
                                     feat_aggregate_mode=opt.feat_aggregate_mode).cuda()
        net.load_state_dict(torch.load(f"{args.model_path}/color_aggregate_checkpoint/{args.iteration}/color_aggregation_network.pth")); net.eval()
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        kw = dict(learnt_normal=opt.learnt_normal, buffer_length=opt.buffer_length, depth_error_threshold=opt.depth_error_threshold, app_model=None)
        # source images: the JPEG round trip IBGS render.py applies before the test pass
        tmp = os.path.join(os.path.dirname(args.out), "srcjpg_" + os.path.basename(args.out).replace(".npz", "")); os.makedirs(tmp, exist_ok=True)
        for i, v in enumerate(train):
            f = f"{tmp}/{v.image_name}.{args.src_image_ext}"; torchvision.utils.save_image(v.original_image, f)
            g = plt.imread(f); g = g / 255.0 if g.max() > 1.0 else g
            scene.original_image_list[i] = v.original_image = torch.from_numpy(g).float().permute(2, 0, 1).cuda()
        # source depths once (identical to the per-call render_depth of the test path)
        for i, v in enumerate(train):
            scene.rendered_depth_list[i] = render_depth(v, gaussians, scene, pipe, args, bg, opt.learnt_normal, 1, opt.buffer_length, opt.depth_error_threshold).view(1, H0, W0)
        win = create_window(11, 3).cuda()
        P = args.patch
        views = tests if not args.max_views else tests[: args.max_views]
        rec = {"names": [], "pool": [], "ibgs_default": [], "subsets": None}
        per_view = []
        t0 = time.time()
        for vi, view in enumerate(views):
            gt = q8(view.original_image.cuda()); _, H, W = gt.shape
            Hc, Wc = (H // P) * P, (W // P) * P
            pool, dis, ang = candidate_pool(view, scene, args, args.pool)
            dflt, _, _ = candidate_pool(view, scene, args, args.multi_view_num); dflt = dflt[: opt.number_src_frames]
            subs = [()]  # K=0
            for k in range(1, args.kmax + 1):
                subs += list(itertools.combinations(range(len(pool)), k))
            subs.append(tuple(range(min(5, len(pool)))))             # nearest-5 (kernel MAX_M = 5 sources)
            entries = [[pool[j] for j in s] for s in subs] + [dflt]   # last = IBGS default
            sse, ssm, lpm, img_mse = [], [], [], []
            feats = None
            for ei, src in enumerate(entries):
                view.nearest_id = src
                out = render(view, gaussians, scene, pipe, args, bg, nb_src_frames=max(1, len(src)), do_find_closest_frame=False,
                             do_render_src_depth=False, render_geo=True, return_depth_normal=False, **kw)
                img = out["render"]
                if len(src) > 0:
                    with (autocast() if opt.enable_mix_precision else nullcontext()):   # as render.py
                        fo = fuse_color(out, color_aggregation_network=net, iter_count=None, burn_start=None, burn_end=None, iteration=args.iteration, opts=opt)
                    if fo is not None:
                        img = fo["image_pred"].float()
                    if len(src) == 1:   # per-source patch features from the single-source pass
                        valid = (out["cam_feat"].view(-1, 4, H, W)[0].abs().sum(0) > 0).float()
                        wimg = out["warped_image"].view(-1, 3, H, W)[0]
                        dif = (wimg - out["render"]).abs().mean(0) * valid
                        vf = F.avg_pool2d(valid[:Hc, :Wc][None, None], P)[0, 0]
                        dm = F.avg_pool2d(dif[:Hc, :Wc][None, None], P)[0, 0] / vf.clamp_min(1e-6)
                        if feats is None: feats = np.zeros((len(pool), 2, Hc // P, Wc // P), np.float32)
                        j = subs[ei][0]; feats[j, 0] = vf.cpu().numpy(); feats[j, 1] = dm.cpu().numpy()
                img = q8(img)
                e = ((img - gt) ** 2).mean(0)
                img_mse.append(float(e.mean()))
                sse.append(F.avg_pool2d(e[:Hc, :Wc][None, None], P)[0, 0].cpu().numpy())
                ssm.append(F.avg_pool2d(ssim_map(img, gt, win)[:Hc, :Wc][None, None], P)[0, 0].cpu().numpy())
                if lp is not None:
                    l = lp(img.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1)[0, 0]
                    lpm.append(F.avg_pool2d(l[:Hc, :Wc][None, None], P)[0, 0].cpu().numpy())
            per_view.append(dict(mse=np.stack(sse).astype(np.float32), ssim=np.stack(ssm).astype(np.float16),
                                 lpips=np.stack(lpm).astype(np.float16) if lpm else np.zeros(1, np.float16),
                                 img_mse=np.array(img_mse, np.float64), feats=feats if feats is not None else np.zeros(1, np.float32),
                                 dis=dis[pool].astype(np.float32), ang=ang[pool].astype(np.float32),
                                 subs=np.array([",".join(map(str, x)) for x in subs])))
            rec["names"].append(view.image_name); rec["pool"].append(pool); rec["ibgs_default"].append(dflt)
            rec["subsets"] = [list(s) for s in subs]
            psnr = lambda m: -10 * np.log10(m)
            print(f"[r0] {vi + 1}/{len(views)} {view.image_name} pool={len(pool)} raw {psnr(img_mse[0]):.2f} ibgs {psnr(img_mse[-1]):.2f} "
                  f"best-subset {psnr(min(img_mse[1:-1])):.2f} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(args.out, names=np.array(rec["names"]), subsets=np.array([",".join(map(str, s)) for s in rec["subsets"]]),
                            pool=np.array([",".join(map(str, p)) for p in rec["pool"]]), ibgs_default=np.array([",".join(map(str, p)) for p in rec["ibgs_default"]]),
                            **{f"{k}_{i}": v for i, d in enumerate(per_view) for k, v in d.items()})
        print(f"[r0] wrote {args.out} ({len(views)} views)")


if __name__ == "__main__":
    main()

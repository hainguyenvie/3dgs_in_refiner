"""Per-view tensors for support-aware arbitration between an explicit model (3DGS-MCMC raw) and image-based evidence
(IBGS released checkpoint). For every TEST view, at the IBGS evaluation resolution, writes <out>/<name>.npz (fp16):
  gt, mcmc (MCMC raw render, resampled if its eval resolution differs), ibgs_raw, ibgs_final (authors' default sources),
  mcmc_res (IBGS residual network applied on the MCMC render instead of its own base — no retraining),
  feats: n_valid (valid warp slots, 0..5), min_depth_diff, warp_raw (mean |warp - ibgs_raw| over valid slots),
         warp_std (std across valid warps), resid (|final - raw|), mcmc_ibgs (|mcmc - ibgs_final|), depth
Images pass through the 8-bit quantisation of torchvision.save_image (as the metric does).

    cd src/irgs && ../../.venv_ibgs/bin/python ../route/dump_hybrid.py -s <scene> -m <ibgs model> <ibgs flags> --mcmc_dir <mcmc model>/test/ours_30000 --out <dir>
"""
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src", "irgs")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args  # noqa: E402
from color_aggregation_network import ColorFusionResidualNet, fuse_color  # noqa: E402
from gaussian_renderer import render, render_depth  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from r0_probe import q8, candidate_pool  # noqa: E402


def load_png(path, H, W):
    im = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).float().permute(2, 0, 1) / 255
    if im.shape[1:] != (H, W):
        im = F.interpolate(im[None], size=(H, W), mode="area")[0]
    return im.cuda()


def main():
    parser = ArgumentParser(); model = ModelParams(parser); op = OptimizationParams(parser); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--src_image_ext", default="jpg"); parser.add_argument("--render_geo", action="store_true")
    parser.add_argument("--mcmc_dir", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--geom_ply", default="", help="single-model variant: replace the IBGS Gaussians by this (MCMC) ply "
                        "for rendering, depth and warping; writes <name>.geo.npz with mcmc_g / mres_g / feats_g")
    args = get_combined_args(parser)
    seed = 678 if "kitchen" in args.model_path else 22
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
    dataset, opt, pipe = model.extract(args), op.extract(args), pipeline.extract(args)
    args.shuffle_source_frame = False
    os.makedirs(args.out, exist_ok=True)
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
        tmp = os.path.join(args.out, "_srcjpg"); os.makedirs(tmp, exist_ok=True)
        for i, v in enumerate(train):
            f = f"{tmp}/{v.image_name}.{args.src_image_ext}"; torchvision.utils.save_image(v.original_image, f)
            g = plt.imread(f); g = g / 255.0 if g.max() > 1.0 else g
            scene.original_image_list[i] = v.original_image = torch.from_numpy(g).float().permute(2, 0, 1).cuda()
        for i, v in enumerate(train):
            scene.rendered_depth_list[i] = render_depth(v, gaussians, scene, pipe, args, bg, opt.learnt_normal, 1, opt.buffer_length, opt.depth_error_threshold).view(1, H0, W0)
        if args.geom_ply:   # MCMC Gaussians through the IBGS rasterizer (normals = smallest axis; MCMC ply has no 'nd')
            from plyfile import PlyData, PlyElement
            pd = PlyData.read(args.geom_ply); el = pd.elements[0]
            if "nd" not in el.data.dtype.names:
                import numpy.lib.recfunctions as rf
                d = rf.append_fields(el.data, "nd", np.zeros(len(el.data), np.float32), usemask=False)
                tmp_ply = os.path.join(args.out, "_geom.ply"); PlyData([PlyElement.describe(d, "vertex")]).write(tmp_ply)
            else:
                tmp_ply = args.geom_ply
            gaussians = GaussianModel(3); gaussians.load_ply(tmp_ply); opt.learnt_normal = False   # MCMC: SH degree 3
            kw["learnt_normal"] = False
            for i, v in enumerate(train):
                scene.rendered_depth_list[i] = render_depth(v, gaussians, scene, pipe, args, bg, False, 1, opt.buffer_length, opt.depth_error_threshold).view(1, H0, W0)
        mc = sorted(os.listdir(os.path.join(args.mcmc_dir, "renders")))
        assert len(mc) == len(tests), f"MCMC renders {len(mc)} != test views {len(tests)}"
        amp = autocast() if opt.enable_mix_precision else nullcontext()
        for vi, view in enumerate(tests):
            gt = q8(view.original_image.cuda()); _, H, W = gt.shape
            mg = load_png(os.path.join(args.mcmc_dir, "gt", mc[vi]), H, W)
            gt_match = float(((mg - gt) ** 2).mean())        # alignment check of the two test orderings
            mcmc = q8(load_png(os.path.join(args.mcmc_dir, "renders", mc[vi]), H, W))
            dflt, _, _ = candidate_pool(view, scene, args, args.multi_view_num); dflt = dflt[: opt.number_src_frames]
            view.nearest_id = dflt
            out = render(view, gaussians, scene, pipe, args, bg, nb_src_frames=max(1, len(dflt)), do_find_closest_frame=False,
                         do_render_src_depth=False, render_geo=True, return_depth_normal=False, **kw)
            raw = out["render"]
            with amp:
                fo = fuse_color(out, color_aggregation_network=net, iter_count=None, burn_start=None, burn_end=None, iteration=args.iteration, opts=opt)
            final = fo["image_pred"].float() if fo is not None else raw
            out2 = dict(out); out2["render"] = mcmc
            with amp:
                fo2 = fuse_color(out2, color_aggregation_network=net, iter_count=None, burn_start=None, burn_end=None, iteration=args.iteration, opts=opt)
            mres = fo2["image_pred"].float() if fo2 is not None else mcmc
            cf = out["cam_feat"].view(-1, 4, H, W); wi = out["warped_image"].view(-1, 3, H, W)
            valid = (cf.abs().sum(1) > 0).float()                       # slots x H x W
            nv = valid.sum(0)
            wr = ((wi - raw[None]).abs().mean(1) * valid).sum(0) / nv.clamp_min(1)
            mu = (wi * valid[:, None]).sum(0) / nv.clamp_min(1)
            ws = ((((wi - mu[None]) ** 2).mean(1) * valid).sum(0) / nv.clamp_min(1)).sqrt()
            raw, final, mres = q8(raw), q8(final), q8(mres)
            feats = torch.stack([nv, out["min_depth_diff"].view(H, W), wr, ws, (final - raw).abs().mean(0), (mcmc - final).abs().mean(0),
                                 out["median_intersected_depth"].view(H, W)])
            h = lambda t: t.half().cpu().numpy()
            if args.geom_ply:   # raw = MCMC through IBGS rasterizer; mres = IBGS net on MCMC-geometry warps over the MCMC render
                np.savez_compressed(os.path.join(args.out, f"{view.image_name}.geo.npz"), mcmc_g=h(raw), mres_g=h(final), mres_gm=h(mres), feats_g=h(feats))
                ps = lambda a: float(-10 * torch.log10(((a - gt) ** 2).mean()))
                print(f"[geo] {vi + 1}/{len(tests)} {view.image_name} | mcmc(png) {ps(mcmc):.2f} mcmc(ibgs-rast) {ps(raw):.2f} "
                      f"res-on-own-render {ps(final):.2f} res-on-mcmc-png {ps(mres):.2f}", flush=True)
                continue
            np.savez_compressed(os.path.join(args.out, f"{view.image_name}.npz"), gt=h(gt), mcmc=h(mcmc), ibgs_raw=h(raw), ibgs_final=h(final),
                                mcmc_res=h(mres), feats=h(feats), src=np.array(dflt), gt_match=gt_match,
                                warps=h(wi * valid[:, None]), valid=valid.bool().cpu().numpy())   # per-slot warps (band disagreement)
            ps = lambda a: float(-10 * torch.log10(((a - gt) ** 2).mean()))
            print(f"[hyb] {vi + 1}/{len(tests)} {view.image_name} gtΔ {gt_match:.1e} | mcmc {ps(mcmc):.2f} ibgs_raw {ps(raw):.2f} "
                  f"ibgs {ps(final):.2f} mcmc+res {ps(mres):.2f} | oracle-px {float(-10 * torch.log10(torch.minimum(((mcmc - gt) ** 2).mean(0), ((final - gt) ** 2).mean(0)).mean())):.2f}", flush=True)


if __name__ == "__main__":
    main()

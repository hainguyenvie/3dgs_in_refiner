"""Gaussian-level failure analysis of a trained 3DGS(-MCMC) model at its TEST views (analysis only; uses GT).

Per Gaussian: max scale, anisotropy, opacity, view-dependent colour energy (norm of SH coefficients beyond DC), number of
training views whose frustum contains it, and — per test view — its *angular novelty*: the angle between the test ray to
the Gaussian and the closest training ray to it (among training views that see it). Attributes are splatted to the test
view (alpha-normalised override colours) and related to the per-pixel squared error. Also renders with SH degree 0
(DC only) to test whether view-dependent colour EXTRAPOLATES badly where the test direction was never observed.

Reports per scene: PSNR full vs DC-only vs per-pixel oracle(full, DC-only); share of squared error and DC-vs-full
difference by novelty decile, by training-view-count bins and by Gaussian-scale decile.

    cd third_party/3dgs-mcmc && python ../../src/route/gauss_diag.py -s <data> -m <model> -r <res> --eval [--max_views 10]
"""
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


def visibility(X, cams, chunk=200000):
    """bool N x V: Gaussian centre inside the view frustum (no occlusion test)."""
    out = []
    P = torch.stack([c.full_proj_transform for c in cams])                       # V x 4 x 4 (row-vector convention)
    for i in range(0, X.shape[0], chunk):
        Xh = torch.cat([X[i:i + chunk], torch.ones_like(X[i:i + chunk, :1])], 1)   # n x 4
        p = torch.einsum("nk,vkj->nvj", Xh, P)                                       # n x V x 4
        w = p[..., 3:4]; ndc = p[..., :2] / w.clamp_min(1e-6)
        out.append((w[..., 0] > 0.01) & (ndc.abs() < 1).all(-1))
    return torch.cat(out)


def splat(cam, g, pipe, bg, attrs):
    """alpha-normalised per-pixel average of up to two per-Gaussian attributes."""
    a = list(attrs) + [torch.zeros_like(attrs[0])] * (2 - len(attrs))
    col = torch.stack([a[0], a[1], torch.ones_like(a[0])], 1)
    o = render(cam, g, pipe, bg, override_color=col)["render"]
    return o[0] / o[2].clamp_min(1e-6), o[1] / o[2].clamp_min(1e-6), o[2]


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--max_views", type=int, default=10); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    ds = model.extract(args); g = GaussianModel(ds.sh_degree); pipe = pipeline.extract(args)
    with torch.no_grad():
        scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda")
        tr, te = scene.getTrainCameras(), scene.getTestCameras()[: args.max_views]
        X = g.get_xyz; N = X.shape[0]
        sc = g.get_scaling; smax = sc.max(1).values; aniso = smax / sc.min(1).values.clamp_min(1e-9)
        op = g.get_opacity[:, 0]
        sh_e = g._features_rest.flatten(1).norm(dim=1)                                # view-dependent colour energy
        V = visibility(X, tr)                                                          # N x Vtr
        nvis = V.float().sum(1)
        Ctr = torch.stack([c.camera_center for c in tr])                                # Vtr x 3
        print(f"[stats] N={N} | train views seeing a Gaussian: median {nvis.median():.0f}, share seen by <3 views {(nvis < 3).float().mean():.3f} | "
              f"opacity median {op.median():.3f} | max-scale p50/p99 {smax.median():.4f}/{torch.quantile(smax[:1000000], 0.99):.4f} | aniso p99 {torch.quantile(aniso[:1000000], 0.99):.1f} | "
              f"SH-rest energy p50/p99 {sh_e.median():.3f}/{torch.quantile(sh_e[:1000000], 0.99):.3f}", flush=True)
        acc = {k: [] for k in ("psnr_full", "psnr_dc", "psnr_or")}
        bins_nov = np.zeros((10, 3)); bins_vis = np.zeros((4, 3)); bins_sc = np.zeros((10, 3))   # [sum err full, sum err dc, count]
        for cam in te:
            gt = cam.original_image.cuda()
            full = render(cam, g, pipe, bg)["render"].clamp(0, 1)
            deg = g.active_sh_degree; g.active_sh_degree = 0
            dc = render(cam, g, pipe, bg)["render"].clamp(0, 1); g.active_sh_degree = deg
            ef = ((full - gt) ** 2).mean(0); ed = ((dc - gt) ** 2).mean(0)
            acc["psnr_full"].append(float(-10 * torch.log10(ef.mean()))); acc["psnr_dc"].append(float(-10 * torch.log10(ed.mean())))
            acc["psnr_or"].append(float(-10 * torch.log10(torch.minimum(ef, ed).mean())))
            # angular novelty per Gaussian for this test view
            dt = F.normalize(cam.camera_center[None] - X, dim=1)                          # N x 3
            nov = torch.empty(N, device="cuda")
            for i in range(0, N, 200000):
                d = F.normalize(Ctr[None] - X[i:i + 200000, None], dim=2)               # n x Vtr x 3
                cos = (d * dt[i:i + 200000, None]).sum(-1).masked_fill(~V[i:i + 200000], -1)
                nov[i:i + 200000] = torch.rad2deg(torch.arccos(cos.max(1).values.clamp(-1, 1)))
            m_nov, m_vis, A = splat(cam, g, pipe, bg, [nov / 90, torch.log1p(nvis) / 6])
            m_sc, _, _ = splat(cam, g, pipe, bg, [torch.log10(smax.clamp_min(1e-6)) / 6 + 1])
            ok = A > 0.5
            nv_deg = (m_nov * 90)[ok]; vis_c = torch.expm1(m_vis * 6)[ok]; scl = m_sc[ok]; e1 = ef[ok]; e2 = ed[ok]
            qn = torch.clamp((nv_deg / 3).long(), 0, 9)                                  # 3-degree bins up to 27+
            for b in range(10):
                mm = qn == b; bins_nov[b] += [float(e1[mm].sum()), float(e2[mm].sum()), float(mm.sum())]
            qv = torch.bucketize(vis_c, torch.tensor([3.0, 10.0, 30.0], device="cuda"))
            for b in range(4):
                mm = qv == b; bins_vis[b] += [float(e1[mm].sum()), float(e2[mm].sum()), float(mm.sum())]
            qs = torch.clamp(((scl - scl.min()) / (scl.max() - scl.min() + 1e-9) * 10).long(), 0, 9)
            for b in range(10):
                mm = qs == b; bins_sc[b] += [float(e1[mm].sum()), float(e2[mm].sum()), float(mm.sum())]
        print(f"[psnr] full {np.mean(acc['psnr_full']):.2f} | SH degree 0 (DC only) {np.mean(acc['psnr_dc']):.2f} | per-pixel oracle(full, DC) {np.mean(acc['psnr_or']):.2f} ({len(te)} views)")
        tot = bins_nov[:, 0].sum()
        def show(name, B, labels):
            print(f"[{name}] bin: pixel share | error share (full) | MSE full | MSE DC-only | DC better?")
            for b, lab in enumerate(labels):
                if B[b, 2] == 0: continue
                print(f"   {lab:>10s}: {B[b, 2] / B[:, 2].sum():.3f} | {B[b, 0] / tot:.3f} | {B[b, 0] / B[b, 2] * 1e4:7.2f} | {B[b, 1] / B[b, 2] * 1e4:7.2f} | {'YES' if B[b, 1] < B[b, 0] else ''}")
        show("novelty (deg)", bins_nov, [f"{3 * b}-{3 * b + 3}" if b < 9 else "27+" for b in range(10)])
        show("train views", bins_vis, ["<3", "3-10", "10-30", "30+"])
        show("scale decile", bins_sc, [f"d{b}" for b in range(10)])


if __name__ == "__main__":
    main()

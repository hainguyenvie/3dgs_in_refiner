"""Headroom of render <-> image-evidence fusion (analysis; uses GT). Per scene, per test view, PSNR of:
  render                 : the explicit render I alone
  oracle-cell(I, warps)  : per 8x8 cell pick the best of {I, valid warp slots} (IBGS geometry, hybrid dumps)
  oracle-cell(I, mean)   : per cell best of {I, mean of valid warps}
  oracle-cell(I, own)    : per cell best of {I, own band-limited evidence E0 (MCMC depth, bandwarp dumps)}
  oracle-aligned         : per cell best of {I, slot warps re-registered to GT by RAFT optical flow} — what perfect
                           registration (geometry/pose) would buy
  unsupported share      : share of the render's squared error that sits in pixels no source sees (IBGS geometry)
An 8x8 cell oracle (not per pixel) keeps the winner's curse small.

    .venv_ibgs/bin/python src/route/headroom.py bonsai counter garden bicycle train
"""
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = 8


def cell_mse(img, gt):
    e = ((img - gt) ** 2).mean(0, keepdim=True)[None]
    return F.avg_pool2d(e, C, ceil_mode=True)[0, 0]


def psnr_from_cells(cells, H, W):
    return float(-10 * torch.log10(cells.mean()))


def main():
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).cuda().eval()

    def align(src, gt):   # warp src onto gt with RAFT flow gt -> src (both 3xHxW in [0,1])
        H, W = gt.shape[-2:]; h8, w8 = (H // 8) * 8, (W // 8) * 8
        a = (gt[:, :h8, :w8][None] * 2 - 1); b = (src[:, :h8, :w8][None] * 2 - 1)
        fl = raft(a, b)[-1][0]                                                      # 2 x h8 x w8, flow from gt to src
        fl = F.pad(fl[None], (0, W - w8, 0, H - h8), mode="replicate")[0]
        yy, xx = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
        g = torch.stack([(xx + fl[0]) / (W - 1) * 2 - 1, (yy + fl[1]) / (H - 1) * 2 - 1], -1)[None]
        return F.grid_sample(src[None], g, align_corners=True, padding_mode="border")[0]

    print(f"{'scene':9s} | render | oracle(I,warps) oracle(I,mean) oracle(I,own) | oracle-aligned | unsup err share | IBGS")
    for s in sys.argv[1:]:
        fs = [f for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_r1", "*.npz"))) if not f.endswith(".geo.npz")]
        own = sorted(glob(os.path.join(ROOT, "outputs", "route", "bandwarp", f"{s}_own", "*.npz")))
        R = []
        with torch.no_grad():
            for vi, f in enumerate(fs):
                z = np.load(f); t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).cuda()
                gt, I, Ef = t(z["gt"]), t(z["mcmc"]), t(z["ibgs_final"]); W = t(z["warps"]); V = torch.from_numpy(z["valid"]).cuda()
                H, Wd = gt.shape[-2:]
                ci = cell_mse(I, gt)
                cands = [ci]; mean_c = [ci]; al = [ci]
                nv = V.float().sum(0)
                for k in range(W.shape[0]):
                    wk = torch.where(V[k][None], W[k], I)                              # invalid pixels fall back to I
                    cands.append(cell_mse(wk, gt))
                    if vi % 2 == 0 and V[k].float().mean() > 0.05:                     # RAFT on every other view
                        al.append(cell_mse(torch.where(V[k][None], align(W[k].clamp(0, 1), gt), I), gt))
                if W.shape[0]:
                    mu = torch.where(nv[None] > 0, (W * V[:, None]).sum(0) / nv.clamp_min(1)[None], I)
                    mean_c.append(cell_mse(mu, gt))
                own_c = [ci]
                if vi < len(own):
                    o = np.load(own[vi]); E0 = t(o["E0"]); own_c.append(cell_mse(E0, gt))
                e_pix = ((I - gt) ** 2).mean(0)
                unsup = float(e_pix[nv < 0.5].sum() / e_pix.sum())
                row = [psnr_from_cells(ci, H, Wd), psnr_from_cells(torch.stack(cands).min(0).values, H, Wd),
                       psnr_from_cells(torch.stack(mean_c).min(0).values, H, Wd), psnr_from_cells(torch.stack(own_c).min(0).values, H, Wd),
                       psnr_from_cells(torch.stack(al).min(0).values, H, Wd) if vi % 2 == 0 else np.nan, unsup,
                       float(-10 * torch.log10(((Ef - gt) ** 2).mean()))]
                R.append(row)
        R = np.array(R, dtype=np.float64); m = np.nanmean(R, 0)
        print(f"{s:9s} | {m[0]:6.2f} |     {m[1]:6.2f}         {m[2]:6.2f}        {m[3]:6.2f}   |     {m[4]:6.2f}     |      {m[5]:.2f}       | {m[6]:.2f}", flush=True)


if __name__ == "__main__":
    main()

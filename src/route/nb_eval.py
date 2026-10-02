"""Masked evaluation on a Nerfbusters scene (official-style visibility masks from nb_mask.py): PSNR / SSIM / LPIPS on
masked pixels for every candidate in the hybrid dump (3DGS, IBGS) and, if present, the Difix outputs saved by d0
(not saved by default) — here we score the dump candidates and report coverage and support-class breakdowns.

    .venv_ibgs/bin/python src/route/nb_eval.py <scene>
"""
import os
import sys
from glob import glob

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    s = sys.argv[1]
    import lpips
    lp = lpips.LPIPS(net="alex").cuda().eval()
    D = os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_nb"); Mdir = os.path.join(ROOT, "outputs", "nb", "masks", s)
    acc = {}
    for f in sorted(glob(D + "/*.npz")):
        name = os.path.basename(f)[:-4]; mp = os.path.join(Mdir, f"{name}_mask.png")
        if not os.path.exists(mp):
            continue
        z = np.load(f); gt = torch.from_numpy(z["gt"].astype(np.float32)).cuda()
        m = torch.from_numpy(np.asarray(Image.open(mp)) > 127).cuda()
        if m.shape != gt.shape[-2:] or not m.any():
            continue
        for k in ("mcmc", "ibgs_final", "mcmc_res"):
            im = torch.from_numpy(z[k].astype(np.float32)).cuda()
            e = ((im - gt) ** 2).mean(0)
            l = lp((im * m)[None] * 2 - 1, (gt * m)[None] * 2 - 1).item()
            acc.setdefault(k, []).append((float(-10 * torch.log10(e[m].mean())), float(-10 * torch.log10(e.mean())), l, float(m.float().mean())))
    names = {"mcmc": "3DGS", "ibgs_final": "IBGS", "mcmc_res": "IBGS-net on 3DGS"}
    for k, v in acc.items():
        v = np.array(v)
        print(f"[{s}] {names[k]:18s} masked PSNR {v[:, 0].mean():6.2f} | full PSNR {v[:, 1].mean():6.2f} | LPIPS(alex, masked) {v[:, 2].mean():.3f} | coverage {v[:, 3].mean():.3f} | {len(v)} views")


if __name__ == "__main__":
    main()

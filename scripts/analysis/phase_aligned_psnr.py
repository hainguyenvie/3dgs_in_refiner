"""Phase-aligned PSNR — a diagnostic metric for the paper: PSNR/SSIM/LPIPS after removing the per-view SMOOTH
sub-pixel misalignment between the test render and the test photo (RAFT flow -> degree-3 polynomial field,
render warped by it). Standard PSNR penalises a sharper model that is phase-shifted w.r.t. an imperfectly
posed test photo (E3/N2); this metric measures the model's fidelity up to that smooth phase term. Report it
NEXT TO standard metrics, clearly labelled; never instead of them (it uses the test GT to fit 20 coefficients).

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/phase_aligned_psnr.py outputs/protocolR/mcmc/bicycle_r1 outputs/p4/bicycle_p6_m1 ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_misalignment import dev, flow_fb, load, warp  # noqa: E402
from e3_view_consistency import poly_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def main():
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="vgg").to(dev)
    except Exception:
        pass
    out = {}
    for mdir in sys.argv[1:]:
        d = Path(mdir) / "test" / "ours_30000"
        names = sorted(os.listdir(d / "gt"))
        ps, pa, la, lb = [], [], [], []
        for nm in names:
            g, r = load(d / "gt" / nm), load(d / "renders" / nm)
            f, ok = flow_fb(model, g, r)
            pol = poly_fit(f, ok)
            G, R = g[None].to(dev), r[None].to(dev)
            Ra = warp(R, pol)
            ps.append(psnr(R, G)); pa.append(psnr(Ra, G))
            if lp is not None:
                lb.append(float(lp(R * 2 - 1, G * 2 - 1))); la.append(float(lp(Ra * 2 - 1, G * 2 - 1)))
        out[mdir] = {"psnr": float(np.mean(ps)), "psnr_phase_aligned": float(np.mean(pa)),
                     "lpips": float(np.mean(lb)) if lb else None, "lpips_phase_aligned": float(np.mean(la)) if la else None, "views": len(names)}
        r = out[mdir]
        print(f"{mdir}: PSNR {r['psnr']:.2f} -> aligned {r['psnr_phase_aligned']:.2f} (+{r['psnr_phase_aligned']-r['psnr']:.2f})"
              + (f"  LPIPS {r['lpips']:.3f} -> {r['lpips_phase_aligned']:.3f}" if lb else ""), flush=True)
    p = ROOT / "reports" / "phase_aligned_psnr.json"
    prev = json.load(open(p)) if p.exists() else {}
    prev.update(out); json.dump(prev, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()

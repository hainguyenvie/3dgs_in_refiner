"""Per-test-view misalignment render<->photo (RAFT -> poly3 field, as in phase_aligned_psnr) for several models of the
same scene: is a phase-refined model MORE misaligned w.r.t. the fixed test cameras (gauge drift) or equally misaligned
but sharper (hence penalised more per pixel)? Reports median |field| px, the poly3-affine (pose-like) share, and the
PSNR penalty (aligned - standard) per model.
    CUDA_VISIBLE_DEVICES=.. .venv_ibgs/bin/python scripts/analysis/n2b_test_misalignment.py <model_dir>[:<subdir>] ...
"""
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
    per_view = {}
    for spec in sys.argv[1:]:
        mdir, sub = (spec.split(":") + ["ours_30000"])[:2]
        d = Path(mdir) / "test" / sub
        names = sorted(os.listdir(d / "gt"))
        mags, pen, hf = [], [], []
        for nm in names:
            g, r = load(d / "gt" / nm), load(d / "renders" / nm)
            f, ok = flow_fb(model, g, r)
            pol = poly_fit(f, ok)
            m = float(pol.norm(dim=1).mean()) if pol.dim() == 4 else float(pol.norm(dim=0).mean())
            G, R = g[None].to(dev), r[None].to(dev)
            p = psnr(warp(R, pol), G) - psnr(R, G)
            # sharpness proxy: high-frequency energy of the render (Laplacian rms)
            lap = R[:, :, 1:-1, 1:-1] * 4 - R[:, :, :-2, 1:-1] - R[:, :, 2:, 1:-1] - R[:, :, 1:-1, :-2] - R[:, :, 1:-1, 2:]
            mags.append(m); pen.append(p); hf.append(float(lap.pow(2).mean().sqrt()))
            per_view.setdefault(nm, {})[spec] = m
        print(f"{spec}: test-view misalignment median {np.median(mags):.3f} px (p90 {np.percentile(mags, 90):.3f}) | "
              f"PSNR penalty of that misalignment (aligned-std) {np.mean(pen):+.2f} dB | render HF rms {np.mean(hf):.4f}", flush=True)
    if len(sys.argv) > 2:   # are the per-view misalignments the same views across models (data property) or model-specific?
        specs = sys.argv[1:]
        M = np.array([[per_view[n].get(s, np.nan) for s in specs] for n in sorted(per_view)])
        for i in range(1, len(specs)):
            c = np.corrcoef(M[:, 0], M[:, i])[0, 1]
            print(f"corr of per-view misalignment {specs[0]} vs {specs[i]}: {c:.2f}")


if __name__ == "__main__":
    main()

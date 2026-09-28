"""N1 + N3 — what part of the remaining error is (N1) sensor/JPEG noise in the GT itself (irreducible), and (N3)
per-view PHOTOMETRIC inconsistency (exposure / white balance) that a per-view colour model could remove.

N1: Immerkaer fast noise estimate on test GT (sigma in [0,1] units) -> PSNR ceiling of a perfect noiseless model
    against noisy GT ~ -20 log10(sigma).
N3: on TRAIN views of the Protocol R MCMC model: PSNR gain from (a) a single GLOBAL 3x4 colour affine (render->gt,
    fitted over all views) and (b) a PER-VIEW 3x4 affine. (b)-(a) = per-view photometric inconsistency budget.

    .venv_ibgs/bin/python scripts/analysis/n13_noise_photometric.py bicycle garden stump counter bonsai train
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_misalignment import dev, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def immerkaer(img):
    """Immerkaer 1996 noise estimator (per channel, then mean), img (3,H,W) in [0,1]."""
    k = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]], device=img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    r = F.conv2d(img[None], k, groups=3)
    H, W = img.shape[-2:]
    return float((torch.sqrt(torch.tensor(np.pi / 2)) / (6 * (W - 2) * (H - 2)) * r.abs().sum(dim=(2, 3))).mean())


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def affine_fit(X, Y):  # X,Y: (N,3) -> A (4,3)
    Xa = torch.cat([X, torch.ones(X.shape[0], 1, device=dev)], 1)
    return torch.linalg.lstsq(Xa, Y).solution


def apply_affine(img, A):
    X = torch.cat([img.reshape(3, -1).T, torch.ones(img.shape[1] * img.shape[2], 1, device=dev)], 1)
    return (X @ A).T.reshape(img.shape).clamp(0, 1)


def main():
    out = {}
    for scene in sys.argv[1:]:
        base = ROOT / "outputs" / "protocolR" / "mcmc" / (f"{scene}_oreg001" if scene == "playroom" else f"{scene}_r1")
        # N1 on test GT
        tg = base / "test" / "ours_30000" / "gt"
        sig = [immerkaer(load(tg / n).to(dev)) for n in sorted(os.listdir(tg))[:20]]
        sigma = float(np.median(sig)); ceiling = -20 * np.log10(max(sigma, 1e-6))
        # N3 on train views
        tr = base / "train" / "ours_30000"
        names = sorted(os.listdir(tr / "gt"))[::2]
        pairs = [(load(tr / "gt" / n).to(dev), load(tr / "renders" / n).to(dev)) for n in names]
        p_raw = float(np.mean([psnr(r, g) for g, r in pairs]))
        # global affine: fit on subsampled pixels of all views
        Xs = torch.cat([r.reshape(3, -1).T[::97] for g, r in pairs]); Ys = torch.cat([g.reshape(3, -1).T[::97] for g, r in pairs])
        Ag = affine_fit(Xs, Ys)
        p_glob = float(np.mean([psnr(apply_affine(r, Ag), g) for g, r in pairs]))
        p_view = float(np.mean([psnr(apply_affine(r, affine_fit(r.reshape(3, -1).T[::13], g.reshape(3, -1).T[::13])), g) for g, r in pairs]))
        # exposure-only per view (gain+bias per channel = diagonal affine)
        out[scene] = {"gt_noise_sigma": sigma, "psnr_ceiling_noise": ceiling, "train_psnr_raw": p_raw,
                      "train_psnr_global_affine": p_glob, "train_psnr_perview_affine": p_view,
                      "perview_minus_global": p_view - p_glob, "views": len(pairs)}
        print(scene, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out[scene].items()}), flush=True)
    json.dump(out, open(ROOT / "reports" / "n13_noise_photometric.json", "w"), indent=1)
    md = ["# N1/N3 — GT noise floor and per-view photometric inconsistency (MCMC Protocol R)", "",
          "| scene | GT noise σ (8-bit) | PSNR ceiling (noise) | train PSNR raw | +global colour affine | +per-view affine | per-view − global |",
          "|---|---|---|---|---|---|---|"]
    for s, r in out.items():
        md.append(f"| {s} | {r['gt_noise_sigma']*255:.2f} | {r['psnr_ceiling_noise']:.1f} | {r['train_psnr_raw']:.2f} | {r['train_psnr_global_affine']-r['train_psnr_raw']:+.2f} | "
                  f"{r['train_psnr_perview_affine']-r['train_psnr_raw']:+.2f} | **{r['perview_minus_global']:+.2f}** |")
    (ROOT / "reports" / "n13_noise_photometric.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()

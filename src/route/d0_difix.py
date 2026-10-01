"""D0 — does single-step diffusion (Difix, CVPR25, official weights) help on top of a strong render, and where?
Inputs: the hybrid dump of a scene (dump_hybrid.py: gt, mcmc, ibgs_final, feats, src ids) and its JPEG train images.
For every test view: Difix(mcmc) without reference, Difix-ref(mcmc, ref = nearest train image = IBGS's first source),
Difix-ref(ibgs_final). Scores PSNR / SSIM / LPIPS(vgg) on the full image and split by support (#valid warps 0 / 1-2 / >=3),
plus an evidence-gated variant that keeps the input where #valid >= 1 and takes Difix only where no source sees the pixel.
Run at native resolution (floored to a multiple of 8), not Difix's default 576x1024.

    .venv_difix/bin/python src/route/d0_difix.py outputs/route/hybrid/<scene>_r1 [--max_views N]
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "Difix3D", "src"))


def ssim(a, b):
    import torchmetrics.functional as tmf
    return float(tmf.structural_similarity_index_measure(a[None], b[None], data_range=1.0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--max_views", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob(os.path.join(a.dump, "*.npz")))
    if a.max_views: files = files[: a.max_views]
    train_imgs = sorted(glob(os.path.join(a.dump, "_srcjpg", "*")))
    to_t = lambda p: torch.from_numpy(np.asarray(p, dtype=np.float32) / 255).permute(2, 0, 1).cuda()
    to_p = lambda t: Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    rows = []
    out_dir = os.path.join(a.dump, "_difix"); os.makedirs(out_dir, exist_ok=True)
    for f in files:
        z = np.load(f); gt = torch.from_numpy(z["gt"].astype(np.float32)).cuda(); _, H, W = gt.shape
        nv = torch.from_numpy(z["feats"][0].astype(np.float32)).cuda()
        ref = Image.open(train_imgs[int(z["src"][0])]).convert("RGB") if len(z["src"]) else None
        h8, w8 = H - H % 8, W - W % 8
        cand = {"mcmc": torch.from_numpy(z["mcmc"].astype(np.float32)).cuda(), "ibgs": torch.from_numpy(z["ibgs_final"].astype(np.float32)).cuda()}
        outs = dict(cand)
        def run(m, x, r=None):   # native resolution: crop to a multiple of 8, keep the input in the leftover border
            kw = dict(image=to_p(x[:, :h8, :w8]), num_inference_steps=1, timesteps=[199], guidance_scale=0.0)
            if r is not None: kw["ref_image"] = r.crop((0, 0, w8, h8))
            y = x.clone(); y[:, :h8, :w8] = to_t(m(prompt, **kw).images[0]); return y
        with torch.no_grad():
            outs["difix(mcmc)"] = run(m0, cand["mcmc"])
            if ref is not None:
                outs["difix_ref(mcmc)"] = run(m1, cand["mcmc"], ref)
                outs["difix_ref(ibgs)"] = run(m1, cand["ibgs"], ref)
        unsup = (nv < 0.5).float()
        for k in [k for k in outs if k.startswith("difix")]:
            base = cand["ibgs"] if "ibgs" in k else cand["mcmc"]
            outs[f"gated[{k}]"] = base * (1 - unsup) + outs[k] * unsup
        r = {"view": os.path.basename(f)[:-4], "unsup_frac": float(unsup.mean())}
        bins = {"u0": nv < 0.5, "s12": (nv >= 0.5) & (nv < 2.5), "s3": nv >= 2.5}
        for k, im in outs.items():
            im = (im.clamp(0, 1) * 255 + 0.5).floor() / 255
            e = ((im - gt) ** 2).mean(0)
            r[f"{k}|psnr"] = float(-10 * torch.log10(e.mean()))
            r[f"{k}|ssim"] = ssim(im, gt)
            r[f"{k}|lpips"] = float(lp(im[None] * 2 - 1, gt[None] * 2 - 1))
            for b, msk in bins.items():
                r[f"{k}|mse_{b}"] = float(e[msk].mean()) if msk.any() else float("nan")
        rows.append(r)
        print(r["view"], f"unsup {r['unsup_frac']:.3f}", " ".join(f"{k}:{r[k + '|psnr']:.2f}" for k in outs), flush=True)
    json.dump(rows, open(os.path.join(out_dir, "d0_rows.json"), "w"), indent=0)
    keys = [k[:-5] for k in rows[0] if k.endswith("|psnr")]
    print("\nmean over views:")
    for k in keys:
        vals = [r for r in rows if f"{k}|psnr" in r]
        mb = {b: np.nanmean([r[f"{k}|mse_{b}"] for r in vals]) for b in ("u0", "s12", "s3")}
        print(f"  {k:24s} PSNR {np.mean([r[k + '|psnr'] for r in vals]):6.2f} SSIM {np.mean([r[k + '|ssim'] for r in vals]):.4f} "
              f"LPIPS {np.mean([r[k + '|lpips'] for r in vals]):.4f} | PSNR by support u0 {-10 * np.log10(mb['u0']):.2f} "
              f"s1-2 {-10 * np.log10(mb['s12']):.2f} s3+ {-10 * np.log10(mb['s3']):.2f}")


if __name__ == "__main__":
    main()

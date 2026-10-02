"""D0b — is Difix's PSNR loss a resolution artefact, and does keeping only its LOW frequencies help where no source sees?
For a few views of a hybrid dump: Difix-ref at native resolution vs at its training resolution (576x1024, resized back,
the official usage), and fusions on the unsupported region (#valid warps = 0):
  full         : Difix output everywhere
  u0-full      : base outside u0, Difix inside u0
  u0-LF        : base + low-pass(Difix - base) inside u0 (Gaussian sigma 4 px) — the band-limited hypothesis
Scores PSNR on the full image and on u0 pixels.

    HF_HOME=... .venv_difix/bin/python src/route/d0b_difix_lf.py outputs/route/hybrid/<scene>_r1 [--views 6] [--base mcmc|ibgs_final]
"""
import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "Difix3D", "src"))


def gblur(x, s=4.0):
    k = int(3 * s) * 2 + 1; t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x[None], (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--views", type=int, default=6); ap.add_argument("--base", default="mcmc")
    a = ap.parse_args()
    from pipeline_difix import DifixPipeline
    m = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True).to("cuda"); m.set_progress_bar_config(disable=True)
    files = [f for f in sorted(glob(os.path.join(a.dump, "*.npz"))) if not f.endswith(".geo.npz")][:: max(1, 1)][: a.views]
    train_imgs = sorted(glob(os.path.join(a.dump, "_srcjpg", "*")))
    to_t = lambda p: torch.from_numpy(np.asarray(p, dtype=np.float32) / 255).permute(2, 0, 1).cuda()
    to_p = lambda t: Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))
    q = lambda x: (x.clamp(0, 1) * 255 + 0.5).floor() / 255
    acc = {}
    for f in files:
        z = np.load(f); gt = torch.from_numpy(z["gt"].astype(np.float32)).cuda(); base = torch.from_numpy(z[a.base].astype(np.float32)).cuda()
        u0 = torch.from_numpy(z["feats"][0].astype(np.float32)).cuda() < 0.5; H, W = gt.shape[-2:]
        if len(z["src"]) == 0: continue   # sector views outside every source cone
        ref = Image.open(train_imgs[int(z["src"][0])]).convert("RGB")
        outs = {}
        with torch.no_grad():
            h8, w8 = H - H % 8, W - W % 8
            y = base.clone(); y[:, :h8, :w8] = to_t(m("remove degradation", image=to_p(base[:, :h8, :w8]), ref_image=ref.crop((0, 0, w8, h8)), num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0])
            outs["native"] = y
            small = m("remove degradation", image=to_p(base).resize((1024, 576), Image.LANCZOS), ref_image=ref.resize((1024, 576), Image.LANCZOS),
                      num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0].resize((W, H), Image.LANCZOS)
            outs["576x1024"] = to_t(small)
        rows = {"base": base}
        for k, o in outs.items():
            rows[f"{k}|full"] = o
            rows[f"{k}|u0-full"] = torch.where(u0[None], o, base)
            rows[f"{k}|u0-LF"] = base + u0[None] * gblur(o - base)
            rows[f"{k}|LF-all"] = base + gblur(o - base)
        for k, im in rows.items():
            e = ((q(im) - gt) ** 2).mean(0)
            acc.setdefault(k, []).append((float(-10 * torch.log10(e.mean())), float(-10 * torch.log10(e[u0].mean())) if u0.any() else np.nan, float(u0.float().mean())))
    print(f"== {a.dump} base={a.base} ({len(files)} views, u0 share {np.mean([r[2] for r in acc['base']]):.3f})")
    for k, v in acc.items():
        v = np.array(v); print(f"  {k:22s} PSNR full {v[:, 0].mean():6.2f} | u0 {np.nanmean(v[:, 1]):6.2f}")


if __name__ == "__main__":
    main()

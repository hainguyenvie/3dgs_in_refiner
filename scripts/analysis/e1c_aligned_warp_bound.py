"""E1c — upper bounds for a single warped source: what PSNR would src0 reach on its valid pixels if it were
(a) colour-matched to GT (per-view 3x4 affine, oracle), (b) perfectly aligned (RAFT flow GT->warp applied to the
warp, i.e. the misalignment removed), (c) both. Tells whether depth-induced misalignment or photometric
mismatch is the bottleneck of texture transfer. Uses E1b dumps. Diagnostic only (uses GT).

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/e1c_aligned_warp_bound.py bicycle garden bonsai counter train
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
from e4_frequency_color import bands  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def mse_m(a, b, m):
    return float((((a - b) ** 2).mean(1, keepdim=True) * m).sum() / m.sum().clamp_min(1))


def psnr(x):
    return float(-10 * np.log10(max(x, 1e-12)))


def color_affine_masked(src, gt, m):
    mm = m[0, 0] > 0
    X = torch.cat([src[0][:, mm], torch.ones(1, int(mm.sum()), device=dev)], 0).T
    A = torch.linalg.lstsq(X, gt[0][:, mm].T).solution
    Xf = torch.cat([src[0].reshape(3, -1), torch.ones(1, src[0].numel() // 3, device=dev)], 0).T
    return (Xf @ A).T.reshape(1, 3, *src.shape[-2:]).clamp(0, 1)


def main():
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    out = {}
    for scene in sys.argv[1:]:
        d = Path(os.environ["E_BASE"].format(scene=scene)) / "test" / "ours_30000" if os.environ.get("E_BASE") else ROOT / "outputs" / "protocolR" / "ibgs" / f"{scene}_pre" / "test" / "ours_30000"
        acc = {k: [] for k in ["raw", "fin", "w", "w_ca", "w_al", "w_al_ca", "raw_al", "b_w", "b_w_al"]}
        for nm in sorted(os.listdir(d / "gt")):
            stem = os.path.splitext(nm)[0]
            wp = d / "warps" / f"{stem}_s0.png"
            if not wp.exists():
                continue
            gt, raw, fin = load(d / "gt" / nm)[None].to(dev), load(d / "renders" / nm)[None].to(dev), load(d / "renders_aggregate" / nm)[None].to(dev)
            w = load(wp)[None].to(dev); m = (load(d / "warpmask" / f"{stem}_s0.png")[None, :1].to(dev) > 0.5).float()
            f, ok = flow_fb(model, gt[0].cpu(), w[0].cpu())          # gt(p) ~ w(p+f)
            mm = m * ok[None, None].float()                            # valid & flow-consistent
            if mm.sum() < 0.05 * mm.numel():
                continue
            w_al = warp(w, f)
            fr, okr = flow_fb(model, gt[0].cpu(), raw[0].cpu()); raw_al = warp(raw, fr)
            acc["raw"].append(mse_m(raw, gt, mm)); acc["fin"].append(mse_m(fin, gt, mm)); acc["w"].append(mse_m(w, gt, mm))
            acc["w_ca"].append(mse_m(color_affine_masked(w, gt, mm), gt, mm))
            acc["w_al"].append(mse_m(w_al, gt, mm)); acc["w_al_ca"].append(mse_m(color_affine_masked(w_al, gt, mm), gt, mm))
            acc["raw_al"].append(mse_m(raw_al, gt, mm))
            acc["b_w"].append(bands((w - gt) * mm)); acc["b_w_al"].append(bands((w_al - gt) * mm))
        r = {k: psnr(np.mean(v)) for k, v in acc.items() if not k.startswith("b_")}
        bw, bwa = np.mean(acc["b_w"], 0), np.mean(acc["b_w_al"], 0)
        r["band_share_warp_err"] = (bw / bw.sum()).round(3).tolist(); r["band_share_aligned_warp_err"] = (bwa / bwa.sum()).round(3).tolist()
        r["band_err_reduction_by_alignment"] = (1 - bwa / bw).round(3).tolist()
        out[scene] = r; print(scene, json.dumps(r), flush=True)
    json.dump(out, open(ROOT / "reports" / f"e1c_aligned_warp_bound{os.environ.get('E_TAG','')}.json", "w"), indent=1)
    md = ["# E1c — single-source upper bounds on src0 valid & flow-consistent pixels (PSNR from mean MSE)", "",
          "| scene | raw | raw aligned | final | warp0 | warp0 +colour | **warp0 aligned** | aligned +colour | warp err share HF/MF/LFm/LF | after align | per-band reduction by alignment |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in out.items():
        f = lambda v: "/".join(f"{x*100:.0f}" for x in v)
        md.append(f"| {s} | {r['raw']:.2f} | {r['raw_al']:.2f} | {r['fin']:.2f} | {r['w']:.2f} | {r['w_ca']:.2f} | **{r['w_al']:.2f}** | {r['w_al_ca']:.2f} | "
                  f"{f(r['band_share_warp_err'])} | {f(r['band_share_aligned_warp_err'])} | {f(r['band_err_reduction_by_alignment'])} |")
    (ROOT / "reports" / f"e1c_aligned_warp_bound{os.environ.get('E_TAG','')}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()

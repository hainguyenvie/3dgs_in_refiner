"""E1b — what limits HF transfer: are the warped SOURCES aligned with GT at test views, and how much would an
oracle source selection recover? Needs `render.py --dump_warps` output (test/ours_30000/{warps,warpmask}).

Per test view and per source k: PSNR on valid pixels, RAFT misalignment warp_k -> GT on valid pixels, error by
frequency band, spectral transfer T(f) of the warp. Oracles (use GT, diagnostic only): per-pixel best source,
per-32x32-patch best source, mean of valid sources. All compared with raw and final.

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/e1b_source_warps.py bicycle garden bonsai counter
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_misalignment import dev, flow_fb, load  # noqa: E402
from e4_frequency_color import bands  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PATCH = 32


def mse(a, b, m=None):
    e = ((a - b) ** 2).mean(1, keepdim=True)
    return float(e.mean()) if m is None else float((e * m).sum() / m.sum().clamp_min(1))


def psnr_from_mse(x):
    return float(-10 * np.log10(max(x, 1e-12)))


def patch_oracle(cands, masks, gt, raw):
    """choose per PATCH the candidate (or raw) with lowest MSE; returns composite."""
    _, _, H, W = gt.shape
    best = raw.clone(); best_err = ((raw - gt) ** 2).mean(1, keepdim=True)
    best_err = F.avg_pool2d(best_err, PATCH, ceil_mode=True)
    for c, m in zip(cands, masks):
        cm = torch.where(m > 0, c, raw)
        e = F.avg_pool2d(((cm - gt) ** 2).mean(1, keepdim=True), PATCH, ceil_mode=True)
        sel = (e < best_err)
        best_err = torch.where(sel, e, best_err)
        sel_full = F.interpolate(sel.float(), scale_factor=PATCH, mode="nearest")[..., :H, :W]
        best = torch.where(sel_full > 0, cm, best)
    return best


def main():
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    out = {}
    for scene in sys.argv[1:]:
        d = ROOT / "outputs" / "protocolR" / "ibgs" / f"{scene}_pre" / "test" / "ours_30000"
        if not (d / "warps").is_dir():
            print("no warps for", scene); continue
        names = sorted(os.listdir(d / "gt"))
        acc = {"raw": [], "fin": [], "src": [[] for _ in range(4)], "src_valid": [[] for _ in range(4)],
               "mis_med": [[] for _ in range(4)], "mis_gt05": [[] for _ in range(4)], "mis_gt1": [[] for _ in range(4)],
               "band_src": [[] for _ in range(4)], "band_raw": [], "band_fin": [],
               "oracle_px": [], "oracle_patch": [], "mean_valid": [], "raw_on_valid0": [], "n_valid_src": []}
        for nm in names:
            stem = os.path.splitext(nm)[0]
            gt, raw, fin = load(d / "gt" / nm)[None].to(dev), load(d / "renders" / nm)[None].to(dev), load(d / "renders_aggregate" / nm)[None].to(dev)
            acc["raw"].append(mse(raw, gt)); acc["fin"].append(mse(fin, gt))
            acc["band_raw"].append(bands(raw - gt)); acc["band_fin"].append(bands(fin - gt))
            cands, masks = [], []
            for k in range(4):
                wp, mp = d / "warps" / f"{stem}_s{k}.png", d / "warpmask" / f"{stem}_s{k}.png"
                if not wp.exists():
                    continue
                w = load(wp)[None].to(dev); m = (load(mp)[None, :1].to(dev) > 0.5).float()
                if m.sum() < 0.05 * m.numel():
                    continue
                cands.append(w); masks.append(m)
                acc["src"][k].append(mse(w, gt, m)); acc["src_valid"][k].append(float(m.mean()))
                if k == 0:
                    acc["raw_on_valid0"].append(mse(raw, gt, m))
                # misalignment of the warp w.r.t. GT on valid pixels (RAFT gt->warp)
                f, ok = flow_fb(model, gt[0].cpu(), w[0].cpu())
                mag = f.norm(dim=1)[0][(ok & (m[0, 0] > 0))]
                if mag.numel() > 100:
                    acc["mis_med"][k].append(float(mag.median())); acc["mis_gt05"][k].append(float((mag > 0.5).float().mean()))
                    acc["mis_gt1"][k].append(float((mag > 1).float().mean()))
                acc["band_src"][k].append(bands((w - gt) * m))
            acc["n_valid_src"].append(len(cands))
            if cands:
                stack = torch.stack([torch.where(m > 0, c, raw) for c, m in zip(cands, masks)])   # S,1,3,H,W
                errs = ((stack - gt) ** 2).mean(2, keepdim=True)                                   # S,1,1,H,W
                raw_err = ((raw - gt) ** 2).mean(1, keepdim=True)[None]
                allerr = torch.cat([errs, raw_err], 0)
                acc["oracle_px"].append(float(allerr.min(0).values.mean()))
                acc["oracle_patch"].append(mse(patch_oracle(cands, masks, gt, raw), gt))
                msum = torch.stack(masks).sum(0)
                mean_valid = torch.where(msum > 0, (torch.stack([c * m for c, m in zip(cands, masks)]).sum(0)) / msum.clamp_min(1), raw)
                acc["mean_valid"].append(mse(mean_valid, gt))
        r = {"views": len(names), "psnr_raw": psnr_from_mse(np.mean(acc["raw"])), "psnr_fin": psnr_from_mse(np.mean(acc["fin"])),
             "psnr_oracle_pixel": psnr_from_mse(np.mean(acc["oracle_px"])), "psnr_oracle_patch32": psnr_from_mse(np.mean(acc["oracle_patch"])),
             "psnr_mean_valid_sources": psnr_from_mse(np.mean(acc["mean_valid"])), "n_valid_src_mean": float(np.mean(acc["n_valid_src"])),
             "band_share_raw_err": (np.mean(acc["band_raw"], 0) / np.mean(acc["band_raw"], 0).sum()).round(3).tolist(),
             "band_share_fin_err": (np.mean(acc["band_fin"], 0) / np.mean(acc["band_fin"], 0).sum()).round(3).tolist(), "sources": []}
        for k in range(4):
            if acc["src"][k]:
                bs = np.mean(acc["band_src"][k], 0)
                r["sources"].append({"k": k, "valid_frac": float(np.mean(acc["src_valid"][k])),
                                     "psnr_on_valid": psnr_from_mse(np.mean(acc["src"][k])),
                                     "psnr_raw_on_same_valid": psnr_from_mse(np.mean(acc["raw_on_valid0"])) if k == 0 else None,
                                     "mis_median_px": float(np.mean(acc["mis_med"][k])) if acc["mis_med"][k] else None,
                                     "frac_mis_gt0.5px": float(np.mean(acc["mis_gt05"][k])) if acc["mis_gt05"][k] else None,
                                     "frac_mis_gt1px": float(np.mean(acc["mis_gt1"][k])) if acc["mis_gt1"][k] else None,
                                     "band_share_err": (bs / bs.sum()).round(3).tolist()})
        out[scene] = r
        print(scene, json.dumps(r), flush=True)
    rep = ROOT / "reports"; json.dump(out, open(rep / "e1b_source_warps.json", "w"), indent=1)
    md = ["# E1b — warped sources at test views (IBGS released checkpoints)", "",
          "| scene | raw | final | mean of valid src | oracle patch32 | oracle pixel | src0: valid% / PSNR(valid) / raw PSNR same px / mis. med px / >0.5px / >1px | src1 mis. med / >0.5px | src2 mis. med / >0.5px |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, r in out.items():
        S = {x["k"]: x for x in r["sources"]}; s0 = S.get(0, {}); s1 = S.get(1, {}); s2 = S.get(2, {})
        g = lambda x, k, f="{:.2f}": f.format(x[k]) if x.get(k) is not None else "—"
        md.append(f"| {s} | {r['psnr_raw']:.2f} | {r['psnr_fin']:.2f} | {r['psnr_mean_valid_sources']:.2f} | {r['psnr_oracle_patch32']:.2f} | "
                  f"{r['psnr_oracle_pixel']:.2f} | {g(s0,'valid_frac','{:.0%}')} / {g(s0,'psnr_on_valid')} / {g(s0,'psnr_raw_on_same_valid')} / "
                  f"{g(s0,'mis_median_px')} / {g(s0,'frac_mis_gt0.5px','{:.0%}')} / {g(s0,'frac_mis_gt1px','{:.0%}')} | "
                  f"{g(s1,'mis_median_px')} / {g(s1,'frac_mis_gt0.5px','{:.0%}')} | {g(s2,'mis_median_px')} / {g(s2,'frac_mis_gt0.5px','{:.0%}')} |")
    (rep / "e1b_source_warps.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()

"""E9 — where does the intrinsic warp error live? On the synthetic consistent world (E8 dumps + depth):
(1) error concentration: fraction of valid pixels carrying 50/80% of warp0 squared error;
(2) warp error vs |grad depth| (depth-edge / multi-layer proxy), binned;
(3) warp error vs disagreement between two warped sources |w0 - w1| on jointly valid pixels
    (both sources see the same synthetic scene, so disagreement = per-pixel geometric ambiguity);
(4) same statistics for the raw render error, as a control.

    E_BASE=outputs/e8/{scene}_ibgs_syn .venv_ibgs/bin/python scripts/analysis/e9_warp_error_mechanism.py bicycle garden bonsai
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
QS = [0.0, 0.5, 0.75, 0.9, 0.97, 1.0]   # quantile bins of the conditioning variable


def concentration(err, m):
    e = torch.sort(err[m > 0].flatten(), descending=True).values
    c = torch.cumsum(e, 0) / e.sum()
    return {"px_frac_for_50pct_err": float((c < 0.5).float().mean()), "px_frac_for_80pct_err": float((c < 0.8).float().mean())}


def binned(err, cond, m, qs=QS):
    v = cond[m > 0].flatten(); e = err[m > 0].flatten()
    edges = torch.quantile(v[::13], torch.tensor(qs, device=v.device))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (v >= lo) & (v <= hi)
        out.append({"q": None, "cond_lo": float(lo), "cond_hi": float(hi), "px_frac": float(sel.float().mean()),
                    "mse": float(e[sel].mean()), "err_share": float(e[sel].sum() / e.sum())})
    return out


def main():
    out = {}
    for scene in sys.argv[1:]:
        d = Path(os.environ["E_BASE"].format(scene=scene)) / "test" / "ours_30000"
        acc = {"conc_w": [], "conc_raw": [], "grad_w": [], "grad_raw": [], "dis_w": [], "dis_raw": [], "dis_share_hf": []}
        for nm in sorted(os.listdir(d / "gt")):
            stem = os.path.splitext(nm)[0]
            dp = d / "warpmask" / f"{stem}_depth.npy"
            if not dp.exists() or not (d / "warps" / f"{stem}_s1.png").exists():
                continue
            gt, raw = load(d / "gt" / nm)[None].to(dev), load(d / "renders" / nm)[None].to(dev)
            w0, w1 = load(d / "warps" / f"{stem}_s0.png")[None].to(dev), load(d / "warps" / f"{stem}_s1.png")[None].to(dev)
            m0 = (load(d / "warpmask" / f"{stem}_s0.png")[None, :1].to(dev) > 0.5).float()
            m1 = (load(d / "warpmask" / f"{stem}_s1.png")[None, :1].to(dev) > 0.5).float()
            depth = torch.from_numpy(np.load(dp)).to(dev)[None, None]
            dz = torch.log(depth.clamp_min(1e-3))
            gx = F.pad(dz[..., :, 1:] - dz[..., :, :-1], (0, 1, 0, 0)); gy = F.pad(dz[..., 1:, :] - dz[..., :-1, :], (0, 0, 0, 1))
            grad = torch.sqrt(gx ** 2 + gy ** 2)                        # log-depth gradient (relative depth jump)
            grad = F.max_pool2d(grad, 5, 1, 2)                           # dilate: pixels near an edge
            e_w = ((w0 - gt) ** 2).mean(1, keepdim=True); e_raw = ((raw - gt) ** 2).mean(1, keepdim=True)
            acc["conc_w"].append(concentration(e_w, m0)); acc["conc_raw"].append(concentration(e_raw, m0))
            acc["grad_w"].append(binned(e_w, grad, m0)); acc["grad_raw"].append(binned(e_raw, grad, m0))
            m01 = m0 * m1
            dis = ((w0 - w1) ** 2).mean(1, keepdim=True)
            acc["dis_w"].append(binned(e_w, dis, m01)); acc["dis_raw"].append(binned(e_raw, dis, m01))
        def avg_bins(lst):
            return [{k: float(np.mean([b[i][k] for b in lst])) for k in lst[0][i] if lst[0][i][k] is not None} for i in range(len(lst[0]))]
        r = {"views": len(acc["conc_w"]),
             "warp_err_concentration": {k: float(np.mean([c[k] for c in acc["conc_w"]])) for k in acc["conc_w"][0]},
             "raw_err_concentration": {k: float(np.mean([c[k] for c in acc["conc_raw"]])) for k in acc["conc_raw"][0]},
             "warp_err_by_depth_grad_quantile": avg_bins(acc["grad_w"]), "raw_err_by_depth_grad_quantile": avg_bins(acc["grad_raw"]),
             "warp_err_by_source_disagreement_quantile": avg_bins(acc["dis_w"]), "raw_err_by_source_disagreement_quantile": avg_bins(acc["dis_raw"])}
        out[scene] = r
        print(scene, json.dumps(r)[:1500], flush=True)
    json.dump(out, open(ROOT / "reports" / "e9_warp_error_mechanism.json", "w"), indent=1)
    md = ["# E9 — mechanism of the intrinsic warp error (synthetic consistent world)", "",
          "Quantile bins of the conditioning variable (0–50 / 50–75 / 75–90 / 90–97 / 97–100 %). `share` = share of total squared error in the bin.", ""]
    for s, r in out.items():
        md += [f"## {s}  ({r['views']} views)", "",
               f"Error concentration — warp0: {r['warp_err_concentration']['px_frac_for_50pct_err']*100:.1f}% of pixels carry 50% of error, "
               f"{r['warp_err_concentration']['px_frac_for_80pct_err']*100:.1f}% carry 80%. Raw render: "
               f"{r['raw_err_concentration']['px_frac_for_50pct_err']*100:.1f}% / {r['raw_err_concentration']['px_frac_for_80pct_err']*100:.1f}%.", "",
               "| conditioning | bin | px% | warp0 MSE ×1e3 | warp0 err share | raw MSE ×1e3 | raw err share |", "|---|---|---|---|---|---|---|"]
        for name, kw, kr in [("|∇ log depth| (edge/multi-layer proxy)", "warp_err_by_depth_grad_quantile", "raw_err_by_depth_grad_quantile"),
                             ("|w0 − w1| source disagreement", "warp_err_by_source_disagreement_quantile", "raw_err_by_source_disagreement_quantile")]:
            for i, (bw, br) in enumerate(zip(r[kw], r[kr])):
                lab = f"{QS[i]*100:.0f}–{QS[i+1]*100:.0f}%"
                md.append(f"| {name if i == 0 else ''} | {lab} | {bw['px_frac']*100:.0f} | {bw['mse']*1e3:.2f} | {bw['err_share']*100:.0f}% | {br['mse']*1e3:.2f} | {br['err_share']*100:.0f}% |")
        md.append("")
    (ROOT / "reports" / "e9_warp_error_mechanism.md").write_text("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()

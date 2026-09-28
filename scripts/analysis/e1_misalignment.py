"""E1 — is the raw-render error of image-based GS mostly *geometric misalignment*, and does the warp branch
fail exactly where the geometry is misaligned?

For every test view: RAFT flow GT<->render (forward-backward consistent pixels only), then
  * misalignment |f| per pixel,
  * share of the raw error explained by misalignment: 1 - |warp_f(raw) - gt| / |raw - gt|,
  * warp-branch gain (|raw-gt| - |final-gt|) binned by misalignment,
  * high-frequency share of the residual (final - raw) and of the raw error.
Inputs are Protocol R test renders (no training, no test-set tuning). Output: reports/e1_misalignment.{json,md}.

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/e1_misalignment.py
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

ROOT = Path(__file__).resolve().parents[2]
PR = ROOT / "outputs" / "protocolR"
SCENES = {"outdoor": ["bicycle", "flowers", "garden", "stump", "treehill"],
          "indoor": ["bonsai", "counter", "kitchen", "room"],
          "tnt": ["train", "truck"], "db": ["drjohnson", "playroom"]}
BINS = [0.0, 0.5, 1.0, 2.0, 4.0, 1e9]
dev = torch.device("cuda")


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)


def pad8(x):
    H, W = x.shape[-2:]
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    return F.pad(x, (0, pw, 0, ph), mode="replicate"), (H, W)


def warp(img, flow):
    """sample img at p + flow(p) (backward warp). img (1,C,H,W), flow (1,2,H,W)."""
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    gx = (xx + flow[:, 0]) / (W - 1) * 2 - 1
    gy = (yy + flow[:, 1]) / (H - 1) * 2 - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def flow_fb(model, a, b):
    """flow a->b (a(p) ~ b(p + f)), plus forward-backward consistency mask."""
    A, (H, W) = pad8(a[None].to(dev) * 2 - 1)
    B, _ = pad8(b[None].to(dev) * 2 - 1)
    fab = model(A, B)[-1][..., :H, :W]
    fba = model(B, A)[-1][..., :H, :W]
    back = warp(fba, fab)
    err = (fab + back).norm(dim=1)
    ok = err < 0.5 + 0.05 * (fab.norm(dim=1) + back.norm(dim=1))
    return fab, ok[0]


def blur(x, s=2.0):
    k = int(4 * s) | 1
    t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def hf_share(x):
    x = x[None].to(dev) if x.dim() == 3 else x
    hf = x - blur(x)
    return float((hf ** 2).sum() / ((x ** 2).sum() + 1e-12))


def analyse(method, scene, model):
    d = PR / method / f"{scene}_r1" / "test" / "ours_30000"
    if not (d / "gt").is_dir():
        return None
    names = sorted(os.listdir(d / "gt"))
    has_final = (d / "renders_aggregate").is_dir()
    acc = {"n_px": 0, "n_valid": 0, "mis_sum": 0.0, "err_raw": 0.0, "err_raw_aligned": 0.0,
           "hf_err_raw": [], "hf_resid": [],
           "bins": [{"n": 0, "err_raw": 0.0, "err_fin": 0.0} for _ in BINS[:-1]]}
    if has_final:
        acc.update({"err_fin": 0.0, "err_fin_aligned": 0.0})
    mis_all = []
    for nm in names:
        gt = load(d / "gt" / nm)
        raw = load(d / "renders" / nm)
        f, ok = flow_fb(model, gt, raw)             # gt(p) ~ raw(p + f)
        G, R = gt[None].to(dev), raw[None].to(dev)
        e_raw = (R - G).abs().mean(1)[0]
        e_raw_al = (warp(R, f) - G).abs().mean(1)[0]
        mis = f.norm(dim=1)[0]
        acc["n_px"] += ok.numel(); acc["n_valid"] += int(ok.sum())
        acc["mis_sum"] += float(mis[ok].sum()); mis_all.append(mis[ok].flatten()[::7].cpu())
        acc["err_raw"] += float(e_raw[ok].sum()); acc["err_raw_aligned"] += float(e_raw_al[ok].sum())
        acc["hf_err_raw"].append(hf_share(R - G))
        if has_final:
            fin = load(d / "renders_aggregate" / nm)
            Fi = fin[None].to(dev)
            e_fin = (Fi - G).abs().mean(1)[0]
            ff, okf = flow_fb(model, gt, fin)
            e_fin_al = (warp(Fi, ff) - G).abs().mean(1)[0]
            acc["err_fin"] += float(e_fin[ok].sum()); acc["err_fin_aligned"] += float(e_fin_al[ok].sum())
            acc["hf_resid"].append(hf_share(Fi - R))
        for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
            m = ok & (mis >= lo) & (mis < hi)
            b = acc["bins"][i]; b["n"] += int(m.sum()); b["err_raw"] += float(e_raw[m].sum())
            if has_final:
                b["err_fin"] += float(e_fin[m].sum())
    nv = max(acc["n_valid"], 1)
    mis_all = torch.cat(mis_all)
    out = {"views": len(names), "valid_frac": acc["n_valid"] / acc["n_px"],
           "mis_mean_px": acc["mis_sum"] / nv, "mis_median_px": float(mis_all.median()),
           "frac_mis_gt1px": float((mis_all > 1).float().mean()), "frac_mis_gt2px": float((mis_all > 2).float().mean()),
           "err_raw": acc["err_raw"] / nv,
           "raw_err_explained_by_misalignment": 1 - acc["err_raw_aligned"] / acc["err_raw"],
           "hf_share_raw_error": float(np.mean(acc["hf_err_raw"]))}
    if has_final:
        out.update({"err_fin": acc["err_fin"] / nv,
                    "fin_err_explained_by_misalignment": 1 - acc["err_fin_aligned"] / acc["err_fin"],
                    "rel_gain": 1 - acc["err_fin"] / acc["err_raw"],
                    "hf_share_residual": float(np.mean(acc["hf_resid"]))})
    out["bins"] = []
    for (lo, hi), b in zip(zip(BINS[:-1], BINS[1:]), acc["bins"]):
        row = {"range": f"[{lo},{hi if hi < 1e8 else 'inf'})", "px_frac": b["n"] / nv,
               "err_raw": b["err_raw"] / max(b["n"], 1)}
        if has_final:
            row["err_fin"] = b["err_fin"] / max(b["n"], 1)
            row["rel_gain"] = 1 - b["err_fin"] / max(b["err_raw"], 1e-12)
        out["bins"].append(row)
    return out


def main():
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    res = {}
    for grp, scenes in SCENES.items():
        for s in scenes:
            for m in ("ibgs", "mcmc"):
                r = analyse(m, s, model)
                if r is not None:
                    res[f"{m}/{s}"] = {"group": grp, **r}
                    print(m, s, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    rep = ROOT / "reports"
    json.dump(res, open(rep / "e1_misalignment.json", "w"), indent=1)
    # markdown summary
    md = ["# E1 — misalignment vs warp gain (Protocol R test renders, RAFT fwd-bwd consistent pixels)", "",
          "| scene | grp | mis. median px | >1px | >2px | raw err expl. by misalign. (IBGS) | (MCMC) | "
          "rel. gain final vs raw | gain @<0.5px | gain @1-2px | gain @>4px | HF share resid. |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for grp, scenes in SCENES.items():
        for s in scenes:
            i, m = res.get(f"ibgs/{s}"), res.get(f"mcmc/{s}")
            if not i:
                continue
            g = lambda k: f"{i['bins'][k]['rel_gain']*100:+.1f}%"
            md.append(f"| {s} | {grp} | {i['mis_median_px']:.2f} | {i['frac_mis_gt1px']*100:.0f}% | "
                      f"{i['frac_mis_gt2px']*100:.0f}% | {i['raw_err_explained_by_misalignment']*100:.0f}% | "
                      f"{(m['raw_err_explained_by_misalignment']*100 if m else float('nan')):.0f}% | "
                      f"{i['rel_gain']*100:+.1f}% | {g(0)} | {g(2)} | {g(4)} | {i['hf_share_residual']*100:.0f}% |")
    (rep / "e1_misalignment.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

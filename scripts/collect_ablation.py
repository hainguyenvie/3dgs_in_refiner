#!/usr/bin/env python3
"""Ablation table across all P6-family variants per scene (PSNR, Δ vs MCMC baseline, LPIPS, learned field px) plus
train wall-clock from file mtimes (run_meta.txt -> point_cloud.ply). Stdlib only.

    python3 scripts/collect_ablation.py            # -> reports/ablation_variants.md
"""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ["bicycle", "flowers", "garden", "stump", "treehill", "bonsai", "counter", "kitchen", "room", "train", "truck", "drjohnson", "playroom"]


def res(d):
    p = Path(d) / "results.json"
    if not p.exists():
        return None
    v = list(json.load(open(p)).values())[0]
    return v["PSNR"], v["SSIM"], v["LPIPS"]


def train_minutes(d):
    m, ply = Path(d) / "run_meta.txt", Path(d) / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if m.exists() and ply.exists():
        return (ply.stat().st_mtime - m.stat().st_mtime) / 60
    return None


def field(d):
    p = Path(d) / "phase_summary.json"
    return json.load(open(p))["field_px_median"] if p.exists() else None


base = {}
for s in SCENES:
    d = ROOT / "outputs" / "protocolR" / "mcmc" / ("playroom_oreg001" if s == "playroom" else f"{s}_r1")
    base[s] = (res(d), train_minutes(d))
variants = sorted({os.path.basename(d).split("_p6_")[1] for d in glob.glob(str(ROOT / "outputs" / "p4" / "*_p6_*")) if "_p6_" in d})
md = ["# Ablation — variants of the learnable phase field (Δ PSNR vs MCMC baseline; LPIPS; learned field px; train min)", "",
      "Variants: m1 = reg 1e-4 · m1_reg1e-3/1e-2 · m1_zm = reg 1e-3 + zero-mean · m1_anc = reg 1e-3 + anchor views · m1_deg1 = affine-only · "
      "m1s = reg 1e-3 + zero-mean (13 scenes) · m1n = reg 1e-3 + anchor (13 scenes, FINAL) · m3 = colour affine · m1a = affine-only + zero-mean", "",
      "| scene | MCMC PSNR (min) | " + " | ".join(variants) + " |", "|---|---|" + "---|" * len(variants)]
for s in SCENES:
    b, bt = base[s]
    row = [s, f"{b[0]:.2f} ({bt:.0f})" if b and bt else (f"{b[0]:.2f}" if b else "—")]
    for v in variants:
        d = ROOT / "outputs" / "p4" / f"{s}_p6_{v}"
        r = res(d)
        if r is None:
            row.append("—"); continue
        f = field(d); t = train_minutes(d)
        row.append(f"{r[0]-b[0]:+.2f} / {r[2]:.3f}" + (f" / {f:.2f}px" if f is not None else "") + (f" / {t:.0f}m" if t else ""))
    md.append("| " + " | ".join(row) + " |")
# averages per variant over scenes with results
md += ["", "| variant | n | mean Δ PSNR | mean Δ LPIPS |", "|---|---|---|---|"]
for v in variants:
    dp, dl = [], []
    for s in SCENES:
        r = res(ROOT / "outputs" / "p4" / f"{s}_p6_{v}"); b = base[s][0]
        if r and b:
            dp.append(r[0] - b[0]); dl.append(r[2] - b[2])
    if dp:
        md.append(f"| {v} | {len(dp)} | {sum(dp)/len(dp):+.3f} | {sum(dl)/len(dl):+.4f} |")
(ROOT / "reports" / "ablation_variants.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

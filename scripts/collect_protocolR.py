#!/usr/bin/env python3
"""Collect Protocol R results into reports/protocolR_table.{md,csv}.

For every (method, scene): paper number, authors' released number (IBGS pretrained JSON), our
reproduction (tag r1), IBGS pretrained re-rendered on our env (tag pre), and deltas. IBGS rows carry
both `raw` (renders/) and `final` (renders_aggregate/). Stdlib only, runs with system python3.

    python3 scripts/collect_protocolR.py            # from the project root on the server
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "protocolR"
PRE = ROOT / "checkpoints" / "ibgs_pretrained" / "output"

MIP = ["bicycle", "flowers", "garden", "stump", "treehill", "bonsai", "counter", "kitchen", "room"]
TNT = ["train", "truck"]
DB = ["drjohnson", "playroom"]
DATASETS = {"mip360": MIP, "tnt": TNT, "db": DB}
PRE_DS = {**{s: "mip_nerf_360" for s in MIP}, **{s: "tanks" for s in TNT}, **{s: "deep_blending" for s in DB}}

# paper numbers, PSNR/SSIM/LPIPS (protocol/protocol_R.md §4)
PAPER_3DGS = {
    "bicycle": (25.246, .771, .205), "flowers": (21.520, .605, .336), "garden": (27.410, .868, .103),
    "stump": (26.550, .775, .210), "treehill": (22.490, .638, .317), "room": (30.632, .914, .220),
    "counter": (28.700, .905, .204), "kitchen": (30.317, .922, .129), "bonsai": (31.980, .938, .205),
    "truck": (25.187, .879, .148), "train": (21.097, .802, .218), "drjohnson": (28.766, .899, .244),
    "playroom": (30.044, .906, .241),
}
PAPER_MCMC = {  # Tab. 5, Ours (SfM), mean of 3 runs
    "counter": (29.51, .92, .22), "stump": (27.80, .82, .19), "kitchen": (32.27, .94, .14),
    "bicycle": (26.15, .81, .18), "bonsai": (32.88, .95, .22), "room": (32.48, .94, .25),
    "garden": (28.16, .89, .10), "train": (22.47, .83, .24), "truck": (26.11, .89, .14),
    "drjohnson": (29.00, .89, .33), "playroom": (30.33, .90, .31),
}
PAPER_IBGS_DS = {"mip360": (28.33, .837, .186), "tnt": (24.84, .869, .148), "db": (30.12, .912, .237)}


def load(path):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return None
    v = list(d.values())[0]
    return (v["PSNR"], v["SSIM"], v["LPIPS"])


def ours(method, scene, tag, kind):
    d = OUT / method / f"{scene}_{tag}"
    if method == "ibgs":
        return load(d / ("results_renders.json" if kind == "raw" else "results_renders_aggregate.json"))
    return load(d / "results.json")


def released(scene, kind):
    d = PRE / PRE_DS[scene] / scene
    return load(d / ("results_renders.json" if kind == "raw" else "results_renders_aggregate.json"))


def fmt(t):
    return "—" if t is None else f"{t[0]:.2f} / {t[1]:.3f} / {t[2]:.3f}"


def dpsnr(a, b):
    return "—" if a is None or b is None else f"{a[0] - b[0]:+.2f}"


def mean(ts):
    ts = [t for t in ts]
    if not ts or any(t is None for t in ts):
        return None
    return tuple(sum(x[i] for x in ts) / len(ts) for i in range(3))


def main():
    rows, md = [], ["# Protocol R — kết quả tái hiện", "",
                    "PSNR / SSIM / LPIPS. `Δ` = ta − mốc (PSNR, dB). IBGS: `released` = JSON trong pretrained "
                    "tác giả phát hành; `pre` = checkpoint đó render + chấm lại trên env của ta.", ""]
    # IBGS
    for kind in ("final", "raw"):
        md += [f"## IBGS — {kind}", "",
               "| scene | released | pre (ta chấm lại) | Δ pre−rel | r1 (ta train) | Δ r1−rel |",
               "|---|---|---|---|---|---|"]
        for ds, scenes in DATASETS.items():
            for s in scenes:
                rel, pre, r1 = released(s, kind), ours("ibgs", s, "pre", kind), ours("ibgs", s, "r1", kind)
                md.append(f"| {s} | {fmt(rel)} | {fmt(pre)} | {dpsnr(pre, rel)} | {fmt(r1)} | {dpsnr(r1, rel)} |")
                rows.append(["ibgs", kind, ds, s, *(rel or ("",) * 3), *(pre or ("",) * 3), *(r1 or ("",) * 3)])
            rel = mean([released(s, kind) for s in scenes])
            pre = mean([ours("ibgs", s, "pre", kind) for s in scenes])
            r1 = mean([ours("ibgs", s, "r1", kind) for s in scenes])
            paper = PAPER_IBGS_DS[ds] if kind == "final" else None
            md.append(f"| **{ds} avg** (paper {fmt(paper)}) | {fmt(rel)} | {fmt(pre)} | {dpsnr(pre, rel)} | "
                      f"{fmt(r1)} | {dpsnr(r1, rel)} |")
        md.append("")
    # 3DGS, MCMC
    for method, paper in (("3dgs", PAPER_3DGS), ("mcmc", PAPER_MCMC)):
        md += [f"## {method.upper()}", "", "| scene | paper | r1 (ta) | Δ |", "|---|---|---|---|"]
        for ds, scenes in DATASETS.items():
            sc = [s for s in scenes if s in paper]
            for s in sc:
                r1 = ours(method, s, "r1", "final")
                md.append(f"| {s} | {fmt(paper[s])} | {fmt(r1)} | {dpsnr(r1, paper[s])} |")
                rows.append([method, "final", ds, s, *paper[s], "", "", "", *(r1 or ("",) * 3)])
            p, r = mean([paper[s] for s in sc]), mean([ours(method, s, "r1", "final") for s in sc])
            md.append(f"| **{ds} avg ({len(sc)})** | {fmt(p)} | {fmt(r)} | {dpsnr(r, p)} |")
        md.append("")
    rep = ROOT / "reports"
    rep.mkdir(exist_ok=True)
    (rep / "protocolR_table.md").write_text("\n".join(md) + "\n")
    with open(rep / "protocolR_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "kind", "dataset", "scene", "ref_psnr", "ref_ssim", "ref_lpips",
                    "pre_psnr", "pre_ssim", "pre_lpips", "r1_psnr", "r1_ssim", "r1_lpips"])
        w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()

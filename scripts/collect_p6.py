#!/usr/bin/env python3
"""P6 table: our one-stage method (outputs/p4/<scene>_p6_<variant>/results.json) vs MCMC baseline (Protocol R),
IBGS raw/final (Protocol R r1) and GADA published final. PSNR / SSIM / LPIPS. Stdlib only.

    python3 scripts/collect_p6.py m1 [ours_30000_sh]   # -> reports/p6_<variant>[_sh].md  (2nd arg: results.json method key)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V = sys.argv[1] if len(sys.argv) > 1 else "m1"
KEY = sys.argv[2] if len(sys.argv) > 2 else None   # e.g. ours_30000_sh (shared field applied at test)
GROUPS = {"mip360": ["bicycle", "flowers", "garden", "stump", "treehill", "bonsai", "counter", "kitchen", "room"],
          "tnt": ["train", "truck"], "db": ["drjohnson", "playroom"]}
GADA = {"bicycle": 26.16, "flowers": 22.29, "garden": 27.74, "stump": 27.33, "treehill": 23.16, "bonsai": 35.37, "counter": 30.84,
        "kitchen": 32.09, "room": 32.67, "train": 23.67, "truck": 26.19}   # published JSON (final), no DB


def load(p, key=None):
    try:
        d = json.load(open(p)); v = d[key] if key else list(d.values())[0]; return (v["PSNR"], v["SSIM"], v["LPIPS"])
    except (OSError, ValueError, IndexError, KeyError):
        return None


def fmt(t):
    return "—" if t is None else f"{t[0]:.2f} / {t[1]:.3f} / {t[2]:.3f}"


def mean(ts):
    ts = [t for t in ts if t is not None]
    return None if not ts else tuple(sum(x[i] for x in ts) / len(ts) for i in range(3))


TAG = V + ("_" + KEY.replace("ours_30000_", "") if KEY else "")
md = [f"# P6 — phase-consistent MCMC ({TAG}) vs baselines (PSNR / SSIM / LPIPS; 1 seed)", "",
      "| scene | MCMC (Protocol R) | **ours raw (one-stage)** | Δ vs MCMC | IBGS raw | IBGS final | GADA final | Δ ours − GADA |", "|---|---|---|---|---|---|---|---|"]
for g, scenes in GROUPS.items():
    rows = []
    for s in scenes:
        m = load(ROOT / "outputs" / "protocolR" / "mcmc" / (f"{s}_oreg001" if s == "playroom" else f"{s}_r1") / "results.json")
        o = load(ROOT / "outputs" / "p4" / f"{s}_p6_{V}" / "results.json", KEY)
        ir = load(ROOT / "outputs" / "protocolR" / "ibgs" / f"{s}_r1" / "results_renders.json")
        ifn = load(ROOT / "outputs" / "protocolR" / "ibgs" / f"{s}_r1" / "results_renders_aggregate.json")
        gd = GADA.get(s)
        rows.append((m, o, ir, ifn, gd))
        dm = "—" if (m is None or o is None) else f"{o[0]-m[0]:+.2f}"
        dg = "—" if (gd is None or o is None) else f"{o[0]-gd:+.2f}"
        md.append(f"| {s} | {fmt(m)} | **{fmt(o)}** | {dm} | {fmt(ir)} | {fmt(ifn)} | {gd if gd else '—'} | {dg} |")
    full = [r for r in rows if r[0] and r[1]]
    if full:
        M, O = mean([r[0] for r in full]), mean([r[1] for r in full])
        IR, IF = mean([r[2] for r in full]), mean([r[3] for r in full])
        gds = [r[4] for r in full if r[4]]; G = sum(gds) / len(gds) if gds else None
        md.append(f"| **{g} avg ({len(full)})** | {fmt(M)} | **{fmt(O)}** | {O[0]-M[0]:+.2f} | {fmt(IR)} | {fmt(IF)} | {G if G is None else f'{G:.2f}'} | "
                  f"{'—' if G is None else f'{O[0]-G:+.2f}'} |")
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"p6_{TAG}.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

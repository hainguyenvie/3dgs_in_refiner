#!/usr/bin/env bash
# One-shot status of the running experiment lines (P1, D1). Run on the server.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
date -u +%H:%M
echo "--- P1 estimation:"; grep -h SUMMARY logs/p1/est_*.log 2>/dev/null | cut -c1-260; tail -1 logs/p1/chain.log 2>/dev/null
echo "--- P1 retrains:"
for f in logs/p1/*_warp.log logs/p1/*_render.log; do
  [ -f "$f" ] && echo "$(basename $f .log): $(tail -c 600 $f | tr '\r' '\n' | grep -oE '[0-9]+/30000|JOB_DONE|PSNR":[0-9.]{5}' | tail -1)"
done
echo "--- D1 ($(ls outputs/irgs/*/*_r1/results_renders.json 2>/dev/null | wc -l)/32 done):"
python3 - <<'EOF'
import json, glob, os
ref = {"train": (22.61, 20.82, 23.69, 23.67), "garden": (28.19, 27.26, 27.57, 27.74), "bicycle": (26.13, 25.71, 26.06, 26.16),
       "stump": (27.69, 27.22, 27.31, 27.33), "bonsai": (32.78, 30.99, 34.98, 35.37), "counter": (29.43, 28.29, 30.65, 30.84)}
for d in sorted(glob.glob("outputs/irgs/*/*_r1")):
    if not os.path.exists(d + "/results_renders.json"):
        continue
    v, s = d.split("/")[2], os.path.basename(d)[:-3]
    raw = list(json.load(open(d + "/results_renders.json")).values())[0]["PSNR"]
    fp = d + "/results_renders_aggregate.json"
    fin = list(json.load(open(fp)).values())[0]["PSNR"] if os.path.exists(fp) else None
    m, ir, ifn, g = ref[s]
    ftxt = "-" if fin is None else f"{fin:.2f} ({fin-ifn:+.2f} vs ibgs, {fin-g:+.2f} vs GADA)"
    print(f"{v:12s} {s:8s} raw {raw:6.2f} ({raw-ir:+.2f} vs ibgs, {raw-m:+.2f} vs mcmc)  final {ftxt}")
EOF

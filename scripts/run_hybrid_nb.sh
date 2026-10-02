#!/usr/bin/env bash
# Nerfbusters dumps: explicit = vanilla 3DGS (eval renders), IBR = IBGS trained on the same split.  G=0 bash scripts/run_hybrid_nb.sh <scene>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; S="$1"; PY="$ROOT/.venv_ibgs/bin/python"
SRC=$ROOT/data/nb/$S; M=$ROOT/outputs/nb/ibgs/$S; MC=$ROOT/outputs/nb/3dgs/$S/test/ours_30000; OUT=$ROOT/outputs/route/hybrid/${S}_nb
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G}" OMP_NUM_THREADS=8
cd "$ROOT/src/irgs"
$PY -u ../route/dump_hybrid.py -s "$SRC" -m "$M" -r 2 --eval --multi_view_max_angle 50 --multi_view_max_dis 1000 --iteration 30000 --mcmc_dir "$MC" --out "$OUT"
touch "$OUT/.done"; echo JOB_DONE

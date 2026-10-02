#!/usr/bin/env bash
# Sector-holdout dumps: explicit candidate = vanilla 3DGS (same base as IBGS/GADA), IBR = IBGS trained on the sector split.
#   G=0 bash scripts/run_hybrid_sector.sh <scene>   -> outputs/route/hybrid/<scene>_sector
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; S="$1"; PY="$ROOT/.venv_ibgs/bin/python"
case "$S" in
  bonsai|counter|kitchen|room)            FLAGS="-r 2 --eval" ;;
  bicycle|flowers|garden|stump|treehill)  FLAGS="-r 4 --eval" ;;
  drjohnson|playroom)                     FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5" ;;
  train|truck)                            FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction" ;;
esac
SRC=$ROOT/data/sector/$S; M=$ROOT/outputs/sector/ibgs/$S; MC=$ROOT/outputs/sector/3dgs/$S/test/ours_30000; OUT=$ROOT/outputs/route/hybrid/${S}_sector
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G}" OMP_NUM_THREADS=8
cd "$ROOT/src/irgs"
$PY -u ../route/dump_hybrid.py -s "$SRC" -m "$M" $FLAGS --iteration 30000 --mcmc_dir "$MC" --out "$OUT"
touch "$OUT/.done"; echo JOB_DONE

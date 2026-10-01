#!/usr/bin/env bash
# R0 oracle routing probe on the IBGS released checkpoint of each scene (flags as run_ibgs.sh / exp_script.py).
#   G=0 bash scripts/run_r0.sh <scene> [extra probe args]      -> outputs/route/r0/<scene>.npz
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; S="$1"; shift
PY="$ROOT/.venv_ibgs/bin/python"; D="$ROOT/data"
case "$S" in
  bonsai|counter|kitchen|room)            SRC=$D/mipnerf360/$S;     FLAGS="-r 2 --eval"; DS=mip_nerf_360 ;;
  bicycle|flowers|garden|stump|treehill)  SRC=$D/mipnerf360/$S;     FLAGS="-r 4 --eval"; DS=mip_nerf_360 ;;
  drjohnson|playroom)                     SRC=$D/tandt_db/db/$S;    FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5"; DS=deep_blending ;;
  train|truck)                            SRC=$D/tandt_db/tandt/$S; FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction"; DS=tanks ;;
  *) echo "unknown scene $S"; exit 2 ;;
esac
M="$ROOT/checkpoints/ibgs_pretrained/output/$DS/$S"; OUT="$ROOT/outputs/route/r0"; mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G}" OMP_NUM_THREADS=8
cd "$ROOT/src/irgs"
$PY -u ../route/r0_probe.py -s "$SRC" -m "$M" $FLAGS --iteration 30000 --out "$OUT/$S.npz" "$@"
echo JOB_DONE

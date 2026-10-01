#!/usr/bin/env bash
# Dumps for the own fusion network (per-slot warps + camera features), on TEST views (full MCMC model) or on the
# cross-fitting DEV views (MODE=dev: data/dev/<scene> copy + MCMC trained without the dev views).
#   MODE=test|dev|devfull G=0 bash scripts/run_hybrid2.sh <scene>   -> outputs/route/hybrid/<scene>_{cf|dev|devfull}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; S="$1"; MODE="${MODE:-test}"
PY="$ROOT/.venv_ibgs/bin/python"; D="$ROOT/data"
case "$S" in
  bonsai|counter|kitchen|room)            SRC=$D/mipnerf360/$S;     FLAGS="-r 2 --eval"; DS=mip_nerf_360 ;;
  bicycle|flowers|garden|stump|treehill)  SRC=$D/mipnerf360/$S;     FLAGS="-r 4 --eval"; DS=mip_nerf_360 ;;
  drjohnson|playroom)                     SRC=$D/tandt_db/db/$S;    FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5"; DS=deep_blending ;;
  train|truck)                            SRC=$D/tandt_db/tandt/$S; FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction"; DS=tanks ;;
  cd|guitars|lab)                         SRC=$D/shiny/_SHINNY_DATASET_/$S; FLAGS="-r 1008 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5"; DS=shiny ;;
esac
M="$ROOT/checkpoints/ibgs_pretrained/output/$DS/$S"           # used for its GEOMETRY (depth for warping) only in the own method
if [ "$MODE" = devfull ]; then   # dev views with BOTH models cross-fitted (IBGS and MCMC trained without them)
  SRC=$D/dev/$S; M="$ROOT/outputs/protocolR/ibgs_dev/$S"; MC="$ROOT/outputs/protocolR/mcmc_dev/$S/test/ours_30000"; OUT="$ROOT/outputs/route/hybrid/${S}_devfull"
elif [ "$MODE" = dev ]; then
  SRC=$D/dev/$S; MC="$ROOT/outputs/protocolR/mcmc_dev/$S/test/ours_30000"; OUT="$ROOT/outputs/route/hybrid/${S}_dev"
else
  MC="$ROOT/outputs/protocolR/mcmc/${S}_r1/test/ours_30000"; OUT="$ROOT/outputs/route/hybrid/${S}_cf"
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G}" OMP_NUM_THREADS=8
cd "$ROOT/src/irgs"
$PY -u ../route/dump_hybrid.py -s "$SRC" -m "$M" $FLAGS --iteration 30000 --mcmc_dir "$MC" --out "$OUT"
touch "$OUT/.done"; echo JOB_DONE

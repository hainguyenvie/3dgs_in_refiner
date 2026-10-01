#!/usr/bin/env bash
# Dump per-view hybrid tensors (MCMC raw vs IBGS released) for one scene.   G=0 bash scripts/run_hybrid.sh <scene> [mcmc_tag]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; S="$1"; TAG="${2:-r1}"
PY="$ROOT/.venv_ibgs/bin/python"; D="$ROOT/data"
case "$S" in
  bonsai|counter|kitchen|room)            SRC=$D/mipnerf360/$S;     FLAGS="-r 2 --eval"; DS=mip_nerf_360 ;;
  bicycle|flowers|garden|stump|treehill)  SRC=$D/mipnerf360/$S;     FLAGS="-r 4 --eval"; DS=mip_nerf_360 ;;
  drjohnson|playroom)                     SRC=$D/tandt_db/db/$S;    FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5"; DS=deep_blending ;;
  train|truck)                            SRC=$D/tandt_db/tandt/$S; FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction"; DS=tanks ;;
esac
M="$ROOT/checkpoints/ibgs_pretrained/output/$DS/$S"; MC="$ROOT/outputs/protocolR/mcmc/${S}_${TAG}/test/ours_30000"
OUT="$ROOT/outputs/route/hybrid/${S}_${TAG}${OUT_SUFFIX:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G}" OMP_NUM_THREADS=8
cd "$ROOT/src/irgs"
GEO_ARGS=""; [ "${GEO:-0}" = 1 ] && GEO_ARGS="--geom_ply $ROOT/outputs/protocolR/mcmc/${S}_${TAG}/point_cloud/iteration_30000/point_cloud.ply"   # single-model variant
$PY -u ../route/dump_hybrid.py -s "$SRC" -m "$M" $FLAGS --iteration 30000 --mcmc_dir "$MC" --out "$OUT" $GEO_ARGS ${EXTRA:-}
echo JOB_DONE

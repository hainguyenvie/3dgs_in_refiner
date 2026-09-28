#!/usr/bin/env bash
# Our fork of IBGS (src/irgs) with diagnostic / method variants. Same data, flags and metric as Protocol R IBGS.
#
#   G=<card> setsid nohup bash scripts/run_irgs.sh <variant> <scene> [tag] > logs/irgs/<variant>_<scene>.log 2>&1 < /dev/null &
#   SMOKE=1 G=7 bash scripts/run_irgs.sh mcmc_agg bicycle smoke
#
# Variants (flags are defined HERE so every run is reproducible from the variant name):
#   ibgs         fork with default flags — must reproduce IBGS (numerics check of the fork)
#   noagg        base only: IBGS geometry losses, no residual branch            (co-adaptation study)
#   detach       residual branch trained on top, final loss never reaches the Gaussians; raw loss weight 1
#   mcmc_agg     MCMC relocation densification (cap_max per scene) + IBGS residual branch
#   mcmc_noagg   MCMC densification, base only
#   mcmc_detach  MCMC densification + detach
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="$1"; SCENE="$2"; TAG="${3:-r1}"
PY="$ROOT/.venv_ibgs/bin/python"
CODE="$ROOT/src/irgs"
D="$ROOT/data"
case "$SCENE" in
  bonsai|counter|kitchen|room)            SRC=$D/mipnerf360/$SCENE;     FLAGS="-r 2 --eval" ;;
  bicycle|flowers|garden|stump|treehill)  SRC=$D/mipnerf360/$SCENE;     FLAGS="-r 4 --eval" ;;
  drjohnson|playroom)                     SRC=$D/tandt_db/db/$SCENE;    FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5" ;;
  train|truck)                            SRC=$D/tandt_db/tandt/$SCENE; FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction" ;;
  *) echo "unknown scene $SCENE"; exit 2 ;;
esac
# cap_max: 3DGS-MCMC author configs; flowers/treehill (no author config) = final #Gaussians of our 3DGS run,
# i.e. the rule the MCMC paper states ("final number of Gaussians reached by the original 3DGS run").
declare -A CAP=([bicycle]=5900000 [garden]=5200000 [stump]=4750000 [room]=1500000 [counter]=1200000
                [kitchen]=1800000 [bonsai]=1300000 [train]=1100000 [truck]=2600000 [drjohnson]=3400000
                [playroom]=2500000 [flowers]=2870000 [treehill]=3210000)
# DB: opacity_reg 0.001 (MCMC paper; Protocol R finding — repo playroom config lacks it)
OREG=0.01; case "$SCENE" in drjohnson|playroom) OREG=0.001 ;; esac
MCMC="--densify_mode mcmc --cap_max ${CAP[$SCENE]} --opacity_reg $OREG"
case "$VARIANT" in
  ibgs)        VFLAGS="" ;;
  noagg)       VFLAGS="--disable_color_aggregation" ;;
  detach)      VFLAGS="--agg_detach_base --raw_loss_weight 1.0 --agg_loss_weight 1.0" ;;
  mcmc_agg)    VFLAGS="$MCMC" ;;
  mcmc_noagg)  VFLAGS="$MCMC --disable_color_aggregation" ;;
  mcmc_detach) VFLAGS="$MCMC --agg_detach_base --raw_loss_weight 1.0 --agg_loss_weight 1.0" ;;
  *) echo "unknown variant $VARIANT"; exit 2 ;;
esac
EXTRA=""; IT=30000
if [ "${SMOKE:-0}" = 1 ]; then
  IT=800
  EXTRA="--iterations 800 --densify_until_iter 500 --densify_from_iter 100 --mcmc_densify_until_iter 700
         --start_color_aggregation_iter 300 --color_aggregate_burnin_steps 100
         --single_view_weight_from_iter 300 --multi_view_weight_from_iter 300 --test_iterations 800 --save_iterations 800"
fi
OUT="$ROOT/outputs/irgs/$VARIANT/${SCENE}_${TAG}"
mkdir -p "$OUT" "$ROOT/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G=<card>}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
vram_sampler() {
  while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i $G 2>/dev/null \
      | sed "s/^/$(date +%s),/" >> "$OUT/vram_samples.csv" || true
    sleep 10
  done
}
stage() {
  local t0 pid rc=0; t0=$(date +%s)
  "${@:2}" & pid=$!
  echo "$1 pid=$pid start=$t0" >> "$OUT/stages.txt"
  wait $pid || rc=$?
  echo "$1 pid=$pid end=$(date +%s) secs=$(( $(date +%s) - t0 )) rc=$rc" >> "$OUT/stages.txt"
  return $rc
}
{
  echo "variant=$VARIANT scene=$SCENE tag=$TAG card=$G src=$SRC flags=$FLAGS vflags=$VFLAGS extra=$EXTRA"
  echo "host_utc=$(date -u +%FT%TZ) code_md5=$(cat $CODE/train.py $CODE/scene/gaussian_model.py $CODE/color_aggregation_network.py | md5sum | cut -c1-12)"
} | tee "$OUT/run_meta.txt"

cd "$CODE"
vram_sampler & VS=$!
trap 'kill $VS 2>/dev/null || true' EXIT
[ -f "$OUT/point_cloud/iteration_$IT/point_cloud.ply" ] || stage train $PY -u train.py -s "$SRC" -m "$OUT" $FLAGS $VFLAGS $EXTRA
[ -f "$OUT/result_fps_mem.json" ] || stage render $PY -u render.py -s "$SRC" -m "$OUT" $FLAGS $VFLAGS --skip_train --iteration $IT
[ -f "$OUT/results_renders.json" ] || stage metrics $PY -u metrics.py -m "$OUT"
cat "$OUT/stages.txt"; cat "$OUT"/results_renders*.json "$OUT/result_fps_mem.json" 2>/dev/null || true
echo JOB_DONE

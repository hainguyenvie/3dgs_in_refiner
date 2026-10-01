#!/usr/bin/env bash
# Protocol R — 3DGS (graphdeco full_eval.py flags) and 3DGS-MCMC (author configs, SfM init).
#
#   G=<card> setsid nohup bash scripts/run_gs_baseline.sh <3dgs|mcmc> <scene> [tag] > logs/<m>_<scene>.log 2>&1 < /dev/null &
#   SMOKE=1 G=7 bash scripts/run_gs_baseline.sh mcmc bicycle smoke    # 1000-iter crash test, never a result
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="$1"; SCENE="$2"; TAG="${3:-r1}"
D="$ROOT/data"
case "$SCENE" in
  bicycle|flowers|garden|stump|treehill) SRC=$D/mipnerf360/$SCENE;     GRP=outdoor ;;
  bonsai|counter|kitchen|room)           SRC=$D/mipnerf360/$SCENE;     GRP=indoor ;;
  train|truck)                           SRC=$D/tandt_db/tandt/$SCENE; GRP=tnt ;;
  drjohnson|playroom)                    SRC=$D/tandt_db/db/$SCENE;    GRP=db ;;
  cd|guitars|lab)                        SRC=$D/shiny/_SHINNY_DATASET_/$SCENE; GRP=shiny ;;   # IBGS-processed Shiny (gate training only)
  *) echo "unknown scene $SCENE"; exit 2 ;;
esac
[ -n "${SRC_OVERRIDE:-}" ] && SRC="$SRC_OVERRIDE"          # e.g. a re-calibrated copy of the scene (data/calib/<scene>_pf)
case "$METHOD" in
  3dgs)
    REPO=$ROOT/third_party/gaussian-splatting; PY=$ROOT/.venv_3dgs/bin/python
    case $GRP in outdoor) FLAGS="-i images_4" ;; indoor) FLAGS="-i images_2" ;; *) FLAGS="" ;; esac
    TRAIN_FLAGS="$FLAGS --disable_viewer --quiet --eval --test_iterations -1"
    RENDER_FLAGS="--quiet --eval --skip_train" ;;
  mcmc)
    REPO=$ROOT/third_party/3dgs-mcmc; PY=$ROOT/.venv_mcmc/bin/python
    CFG=$REPO/configs/$SCENE.json
    [ -f "$CFG" ] || CFG=$ROOT/configs/mcmc/$SCENE.json     # our config for scenes the repo lacks (flowers, treehill)
    [ -n "${CFG_OVERRIDE:-}" ] && CFG="$CFG_OVERRIDE"       # e.g. playroom: repo config lacks opacity_reg (week-1 finding)
    [ -f "$CFG" ] || { echo "no config (cap_max) for $SCENE"; exit 3; }
    RES=$($PY -c "import json;print(json.load(open('$CFG'))['resolution'])")
    TRAIN_FLAGS="--config $CFG --eval --init_type sfm --quiet"
    RENDER_FLAGS="--quiet --eval --skip_train -r $RES" ;;
  *) echo "method must be 3dgs|mcmc"; exit 2 ;;
esac
# EXTRA_TRAIN: extra train flags for a DIAGNOSTIC run (always give it its own tag; never overwrite r1)
[ -n "${EXTRA_TRAIN:-}" ] && TRAIN_FLAGS="$TRAIN_FLAGS $EXTRA_TRAIN"
IT=30000
if [ "${SMOKE:-0}" = 1 ]; then IT=1000; TRAIN_FLAGS="$TRAIN_FLAGS --iterations 1000 --save_iterations 1000 --densify_until_iter 800"; fi
OUT="${OUT_OVERRIDE:-$ROOT/outputs/protocolR/$METHOD/${SCENE}_${TAG}}"
mkdir -p "$OUT" "$ROOT/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G=<card>}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
vgate() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -le "${VGATE_MIB:-1000}" ]; do sleep 120; done; }
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
  echo "method=$METHOD scene=$SCENE tag=$TAG card=$G src=$SRC train_flags=$TRAIN_FLAGS render_flags=$RENDER_FLAGS"
  echo "host_utc=$(date -u +%FT%TZ) commit=$(git -C "$REPO" rev-parse HEAD)"
  $PY -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0))"
} | tee "$OUT/run_meta.txt"

cd "$REPO"
[ "${NO_VGATE:-0}" = 1 ] || vgate
vram_sampler & VS=$!
trap 'kill $VS 2>/dev/null || true' EXIT
[ -f "$OUT/point_cloud/iteration_$IT/point_cloud.ply" ] || stage train $PY -u train.py -s "$SRC" -m "$OUT" $TRAIN_FLAGS
[ -d "$OUT/test/ours_$IT/renders" ] && [ "$(ls "$OUT/test/ours_$IT/renders" | wc -l)" -gt 0 ] || \
  stage render $PY -u render.py -s "$SRC" -m "$OUT" --iteration $IT $RENDER_FLAGS
[ -f "$OUT/results.json" ] || stage metrics $PY -u metrics.py -m "$OUT"
echo "num_gaussians=$(head -c 400 "$OUT/point_cloud/iteration_$IT/point_cloud.ply" | grep -a -m1 'element vertex' | awk '{print $3}')" >> "$OUT/stages.txt"
cat "$OUT/stages.txt" "$OUT/results.json"
echo JOB_DONE

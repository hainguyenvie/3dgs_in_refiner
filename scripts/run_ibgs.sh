#!/usr/bin/env bash
# Protocol R — IBGS (NeurIPS'25) exactly as the authors' exp_script.py: train -> render -> metrics.
#
#   G=<card> setsid nohup bash scripts/run_ibgs.sh <scene> [tag] > logs/ibgs_<scene>.log 2>&1 < /dev/null &
#   SMOKE=1 G=7 bash scripts/run_ibgs.sh bicycle smoke      # 800-iter crash test, never a result
#
# Flags per scene group are copied from third_party/ibgs/exp_script.py (commit in run_meta.txt).
# Only deviation: render.py gets --skip_train (train-view renders do not enter the test metric).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$1"; TAG="${2:-r1}"
PY="$ROOT/.venv_ibgs/bin/python"
D="$ROOT/data"
case "$SCENE" in
  bonsai|counter|kitchen|room)            SRC=$D/mipnerf360/$SCENE;     FLAGS="-r 2 --eval" ;;
  bicycle|flowers|garden|stump|treehill)  SRC=$D/mipnerf360/$SCENE;     FLAGS="-r 4 --eval" ;;
  drjohnson|playroom)                     SRC=$D/tandt_db/db/$SCENE;    FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5" ;;
  train|truck)                            SRC=$D/tandt_db/tandt/$SCENE; FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction" ;;
  *) echo "unknown scene $SCENE"; exit 2 ;;
esac
EXTRA=""; IT=30000
if [ "${SMOKE:-0}" = 1 ]; then
  IT=800
  EXTRA="--iterations 800 --densify_until_iter 500 --start_color_aggregation_iter 300 --color_aggregate_burnin_steps 100
         --single_view_weight_from_iter 300 --multi_view_weight_from_iter 300 --test_iterations 800 --save_iterations 800"
fi
OUT="$ROOT/outputs/protocolR/ibgs/${SCENE}_${TAG}"
mkdir -p "$OUT" "$ROOT/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G=<card>}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
vgate() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -le "${VGATE_MIB:-1000}" ]; do sleep 120; done; }

# VRAM on our card every 10 s: "unix_ts, pid, MiB" for every process; filter later by the PIDs in stages.txt
vram_sampler() {
  while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i $G 2>/dev/null \
      | sed "s/^/$(date +%s),/" >> "$OUT/vram_samples.csv" || true
    sleep 10
  done
}
stage() {  # stage <name> <cmd...>: run, record PID + wall time, propagate exit code
  local t0 pid rc=0; t0=$(date +%s)
  "${@:2}" & pid=$!
  echo "$1 pid=$pid start=$t0" >> "$OUT/stages.txt"
  wait $pid || rc=$?
  echo "$1 pid=$pid end=$(date +%s) secs=$(( $(date +%s) - t0 )) rc=$rc" >> "$OUT/stages.txt"
  return $rc
}

{
  echo "scene=$SCENE tag=$TAG card=$G src=$SRC flags=$FLAGS extra=$EXTRA"
  echo "host_utc=$(date -u +%FT%TZ) ibgs_commit=$(git -C "$ROOT/third_party/ibgs" rev-parse HEAD)"
  $PY -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0))"
} | tee "$OUT/run_meta.txt"

cd "$ROOT/third_party/ibgs"
[ "${NO_VGATE:-0}" = 1 ] || vgate
vram_sampler & VS=$!
trap 'kill $VS 2>/dev/null || true' EXIT

[ -f "$OUT/point_cloud/iteration_$IT/point_cloud.ply" ] || stage train $PY -u train.py -s "$SRC" -m "$OUT" $FLAGS $EXTRA
[ -f "$OUT/result_fps_mem.json" ] || stage render $PY -u render.py -s "$SRC" -m "$OUT" $FLAGS --skip_train --iteration $IT
[ -f "$OUT/results_renders_aggregate.json" ] || stage metrics $PY -u metrics.py -m "$OUT"
cat "$OUT/stages.txt" "$OUT/results_renders.json" "$OUT/results_renders_aggregate.json" "$OUT/result_fps_mem.json"
echo JOB_DONE

#!/usr/bin/env bash
# E7 self-consistency oracle: train MCMC (author config for the scene) on the synthetic dataset data/e7/<scene>_syn_s<σ>.
#   G=<card> setsid nohup bash scripts/run_e7.sh <scene> <sigma> > logs/e7/<scene>_s<sigma>.log 2>&1 < /dev/null &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$1"; SIG="$2"
REPO=$ROOT/third_party/3dgs-mcmc; PY=$ROOT/.venv_mcmc/bin/python
SRC=$ROOT/data/e7/${SCENE}_syn_s${SIG}
OUT=$ROOT/outputs/e7/${SCENE}_syn_s${SIG}
mkdir -p "$OUT" "$ROOT/logs/e7"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$REPO"
# author config but resolution 1: the synthetic images are already at eval resolution
CAP=$(python3 -c "import json;print(json.load(open('configs/$SCENE.json'))['cap_max'])")
OREG=$(python3 -c "import json;print(json.load(open('configs/$SCENE.json')).get('opacity_reg',0.01))")
echo "scene=$SCENE sigma=$SIG cap_max=$CAP opacity_reg=$OREG src=$SRC utc=$(date -u +%FT%TZ)" | tee "$OUT/run_meta.txt"
[ -f "$OUT/point_cloud/iteration_30000/point_cloud.ply" ] || \
  $PY -u train.py -s "$SRC" -m "$OUT" -r 1 --eval --init_type sfm --cap_max $CAP --opacity_reg $OREG --quiet
[ -d "$OUT/test/ours_30000/renders" ] || $PY -u render.py -s "$SRC" -m "$OUT" --iteration 30000 -r 1 --eval --skip_train --quiet
[ -f "$OUT/results.json" ] || $PY -u metrics.py -m "$OUT"
cat "$OUT/results.json"; echo JOB_DONE

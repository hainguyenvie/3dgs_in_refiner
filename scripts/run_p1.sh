#!/usr/bin/env bash
# P1 retrain: 3DGS-MCMC (author config) on a pose-corrected scene copy data/p1/<tag> (test poses untouched).
#   G=<card> setsid nohup bash scripts/run_p1.sh <scene> <tag> > logs/p1/<tag>.log 2>&1 < /dev/null &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$1"; TAG="$2"
REPO=$ROOT/third_party/3dgs-mcmc; PY=$ROOT/.venv_mcmc/bin/python
SRC=$ROOT/data/p1/$TAG; OUT=$ROOT/outputs/p1/$TAG
mkdir -p "$OUT" "$ROOT/logs/p1"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$REPO"
CFG=$REPO/configs/$SCENE.json; RES=$(python3 -c "import json;print(json.load(open('$CFG'))['resolution'])")
echo "scene=$SCENE tag=$TAG src=$SRC cfg=$CFG utc=$(date -u +%FT%TZ)" | tee "$OUT/run_meta.txt"
[ -f "$OUT/point_cloud/iteration_30000/point_cloud.ply" ] || $PY -u train.py -s "$SRC" -m "$OUT" --config "$CFG" --eval --init_type sfm --quiet
[ -d "$OUT/test/ours_30000/renders" ] || $PY -u render.py -s "$SRC" -m "$OUT" --iteration 30000 -r $RES --eval --skip_train --quiet
[ -f "$OUT/results.json" ] || $PY -u metrics.py -m "$OUT"
cat "$OUT/results.json"; echo JOB_DONE

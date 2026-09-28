#!/usr/bin/env bash
# P4: MCMC (author config, original data) with the phase-aware loss (render warped by the per-view smooth field).
#   G=<card> [EXTRA="--learn_phase ..."] setsid nohup bash scripts/run_p4.sh <scene> <flow_scale> <tag> > logs/p4/<tag>.log 2>&1 < /dev/null &
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$1"; SCALE="$2"; TAG="$3"
REPO=$ROOT/third_party/3dgs-mcmc; PY=$ROOT/.venv_mcmc/bin/python
case "$SCENE" in train|truck) SRC=$ROOT/data/tandt_db/tandt/$SCENE;; drjohnson|playroom) SRC=$ROOT/data/tandt_db/db/$SCENE;; *) SRC=$ROOT/data/mipnerf360/$SCENE;; esac
OUT=$ROOT/outputs/p4/$TAG; mkdir -p "$OUT" "$ROOT/logs/p4"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$REPO"
# author config, or our config for scenes the MCMC repo lacks (flowers, treehill) / fixes (playroom opacity_reg)
CFG=$ROOT/configs/mcmc/$SCENE.json; [ -f "$CFG" ] || CFG=$REPO/configs/$SCENE.json
RES=$(python3 -c "import json;print(json.load(open('$CFG'))['resolution'])")
FIELDS=$ROOT/data/p4/${SCENE}_fields.json; [ "${NO_FIELDS:-0}" = 1 ] && FIELDS=none
echo "scene=$SCENE scale=$SCALE tag=$TAG fields=$FIELDS extra=${EXTRA:-} utc=$(date -u +%FT%TZ) code_md5=$(md5sum $ROOT/src/phase/train_mcmc_phase.py | cut -c1-12)" | tee "$OUT/run_meta.txt"
[ -f "$OUT/point_cloud/iteration_30000/point_cloud.ply" ] || \
  $PY -u $ROOT/src/phase/train_mcmc_phase.py -s "$SRC" -m "$OUT" --config "$CFG" --eval --init_type sfm --quiet \
      --view_flow $FIELDS --flow_scale $SCALE ${EXTRA:-}
[ -d "$OUT/test/ours_30000/renders" ] || $PY -u render.py -s "$SRC" -m "$OUT" --iteration 30000 -r $RES --eval --skip_train --quiet
[ -f "$OUT/color_affine.json" ] && [ ! -d "$OUT/test/ours_30000_m3/renders" ] && $PY $ROOT/scripts/analysis/m3_apply_test.py "$OUT" "$SRC" 3
[ -f "$OUT/phase_shared.json" ] && [ ! -d "$OUT/test/ours_30000_sh/renders" ] && $PY $ROOT/scripts/analysis/shared_apply_test.py "$OUT"
[ -f "$OUT/results.json" ] || $PY -u metrics.py -m "$OUT"
cat "$OUT/results.json"; echo JOB_DONE

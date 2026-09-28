#!/usr/bin/env bash
# E8: (1) render TRAIN views of the IBGS released model (png), (2) build the synthetic consistent scene,
# (3) dump per-source warps of the synthetic sources with the model's own depth, (4) E1b/E1c on them.
#   G=<card> setsid nohup bash scripts/run_e8.sh bicycle garden > logs/e8.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY=$ROOT/.venv_ibgs/bin/python; D=$ROOT/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS=4
cd $ROOT/src/irgs
for s in "$@"; do
  case "$s" in bonsai|counter|kitchen|room) SRC=$D/mipnerf360/$s; R=2;; *) SRC=$D/mipnerf360/$s; R=4;; esac
  PRE=$ROOT/outputs/protocolR/ibgs/${s}_pre
  # (1) train renders (png, full quality)
  [ -d $PRE/train/ours_30000/renders ] && [ "$(ls $PRE/train/ours_30000/renders | wc -l)" -gt 100 ] || \
    $PY render.py -s $SRC -m $PRE -r $R --eval --skip_test --iteration 30000 --train_render_png
  # (2) synthetic scene
  python3 $ROOT/scripts/analysis/e8_build_synthetic_ibgs.py --scene $s --res $R
  # (3) warps of synthetic sources: model dir = symlinks to the released checkpoint, separate output
  M=$ROOT/outputs/e8/${s}_ibgs_syn; mkdir -p $M
  for f in cfg_args config.json multi_view.json multi_view_test.json input.ply point_cloud app_model color_aggregate_checkpoint; do
    [ -e $PRE/$f ] && [ ! -e $M/$f ] && ln -s $(readlink -f $PRE/$f) $M/$f; done
  [ -d $M/test/ours_30000/warps ] || $PY render.py -s $D/e8/${s}_ibgs_syn -m $M -r 1 --eval --skip_train --iteration 30000 --dump_warps
  echo "e8 dumped $s $(ls $M/test/ours_30000/warps | wc -l)"
done
echo JOB_DONE

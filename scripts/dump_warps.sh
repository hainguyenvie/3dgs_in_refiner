#!/usr/bin/env bash
# Re-render test views of the IBGS *released* checkpoints (outputs/protocolR/ibgs/<scene>_pre) with the fork's
# render.py --dump_warps, then run the E1b analysis.   G=<card> bash scripts/dump_warps.sh bicycle garden ...
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY=$ROOT/.venv_ibgs/bin/python; D=$ROOT/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS=4
cd $ROOT/src/irgs
for s in "$@"; do
  case "$s" in
    bonsai|counter|kitchen|room)            SRC=$D/mipnerf360/$s;     FLAGS="-r 2 --eval" ;;
    bicycle|flowers|garden|stump|treehill)  SRC=$D/mipnerf360/$s;     FLAGS="-r 4 --eval" ;;
    drjohnson|playroom)                     SRC=$D/tandt_db/db/$s;    FLAGS="-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5" ;;
    train|truck)                            SRC=$D/tandt_db/tandt/$s; FLAGS="-r 2 --eval --exposure_compensation --enable_exposure_correction" ;;
  esac
  M=$ROOT/outputs/protocolR/ibgs/${s}_pre
  [ -d $M/test/ours_30000/warps ] && [ "$(ls $M/test/ours_30000/warps | wc -l)" -gt 0 ] || \
    $PY render.py -s $SRC -m $M $FLAGS --skip_train --iteration 30000 --dump_warps
  echo "dumped $s $(ls $M/test/ours_30000/warps | wc -l)"
done
$PY -u $ROOT/scripts/analysis/e1b_source_warps.py "$@"
echo JOB_DONE

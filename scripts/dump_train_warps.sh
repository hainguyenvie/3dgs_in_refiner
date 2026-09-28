#!/usr/bin/env bash
# Dump per-source warps, masks, depth and GT for TRAIN views of the IBGS released checkpoints (P1 input).
#   G=6 setsid nohup bash scripts/dump_train_warps.sh bicycle garden counter > logs/dump_train_warps.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY=$ROOT/.venv_ibgs/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS=4
cd $ROOT/src/irgs
for s in "$@"; do
  case "$s" in bonsai|counter|kitchen|room) R=2;; *) R=4;; esac
  M=$ROOT/outputs/protocolR/ibgs/${s}_pre; T=$M/train/ours_30000
  if [ ! -d $T/gt ] || [ "$(ls $T/gt 2>/dev/null | wc -l)" -lt 10 ]; then
    rm -rf $T/warps $T/warpmask $T/gt
    $PY render.py -s $ROOT/data/mipnerf360/$s -m $M -r $R --eval --skip_test --iteration 30000 --dump_warps --train_render_png
  fi
  echo "dumped-train $s warps=$(ls $T/warps | wc -l) gt=$(ls $T/gt | wc -l)"
done
echo JOB_DONE

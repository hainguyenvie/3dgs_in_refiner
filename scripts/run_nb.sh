#!/usr/bin/env bash
# Nerfbusters scene (official train/eval videos via split.json, downscale 2 as published): vanilla 3DGS and IBGS.
#   G=0 bash scripts/run_nb.sh <scene> [3dgs|ibgs|both]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; S="$1"; WHAT="${2:-both}"; SRC=$ROOT/data/nb/$S
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?set G}" OMP_NUM_THREADS=8
if [[ "$WHAT" == 3dgs || "$WHAT" == both ]]; then
  O=$ROOT/outputs/nb/3dgs/$S; mkdir -p $O; cd $ROOT/third_party/gaussian-splatting
  [ -f $O/point_cloud/iteration_30000/point_cloud.ply ] || $ROOT/.venv_3dgs/bin/python train.py -s $SRC -m $O -r 2 --eval --disable_viewer --quiet --test_iterations -1
  [ -d $O/test/ours_30000/renders ] || $ROOT/.venv_3dgs/bin/python render.py -s $SRC -m $O -r 2 --eval --skip_train --quiet
  [ -f $O/results.json ] || $ROOT/.venv_3dgs/bin/python metrics.py -m $O
  echo "3DGS $S done"
fi
if [[ "$WHAT" == ibgs || "$WHAT" == both ]]; then
  O=$ROOT/outputs/nb/ibgs/$S; mkdir -p $O; cd $ROOT/third_party/ibgs
  F="-r 2 --eval --multi_view_max_angle 50 --multi_view_max_dis 1000"
  [ -f $O/point_cloud/iteration_30000/point_cloud.ply ] || $ROOT/.venv_ibgs/bin/python -u train.py -s $SRC -m $O $F
  [ -f $O/result_fps_mem.json ] || $ROOT/.venv_ibgs/bin/python -u render.py -s $SRC -m $O $F --skip_train --iteration 30000
  [ -f $O/results_renders_aggregate.json ] || $ROOT/.venv_ibgs/bin/python -u metrics.py -m $O
  echo "IBGS $S done"
fi

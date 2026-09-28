#!/bin/bash
# Render TRAIN views of Protocol R MCMC checkpoints (for the E3 consistency audit). No training.
#   G=3 setsid nohup bash scripts/render_train_views.sh bicycle garden ... > logs/render_train_views.log 2>&1 < /dev/null &
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY=$R/.venv_mcmc/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${G:-3} OMP_NUM_THREADS=4
cd $R/third_party/3dgs-mcmc
for s in "$@"; do
  M=$R/outputs/protocolR/mcmc/${s}_r1
  case $s in train|truck) SRC=$R/data/tandt_db/tandt/$s;; drjohnson|playroom) SRC=$R/data/tandt_db/db/$s;; *) SRC=$R/data/mipnerf360/$s;; esac
  RES=$(python3 -c "import json;print(json.load(open('$R/third_party/3dgs-mcmc/configs/$s.json'))['resolution'])")
  if [ ! -d $M/train/ours_30000/renders ] || [ "$(ls $M/train/ours_30000/renders | wc -l)" -eq 0 ]; then
    $PY render.py -s $SRC -m $M --iteration 30000 --skip_test --quiet --eval -r $RES
  fi
  echo "done $s $(ls $M/train/ours_30000/renders 2>/dev/null | wc -l)"
done
echo JOB_DONE

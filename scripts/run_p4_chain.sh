#!/usr/bin/env bash
# P4 chain: fields for the scenes, then trainings (scale 1) + one code-path control (scale 0) on the first scene.
#   setsid nohup bash scripts/run_p4_chain.sh bicycle garden stump > logs/p4/chain.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd $ROOT; mkdir -p logs/p4
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G_EST:-5}" OMP_NUM_THREADS=4 .venv_ibgs/bin/python scripts/analysis/p4_fields.py "$@" 2>&1 | grep -vi warning
i=0
for s in "$@"; do
  [ -f data/p4/${s}_fields.json ] || { echo "no fields for $s"; continue; }
  (G=$(( (i + ${G0:-4}) % 8 )) setsid nohup bash scripts/run_p4.sh $s 1.0 ${s}_phase1 > logs/p4/${s}_phase1.log 2>&1 < /dev/null &); i=$((i+1))
done
s="$1"; (G=$(( (i + ${G0:-4}) % 8 )) setsid nohup bash scripts/run_p4.sh $s 0.0 ${s}_phase0 > logs/p4/${s}_phase0.log 2>&1 < /dev/null &)
echo "P4_TRAIN_LAUNCHED"

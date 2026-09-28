#!/usr/bin/env bash
# P1 chain: estimate per-view corrections (warp signal + render-signal control) then retrain MCMC on each.
#   setsid nohup bash scripts/run_p1_chain.sh bicycle garden > logs/p1/chain.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd $ROOT; mkdir -p logs/p1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G_EST:-6}" OMP_NUM_THREADS=4
TAGS=()
for s in "$@"; do
  for sig in warp render; do
    .venv_ibgs/bin/python scripts/analysis/p1_estimate_view_corrections.py $s --signal $sig > logs/p1/est_${s}_${sig}.log 2>&1
    grep SUMMARY logs/p1/est_${s}_${sig}.log || { echo "estimation failed: $s $sig"; continue; }
    TAGS+=("${s}_${sig}${P1_SUFFIX:-}")
  done
done
i=0
for t in "${TAGS[@]}"; do
  s=${t%%_*}
  (G=$(( (i + ${G0:-0}) % 8 )) setsid nohup bash scripts/run_p1.sh $s $t > logs/p1/$t.log 2>&1 < /dev/null &)
  i=$((i+1))
done
echo "P1_TRAIN_LAUNCHED ${TAGS[*]}"

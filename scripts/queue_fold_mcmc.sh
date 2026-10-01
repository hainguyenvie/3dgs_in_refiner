#!/usr/bin/env bash
# MCMC on the extra cross-fitting folds (data/devf/<scene>_f<k>), <= MAXJ concurrent trainings on the card.
#   setsid nohup bash scripts/queue_fold_mcmc.sh > logs/queue_fold_mcmc.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; MAXJ=${MAXJ:-3}
for k in 2 4 6; do for s in ${FOLD_SCENES:-bonsai counter garden}; do
  d=${s}_f$k; [ -f outputs/protocolR/mcmc_devf/$d/results.json ] && continue
  while [ "$(pgrep -fc 'venv_mcmc/bin/python -u train.py')" -ge "$MAXJ" ]; do sleep 60; done
  echo "[$(date -u +%FT%TZ)] launch fold MCMC $d"
  G=0 NO_VGATE=1 SRC_OVERRIDE=$ROOT/data/devf/$d OUT_OVERRIDE=$ROOT/outputs/protocolR/mcmc_devf/$d \
    setsid nohup bash scripts/run_gs_baseline.sh mcmc $s f$k > logs/mcmc_devf_$d.log 2>&1 < /dev/null &
  sleep 90
done; done
wait; echo "[$(date -u +%FT%TZ)] FOLD_QUEUE_DONE"

#!/usr/bin/env bash
# Cross-fitting models: MCMC trained on the dev copy (official test removed, 1/8 of train held out as dev views).
# Launches the next scene whenever fewer than MAXJ MCMC trainings run on the card.
#   setsid nohup bash scripts/queue_dev_mcmc.sh > logs/queue_dev_mcmc.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; MAXJ=${MAXJ:-3}
for s in ${DEV_SCENES:-bonsai counter train bicycle garden}; do
  [ -f outputs/protocolR/mcmc_dev/$s/results.json ] && continue
  while [ "$(pgrep -fc 'venv_mcmc/bin/python -u train.py')" -ge "$MAXJ" ]; do sleep 60; done
  echo "[$(date -u +%FT%TZ)] launch dev MCMC $s"
  G=0 NO_VGATE=1 SRC_OVERRIDE=$ROOT/data/dev/$s OUT_OVERRIDE=$ROOT/outputs/protocolR/mcmc_dev/$s \
    setsid nohup bash scripts/run_gs_baseline.sh mcmc $s dev > logs/mcmc_dev_$s.log 2>&1 < /dev/null &
  sleep 120
done
wait; echo "[$(date -u +%FT%TZ)] DEV_QUEUE_DONE"

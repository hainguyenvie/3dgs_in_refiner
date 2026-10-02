#!/usr/bin/env bash
# Ensemble control: a second 3DGS-MCMC run (GS_SEED=1) per scene, same config/data -> outputs/protocolR/mcmc/<s>_s1.
#   setsid nohup bash scripts/queue_seed_mcmc.sh > logs/queue_seed_mcmc.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; MAXJ=${MAXJ:-4}
for s in ${SEED_SCENES:-bicycle garden stump bonsai counter train}; do
  [ -f outputs/protocolR/mcmc/${s}_s1/results.json ] && continue
  while [ "$(pgrep -fc 'venv_mcmc/bin/python -u train.py')" -ge "$MAXJ" ]; do sleep 60; done
  echo "[$(date -u +%FT%TZ)] launch MCMC seed1 $s"
  GS_SEED=1 G=0 NO_VGATE=1 setsid nohup bash scripts/run_gs_baseline.sh mcmc $s s1 > logs/mcmc_${s}_s1.log 2>&1 < /dev/null &
  sleep 90
done
wait; echo "[$(date -u +%FT%TZ)] SEED_QUEUE_DONE"

#!/usr/bin/env bash
# IBGS trained on the dev copies (cross-fitting): its outputs at the dev views are then test-like candidates.
#   setsid nohup bash scripts/queue_dev_ibgs.sh > logs/queue_dev_ibgs.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
for s in ${DEV_SCENES:-bonsai counter train bicycle garden}; do
  [ -f outputs/protocolR/ibgs_dev/$s/results_renders_aggregate.json ] && continue
  echo "[$(date -u +%FT%TZ)] IBGS dev $s"
  NO_VGATE=1 G=0 SRC_OVERRIDE=$ROOT/data/dev/$s OUT_OVERRIDE=$ROOT/outputs/protocolR/ibgs_dev/$s bash scripts/run_ibgs.sh $s dev > logs/ibgs_dev_$s.log 2>&1
  echo "[$(date -u +%FT%TZ)] IBGS dev $s rc=$?"
done
echo DEV_IBGS_QUEUE_DONE

#!/usr/bin/env bash
# Parallel lanes for IBGS-on-dev-copy trainings, with a per-scene lock (mkdir) so lanes never duplicate a scene.
#   LANES=2 setsid nohup bash scripts/queue_dev_ibgs2.sh > logs/queue_dev_ibgs2.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p outputs/protocolR/ibgs_dev/.locks
lane() {
  for s in ${DEV_SCENES:-train bicycle garden counter bonsai}; do
    [ -f outputs/protocolR/ibgs_dev/$s/results_renders_aggregate.json ] && continue
    mkdir outputs/protocolR/ibgs_dev/.locks/$s 2>/dev/null || continue
    echo "[$(date -u +%FT%TZ)] lane $1: IBGS dev $s"
    NO_VGATE=1 G=0 SRC_OVERRIDE=$ROOT/data/dev/$s OUT_OVERRIDE=$ROOT/outputs/protocolR/ibgs_dev/$s bash scripts/run_ibgs.sh $s dev > logs/ibgs_dev_$s.log 2>&1
    echo "[$(date -u +%FT%TZ)] lane $1: IBGS dev $s rc=$?"
  done
}
for i in $(seq 1 ${LANES:-2}); do lane $i & sleep 30; done
wait; echo DEV_IBGS2_DONE

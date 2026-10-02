#!/usr/bin/env bash
# Sector-holdout baselines (same setup for everyone): vanilla 3DGS and IBGS trained on data/sector/<scene> (split.json).
#   setsid nohup bash scripts/queue_sector.sh > logs/queue_sector.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SC="${SECTOR_SCENES:-garden bonsai bicycle counter truck}"
until [ -x .venv_3dgs/bin/python ] && grep -q 3DGS_ENV_DONE logs/setup_3dgs_env.log; do sleep 60; done
lane3dgs() { for s in $SC; do [ -f outputs/sector/3dgs/$s/results.json ] && continue
  echo "[$(date -u +%FT%TZ)] 3dgs sector $s"
  G=0 NO_VGATE=1 SRC_OVERRIDE=$ROOT/data/sector/$s OUT_OVERRIDE=$ROOT/outputs/sector/3dgs/$s bash scripts/run_gs_baseline.sh 3dgs $s sector > logs/sector_3dgs_$s.log 2>&1; done; }
laneibgs() { for s in $1; do [ -f outputs/sector/ibgs/$s/results_renders_aggregate.json ] && continue
  echo "[$(date -u +%FT%TZ)] ibgs sector $s"
  NO_VGATE=1 G=0 SRC_OVERRIDE=$ROOT/data/sector/$s OUT_OVERRIDE=$ROOT/outputs/sector/ibgs/$s bash scripts/run_ibgs.sh $s sector > logs/sector_ibgs_$s.log 2>&1; done; }
lane3dgs & sleep 60
laneibgs "garden bicycle truck" & sleep 60
laneibgs "bonsai counter" &
wait; echo "[$(date -u +%FT%TZ)] SECTOR_QUEUE_DONE"

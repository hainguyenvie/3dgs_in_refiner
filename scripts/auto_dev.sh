#!/usr/bin/env bash
# Cross-fitting pipeline for the focus scenes: dev MCMC done -> dev dump -> BandFuse with dev / self protocols.
#   setsid nohup bash scripts/auto_dev.sh > logs/auto_dev.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SC="${DEV_SCENES:-bonsai counter train bicycle garden}"; O=outputs/route/hybrid
log() { echo "[$(date -u +%FT%TZ)] $*"; }
while true; do
  left=0
  for s in $SC; do
    [ -f $O/${s}_dev/.done ] && continue; left=1
    [ -f outputs/protocolR/mcmc_dev/$s/results.json ] || continue
    log "dev dump $s"; MODE=dev G=0 bash scripts/run_hybrid2.sh $s > logs/hyb2_dev_$s.log 2>&1 && log "dev dump $s ok"
  done
  [ $left = 0 ] && break; sleep 120
done
for v in "" "--attn" "--attn --aug"; do
  log "bandfuse dev $v"; .venv_ibgs/bin/python src/route/bandfuse.py --scenes $SC --protocol dev --iters 4000 $v > logs/bandfuse_dev$(echo $v | tr -d ' -').log 2>&1
done
log "bandfuse self"; .venv_ibgs/bin/python src/route/bandfuse.py --scenes $SC --protocol self --iters 3000 --attn > logs/bandfuse_self_attn.log 2>&1
log "AUTO_DEV_DONE"

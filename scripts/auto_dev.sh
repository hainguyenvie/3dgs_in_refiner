#!/usr/bin/env bash
# Cross-fitting pipeline for the focus scenes: dev MCMC done -> dev dump -> BandFuse with dev / devft / self protocols.
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
B=".venv_ibgs/bin/python src/route/bandfuse.py --scenes $SC"
log "bandfuse devft attn resid";  $B --protocol devft --iters 6000 --ft_iters 1500 --attn --resid > logs/bandfuse_devft_attn_resid.log 2>&1
log "bandfuse self attn resid";   $B --protocol self --iters 6000 --attn --resid --tag self_attn_resid_6k > logs/bandfuse_self_attn_resid.log 2>&1
log "bandfuse dev attn resid";    $B --protocol dev --iters 6000 --attn --resid > logs/bandfuse_dev_attn_resid.log 2>&1
log "bandfuse devft attn resid align"; $B --protocol devft --iters 6000 --ft_iters 1500 --attn --resid --align > logs/bandfuse_devft_attn_resid_align.log 2>&1
log "bandfuse devft attn";        $B --protocol devft --iters 6000 --ft_iters 1500 --attn > logs/bandfuse_devft_attn.log 2>&1
log "AUTO_DEV_DONE"

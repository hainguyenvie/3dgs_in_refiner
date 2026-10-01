#!/usr/bin/env bash
# Fully cross-fitted gate per focus scene: IBGS_dev + MCMC_dev done -> devfull dump -> gate trained on the scene's own
# dev views (all candidates test-like), evaluated on its official test views.
#   setsid nohup bash scripts/auto_devfull.sh > logs/auto_devfull.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
SC="${DEV_SCENES:-bonsai counter train bicycle garden}"; O=outputs/route/hybrid
log() { echo "[$(date -u +%FT%TZ)] $*"; }
while true; do
  left=0
  for s in $SC; do
    [ -f $O/${s}_devfull/.gated ] && continue; left=1
    [ -f outputs/protocolR/ibgs_dev/$s/point_cloud/iteration_30000/point_cloud.ply ] && [ -f outputs/protocolR/ibgs_dev/$s/color_aggregate_checkpoint/30000/color_aggregation_network.pth ] || continue
    [ -f outputs/protocolR/mcmc_dev/$s/results.json ] || continue
    if [ ! -f $O/${s}_devfull/.done ]; then log "devfull dump $s"; MODE=devfull G=0 bash scripts/run_hybrid2.sh $s > logs/hyb2_devfull_$s.log 2>&1; fi
    for m in band pixel; do
      log "gate self-dev $s $m"
      CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python src/route/gate_loso.py --scenes $s --held $s --self_dev devfull --tag r1 --mode $m --iters 3000 \
        --out outputs/route/gate_selfdev_${m}_$s.json > logs/gate_selfdev_${m}_$s.log 2>&1
    done
    touch $O/${s}_devfull/.gated
  done
  [ $left = 0 ] && break; sleep 300
done
log "AUTO_DEVFULL_DONE"

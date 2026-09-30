#!/usr/bin/env bash
# Train MCMC on the per-image-focal re-calibrated scenes (data/calib/<scene>_pf) once the given cards are free.
#   NOT_BEFORE_UTC=04:30 CARDS="0 1 5 6" setsid nohup bash scripts/launch_calib_pf.sh flowers bicycle truck drjohnson > logs/calib_pf_launcher.log 2>&1 < /dev/null &
# A card counts as free only if it has NO compute process and < 1000 MiB used on two checks 60 s apart; one job per card.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
CARDS="${CARDS:?set CARDS}"; NB="${NOT_BEFORE_UTC:-00:00}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
until [ "$(date -u +%H:%M)" \> "$NB" ] || [ "$(date -u +%H:%M)" = "$NB" ]; do sleep 60; done
echo "[$(date -u +%FT%TZ)] time gate passed ($NB UTC); cards: $CARDS"
free_card() {   # $1 = index -> 0 if free
  local bus mem; bus=$(nvidia-smi -i $1 --query-gpu=pci.bus_id --format=csv,noheader); mem=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$mem" -lt 1000 ] && ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -qi "${bus#00000000:}"
}
declare -A BUSY
for sc in "$@"; do
  while true; do
    for c in $CARDS; do                      # release cards whose job ended
      [ -n "${BUSY[$c]:-}" ] && ! kill -0 "${BUSY[$c]}" 2>/dev/null && unset "BUSY[$c]"
    done
    pick=""
    for c in $CARDS; do
      [ -n "${BUSY[$c]:-}" ] && continue
      if free_card $c; then sleep 60; free_card $c && { pick=$c; break; }; fi
    done
    [ -n "$pick" ] && break
    sleep 120
  done
  G=$pick NO_VGATE=1 SRC_OVERRIDE=$ROOT/data/calib/${sc}_pf OUT_OVERRIDE=$ROOT/outputs/calib/runs/${sc}_mcmc_pf \
    setsid nohup bash scripts/run_gs_baseline.sh mcmc $sc pf > logs/calib/${sc}_mcmc_pf.log 2>&1 < /dev/null &
  BUSY[$pick]=$!
  echo "[$(date -u +%FT%TZ)] launched $sc on card $pick (pid ${BUSY[$pick]})"
done
wait; echo "[$(date -u +%FT%TZ)] LAUNCHER_DONE"

#!/usr/bin/env bash
# Generic GPU queue: run each line of a job file (a shell command using $G for the card) on the first free card.
#   CARDS="0 1 5 6" setsid nohup bash scripts/launch_queue.sh protocol/jobs/<file>.txt > logs/queue_<file>.log 2>&1 < /dev/null &
# Job file lines:  <name> | <command using $G>     (blank lines and '#' comments ignored)
# A card is free only with NO compute process and < 1000 MiB used on two checks 60 s apart; one job per card.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
CARDS="${CARDS:?set CARDS}"; export CUDA_DEVICE_ORDER=PCI_BUS_ID
free_card() {
  local bus mem; bus=$(nvidia-smi -i $1 --query-gpu=pci.bus_id --format=csv,noheader); mem=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$mem" -lt 1000 ] && ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -qi "${bus#00000000:}"
}
declare -A BUSY; mkdir -p logs/queue
while IFS= read -r line; do
  line="${line%%#*}"; [ -z "${line// }" ] && continue
  name=$(echo "${line%%|*}" | xargs); cmd="${line#*|}"
  while true; do
    for c in $CARDS; do [ -n "${BUSY[$c]:-}" ] && ! kill -0 "${BUSY[$c]}" 2>/dev/null && unset "BUSY[$c]"; done
    pick=""
    for c in $CARDS; do
      [ -n "${BUSY[$c]:-}" ] && continue
      if free_card $c; then sleep 60; free_card $c && { pick=$c; break; }; fi
    done
    [ -n "$pick" ] && break; sleep 120
  done
  G=$pick setsid nohup bash -c "export G=$pick; $cmd; echo JOB_DONE rc=\$?" > logs/queue/$name.log 2>&1 < /dev/null &
  BUSY[$pick]=$!; echo "[$(date -u +%FT%TZ)] launched $name on card $pick (pid ${BUSY[$pick]})"
done < "$1"
wait; echo "[$(date -u +%FT%TZ)] QUEUE_DONE"

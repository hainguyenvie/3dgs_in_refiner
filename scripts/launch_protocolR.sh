#!/usr/bin/env bash
# Distribute Protocol R runs over cards: SLOTS queues per card, each queue runs its jobs sequentially.
#
#   CARDS=0,1,2,3,4,5,6,7 SLOTS=2 bash scripts/launch_protocolR.sh jobs.txt
#   jobs.txt: one job per line, "<ibgs|3dgs|mcmc|irgs.<variant>> <scene> [tag]"
# Every job is idempotent (guards in run_*.sh), so relaunching the same list only redoes missing work.
# Quality numbers are what Protocol R is for; timings of co-scheduled jobs are contended (noted in reports).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$1"; IFS=, read -ra CARDS <<< "${CARDS:-0,1,2,3,4,5,6,7}"; SLOTS="${SLOTS:-2}"
mkdir -p "$ROOT/logs/protocolR"
Q=()                                   # queue id -> card
for s in $(seq 1 "$SLOTS"); do for c in "${CARDS[@]}"; do Q+=("$c"); done; done
declare -A LIST
i=0
while read -r m sc tag; do
  [ -z "${m:-}" ] || [[ "$m" == \#* ]] && continue
  q=$(( i % ${#Q[@]} )); LIST[$q]+="$m $sc ${tag:-r1};"; i=$((i+1))
done < "$JOBS"
for q in "${!LIST[@]}"; do
  card=${Q[$q]}
  setsid nohup bash -c '
    ROOT=$1; card=$2; IFS=";" read -ra J <<< "$3"
    for j in "${J[@]}"; do
      set -- $j; m=$1; sc=$2; tag=$3
      log=$ROOT/logs/protocolR/${m}_${sc}_${tag}.log
      echo "[$(date -u +%FT%TZ)] card=$card start $m $sc $tag"
      if [ "$m" = ibgs ]; then G=$card NO_VGATE=1 OMP_NUM_THREADS=2 bash $ROOT/scripts/run_ibgs.sh $sc $tag > $log 2>&1
      elif [[ "$m" == irgs.* ]]; then mkdir -p $ROOT/logs/irgs; log=$ROOT/logs/irgs/${m#irgs.}_${sc}_${tag}.log
        G=$card OMP_NUM_THREADS=2 bash $ROOT/scripts/run_irgs.sh ${m#irgs.} $sc $tag > $log 2>&1
      else G=$card NO_VGATE=1 OMP_NUM_THREADS=2 bash $ROOT/scripts/run_gs_baseline.sh $m $sc $tag > $log 2>&1; fi
      echo "[$(date -u +%FT%TZ)] card=$card end $m $sc $tag rc=$?"
    done
    echo QUEUE_DONE' _ "$ROOT" "$card" "${LIST[$q]}" > "$ROOT/logs/protocolR/queue_$(basename "$JOBS" .txt)_${q}_card${card}.log" 2>&1 < /dev/null &
  echo "queue $q -> card $card: ${LIST[$q]}"
done

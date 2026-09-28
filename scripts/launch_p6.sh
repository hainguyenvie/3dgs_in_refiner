#!/usr/bin/env bash
# P6: the method (phase-consistent MCMC) on a scene list, queued over the allowed cards (RIT_CARDS), SLOTS per card.
#   VARIANT=m1 bash scripts/launch_p6.sh protocol/jobs/P6_scenes.txt          # m1 = learnable field from zero
#   VARIANT=m1m2 ...                                                             # + shift-tolerant L1
# Idempotent (run_p4.sh guards on the final artefacts). Tags: <scene>_p6_<variant>.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd $ROOT; mkdir -p logs/p6
LIST="$1"; VARIANT="${VARIANT:-m1}"; SLOTS="${SLOTS:-2}"
[ -z "${CARDS:-}" ] && [ -f infra.env ] && CARDS=$(grep -E "^RIT_CARDS=" infra.env | cut -d= -f2 | cut -d" " -f1)
[ -z "${CARDS:-}" ] && { echo "set CARDS or RIT_CARDS"; exit 2; }
case "$VARIANT" in
  m1)   EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-4 --phase_start 1000" ;;
  m1s)  EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-3 --phase_start 1000 --phase_zero_mean" ;;   # safe: reg at the measured-field scale + gauge fixing
  m1a)  EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-3 --phase_start 1000 --phase_zero_mean --phase_deg 1" ;;   # affine-only fields
  m1n)  EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-3 --phase_start 1000 --phase_anchor_every 8" ;;   # v1: anchor-view gauge fixing
  m1w)  EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-4 --phase_start 1000 --phase_anchor_every 8 --phase_shared" ;;   # v2 candidate: weak reg + anchors + shared (test-applied) field
  m1wc) EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-4 --phase_start 1000 --phase_anchor_every 8 --phase_shared --phase_cap 0.5" ;;   # m1w + per-view cap (robust to a mis-posed view: kitchen)
  m1m2) EX="--learn_phase --phase_lr 1e-3 --phase_reg 1e-4 --phase_start 1000 --shift_tol 0.5 --shift_grid 3 --shift_patch 32 --shift_start 7000" ;;
  m2)   EX="--shift_tol 0.5 --shift_grid 3 --shift_patch 32 --shift_start 7000" ;;
  *) echo "unknown VARIANT"; exit 2 ;;
esac
IFS=, read -ra C <<< "$CARDS"; Q=(); for s in $(seq 1 $SLOTS); do for c in "${C[@]}"; do Q+=("$c"); done; done
declare -A LIST_Q; i=0
while read -r sc; do [ -z "$sc" ] && continue; q=$(( i % ${#Q[@]} )); LIST_Q[$q]+="$sc "; i=$((i+1)); done < "$LIST"
for q in "${!LIST_Q[@]}"; do
  card=${Q[$q]}
  (setsid nohup bash -c 'ROOT=$1; card=$2; EX=$3; V=$4; for sc in $5; do echo "[$(date -u +%FT%TZ)] card=$card start $sc"; G=$card NO_FIELDS=1 EXTRA="$EX" OMP_NUM_THREADS=3 bash $ROOT/scripts/run_p4.sh $sc 1.0 ${sc}_p6_$V > $ROOT/logs/p6/${sc}_p6_$V.log 2>&1; echo "[$(date -u +%FT%TZ)] card=$card end $sc rc=$?"; done; echo QUEUE_DONE' _ "$ROOT" "$card" "$EX" "$VARIANT" "${LIST_Q[$q]}" > logs/p6/queue_${VARIANT}_${q}_card${card}.log 2>&1 < /dev/null &)
  echo "queue $q -> card $card: ${LIST_Q[$q]}"
done

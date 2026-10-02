#!/usr/bin/env bash
# When both sector baselines of a scene exist: dump -> Difix D0 -> band spectrum + headroom (analysis on the sector split).
#   setsid nohup bash scripts/auto_sector_analysis.sh > logs/auto_sector.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; O=outputs/route/hybrid
SC="${SECTOR_SCENES:-garden bonsai bicycle counter truck}"
while true; do
  left=0
  for s in $SC; do
    [ -f $O/${s}_sector/.analysed ] && continue; left=1
    [ -f outputs/sector/3dgs/$s/results.json ] && [ -f outputs/sector/ibgs/$s/color_aggregate_checkpoint/30000/color_aggregation_network.pth ] || continue
    echo "[$(date -u +%FT%TZ)] sector dump $s"
    [ -f $O/${s}_sector/.done ] || G=0 bash scripts/run_hybrid_sector.sh $s > logs/hybrid_sector_$s.log 2>&1
    HF_HOME=$ROOT/checkpoints/hf CUDA_VISIBLE_DEVICES=0 .venv_difix/bin/python src/route/d0_difix.py $O/${s}_sector > logs/d0_sector_$s.log 2>&1
    HF_HOME=$ROOT/checkpoints/hf CUDA_VISIBLE_DEVICES=0 .venv_difix/bin/python src/route/d0b_difix_lf.py $O/${s}_sector --views 10 > logs/d0b_sector_$s.log 2>&1
    echo "[$(date -u +%FT%TZ)] sector analysed $s"; touch $O/${s}_sector/.analysed
  done
  [ $left = 0 ] && break; sleep 180
done
echo SECTOR_ANALYSIS_DONE

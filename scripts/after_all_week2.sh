#!/usr/bin/env bash
# Week-2 evaluation chain, run unattended once every hybrid dump exists (13 benchmark scenes + 3 Shiny).
#   setsid nohup bash scripts/after_all_week2.sh > logs/after_all_week2.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
B="bicycle flowers garden stump treehill bonsai counter kitchen room train truck drjohnson playroom"; SH="cd guitars lab"
PY=.venv_ibgs/bin/python; GL="src/route/gate_loso.py"; O=outputs/route; export CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date -u +%FT%TZ)] $*"; }
for i in $(seq 1 600); do   # wait up to ~20 h
  n=0; for s in $B $SH; do [ -f $O/hybrid/${s}_r1/.done ] && n=$((n+1)); done
  [ $n -eq 16 ] && break; [ $((i % 10)) = 0 ] && log "waiting: $n/16 hybrid dumps"; sleep 120
done
log "hybrid dumps: $n/16 — starting evaluation chain"
HAVE=""; for s in $B; do [ -f $O/hybrid/${s}_r1/.done ] && HAVE="$HAVE $s"; done
HSH=""; for s in $SH; do [ -f $O/hybrid/${s}_r1/.done ] && HSH="$HSH $s"; done
run() { local tag=$1; shift; log "start $tag"; $PY $GL "$@" --out $O/$tag.json > logs/$tag.log 2>&1; log "done $tag rc=$?"; .venv_tools/bin/python src/route/gate_summary.py $O/$tag.json >> logs/week2_summary.txt 2>&1; }
run gate13_loso_band  --scenes $HAVE --mode band --iters 3000
[ -n "$HSH" ] && run gate_shiny_band --scenes $HAVE $HSH --train_scenes $HSH --mode band --iters 3000
run gate13_lodo_band  --scenes $HAVE --mode band --iters 3000 --lodo
run gate13_loso_pixel --scenes $HAVE --mode pixel --iters 3000
run abl_drop_support  --scenes $HAVE --mode band --iters 3000 --drop support
run abl_drop_disagree --scenes $HAVE --mode band --iters 3000 --drop disagree resid
run abl_drop_models   --scenes $HAVE --mode band --iters 3000 --drop models
run abl_drop_color    --scenes $HAVE --mode band --iters 3000 --drop color
run abl_cands_mcmc_ibgs  --scenes $HAVE --mode band --iters 3000 --cands mcmc ibgs_final
run abl_cands_mcmc_res   --scenes $HAVE --mode band --iters 3000 --cands mcmc mcmc_res
log "gate chain done; resuming outdoor R0"
bash scripts/run_r0_pilot.sh > logs/r0_pilot_resume.log 2>&1
for s in bicycle garden stump; do [ -f $O/r0/$s.npz ] && .venv_tools/bin/python src/route/r0_analyze.py $O/r0/$s.npz >> logs/week2_summary.txt; done
log "AFTER_ALL_DONE"

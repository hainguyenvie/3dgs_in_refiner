#!/usr/bin/env bash
# Own single-model evidence (MCMC colour + MCMC depth, no IBR checkpoint): band-limited warps for every scene's TEST
# views, then the per-level low-pass fusion evaluated leave-one-scene-out.
#   setsid nohup bash scripts/run_bandwarp_all.sh > logs/bandwarp_all.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; D=$ROOT/data
ARGS="--n_src 3 --max_angle 30 --relax 1"
for s in bonsai counter kitchen room bicycle flowers garden stump treehill train truck drjohnson playroom cd guitars lab; do
  case $s in
    bonsai|counter|kitchen|room|bicycle|flowers|garden|stump|treehill) SRC=$D/mipnerf360/$s ;;
    train|truck) SRC=$D/tandt_db/tandt/$s ;; drjohnson|playroom) SRC=$D/tandt_db/db/$s ;; *) SRC=$D/shiny/_SHINNY_DATASET_/$s ;;
  esac
  M=$ROOT/outputs/protocolR/mcmc/${s}_r1; [ $s = playroom ] && M=$ROOT/outputs/protocolR/mcmc/playroom_fix
  RES=$(.venv_tools/bin/python -c "import re;print(re.search(r'resolution=(-?\d+)', open('$M/cfg_args').read()).group(1))")
  O=$ROOT/outputs/route/bandwarp/${s}_own; [ -f $O/.done ] && continue
  echo "[$(date -u +%FT%TZ)] bandwarp $s (r $RES)"
  (cd third_party/3dgs-mcmc && CUDA_VISIBLE_DEVICES=0 ../../.venv_mcmc/bin/python ../../src/route/bandwarp.py -s $SRC -m $M -r $RES --eval $ARGS --out $O > ../../logs/bandwarp_$s.log 2>&1) && touch $O/.done
done
B="bicycle flowers garden stump treehill bonsai counter kitchen room train truck drjohnson playroom"
CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python src/route/bandwarp_eval.py $(for s in $B; do echo outputs/route/bandwarp/${s}_own; done) > logs/bandwarp_eval13.log 2>&1
echo "[$(date -u +%FT%TZ)] BANDWARP_ALL_DONE"

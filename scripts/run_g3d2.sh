#!/usr/bin/env bash
# Apply the fitted global sim3 gauge to a trained phase model, render test views, apply shared field, score.
#   bash scripts/run_g3d.sh <scene> <model_tag e.g. p6_m1w>      (picks a free card among RIT_CARDS)
set -euo pipefail
ROOT=~/projects/3dgs-refiner-it; cd $ROOT
s=$1; tag=$2; M=$ROOT/outputs/p4/${s}_$tag
case $s in playroom|drjohnson) SRC=$ROOT/data/tandt_db/db/$s;; train|truck) SRC=$ROOT/data/tandt_db/tandt/$s;; *) SRC=$ROOT/data/mipnerf360/$s;; esac
case $s in bicycle|flowers|garden|stump|treehill) R=4;; bonsai|counter|kitchen|room) R=2;; *) R=1;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
G=$(for c in ${RIT_CARDS:-2,3,4,7} ; do :; done; for c in $(echo ${RIT_CARDS:-2,3,4,7} | tr , " "); do u=$(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && { echo $c; break; }; done)
[ -n "$G" ] || { echo "no free card"; exit 1; }
export CUDA_VISIBLE_DEVICES=$G
PY=$ROOT/.venv_mcmc/bin/python
$PY scripts/analysis/gauge3d_fit.py $M --apply ${FITARGS:-} 2>&1 | grep -E "^\[(all|apply)\]"
OUT=${M}_g3d${SUF:-}
cd third_party/3dgs-mcmc
$PY render.py -s $SRC -m $OUT -r $R --eval --skip_train --quiet > /dev/null 2>&1
mkdir -p $OUT/test/ours_30000; [ -e $OUT/test/ours_30000/gt ] || true
$PY $ROOT/scripts/analysis/shared_apply_test.py $OUT > /dev/null 2>&1 || echo "shared apply failed"
rm -f $OUT/results.json; $PY metrics.py -m $OUT > /dev/null 2>&1
python3 - <<EOF
import json
for m in ["$M", "$OUT"]:
    d = json.load(open(m + "/results.json")); print(m.split("/")[-1], {k.replace("ours_30000", "o"): round(v["PSNR"], 3) for k, v in d.items()})
EOF

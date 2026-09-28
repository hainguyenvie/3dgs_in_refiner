#!/bin/bash
# IBGS pretrained models + processed Shiny (authors' official Google Drive links in third_party/ibgs/README.md).
set -euo pipefail
R=$HOME/projects/3dgs-refiner-it; Z=$R/data/zips; mkdir -p $Z $R/data/shiny $R/checkpoints/ibgs_pretrained
export UV_CACHE_DIR=$R/.uv-cache
T=$R/.venv_tools
[ -x $T/bin/gdown ] || { ~/.local/bin/uv venv --python 3.12 --allow-existing $T; ~/.local/bin/uv pip install --python $T/bin/python gdown; }
[ -s $Z/ibgs_pretrained.zip ] || $T/bin/gdown -O $Z/ibgs_pretrained.zip 1zFshzLTFaka8Kem6K4gA5Uu_FX5Hz6vC
[ -s $Z/shiny_ibgs.zip ]      || $T/bin/gdown -O $Z/shiny_ibgs.zip 1ZbVkpzbqJMjYyHeXi2WOL0BAdfKzP1av
ls -la $Z
for f in ibgs_pretrained shiny_ibgs; do echo "== $f"; unzip -l $Z/$f.zip | head -30; unzip -l $Z/$f.zip | tail -1; done
echo JOB_DONE

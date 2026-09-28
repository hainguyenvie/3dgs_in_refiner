#!/bin/bash
# GADA published renders + GT (authors' Google Drive link in third_party/GADA/README.md). Code not released.
set -euo pipefail
R=$HOME/projects/3dgs-refiner-it; Z=$R/data/zips; mkdir -p $Z $R/outputs/published/gada
[ -s $Z/gada_renders.zip ] || $R/.venv_tools/bin/gdown -O $Z/gada_renders.zip 1NVFaTcrwE5A1dO_AzEZEU1QBANmPTpo_
ls -la $Z/gada_renders.zip; unzip -l $Z/gada_renders.zip | awk '{print $4}' | cut -d/ -f1-3 | sort -u | head -60
echo JOB_DONE

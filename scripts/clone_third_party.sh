#!/bin/bash
# Clone baseline repos (recursive) into third_party/; idempotent.
set -uo pipefail
R=$HOME/projects/3dgs-refiner-it; cd $R/third_party
for r in HoangChuongNguyen/ibgs graphdeco-inria/gaussian-splatting ubc-vision/3dgs-mcmc siw00-lim/GADA; do
  n=$(basename $r)
  [ -d $n/.git ] || git clone --recursive https://github.com/$r.git $n
  git -C $n submodule update --init --recursive
  echo "$n $(git -C $n log -1 --format='%H %ci')"
done
echo JOB_DONE

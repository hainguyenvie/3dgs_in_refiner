#!/bin/bash
# Stage 0 data: copy byte-verified public zips from ares-gs, fetch extra scenes, extract fresh.
set -euo pipefail
R=$HOME/projects/3dgs-refiner-it
Z=$R/data/zips; mkdir -p $Z $R/data/mipnerf360 $R/data/tandt_db $R/logs
SRC=$HOME/projects/ares-gs/data
[ -f $Z/360_v2.zip ]   || cp $SRC/360_v2.zip $Z/
[ -f $Z/tandt_db.zip ] || cp $SRC/tandt_db.zip $Z/
[ "$(stat -c %s $Z/360_extra_scenes.zip 2>/dev/null || echo 0)" = 4488140217 ] || \
  wget -q -c -O $Z/360_extra_scenes.zip https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip
for f in 360_v2:12535427936 tandt_db:682628995 360_extra_scenes:4488140217; do
  n=${f%%:*}; s=${f##*:}; [ "$(stat -c %s $Z/$n.zip)" = "$s" ] || { echo "SIZE MISMATCH $n"; exit 1; }
done
sha256sum $Z/*.zip > $Z/SHA256SUMS
[ -d $R/data/mipnerf360/bicycle ] || unzip -q $Z/360_v2.zip -d $R/data/mipnerf360
[ -d $R/data/mipnerf360/flowers ] || unzip -q $Z/360_extra_scenes.zip -d $R/data/mipnerf360
[ -d $R/data/tandt_db/tandt ]     || unzip -q $Z/tandt_db.zip -d $R/data/tandt_db
for d in $R/data/mipnerf360/* $R/data/tandt_db/*/*; do
  [ -d "$d" ] && echo "$(basename $d) $(ls $d/images 2>/dev/null | wc -l)"
done
echo JOB_DONE

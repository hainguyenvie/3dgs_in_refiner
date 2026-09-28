#!/bin/bash
# Reference papers into papers/ (idempotent). Pulled to local by tar.
set -uo pipefail
P=$HOME/projects/3dgs-refiner-it/papers; mkdir -p $P; cd $P
dl() { [ -s "$1" ] && head -c4 "$1" | grep -q %PDF && return; curl -sL -A "Mozilla/5.0" -o "$1" "$2"; echo "$1 $(head -c4 "$1") $(stat -c %s "$1")"; }
dl ibgs_neurips25.pdf "https://openreview.net/pdf?id=AZLj6ObEDF"
dl 3dgs_2308.04079.pdf https://arxiv.org/pdf/2308.04079
dl 3dgs_mcmc_2404.09591.pdf https://arxiv.org/pdf/2404.09591
dl pgsr_2406.06521.pdf https://arxiv.org/pdf/2406.06521
dl idesplat_2601.03824.pdf https://arxiv.org/pdf/2601.03824
dl difix3d_2503.01774.pdf https://arxiv.org/pdf/2503.01774
curl -sL https://siw00-lim.github.io/GADA-Project-Page/ | grep -oE 'href="[^"]*"' | grep -iE "arxiv|pdf|openreview|drive|github" | sort -u
echo JOB_DONE

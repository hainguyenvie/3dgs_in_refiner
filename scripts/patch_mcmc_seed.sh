#!/usr/bin/env bash
# 3DGS-MCMC hard-codes seed 0 in utils/general_utils.safe_state; make it GS_SEED (default 0 = unchanged behaviour).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; F=$ROOT/third_party/3dgs-mcmc/utils/general_utils.py
grep -q GS_SEED "$F" && { echo "already patched"; exit 0; }
sed -i 's/^    random.seed(0)$/    import os as _os; _seed = int(_os.environ.get("GS_SEED", 0)); random.seed(_seed)  # [route] GS_SEED/; s/^    np.random.seed(0)$/    np.random.seed(_seed)/; s/^    torch.manual_seed(0)$/    torch.manual_seed(_seed)/' "$F"
grep -n "_seed" "$F"

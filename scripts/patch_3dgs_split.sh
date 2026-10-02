#!/usr/bin/env bash
# Vanilla 3DGS (current graphdeco repo) chooses test views by the LLFF hold or sparse/0/test.txt; add split.json support
# (names without extension, same file the IBGS reader uses). Idempotent; marked [route].
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; F=$ROOT/third_party/gaussian-splatting/scene/dataset_readers.py
grep -q "split.json" "$F" && { echo "already patched"; exit 0; }
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = '    reading_dir = "images" if images == None else images\n    cam_infos_unsorted = readColmapCameras('
assert anchor in s, "anchor not found"
ins = ('    _split = os.path.join(path, "split.json")  # [route] explicit test list (sector holdout)\n'
       '    if eval and os.path.exists(_split):\n'
       '        import json as _json; _test = set(_json.load(open(_split))["test"])\n'
       '        test_cam_names_list = [cam_extrinsics[k].name for k in cam_extrinsics if os.path.splitext(cam_extrinsics[k].name)[0] in _test]\n'
       '        print(f"------------split.json: {len(test_cam_names_list)} test views-------------")\n')
s = s.replace(anchor, ins + anchor, 1); open(p, "w").write(s); print("patched", p)
PY

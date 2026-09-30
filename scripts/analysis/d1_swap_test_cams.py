"""D1 (step 1, pycolmap venv) — express the TEST cameras of one calibration in the frame of another.
Aligns `src` to `tgt` by a Sim3 over their shared 3D points (same track ids), then dumps, for every image, the
world-to-camera rotation/translation in tgt's frame plus the src intrinsics. Step 2 (render_cams_json.py) renders a
model trained on `tgt` with these cameras: separates "better test cameras" from "better-trained model".

    .venv_tools/bin/python scripts/analysis/d1_swap_test_cams.py <src_sparse> <tgt_sparse> <out.json>
"""
import json
import sys

import numpy as np
import pycolmap

src = pycolmap.Reconstruction(sys.argv[1]); tgt = pycolmap.Reconstruction(sys.argv[2])
sim = pycolmap.align_reconstructions_via_points(src, tgt)
src.transform(sim)
out = {"sim3_scale": float(sim.scale), "images": {}}
for im in src.images.values():
    T = im.cam_from_world(); c = src.cameras[im.camera_id]
    out["images"][im.name] = {"R_cw": T.rotation.matrix().tolist(), "t_cw": np.asarray(T.translation).tolist(),
                              "model": c.model.name, "params": [float(x) for x in c.params], "width": c.width, "height": c.height}
json.dump(out, open(sys.argv[3], "w"))
print(f"aligned {sys.argv[1]} -> {sys.argv[2]} (sim3 scale {sim.scale:.6f}); {len(out['images'])} cameras -> {sys.argv[3]}")

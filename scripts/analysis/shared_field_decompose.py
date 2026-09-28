"""Decompose the learned camera-common phase field (phase_shared.json) into affine (scale/translation/rotation/shear)
and higher-order parts: how much could be rendered exactly through intrinsics/extrinsics instead of resampling?
    python3 scripts/analysis/shared_field_decompose.py outputs/p4/bonsai_p6_m1sh ...
"""
import json, sys
from pathlib import Path
import torch
for d in sys.argv[1:]:
    j = json.load(open(Path(d) / "phase_shared.json")); terms = [tuple(t) for t in j["terms"]]; A = torch.tensor(j["A"])
    H, W = 96, 128
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij"); xn, yn = xx / W - .5, yy / H - .5
    B = torch.stack([xn ** i * yn ** j for i, j in terms], -1); f = (B @ A)                      # H W 2 (px, at the coarse grid scale)
    X = torch.stack([xn, yn, torch.ones_like(xn)], -1).reshape(-1, 3); U = f.reshape(-1, 2)
    Aff = torch.linalg.lstsq(X, U).solution; fa = (X @ Aff).reshape(H, W, 2)
    tot = float((f ** 2).mean()); res = float(((f - fa) ** 2).mean())
    M = Aff[:2].T                                    # 2x2 linear part (du/dxn, du/dyn; dv/dxn, dv/dyn) in px per normalised unit
    scale = float((M[0, 0] + M[1, 1]) / 2); rot = float((M[1, 0] - M[0, 1]) / 2); shear = float((M[0, 0] - M[1, 1]) / 2)
    print(f"{Path(d).name}: |f| mean {float(f.norm(dim=-1).mean()):.3f} px | affine explains {(1-res/tot)*100:.0f}% | "
          f"translation ({float(Aff[2,0]):+.3f},{float(Aff[2,1]):+.3f}) px, isotropic scale {scale:+.3f} px/unit, rotation {rot:+.3f}, shear {shear:+.3f}")

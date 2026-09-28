"""Apply the learned camera-common phase field (phase_shared.json) to the TEST renders -> test/ours_30000_sh/renders
(+ gt symlink) so metrics.py scores both. Source-free, one image-space warp per test render.
    python shared_apply_test.py <model_dir>
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
model = Path(sys.argv[1]); j = json.load(open(model / "phase_shared.json")); terms = [tuple(t) for t in j["terms"]]
A = torch.tensor(j["A"], dtype=torch.float32)
src = model / "test" / "ours_30000"; dst = model / "test" / "ours_30000_sh"; (dst / "renders").mkdir(parents=True, exist_ok=True)
if not (dst / "gt").exists(): os.symlink((src / "gt").resolve(), dst / "gt")
for nm in sorted(os.listdir(src / "renders")):
    img = torch.from_numpy(np.asarray(Image.open(src / "renders" / nm).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij"); xn, yn = xx / W - .5, yy / H - .5
    f = (torch.stack([xn ** i * yn ** j for i, j in terms], -1) @ A)          # H W 2, pixels (field learned at the same eval res)
    gx = (xx + f[..., 0]) / (W - 1) * 2 - 1; gy = (yy + f[..., 1]) / (H - 1) * 2 - 1
    out = F.grid_sample(img, torch.stack([gx, gy], -1)[None], mode="bicubic", padding_mode="border", align_corners=True)[0].clamp(0, 1)
    Image.fromarray((out.permute(1, 2, 0).numpy() * 255 + .5).astype(np.uint8)).save(dst / "renders" / nm)
print("shared field applied to", len(os.listdir(dst / "renders")), "test renders; mean |f| coarse =", round(j["mean_px_coarse"], 3), "px")

"""Apply the camera-common field with SUPERSAMPLING: test renders at 2x (renders_2x/, produced by render.py at half the
downscale factor), field scaled to 2x, bicubic warp at 2x, then 2x2 area downsample -> test/ours_30000_sh2/renders.
    python shared_apply_test_ss.py <model_dir>
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
model = Path(sys.argv[1]); j = json.load(open(model / "phase_shared.json")); terms = [tuple(t) for t in j["terms"]]; A = torch.tensor(j["A"], dtype=torch.float32)
src = model / "test" / "ours_30000"; hi = model / "test_2x" / "ours_30000" / "renders"; dst = model / "test" / "ours_30000_sh2"
(dst / "renders").mkdir(parents=True, exist_ok=True)
if not (dst / "gt").exists(): os.symlink((src / "gt").resolve(), dst / "gt")
for nm in sorted(os.listdir(src / "renders")):
    g = Image.open(src / "gt" / nm); W, H = g.size
    img = torch.from_numpy(np.asarray(Image.open(hi / nm).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    H2, W2 = img.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H2).float(), torch.arange(W2).float(), indexing="ij"); xn, yn = xx / W2 - .5, yy / H2 - .5
    f = (torch.stack([xn ** i * yn ** j for i, j in terms], -1) @ A) * torch.tensor([W2 / W, H2 / H])   # field in 2x pixels
    gx = (xx + f[..., 0]) / (W2 - 1) * 2 - 1; gy = (yy + f[..., 1]) / (H2 - 1) * 2 - 1
    w = F.grid_sample(img, torch.stack([gx, gy], -1)[None], mode="bicubic", padding_mode="border", align_corners=True)
    out = F.interpolate(w, size=(H, W), mode="area")[0].clamp(0, 1)
    Image.fromarray((out.permute(1, 2, 0).numpy() * 255 + .5).astype(np.uint8)).save(dst / "renders" / nm)
print("supersampled shared-field application done:", len(os.listdir(dst / "renders")))

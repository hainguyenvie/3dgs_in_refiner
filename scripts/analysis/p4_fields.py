"""P4 — per-view smooth phase field. For each TRAIN view of a Protocol R MCMC model: RAFT flow GT -> render
(fwd-bwd consistent), least-squares 2D polynomial (total degree 3) fit -> coefficients saved for the phase-aware
training loss (src/phase/train_mcmc_phase.py warps the *render* by this field before the loss).

Convention: f(p) in pixels, gt(p) ~ render(p + f(p)); basis in normalised coords x = px/W - 0.5, y = py/H - 0.5.

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/p4_fields.py bicycle garden stump
"""
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_misalignment import dev, flow_fb, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEG = 3
TERMS = [(i, j) for i in range(DEG + 1) for j in range(DEG + 1 - i)]


def basis(xn, yn):
    return torch.stack([xn ** i * yn ** j for i, j in TERMS], -1)


def fit(flow, ok, step=4):
    _, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    xn, yn = xx / W - 0.5, yy / H - 0.5
    m = ok[::step, ::step]
    X = basis(xn[::step, ::step][m], yn[::step, ::step][m])
    U = torch.stack([flow[0, 0, ::step, ::step][m], flow[0, 1, ::step, ::step][m]], 1)
    A = torch.linalg.lstsq(X, U).solution                         # T x 2
    field = (basis(xn, yn) @ A).permute(2, 0, 1)[None]
    r = (flow - field).norm(dim=1)[0][ok]
    return A, float(flow.norm(dim=1)[0][ok].pow(2).mean().sqrt()), float(r.pow(2).mean().sqrt()), float(field.norm(dim=1)[0].pow(2).mean().sqrt())


def image_names(scene):
    p = ROOT / "data" / ("tandt_db/tandt" if scene in ("train", "truck") else "tandt_db/db" if scene in ("drjohnson", "playroom") else "mipnerf360") / scene / "sparse" / "0" / "images.bin"
    names = []
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(4 + 32 + 24 + 4); s = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                s += c
            k = struct.unpack("<Q", f.read(8))[0]; f.read(24 * k); names.append(s.decode())
    names = sorted(names)
    return [os.path.splitext(n)[0] for i, n in enumerate(names) if i % 8 != 0]   # train, in render index order


def main():
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    (ROOT / "data" / "p4").mkdir(exist_ok=True)
    for scene in sys.argv[1:]:
        d = ROOT / "outputs" / "protocolR" / "mcmc" / f"{scene}_r1" / "train" / "ours_30000"
        names = image_names(scene)
        out = {"deg": DEG, "terms": TERMS, "views": {}}
        for k, stem in enumerate(names):
            gp, rp = d / "gt" / f"{k:05d}.png", d / "renders" / f"{k:05d}.png"
            if not gp.exists():
                continue
            gt, rd = load(gp), load(rp)
            f, ok = flow_fb(model, gt, rd)
            if ok.sum() < 5000:
                continue
            A, rms_in, rms_res, rms_field = fit(f, ok)
            out["views"][stem] = {"A": A.cpu().numpy().tolist(), "W": gt.shape[-1], "H": gt.shape[-2],
                                  "flow_rms": rms_in, "resid_rms": rms_res, "field_rms": rms_field}
        v = out["views"]
        s = {k: float(np.median([x[k] for x in v.values()])) for k in ("flow_rms", "resid_rms", "field_rms")}
        json.dump(out, open(ROOT / "data" / "p4" / f"{scene}_fields.json", "w"))
        print(f"{scene}: {len(v)} views; median flow {s['flow_rms']:.3f} px, smooth field {s['field_rms']:.3f} px, residual {s['resid_rms']:.3f} px", flush=True)


if __name__ == "__main__":
    main()

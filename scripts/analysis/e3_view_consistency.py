"""E3 — is the high-frequency loss caused by INCONSISTENT supervision (per-image pose / intrinsics error)
rather than representation capacity?

On TRAIN views a consistent capacity-limited model should show no coherent image-wide flow between GT and render;
per-image pose/intrinsic errors show up as a flow field that a 6-parameter affine model explains.
For each view: RAFT flow GT->render (fwd-bwd consistent), affine fit, and PSNR before / after aligning the render
with (a) the full flow and (b) only its affine part. Protocol R MCMC checkpoints; no training, no test tuning.

    CUDA_VISIBLE_DEVICES=0 .venv_ibgs/bin/python scripts/analysis/e3_view_consistency.py bicycle garden ...
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_misalignment import dev, flow_fb, hf_share, load, warp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


def affine_fit(flow, ok, step=4):
    """least-squares affine flow u = A [x y 1]^T on consistent pixels; returns dense affine field."""
    _, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    xn, yn = xx / W - 0.5, yy / H - 0.5
    m = ok[::step, ::step]
    X = torch.stack([xn[::step, ::step][m], yn[::step, ::step][m], torch.ones_like(xn[::step, ::step][m])], 1)
    U = torch.stack([flow[0, 0, ::step, ::step][m], flow[0, 1, ::step, ::step][m]], 1)
    A = torch.linalg.lstsq(X, U).solution                                       # 3x2
    field = torch.stack([xn, yn, torch.ones_like(xn)], -1) @ A                  # H W 2
    return field.permute(2, 0, 1)[None], A


def poly_fit(flow, ok, deg=3, step=4):
    """least-squares 2D polynomial flow field of total degree `deg` (smooth per-view part: rotation, distortion,
    smooth parallax); returns dense field."""
    _, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    xn, yn = xx / W - 0.5, yy / H - 0.5
    terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    def basis(x, y):
        return torch.stack([x ** i * y ** j for i, j in terms], -1)
    m = ok[::step, ::step]
    X = basis(xn[::step, ::step][m], yn[::step, ::step][m])
    U = torch.stack([flow[0, 0, ::step, ::step][m], flow[0, 1, ::step, ::step][m]], 1)
    A = torch.linalg.lstsq(X, U).solution
    return (basis(xn, yn) @ A).permute(2, 0, 1)[None]


def analyse(scene, split, model, stride):
    d = ROOT / "outputs" / "protocolR" / "mcmc" / f"{scene}_r1" / split / "ours_30000"
    if not (d / "renders").is_dir() or not os.listdir(d / "renders"):
        return None
    names = sorted(os.listdir(d / "gt"))[::stride]
    rows = []
    for nm in names:
        gt, rd = load(d / "gt" / nm), load(d / "renders" / nm)
        f, ok = flow_fb(model, gt, rd)
        G, R = gt[None].to(dev), rd[None].to(dev)
        aff, A = affine_fit(f, ok)
        pol = poly_fit(f, ok)
        mag = f.norm(dim=1)[0][ok]
        amag = aff.norm(dim=1)[0][ok]
        resid = (f - aff).norm(dim=1)[0][ok]
        presid = (f - pol).norm(dim=1)[0][ok]
        rows.append({"name": nm, "psnr": psnr(R, G), "psnr_align_full": psnr(warp(R, f), G),
                     "psnr_align_affine": psnr(warp(R, aff), G), "psnr_align_poly3": psnr(warp(R, pol), G),
                     "flow_rms": float(mag.pow(2).mean().sqrt()), "affine_rms": float(amag.pow(2).mean().sqrt()),
                     "local_rms": float(resid.pow(2).mean().sqrt()), "poly3_resid_rms": float(presid.pow(2).mean().sqrt()),
                     "translation_px": float(aff.mean(dim=(2, 3)).norm()), "hf_share_err": hf_share(R - G)})
    keys = [k for k in rows[0] if k != "name"]
    summ = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summ["affine_explained_var"] = 1 - np.mean([r["local_rms"] ** 2 for r in rows]) / np.mean([r["flow_rms"] ** 2 for r in rows])
    summ["poly3_explained_var"] = 1 - np.mean([r["poly3_resid_rms"] ** 2 for r in rows]) / np.mean([r["flow_rms"] ** 2 for r in rows])
    summ["n"] = len(rows)
    return {"summary": summ, "views": rows}


def main():
    torch.hub.set_dir(str(ROOT / ".torch_hub"))
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(dev).eval()
    out = {}
    for s in sys.argv[1:]:
        for split, stride in (("train", 3), ("test", 1)):
            r = analyse(s, split, model, stride)
            if r:
                out[f"{s}/{split}"] = r
                print(s, split, {k: round(v, 3) for k, v in r["summary"].items()}, flush=True)
    json.dump(out, open(ROOT / "reports" / "e3_view_consistency_poly.json", "w"), indent=1)
    md = ["# E3 — view consistency (MCMC Protocol R, RAFT GT->render)", "",
          "| scene | split | n | PSNR | +align affine | +align poly3 | +align full | flow rms px | affine rms px | "
          "affine expl. var | poly3 expl. var | poly3 resid rms px | transl. px | HF share err |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in out.items():
        s = r["summary"]; sc, sp = k.split("/")
        md.append(f"| {sc} | {sp} | {s['n']} | {s['psnr']:.2f} | {s['psnr_align_affine']-s['psnr']:+.2f} | "
                  f"{s['psnr_align_poly3']-s['psnr']:+.2f} | {s['psnr_align_full']-s['psnr']:+.2f} | {s['flow_rms']:.3f} | {s['affine_rms']:.3f} | "
                  f"{s['affine_explained_var']*100:.0f}% | {s['poly3_explained_var']*100:.0f}% | {s['poly3_resid_rms']:.3f} | "
                  f"{s['translation_px']:.3f} | {s['hf_share_err']*100:.0f}% |")
    (ROOT / "reports" / "e3_view_consistency_poly.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

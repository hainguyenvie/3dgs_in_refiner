"""Support-aware arbitration between an explicit render and image-based evidence — leave-one-scene-out (LOSO).

Per pixel, a small CNN looks at the evidence (number of valid warps, depth-test margin, warp-vs-render and warp-vs-warp
disagreement, size of the IBR residual, disagreement between two independently trained Gaussian models, depth) and the
candidate colours, and outputs softmax weights over the candidates {MCMC raw, IBGS final, IBGS residual on MCMC}.
Trained on the hybrid dumps of all OTHER scenes (their test views), evaluated on the held-out scene: no ground truth of
the evaluated scene is ever used. Reports PSNR / SSIM / LPIPS for every candidate, per-scene oracle choice, pixel oracle,
and the gate.

    .venv_ibgs/bin/python src/route/gate_loso.py --scenes bonsai counter ... [--iters 3000] [--tag r1]
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANDS = ("mcmc", "ibgs_final", "mcmc_res")


def load_view(f):   # kept on the GPU in fp16; cast per crop
    z = np.load(f)
    g = lambda a: torch.from_numpy(a).cuda().half()
    return g(np.stack([z[k] for k in CANDS])), g(z["gt"]), g(z["feats"]), g(z["ibgs_raw"])


def make_input(c, fe, ir):
    nv, mdd, wr, ws, res, mi, dep = fe
    d = dep.clamp_min(1e-3); ld = torch.log(d) - torch.log(d[d > 1e-3].median() if (d > 1e-3).any() else one)
    ens = (c[0] - ir).abs().mean(0)
    one = torch.ones((), device=c.device)                                                       # MCMC vs IBGS-base disagreement
    x = torch.cat([c.reshape(-1, *c.shape[-2:]),                                         # candidate colours (9)
                   torch.stack([nv / 5, (nv == 0).float(), mdd, wr * 5, ws * 5, res * 5, mi * 5, ens * 5, ld.clamp(-3, 3) / 3])])
    return x


class Gate(nn.Module):
    def __init__(self, cin, k, w=48):
        super().__init__()
        L = []
        for i, d in enumerate((1, 2, 4, 8, 1)):
            L += [nn.Conv2d(cin if i == 0 else w, w, 3, padding=d, dilation=d), nn.GELU()]
        self.body = nn.Sequential(*L); self.head = nn.Conv2d(w, k, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.body(x))


def blend(logit, c, bias):
    w = torch.softmax(logit + bias.view(1, -1, 1, 1), 1)                                # B x K x H x W
    return (w.unsqueeze(2) * c).sum(1), w


def lap_pyr(x, L):
    """Laplacian pyramid of x (N x C x H x W, H and W divisible by 2^(L-1)); exact reconstruction by construction."""
    G = [x]
    for _ in range(L - 1):
        G.append(F.avg_pool2d(G[-1], 2))
    up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return [G[l] - up(G[l + 1], G[l]) for l in range(L - 1)] + [G[-1]]


def blend_band(logit, c, bias, L):
    """Band-wise arbitration: logit B x (K*L) x H x W -> per level softmax over the K candidates, applied to that
    level's Laplacian band of every candidate; then collapse the pyramid."""
    B, K = c.shape[:2]
    pyr = lap_pyr(c.flatten(0, 1), L)                                                   # list of (B*K) x 3 x h x w
    lg = logit.view(B, K, L, *logit.shape[-2:]) + bias.view(1, -1, 1, 1, 1)
    out, wl = None, []
    for l in reversed(range(L)):
        b = pyr[l].view(B, K, 3, *pyr[l].shape[-2:])
        w = torch.softmax(F.adaptive_avg_pool2d(lg[:, :, l], b.shape[-2:]), 1)
        y = (w.unsqueeze(2) * b).sum(1)
        out = y if out is None else y + F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
        wl.append(w.mean((2, 3)))
    return out, torch.stack(wl[::-1], 2)                                                # B x 3 x H x W, B x K x L


def ssim_t(a, b):
    from torchmetrics.functional import structural_similarity_index_measure as S
    return float(S(a[None], b[None], data_range=1.0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", nargs="+", required=True); ap.add_argument("--tag", default="r1")
    ap.add_argument("--iters", type=int, default=3000); ap.add_argument("--crop", type=int, default=192); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default=None); ap.add_argument("--lpips", type=int, default=1)
    ap.add_argument("--mode", default="pixel", choices=["pixel", "band"]); ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--within", action="store_true", help="DIAGNOSTIC ONLY: 2-fold over the views of each scene (uses that scene's test GT)")
    a = ap.parse_args()
    a.out = a.out or os.path.join(ROOT, "outputs", "route", f"gate_loso_{a.mode}.json")
    dev = "cuda"; L = a.levels; mult = 2 ** (L - 1)
    def apply(net, X, C):
        if a.mode == "pixel":
            o, w = blend(net(X), C, bias); return o, w.mean((2, 3))
        return blend_band(net(X), C, bias, L)
    data = {s: [load_view(f) for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_{a.tag}", "*.npz"))) if not f.endswith(".geo.npz")] for s in a.scenes}
    data = {s: v for s, v in data.items() if v}
    if a.within:   # diagnostic: folds = even / odd views of each scene
        data = {f"{s}#{p}": [v for i, v in enumerate(vs) if i % 2 == p] for s, vs in data.items() for p in (0, 1)}
    print({s: len(v) for s, v in data.items()}, flush=True)
    lp = None
    if a.lpips:
        import lpips
        lp = lpips.LPIPS(net="vgg").to(dev).eval()
    # candidate prior: start from "IBGS final" (the 2-stage default)
    bias = torch.tensor([0.0, 2.0, 0.0], device=dev)
    results = {}
    for held in data:
        torch.manual_seed(0); rng = np.random.default_rng(0)
        train = [v for s, vs in data.items() if (s.split("#")[0] == held.split("#")[0] and s != held) == a.within and s != held for v in vs]
        net = Gate(9 + 9, len(CANDS) * (L if a.mode == "band" else 1)).to(dev); opt = torch.optim.Adam(net.parameters(), 2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        for it in range(a.iters):
            X, C, G = [], [], []
            for _ in range(a.bs):
                c, gt, fe, ir = train[rng.integers(len(train))]
                H, W = gt.shape[-2:]; cs = min(a.crop, H, W)
                y0, x0 = rng.integers(0, H - cs + 1), rng.integers(0, W - cs + 1)
                sl = (..., slice(y0, y0 + cs), slice(x0, x0 + cs))
                c_, fe_, ir_ = c[sl].float(), fe[sl].float(), ir[sl].float()
                X.append(make_input(c_, fe_, ir_)); C.append(c_); G.append(gt[sl].float())
            X, C, G = torch.stack(X).to(dev), torch.stack(C).to(dev), torch.stack(G).to(dev)
            out, _ = apply(net, X, C)
            loss = F.mse_loss(out, G)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval(); rows = []
        with torch.no_grad():
            for c, gt, fe, ir in data[held]:
                c, gt, fe, ir = c.float(), gt.float(), fe.float(), ir.float()
                H, W = gt.shape[-2:]; ph, pw = (-H) % mult, (-W) % mult            # reflect-pad to a pyramid-friendly size
                pad = lambda t: F.pad(t[None] if t.dim() == 3 else t, (0, pw, 0, ph), mode="replicate")
                X = pad(make_input(c, fe, ir)); Cp = F.pad(c, (0, pw, 0, ph), mode="replicate")[None]
                o, w = apply(net, X, Cp); o = o[0, :, :H, :W]
                ims = {k: c[i] for i, k in enumerate(CANDS)}; ims["gate"] = o
                e = torch.stack([((c[i] - gt) ** 2).mean(0) for i in range(len(CANDS))])
                r = {}
                for k, im in ims.items():
                    im = (im.clamp(0, 1) * 255 + 0.5).floor() / 255
                    r[k] = (float(-10 * torch.log10(((im - gt) ** 2).mean())), ssim_t(im, gt), float(lp(im[None] * 2 - 1, gt[None] * 2 - 1)) if lp else 0.0)
                r["oracle_px_mcmc_ibgs"] = float(-10 * torch.log10(torch.minimum(e[0], e[1]).mean()))
                nv = fe[0]; bins = {"u0": nv < 0.5, "s12": (nv >= 0.5) & (nv < 2.5), "s3": nv >= 2.5}
                eg = ((((o.clamp(0, 1) * 255 + 0.5).floor() / 255) - gt) ** 2).mean(0)
                for b, msk in bins.items():   # per-support-class squared error sums (pooled later) and pixel counts
                    r[f"bin_{b}"] = [float(msk.sum())] + [float(e[i][msk].sum()) for i in range(len(CANDS))] + [float(eg[msk].sum())]
                r["w_mean"] = w[0].tolist()
                rows.append(r)
        S = {k: tuple(np.mean([r[k][j] for r in rows]) for j in range(3)) for k in list(CANDS) + ["gate"]}
        S["oracle_scene"] = max((S[k] for k in ("mcmc", "ibgs_final")), key=lambda t: t[0])
        S["oracle_px_mcmc_ibgs"] = (np.mean([r["oracle_px_mcmc_ibgs"] for r in rows]), 0, 0)
        S["w_mean"] = np.mean([r["w_mean"] for r in rows], 0).tolist()
        for b in ("u0", "s12", "s3"):   # pooled PSNR per support class: [frac, cands..., gate]
            t = np.sum([r[f"bin_{b}"] for r in rows], 0); n = max(t[0], 1)
            S[f"bin_{b}"] = [t[0] / sum(np.sum([r[f"bin_{x}"][0] for r in rows]) for x in ("u0", "s12", "s3"))] + [float(-10 * np.log10(max(v / n, 1e-12))) for v in t[1:]]
        results[held] = {k: list(v) for k, v in S.items()}
        print(f"[{held}] " + " | ".join(f"{k} {v[0]:.2f}/{v[1]:.3f}/{v[2]:.3f}" for k, v in S.items() if k != "w_mean" and not k.startswith("bin_")) + f" | w {np.round(S['w_mean'], 2).tolist()}", flush=True)
        print(f"    support bins [frac | {' '.join(CANDS)} gate]: " + " ; ".join(f"{b} {S['bin_' + b][0]:.3f} | " + " ".join(f"{x:.2f}" for x in S['bin_' + b][1:]) for b in ("u0", "s12", "s3")), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

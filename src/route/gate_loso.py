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


BAND_FEATS = {"on": False, "L": 5, "aff": False}


def mcmc_affine(z):
    """MCMC corrected by a 3x4 colour affine fitted to the nearest source's valid warp (IBGS's exposure model on MCMC)."""
    I = torch.from_numpy(z["mcmc"].astype(np.float32)).cuda()
    if z["warps"].shape[0] == 0 or not z["valid"][0].any():
        return I
    Wp = torch.from_numpy(z["warps"][0].astype(np.float32)).cuda(); m = torch.from_numpy(z["valid"][0]).cuda()
    X = I[:, m].T; Y = Wp[:, m].T
    if X.shape[0] < 100:
        return I
    A = torch.linalg.lstsq(torch.cat([X, torch.ones_like(X[:, :1])], 1), Y).solution
    flat = I.flatten(1).T
    return (torch.cat([flat, torch.ones_like(flat[:, :1])], 1) @ A).T.view_as(I).clamp(0, 1)


def band_evidence(z, L):
    """Per Laplacian band l: log local energy of (IBGS final - MCMC) and of the spread across valid source warps,
    upsampled to full resolution -> 2L x H x W (the two quantities of the Wiener weight w_E = 1 - S_E / (S_I + S_E))."""
    g = lambda k: torch.from_numpy(z[k].astype(np.float32)).cuda()
    I, E = g("mcmc")[None], g("ibgs_final")[None]; Wp = g("warps"); V = torch.from_numpy(z["valid"]).cuda()
    H, W = I.shape[-2:]
    nv = V.float().sum(0)
    mu = (Wp * V[:, None]).sum(0, keepdim=True) / nv.clamp_min(1)[None, None]
    Wf = torch.where(V[:, None], Wp, mu.expand_as(Wp)) if Wp.shape[0] else I
    def pyr(x):
        G = [x]
        for _ in range(L - 1):
            G.append(F.avg_pool2d(G[-1], 2, ceil_mode=True))
        return [G[l] - F.interpolate(G[l + 1], size=G[l].shape[-2:], mode="bilinear", align_corners=False) for l in range(L - 1)] + [G[-1]]
    PI, PE, PW = pyr(I), pyr(E), pyr(Wf)
    out = []
    for l in range(L):
        d = ((PE[l] - PI[l]) ** 2).mean(1, keepdim=True)
        v = ((PW[l] - PW[l].mean(0, keepdim=True)) ** 2).mean(1, keepdim=True).mean(0, keepdim=True)
        for t in (d, v):
            t = F.avg_pool2d(t, 5, stride=1, padding=2, count_include_pad=False)
            out.append(F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0])
    return ((torch.log10(torch.stack(out) + 1e-6) + 4) / 2).clamp(-2, 2)


def load_view(f):   # kept on the GPU in fp16; cast per crop
    z = np.load(f)
    g = lambda a: torch.from_numpy(a).cuda().half()
    bf = band_evidence(z, BAND_FEATS["L"]).half() if BAND_FEATS["on"] else None
    c = g(np.stack([z[k] for k in CANDS]))
    if BAND_FEATS["aff"]:   # extra candidate appended after the three standard ones
        c = torch.cat([c, mcmc_affine(z).half()[None]])
    return c, g(z["gt"]), g(z["feats"]), g(z["ibgs_raw"]), bf


FEAT_GROUPS = {"support": [0, 1, 2], "disagree": [3, 4], "resid": [5], "models": [6, 7], "depth": [8], "color": slice(9, None)}   # colours last (3 x #candidates)


def make_input(c, fe, ir, drop=(), bf=None):
    """c: all 3 candidates (K=3 x 3 x H x W) — the input always sees the same channels; dropped groups are zeroed."""
    nv, mdd, wr, ws, res, mi, dep = fe
    one = torch.ones((), device=c.device)
    d = dep.clamp_min(1e-3); ld = torch.log(d) - torch.log(d[d > 1e-3].median() if (d > 1e-3).any() else one)
    ens = (c[0] - ir).abs().mean(0)                                                       # MCMC vs IBGS-base disagreement
    x = torch.cat([torch.stack([nv / 5, (nv == 0).float(), mdd, wr * 5, ws * 5, res * 5, mi * 5, ens * 5, ld.clamp(-3, 3) / 3]),
                   c.reshape(-1, *c.shape[-2:])])                                        # candidate colours (3 x K)
    for g in drop:
        x[FEAT_GROUPS[g]] = 0
    if bf is not None:
        x = torch.cat([x, bf])
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
    ap.add_argument("--cands", nargs="+", default=list(CANDS)); ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--scene_tags", nargs="*", default=[], help="per-scene dump tag overrides, e.g. playroom=fix")
    ap.add_argument("--self_dev", default="", help="train on <scene>_<this tag> (cross-fitted dev views of the SAME scene), test on --tag")
    ap.add_argument("--held", nargs="*", default=[], help="evaluate only these held-out scenes (training set unchanged)")
    ap.add_argument("--aff_cand", action="store_true", help="add MCMC+affine-exposure (fit to nearest warp) as a 4th candidate")
    ap.add_argument("--band_feats", action="store_true", help="add per-band evidence (|E-I| and warp-spread energy per Laplacian level)")
    ap.add_argument("--train_scenes", nargs="*", default=[], help="fixed training set (e.g. Shiny); every OTHER scene is evaluated zero-shot")
    ap.add_argument("--lodo", action="store_true", help="leave-one-DATASET-out (mip / tnt / db) instead of leave-one-scene-out")
    ap.add_argument("--within", action="store_true", help="DIAGNOSTIC ONLY: 2-fold over the views of each scene (uses that scene's test GT)")
    a = ap.parse_args()
    a.out = a.out or os.path.join(ROOT, "outputs", "route", f"gate_loso_{a.mode}.json")
    dev = "cuda"; L = a.levels; mult = 2 ** (L - 1)
    def apply(net, X, C):
        if a.mode == "pixel":
            o, w = blend(net(X), C, bias); return o, w.mean((2, 3))
        return blend_band(net(X), C, bias, L)
    BAND_FEATS["on"] = a.band_feats; BAND_FEATS["L"] = a.levels; BAND_FEATS["aff"] = a.aff_cand
    if a.aff_cand and "mcmc_aff" not in a.cands: a.cands = list(a.cands) + ["mcmc_aff"]
    stag = dict(x.split("=") for x in a.scene_tags)
    data = {s: [load_view(f) for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_{stag.get(s, a.tag)}", "*.npz"))) if not f.endswith(".geo.npz")] for s in a.scenes}
    data = {s: v for s, v in data.items() if v}
    if a.within:   # diagnostic: folds = even / odd views of each scene
        data = {f"{s}#{p}": [v for i, v in enumerate(vs) if i % 2 == p] for s, vs in data.items() for p in (0, 1)}
    print({s: len(v) for s, v in data.items()}, flush=True)
    lp = None
    if a.lpips:
        import lpips
        lp = lpips.LPIPS(net="vgg").to(dev).eval()
    # candidate prior: start from "IBGS final" (the 2-stage default)
    ALLC = list(CANDS) + (["mcmc_aff"] if a.aff_cand else [])
    sel = [ALLC.index(k) for k in a.cands]
    bias = torch.tensor([2.0 if k == "ibgs_final" else 0.0 for k in a.cands], device=dev)
    results = {}
    held_list = [s for s in data if s not in a.train_scenes] if a.train_scenes else list(data)
    if a.held: held_list = [s for s in held_list if s in a.held]
    fixed_net = None
    for held in held_list:
        torch.manual_seed(0); rng = np.random.default_rng(0)
        grp = lambda x: {"train": "tnt", "truck": "tnt", "drjohnson": "db", "playroom": "db"}.get(x.split("#")[0], "mip")
        if a.self_dev:
            train = [load_view(f) for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{held}_{a.self_dev}", "*.npz"))) if not f.endswith(".geo.npz")]
        elif a.train_scenes:
            train = [v for s in a.train_scenes for v in data[s]]
        elif a.lodo:
            train = [v for s, vs in data.items() if grp(s) != grp(held) for v in vs]
        else:
            train = [v for s, vs in data.items() if (s.split("#")[0] == held.split("#")[0] and s != held) == a.within and s != held for v in vs]
        def train_once(lr, seed):
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            net = Gate(9 + 3 * len(ALLC) + (2 * L if a.band_feats else 0), len(sel) * (L if a.mode == "band" else 1)).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
            hist_g, hist_c = [], []
            for it in range(a.iters):
                X, C, G = [], [], []
                for _ in range(a.bs):
                    c, gt, fe, ir, bf = train[rng.integers(len(train))]
                    H, W = gt.shape[-2:]; cs = min(a.crop, H, W)
                    y0, x0 = rng.integers(0, H - cs + 1), rng.integers(0, W - cs + 1)
                    sl = (..., slice(y0, y0 + cs), slice(x0, x0 + cs))
                    c_, fe_, ir_ = c[sl].float(), fe[sl].float(), ir[sl].float()
                    X.append(make_input(c_, fe_, ir_, a.drop, bf[sl].float() if bf is not None else None)); C.append(c_[sel]); G.append(gt[sl].float())
                X, C, G = torch.stack(X).to(dev), torch.stack(C).to(dev), torch.stack(G).to(dev)
                out, _ = apply(net, X, C)
                loss = F.mse_loss(out, G)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
                if it >= a.iters - 300:   # divergence guard: compare with the best single candidate on the same batches
                    hist_g.append(float(loss)); hist_c.append(float(((C - G[:, None]) ** 2).mean((0, 2, 3, 4)).min()))
            ok = np.isfinite(hist_g).all() and np.mean(hist_g) < np.mean(hist_c)
            return net, ok, float(np.mean(hist_g)), float(np.mean(hist_c))
        if fixed_net is not None:
            net = fixed_net
        else:
            for attempt, (lr, seed) in enumerate(((2e-3, 0), (1e-3, 1), (5e-4, 2))):
                net, ok, lg_, lc_ = train_once(lr, seed)
                if ok: break
                print(f"    [{held}] attempt {attempt}: train loss {lg_:.2e} not below best single candidate {lc_:.2e} -> retry", flush=True)
        net.eval(); rows = []
        if a.train_scenes: fixed_net = net
        with torch.no_grad():
            for c, gt, fe, ir, bf in data[held]:
                c, gt, fe, ir = c.float(), gt.float(), fe.float(), ir.float(); bf = bf.float() if bf is not None else None
                H, W = gt.shape[-2:]; ph, pw = (-H) % mult, (-W) % mult            # reflect-pad to a pyramid-friendly size
                pad = lambda t: F.pad(t[None] if t.dim() == 3 else t, (0, pw, 0, ph), mode="replicate")
                X = pad(make_input(c, fe, ir, a.drop, bf)); Cp = F.pad(c[sel], (0, pw, 0, ph), mode="replicate")[None]
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
                if a.mode == "band":           # full-res per-level weights, averaged inside each support class (sums, pooled later)
                    lg = net(X)[..., :H, :W].reshape(1, len(sel), L, H, W) + bias.view(1, -1, 1, 1, 1)
                    wf = torch.softmax(lg, 1)[0]                                                    # K x L x H x W
                    for b, msk in bins.items():
                        r[f"wbin_{b}"] = (wf[:, :, msk].sum(-1)).tolist()                           # K x L sums
                r["w_mean"] = w[0].tolist()
                rows.append(r)
        S = {k: tuple(np.mean([r[k][j] for r in rows]) for j in range(3)) for k in list(CANDS) + ["gate"]}
        S["oracle_scene"] = max((S[k] for k in ("mcmc", "ibgs_final")), key=lambda t: t[0])
        S["oracle_px_mcmc_ibgs"] = (np.mean([r["oracle_px_mcmc_ibgs"] for r in rows]), 0, 0)
        S["w_mean"] = np.mean([r["w_mean"] for r in rows], 0).tolist()
        if a.mode == "band":
            for b in ("u0", "s12", "s3"):
                S[f"wbin_{b}"] = (np.sum([r[f"wbin_{b}"] for r in rows], 0) / max(np.sum([r[f"bin_{b}"][0] for r in rows]), 1)).tolist()
            print("    band weights per support class (rows = " + "/".join(a.cands) + ", cols = fine..coarse): " +
                  " ; ".join(f"{b} " + str(np.round(S[f'wbin_{b}'], 2).tolist()) for b in ("u0", "s12", "s3")), flush=True)
        for b in ("u0", "s12", "s3"):   # pooled PSNR per support class: [frac, cands..., gate]
            t = np.sum([r[f"bin_{b}"] for r in rows], 0); n = max(t[0], 1)
            S[f"bin_{b}"] = [t[0] / sum(np.sum([r[f"bin_{x}"][0] for r in rows]) for x in ("u0", "s12", "s3"))] + [float(-10 * np.log10(max(v / n, 1e-12))) for v in t[1:]]
        results[held] = {k: list(v) for k, v in S.items()}
        print(f"[{held}] " + " | ".join(f"{k} {v[0]:.2f}/{v[1]:.3f}/{v[2]:.3f}" for k, v in S.items() if k != "w_mean" and not k.startswith(("bin_", "wbin_"))) + f" | w {np.round(S['w_mean'], 2).tolist()}", flush=True)
        print(f"    support bins [frac | {' '.join(CANDS)} gate]: " + " ; ".join(f"{b} {S['bin_' + b][0]:.3f} | " + " ".join(f"{x:.2f}" for x in S['bin_' + b][1:]) for b in ("u0", "s12", "s3")), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

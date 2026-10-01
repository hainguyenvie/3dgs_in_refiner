"""BandFuse — our own evidence-aware, band-wise fusion of an explicit 3DGS render with individual source warps.

No colour network of IBGS is used: only per-slot warps (depth-tested, from a geometry model) and their camera features.
Per pixel: a shared per-source encoder (warp, warp - render, valid, target-source camera offset, ray cosine), masked
mean/max (+ optional self-attention) across sources, a dilated trunk, and per Laplacian band a softmax over
{explicit render, source_1..source_S} (invalid sources masked out). Output = collapse of the band-wise mixtures.

Training data (no GT of the evaluated scene's TEST views is ever used):
  --protocol loso : train on TEST views of the other scenes
  --protocol dev  : train on the cross-fitting DEV views of all listed scenes (the evaluated one included — its dev
                    views are held-out TRAIN images rendered by an MCMC trained without them)
  --protocol self : train only on the evaluated scene's own dev views
Hard cases (--aug): random source dropout (low support), per-source exposure jitter.

    .venv_ibgs/bin/python src/route/bandfuse.py --scenes bonsai counter train bicycle garden --protocol dev
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
S_MAX = 4


def load_dir(d):
    out = []
    for f in sorted(glob(os.path.join(d, "*.npz"))):
        if f.endswith(".geo.npz"):
            continue
        z = np.load(f)
        if "cam_feat" not in z:
            raise RuntimeError(f"{f}: no cam_feat (dump with run_hybrid2.sh)")
        g = lambda a: torch.from_numpy(np.asarray(a)).cuda()
        W = z["warps"].astype(np.float16); V = z["valid"]; CF = z["cam_feat"].astype(np.float16)
        S = W.shape[0]; H, Wd = z["gt"].shape[-2:]
        Wp = np.zeros((S_MAX, 3, H, Wd), np.float16); Vp = np.zeros((S_MAX, H, Wd), bool); Cp = np.zeros((S_MAX, 4, H, Wd), np.float16)
        Wp[:min(S, S_MAX)] = W[:S_MAX]; Vp[:min(S, S_MAX)] = V[:S_MAX]; Cp[:min(S, S_MAX)] = CF[:S_MAX]
        fe = z["feats"].astype(np.float16)
        out.append(dict(gt=g(z["gt"]).half(), I=g(z["mcmc"]).half(), W=g(Wp), V=g(Vp), C=g(Cp),
                        nv=g(fe[0]), mdd=g(fe[1]), dep=g(fe[6]), ibgs=g(z["ibgs_final"]).half(), ires=g(z["mcmc_res"]).half()))
    return out


def lap_pyr(x, L):
    G = [x]
    for _ in range(L - 1):
        G.append(F.avg_pool2d(G[-1], 2, ceil_mode=True))
    up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return [G[l] - up(G[l + 1], G[l]) for l in range(L - 1)] + [G[-1]]


class BandFuse(nn.Module):
    def __init__(self, L=5, w=32, attn=False, align=False, resid=False):
        super().__init__()
        self.L, self.attn, self.align, self.resid = L, attn, align, resid
        if resid:   # per-band additive correction from the pooled source evidence (zero-init, bounded)
            self.head_res = nn.Sequential(nn.Conv2d(48 + 3, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 3 * L, 3, padding=1))
            nn.init.zeros_(self.head_res[-1].weight); nn.init.zeros_(self.head_res[-1].bias)
        if align:   # per-source sub-pixel flow (<= 2 px) predicted from [warp, render, warp - render]; zero-init = identity
            self.flow = nn.Sequential(nn.Conv2d(9, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(32, 2, 3, padding=1))
            nn.init.zeros_(self.flow[-1].weight); nn.init.zeros_(self.flow[-1].bias)
        self.enc = nn.Sequential(nn.Conv2d(11, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
        if attn:
            self.qkv = nn.Linear(w, 3 * w); self.proj = nn.Linear(w, w)
        self.ienc = nn.Sequential(nn.Conv2d(6, w, 3, padding=1), nn.GELU())
        T = []
        for i, d in enumerate((1, 2, 4, 8, 1)):
            T += [nn.Conv2d(3 * w if i == 0 else 48, 48, 3, padding=d, dilation=d), nn.GELU()]
        self.trunk = nn.Sequential(*T)
        self.head_src = nn.Sequential(nn.Conv2d(48 + w, 32, 1), nn.GELU(), nn.Conv2d(32, L, 1))
        self.head_I = nn.Conv2d(48, L, 1)
        for m in (self.head_src[-1], self.head_I):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def realign(self, I, W):
        """Shift every source warp by a predicted sub-pixel flow toward the render (identity at init)."""
        B, S = W.shape[:2]; H, Wd = I.shape[-2:]
        x = torch.cat([W, I[:, None].expand_as(W), W - I[:, None]], 2).flatten(0, 1)
        fl = 2.0 * torch.tanh(self.flow(x))                                            # (B*S) 2 H W, pixels
        yy, xx = torch.meshgrid(torch.arange(H, device=I.device), torch.arange(Wd, device=I.device), indexing="ij")
        gx = (xx[None] + fl[:, 0]) / (Wd - 1) * 2 - 1; gy = (yy[None] + fl[:, 1]) / (H - 1) * 2 - 1
        Wa = F.grid_sample(W.flatten(0, 1), torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=True)
        return Wa.view_as(W), fl.view(B, S, 2, H, Wd)

    def forward(self, I, W, V, C, nv, mdd, dep):
        """I: B3HW, W: BS3HW (invalid filled with I), V: BSHW, C: BS4HW. Returns logits B x (1+S) x L x H x W."""
        B, S = W.shape[:2]; H, Wd = I.shape[-2:]
        vm = V[:, :, None].float()
        cn = C[:, :, :3].norm(dim=2, keepdim=True)                                    # |target - source centre| (0 if invalid)
        scale = ((cn * vm).sum((1, 2, 3, 4)) / vm.sum((1, 2, 3, 4)).clamp_min(1)).clamp_min(1e-6).view(B, 1, 1, 1, 1)
        x = torch.cat([W, W - I[:, None], vm, C[:, :, :3] / scale, (1 - C[:, :, 3:4]) * 50], 2)   # 3+3+1+3+1 = 11 per source
        f = self.enc(x.flatten(0, 1)).view(B, S, -1, H, Wd)
        if self.attn:   # self-attention across the S sources at every pixel (masked)
            t = f.permute(0, 3, 4, 1, 2).reshape(-1, S, f.shape[2])
            q, k, v = self.qkv(t).chunk(3, -1)
            a = (q @ k.transpose(1, 2)) / q.shape[-1] ** 0.5
            mask = V.permute(0, 2, 3, 1).reshape(-1, 1, S)
            a = a.masked_fill(~mask, -1e4).softmax(-1)
            t = t + self.proj(a @ v)
            f = t.view(B, H, Wd, S, -1).permute(0, 3, 4, 1, 2)
        cnt = vm.sum(1).clamp_min(1)
        mean = (f * vm).sum(1) / cnt; mx = (f * vm - (1 - vm) * 1e4).max(1).values * (vm.sum(1) > 0).float()
        d = dep.clamp_min(1e-3); ld = torch.log(d) - torch.log(d.flatten(1).median(1).values.clamp_min(1e-3)).view(B, 1, 1)
        gi = self.ienc(torch.cat([I, (nv / 4)[:, None], mdd[:, None], ld.clamp(-3, 3)[:, None] / 3], 1))
        tr = self.trunk(torch.cat([mean, mx, gi], 1))
        ls = self.head_src(torch.cat([tr[:, None].expand(-1, S, -1, -1, -1), f], 2).flatten(0, 1)).view(B, S, self.L, H, Wd)
        ls = ls.masked_fill(~V[:, :, None].expand(-1, -1, self.L, -1, -1), -1e4)
        li = self.head_I(tr)[:, None] + 2.0                                            # prior: start from the explicit render
        lg = torch.cat([li, ls], 1)
        if self.resid:
            mres = ((W - I[:, None]) * vm).sum(1) / cnt                                 # mean (warp - render) over valid sources
            self._res = 0.25 * torch.tanh(self.head_res(torch.cat([tr, mres], 1))).view(B, self.L, 3, H, Wd)
        return lg


def fuse(logits, I, W, L, res=None):
    """res: optional B x L x 3 x H x W per-band correction, average-pooled to each band's resolution and added."""
    cands = torch.cat([I[:, None], W], 1)                                              # B (1+S) 3 H W
    B, K = cands.shape[:2]
    P = lap_pyr(cands.flatten(0, 1), L)
    out = None
    for l in reversed(range(L)):
        b = P[l].view(B, K, 3, *P[l].shape[-2:])
        w = torch.softmax(F.adaptive_avg_pool2d(logits[:, :, l].flatten(0, 1)[:, None], b.shape[-2:]).view(B, K, *b.shape[-2:]), 1)
        y = (w[:, :, None] * b).sum(1)
        if res is not None:
            y = y + F.adaptive_avg_pool2d(res[:, l], y.shape[-2:])
        out = y if out is None else y + F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return out


def batch(views, rng, bs, crop, aug):
    keys = ("gt", "I", "W", "V", "C", "nv", "mdd", "dep")
    acc = {k: [] for k in keys}
    for _ in range(bs):
        v = views[rng.integers(len(views))]
        H, Wd = v["gt"].shape[-2:]; cs = min(crop, H, Wd) // 16 * 16
        y, x = rng.integers(0, H - cs + 1), rng.integers(0, Wd - cs + 1)
        for k in keys:
            acc[k].append(v[k][..., y:y + cs, x:x + cs])
    b = {k: torch.stack(acc[k]) for k in keys}
    b = {k: (t.float() if t.dtype == torch.float16 else t) for k, t in b.items()}
    if aug:
        drop = torch.rand(b["V"].shape[:2], device="cuda") < 0.3                        # source dropout -> low support
        b["V"] = b["V"] & ~drop[:, :, None, None]
        b["W"] = b["W"] * (1 + 0.06 * torch.randn(b["W"].shape[:3], device="cuda")[..., None, None]).clamp(0.8, 1.2)
        b["nv"] = b["V"].float().sum(1)
    b["W"] = torch.where(b["V"][:, :, None], b["W"], b["I"][:, None].expand_as(b["W"]))
    return b


def psnr(a, gt):
    a = (a.clamp(0, 1) * 255 + 0.5).floor() / 255
    return float(-10 * torch.log10(((a - gt) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--protocol", default="dev", choices=["loso", "dev", "self", "devft"])
    ap.add_argument("--ft_iters", type=int, default=1500, help="devft: per-scene fine-tuning iterations on its own dev views"); ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--attn", action="store_true"); ap.add_argument("--aug", action="store_true"); ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--align", action="store_true", help="per-source learned sub-pixel re-alignment before fusion")
    ap.add_argument("--resid", action="store_true", help="per-band bounded additive correction head (own IBR network)")
    ap.add_argument("--crop", type=int, default=192); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--tag", default="")
    a = ap.parse_args(); L = a.L
    H = os.path.join(ROOT, "outputs", "route", "hybrid")
    test = {s: load_dir(os.path.join(H, f"{s}_cf")) for s in a.scenes}
    dev = {s: load_dir(os.path.join(H, f"{s}_dev")) for s in a.scenes} if a.protocol in ("dev", "self", "devft") else {}
    print({s: (len(test[s]), len(dev.get(s, []))) for s in a.scenes}, flush=True)
    res = {}
    for held in a.scenes:
        if a.protocol == "loso":
            train = [v for s in a.scenes if s != held for v in test[s]]
        elif a.protocol in ("dev", "devft"):
            train = [v for s in a.scenes for v in dev[s]]
        else:
            train = dev[held]
        if a.protocol in ("dev", "devft") and "shared" in res:   # one shared model pretrained on all dev views
            net = res["shared"]
        else:
            for attempt, (lr, seed) in enumerate(((2e-3, 0), (1e-3, 1), (5e-4, 2))):
                torch.manual_seed(seed); rng = np.random.default_rng(seed)
                net = BandFuse(L, attn=a.attn, align=a.align, resid=a.resid).cuda(); opt = torch.optim.Adam(net.parameters(), lr)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters); hg, hi = [], []
                for it in range(a.iters):
                    b = batch(train, rng, a.bs, a.crop, a.aug)
                    if a.align:
                        Wa, _ = net.realign(b["I"], b["W"]); b["W"] = torch.where(b["V"][:, :, None], Wa, b["I"][:, None].expand_as(Wa))
                    lg_ = net(b["I"], b["W"], b["V"], b["C"], b["nv"], b["mdd"], b["dep"])
                    out = fuse(lg_, b["I"], b["W"], L, net._res if a.resid else None)
                    loss = F.mse_loss(out, b["gt"]); opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
                    if it >= a.iters - 300:
                        hg.append(float(loss)); hi.append(float(F.mse_loss(b["I"], b["gt"])))
                if np.isfinite(hg).all() and np.mean(hg) < np.mean(hi):
                    break
                print(f"    [{held}] attempt {attempt}: train loss {np.mean(hg):.2e} not below explicit render {np.mean(hi):.2e} -> retry", flush=True)
            if a.protocol in ("dev", "devft"): res["shared"] = net
        if a.protocol == "devft":   # copy the shared model, fine-tune on the held scene's own dev views
            import copy
            res["shared"]._res = None                                                 # drop the cached graph tensor before copying
            net = copy.deepcopy(res["shared"]); net.train(); opt = torch.optim.Adam(net.parameters(), 5e-4); rng = np.random.default_rng(7)
            for it in range(a.ft_iters):
                b = batch(dev[held], rng, a.bs, a.crop, a.aug)
                if a.align:
                    Wa, _ = net.realign(b["I"], b["W"]); b["W"] = torch.where(b["V"][:, :, None], Wa, b["I"][:, None].expand_as(Wa))
                lg_ = net(b["I"], b["W"], b["V"], b["C"], b["nv"], b["mdd"], b["dep"])
                loss = F.mse_loss(fuse(lg_, b["I"], b["W"], L, net._res if a.resid else None), b["gt"])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval(); rows = []
        with torch.no_grad():
            for v in test[held]:
                Hh, Ww = v["gt"].shape[-2:]; m = 16; ph, pw = (-Hh) % m, (-Ww) % m
                p = lambda t: F.pad(t, (0, pw, 0, ph), mode="replicate")
                gt = v["gt"].float(); I = v["I"].float()[None]; V = v["V"][None]
                W = torch.where(V[:, :, None], v["W"].float()[None], I[:, None].expand(-1, S_MAX, -1, -1, -1))
                Vp = F.pad(V.float(), (0, pw, 0, ph)).bool()
                if a.align:
                    Wpp = p(W.flatten(1, 2)).view(1, S_MAX, 3, Hh + ph, Ww + pw); Wa, _ = net.realign(p(I), Wpp)
                    W = torch.where(Vp[:, :, None], Wa, p(I)[:, None].expand_as(Wa))[..., :Hh, :Ww]
                lg = net(p(I), p(W.flatten(1, 2)).view(1, S_MAX, 3, Hh + ph, Ww + pw), Vp, p(v["C"].float()[None].flatten(1, 2)).view(1, S_MAX, 4, Hh + ph, Ww + pw),
                         p(v["nv"].float()[None, None])[:, 0], p(v["mdd"].float()[None, None])[:, 0], p(v["dep"].float()[None, None])[:, 0])
                o = fuse(lg, p(I), p(W.flatten(1, 2)).view(1, S_MAX, 3, Hh + ph, Ww + pw), L, net._res if a.resid else None)[0, :, :Hh, :Ww]
                wI = torch.softmax(lg, 1)[0, 0, :, :Hh, :Ww].mean((1, 2))
                rows.append({"mcmc": psnr(I[0], gt), "ibgs": psnr(v["ibgs"].float(), gt), "I+r": psnr(v["ires"].float(), gt), "bandfuse": psnr(o, gt), "wI": wI.tolist()})
        r = {k: float(np.mean([x[k] for x in rows])) for k in ("mcmc", "ibgs", "I+r", "bandfuse")}
        r["wI_per_band"] = np.mean([x["wI"] for x in rows], 0).round(3).tolist(); res[held] = r
        print(f"[{held}] {a.protocol}{' attn' if a.attn else ''}{' aug' if a.aug else ''}{' align' if a.align else ''}{' resid' if a.resid else ''}: mcmc {r['mcmc']:.2f} ibgs {r['ibgs']:.2f} I+r {r['I+r']:.2f} | "
              f"BandFuse {r['bandfuse']:.2f} | w(explicit) per band fine->coarse {r['wI_per_band']}", flush=True)
    res.pop("shared", None)
    tag = a.tag or f"{a.protocol}{'_attn' if a.attn else ''}{'_aug' if a.aug else ''}{'_align' if a.align else ''}{'_resid' if a.resid else ''}"
    json.dump(res, open(os.path.join(ROOT, "outputs", "route", f"bandfuse_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

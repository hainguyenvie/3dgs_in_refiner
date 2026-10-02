"""Failure analysis — why does image evidence help indoor but not outdoor? (analysis only; uses GT)

Per scene, from the hybrid dumps (IBGS geometry: per-slot warps + validity) and the own band-limited dumps (MCMC depth):
  H1 geometry/support : fraction of pixels with >=1 valid source — IBGS geometry vs MCMC expected depth (own dumps)
  H2 evidence quality : where slot-0 is valid, low-pass (Gaussian level 3 ~ 8 px) MSE of the raw warp vs GT, compared to
                        the MCMC render's low-pass MSE on the same pixels (ratio < 1 => photos carry better LF content)
  H3 photometric      : per-view 3x4 colour affine fitted from the slot-0 warp to GT on valid pixels — PSNR gain of the
                        affine (large => the source photo's colour/exposure differs from the target photo), and the LF
                        disagreement between slot-0 and slot-1 warps where both are valid (independent photos)
  H4 regions          : the same LF ratio split by depth tercile (near / mid / far) and by texture (top-25% gradient)

    .venv_ibgs/bin/python src/route/failure_outdoor.py bonsai counter garden stump bicycle ...
"""
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def lp(x, k=3):
    for _ in range(k):
        x = F.avg_pool2d(x, 2, ceil_mode=True)
    return x


def main():
    print(f"{'scene':9s} | supp IBGS  supp MCMC | LF ratio warp/MCMC (valid) | affine gain dB | src-src LF MSEx1e4 | LF ratio near/mid/far | textured/flat")
    for s in sys.argv[1:]:
        fs = [f for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_r1", "*.npz"))) if not f.endswith(".geo.npz")]
        own = sorted(glob(os.path.join(ROOT, "outputs", "route", "bandwarp", f"{s}_own", "*.npz")))
        if not fs:
            continue
        sup_i, sup_m, rat, gain, ss, rz, rt = [], [], [], [], [], [], []
        for vi, f in enumerate(fs):
            z = np.load(f); t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).cuda()
            gt, I = t(z["gt"]), t(z["mcmc"]); W = t(z["warps"]); V = torch.from_numpy(z["valid"]).cuda(); dep = t(z["feats"][6])
            sup_i.append(float((V.any(0)).float().mean()))
            if vi < len(own):
                o = np.load(own[vi]); sup_m.append(float((o["n0"] > 0).mean()))
            if V.shape[0] == 0 or not V[0].any():
                continue
            m = V[0].float()[None, None]
            Lg, Li, Lw, Lm = lp(gt[None]), lp(I[None]), lp(W[0][None]), lp(m)
            ok = (Lm[0, 0] > 0.99)                                                     # LF cells fully covered by slot 0
            if ok.sum() < 50:
                continue
            ew = ((Lw - Lg) ** 2).mean(1)[0][ok]; ei = ((Li - Lg) ** 2).mean(1)[0][ok]
            rat.append(float(ew.mean() / ei.mean()))
            # H3 affine colour mismatch of the source photo
            vm = V[0]
            X = W[0][:, vm].T; Y = gt[:, vm].T
            A = torch.linalg.lstsq(torch.cat([X, torch.ones_like(X[:, :1])], 1), Y).solution
            Xa = torch.cat([X, torch.ones_like(X[:, :1])], 1) @ A
            gain.append(float(10 * torch.log10(((X - Y) ** 2).mean() / ((Xa - Y) ** 2).mean())))
            if V.shape[0] > 1 and V[1].any():
                m2 = (V[0] & V[1]).float()[None, None]; ok2 = lp(m2)[0, 0] > 0.99
                if ok2.sum() > 50:
                    ss.append(float(((lp(W[0][None]) - lp(W[1][None])) ** 2).mean(1)[0][ok2].mean()))
            # H4 regions
            Ld = lp(dep[None, None])[0, 0]; q = torch.quantile(Ld[ok], torch.tensor([1 / 3, 2 / 3], device="cuda"))
            r3 = []
            for a, b in ((-1e9, q[0]), (q[0], q[1]), (q[1], 1e9)):
                mm = ok & (Ld > a) & (Ld <= b)
                r3.append(float(((Lw - Lg) ** 2).mean(1)[0][mm].mean() / ((Li - Lg) ** 2).mean(1)[0][mm].mean().clamp_min(1e-12)) if mm.sum() > 20 else np.nan)
            rz.append(r3)
            gx = (gt[:, :, 1:] - gt[:, :, :-1]).abs().mean(0); gx = F.pad(gx, (0, 1))
            Lgr = lp(gx[None, None])[0, 0]; th = torch.quantile(Lgr[ok], 0.75)
            rt.append([float(((Lw - Lg) ** 2).mean(1)[0][ok & (Lgr > th)].mean() / ((Li - Lg) ** 2).mean(1)[0][ok & (Lgr > th)].mean()),
                       float(((Lw - Lg) ** 2).mean(1)[0][ok & (Lgr <= th)].mean() / ((Li - Lg) ** 2).mean(1)[0][ok & (Lgr <= th)].mean())])
        rz = np.nanmean(np.array(rz), 0) if rz else [np.nan] * 3; rt = np.mean(np.array(rt), 0) if rt else [np.nan] * 2
        print(f"{s:9s} |   {np.mean(sup_i):.2f}       {np.mean(sup_m) if sup_m else float('nan'):.2f}   |          {np.median(rat):.2f}             |     {np.mean(gain):.2f}      |"
              f"      {np.mean(ss) * 1e4 if ss else float('nan'):.2f}        |   {rz[0]:.2f} / {rz[1]:.2f} / {rz[2]:.2f}    |  {rt[0]:.2f} / {rt[1]:.2f}", flush=True)


if __name__ == "__main__":
    main()

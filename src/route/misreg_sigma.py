"""Analysis (uses GT): how misregistered are the source warps? Sub-pixel phase correlation between each valid warp slot
and the ground truth on 32x32 luminance patches (>= 95% valid, textured), per scene -> RMS shift sigma (px), and the
misregistration model's predicted crossover: the Laplacian band whose centre frequency w satisfies w*sigma ~ 1.
Pairs with band_spectrum.py (measured crossover) to test "larger sigma => image evidence stops being useful at a
coarser band".

    .venv_tools/bin/python src/route/misreg_sigma.py bonsai counter train bicycle
"""
import os
import sys
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = 32


def subpix_shift(a, b):
    """shift (dy, dx) of b relative to a by phase correlation with parabolic peak refinement."""
    A = np.fft.fft2(a * np.hanning(P)[:, None] * np.hanning(P)[None]); B = np.fft.fft2(b * np.hanning(P)[:, None] * np.hanning(P)[None])
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    r = np.fft.fftshift(np.real(np.fft.ifft2(R)))
    y, x = np.unravel_index(np.argmax(r), r.shape)
    if not (0 < y < P - 1 and 0 < x < P - 1):
        return None
    def par(m, c, p):
        d = m - 2 * c + p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (m - p) / d
    dy = y + par(r[y - 1, x], r[y, x], r[y + 1, x]) - P // 2
    dx = x + par(r[y, x - 1], r[y, x], r[y, x + 1]) - P // 2
    return dy, dx


def main():
    for s in sys.argv[1:]:
        fs = [f for f in sorted(glob(os.path.join(ROOT, "outputs", "route", "hybrid", f"{s}_r1", "*.npz"))) if not f.endswith(".geo.npz")]
        sh = {0: [], 1: [], 2: []}
        for f in fs[::2]:
            z = np.load(f); gt = z["gt"].astype(np.float32).mean(0); W = z["warps"].astype(np.float32).mean(1); V = z["valid"]
            H, Wd = gt.shape
            for k in range(min(3, W.shape[0])):
                for y in range(0, H - P, P):
                    for x in range(0, Wd - P, P):
                        if V[k, y:y + P, x:x + P].mean() < 0.95: continue
                        g = gt[y:y + P, x:x + P]
                        if g.std() < 0.03: continue
                        d = subpix_shift(g - g.mean(), W[k, y:y + P, x:x + P] - W[k, y:y + P, x:x + P].mean())
                        if d is not None and abs(d[0]) < 4 and abs(d[1]) < 4: sh[k].append(np.hypot(*d))
        for k in sh:
            a = np.array(sh[k])
            if len(a) == 0: continue
            sig = float(np.sqrt(np.mean(a ** 2)))
            # band l of a Laplacian pyramid is centred near w = pi / 2^(l+1) rad/px; crossover where w * sigma ~ 1
            lc = np.log2(np.pi / max(sig, 1e-3)) - 1
            print(f"[{s}] slot {k}: {len(a)} patches, |shift| median {np.median(a):.3f} px, RMS sigma {sig:.3f} px, p90 {np.percentile(a, 90):.3f} -> predicted crossover band ~{lc:.1f}")


if __name__ == "__main__":
    main()

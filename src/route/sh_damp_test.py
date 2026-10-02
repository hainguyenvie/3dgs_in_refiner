"""Training-free test of support-aware view-dependence: scale the non-DC SH coefficients of Gaussians seen by few
training views (frustum count) by alpha at render time; PSNR on test views for a small grid (analysis — the grid is
evaluated on test views, so this only tells whether a support-aware treatment of view-dependence has headroom).

    cd third_party/3dgs-mcmc && python ../../src/route/sh_damp_test.py -s <data> -m <model> -r <res> --eval
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "3dgs-mcmc")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from gauss_diag import visibility  # noqa: E402


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    ds = model.extract(args); g = GaussianModel(ds.sh_degree); pipe = pipeline.extract(args)
    with torch.no_grad():
        scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda"); tr, te = scene.getTrainCameras(), scene.getTestCameras()
        nvis = visibility(g.get_xyz, tr).float().sum(1)
        rest0 = g._features_rest.data.clone()
        def score():
            return float(np.mean([-10 * torch.log10(((render(c, g, pipe, bg)["render"].clamp(0, 1) - c.original_image.cuda()) ** 2).mean()).item() for c in te]))
        base = score(); print(f"[sh-damp] baseline {base:.3f} ({len(te)} views)", flush=True)
        for th in (3, 10, 30):
            m = (nvis < th).float()[:, None, None]
            for alpha in (0.0, 0.5):
                g._features_rest.data = rest0 * (1 - m + m * alpha)
                s = score(); print(f"[sh-damp] nvis<{th:>2d} ({float(m.mean()):.4f} of Gaussians) alpha {alpha}: {s:.3f} ({s - base:+.3f})", flush=True)
        g._features_rest.data = rest0


if __name__ == "__main__":
    main()

"""Nerfbusters-style visibility masks for the eval views of a scene (reimplementation of the protocol: a pseudo-GT model
trained on ALL frames renders depth at each eval view; the back-projected point is counted in every TRAIN camera whose
frustum contains it; mask = count >= 1 AND depth < 2 in nerfstudio's normalised frame (cameras centred on their mean and
scaled so that max |coordinate| = 1)). Also stores the count map (support classes 0 / 1-2 / >=3).

    cd third_party/gaussian-splatting && python ../../src/route/nb_mask.py -s data/nb/<scene> -m <all-frames model> -r 2 --out <dir>
"""
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "gaussian-splatting"))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--out", required=True); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    ds = model.extract(args); ds.eval = True                       # split.json of the scene decides train / eval
    g = GaussianModel(ds.sh_degree); pipe = pipeline.extract(args)
    os.makedirs(args.out, exist_ok=True)
    with torch.no_grad():
        scene = Scene(ds, g, load_iteration=args.iteration, shuffle=False)
        bg = torch.zeros(3, device="cuda")
        tr, te = scene.getTrainCameras(), scene.getTestCameras()
        allc = torch.stack([c.camera_center for c in list(tr) + list(te)])
        cen = allc.mean(0); scale = (allc - cen).abs().max()
        summ = []
        for cam in te:
            z = (torch.cat([g.get_xyz, torch.ones_like(g.get_xyz[:, :1])], 1) @ cam.world_view_transform)[:, 2:3]
            o = render(cam, g, pipe, bg, override_color=torch.cat([z, torch.ones_like(z), torch.zeros_like(z)], 1))["render"]
            D = o[0] / o[1].clamp_min(1e-6); A = o[1]; H, W = D.shape
            fx = W / (2 * math.tan(cam.FoVx / 2)); fy = H / (2 * math.tan(cam.FoVy / 2))
            v, u = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
            Xc = torch.stack([(u + 0.5 - W / 2) / fx * D, (v + 0.5 - H / 2) / fy * D, D, torch.ones_like(D)], -1).view(-1, 4)
            Xw = Xc @ torch.inverse(cam.world_view_transform)
            cnt = torch.zeros(H * W, device="cuda")
            for s in tr:
                Xs = Xw @ s.world_view_transform; zs = Xs[:, 2]
                sfx = s.image_width / (2 * math.tan(s.FoVx / 2)); sfy = s.image_height / (2 * math.tan(s.FoVy / 2))
                us = Xs[:, 0] / zs * sfx + s.image_width / 2; vs = Xs[:, 1] / zs * sfy + s.image_height / 2
                cnt += ((zs > 1e-4) & (us >= 0) & (us < s.image_width) & (vs >= 0) & (vs < s.image_height)).float()
            depth_norm = D / scale
            mask = (cnt.view(H, W) >= 1) & (depth_norm < 2) & (A > 0.5)
            Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.out, f"{os.path.splitext(cam.image_name)[0]}_mask.png"))
            np.save(os.path.join(args.out, f"{os.path.splitext(cam.image_name)[0]}_count.npy"), cnt.view(H, W).to(torch.int16).cpu().numpy())
            summ.append(float(mask.float().mean()))
        json.dump({"coverage_mean": float(np.mean(summ)), "per_view": summ}, open(os.path.join(args.out, "coverage.json"), "w"))
        print(f"[nb_mask] {len(te)} eval views, mean coverage {np.mean(summ):.3f}")


if __name__ == "__main__":
    main()

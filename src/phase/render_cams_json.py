"""D1 (step 2) — render the TEST views of a trained model with cameras taken from a json (d1_swap_test_cams.py):
pose (world-to-camera in the model's frame) and focal/principal point per image. GT images are the model's own test
images; output goes to <model>/test/<out_sub>/{renders,gt} for metrics.py.

    cd third_party/3dgs-mcmc && python ../../src/phase/render_cams_json.py -s <scene> -m <model> -r <res> --eval --cams <json> --out_sub ours_30000_swap
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torchvision

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "3dgs-mcmc"))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # noqa: E402


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--cams", required=True); parser.add_argument("--out_sub", default="ours_30000_swap")
    args = get_combined_args(parser); safe_state(args.quiet)
    J = json.load(open(args.cams))["images"]; byname = {os.path.splitext(k)[0]: v for k, v in J.items()}
    dataset = model.extract(args); gaussians = GaussianModel(dataset.sh_degree)
    with torch.no_grad():
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        base = os.path.join(args.model_path, "test", args.out_sub); os.makedirs(os.path.join(base, "renders"), exist_ok=True); os.makedirs(os.path.join(base, "gt"), exist_ok=True)
        n = 0
        for idx, cam in enumerate(scene.getTestCameras()):
            e = byname[cam.image_name]
            R = np.array(e["R_cw"]).T; T = np.array(e["t_cw"])                      # 3DGS convention: R = R_cw^T, T = t_cw
            fx, fy, cx, cy = e["params"][:4]; W, H = e["width"], e["height"]
            cam.FoVx = 2 * math.atan(W / (2 * fx)); cam.FoVy = 2 * math.atan(H / (2 * fy))
            cam.world_view_transform = torch.tensor(getWorld2View2(R, T, cam.trans, cam.scale)).transpose(0, 1).cuda()
            P = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy)
            P[0, 2] = 2 * cx / W - 1; P[1, 2] = 2 * cy / H - 1                   # off-centre principal point (0 if centred)
            cam.projection_matrix = P.transpose(0, 1).cuda()
            cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]
            img = render(cam, gaussians, pipeline.extract(args), bg)["render"]
            torchvision.utils.save_image(img.clamp(0, 1), os.path.join(base, "renders", f"{idx:05d}.png"))
            torchvision.utils.save_image(cam.original_image[0:3], os.path.join(base, "gt", f"{idx:05d}.png")); n += 1
    print(f"[render_cams_json] {n} test views with cameras from {args.cams} -> {base}")


if __name__ == "__main__":
    main()

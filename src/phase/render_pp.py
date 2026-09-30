"""Render TEST views of a trained MCMC/phase model with per-image principal points from COLMAP (off-centre projection;
see pp_patch.py). Writes <model>/test/ours_<iter>/{renders,gt} exactly like MCMC render.py so metrics.py works.

    cd third_party/3dgs-mcmc && python ../../src/phase/render_pp.py -s <scene> -m <model> -r <res> --eval
"""
import os
import sys

import torch
import torchvision

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "3dgs-mcmc")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from argparse import ArgumentParser  # noqa: E402
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import render  # noqa: E402
from scene import Scene, GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from pp_patch import patch_principal_points  # noqa: E402


def main():
    parser = ArgumentParser(); model = ModelParams(parser, sentinel=True); pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int); parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser); safe_state(args.quiet)
    dataset = model.extract(args); gaussians = GaussianModel(dataset.sh_degree)
    with torch.no_grad():
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        patch_principal_points(scene, dataset.source_path)
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
        base = os.path.join(args.model_path, "test", f"ours_{args.iteration}")
        os.makedirs(os.path.join(base, "renders"), exist_ok=True); os.makedirs(os.path.join(base, "gt"), exist_ok=True)
        for idx, cam in enumerate(scene.getTestCameras()):
            img = render(cam, gaussians, pipeline.extract(args), bg)["render"]
            torchvision.utils.save_image(img.clamp(0, 1), os.path.join(base, "renders", f"{idx:05d}.png"))
            torchvision.utils.save_image(cam.original_image[0:3], os.path.join(base, "gt", f"{idx:05d}.png"))
    print(f"[render_pp] {len(scene.getTestCameras())} test views -> {base}")


if __name__ == "__main__":
    main()

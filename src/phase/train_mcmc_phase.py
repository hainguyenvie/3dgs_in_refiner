#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# [phase] copy of ubc-vision/3dgs-mcmc train.py (7b4fc9f) + per-view phase-aware loss. Run from any cwd.
import os
import json
import sys as _sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, os.path.join(_ROOT, "third_party", "3dgs-mcmc"))
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.gaussian_model import build_scaling_rotation
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class PhaseWarp:
    """Warp the render by a fixed per-view smooth field f (pixels, gt(p) ~ render(p + f(p))) before the loss.
    Fields come from scripts/analysis/p4_fields.py. flow_scale 0 -> identity sampling at integer positions."""
    def __init__(self, path, scale, learn=False, lr=0.0, init_from_fields=True, names=None, deg=3):
        if path and os.path.exists(path):
            j = json.load(open(path)); self.terms = [tuple(t) for t in j["terms"]]; self.views = j["views"]
        else:   # no measured fields: learnable from zero for every train view (RAFT-free)
            self.terms = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]; self.views = {}
            assert learn and names, "without a fields json the field must be learnable and needs the train view names"
        self.scale = scale
        self.cache = {}; self.hits = 0; self.misses = 0
        self.learn = learn; self.params = {}; self.opt = None; self.ref = {}
        if learn:   # [phase] one coefficient table (T x 2) per train view, optimised jointly with the Gaussians
            T = len(self.terms)
            for n in (names or list(self.views.keys())):
                v = self.views.get(n)
                init = torch.tensor(v["A"], dtype=torch.float32) if (v is not None and init_from_fields) else torch.zeros(T, 2)
                self.params[n] = torch.nn.Parameter(init.cuda())
                if v is not None: self.ref[n] = (v["W"], v["H"])
            self.opt = torch.optim.Adam(list(self.params.values()), lr=lr, eps=1e-15)
            print(f"[phase] learnable fields: {len(self.params)} views, lr={lr}, init_from_fields={init_from_fields}")
    def field(self, name, H, W):
        key = (name, H, W)
        if key not in self.cache:
            v = self.views.get(name)
            if v is None:
                self.cache[key] = None
            else:
                A = torch.tensor(v["A"], device="cuda", dtype=torch.float32)          # T x 2 (pixels at the field's own W,H)
                yy, xx = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32), torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij")
                xn, yn = xx / W - 0.5, yy / H - 0.5
                B = torch.stack([xn ** i * yn ** j for i, j in self.terms], -1)
                f = (B @ A) * torch.tensor([W / v["W"], H / v["H"]], device="cuda")   # rescale if resolution differs
                self.cache[key] = f.permute(2, 0, 1)                                    # 2 x H x W
        return self.cache[key]
    def basis(self, H, W):
        key = ("B", H, W)
        if key not in self.cache:
            yy, xx = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32), torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij")
            xn, yn = xx / W - 0.5, yy / H - 0.5
            self.cache[key] = torch.stack([xn ** i * yn ** j for i, j in self.terms], -1)
        return self.cache[key]
    def reg(self):
        """L2 on the field magnitude (pixels) — keeps the learnable warp from absorbing scene content."""
        if not self.learn: return 0.0
        return sum((p ** 2).sum() for p in self.params.values()) / max(1, len(self.params))
    def __call__(self, image, name):
        _, H, W = image.shape
        if self.learn:
            if name not in self.params:
                self.misses += 1; return image
            A = self.params[name]
            sc = torch.tensor([W / self.ref[name][0], H / self.ref[name][1]], device="cuda") if name in self.ref else 1.0
            f = ((self.basis(H, W) @ A) * sc).permute(2, 0, 1)
        else:
            f = self.field(name, H, W)
            if f is None:
                self.misses += 1; return image
        self.hits += 1
        yy, xx = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32), torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij")
        gx = (xx + self.scale * f[0]) / (W - 1) * 2 - 1; gy = (yy + self.scale * f[1]) / (H - 1) * 2 - 1
        return F.grid_sample(image[None], torch.stack([gx, gy], -1)[None], mode="bicubic", padding_mode="border", align_corners=True)[0]

PHASE = None
PHASE_ARGS = None
PHASE_REG = 0.0
PHASE_START = 0
SHIFT_TOL = None   # [phase-M2] (tol_px, grid_n, patch): per-patch min over sub-pixel shifts of the render

def shift_tolerant_l1(image, gt, tol, n, patch):
    """L1 where each PATCH may pick the sub-pixel shift (n x n grid in [-tol, tol] px) of the render that fits best.
    Lets the model keep sharp content whose true location in this photo is off by < tol px (non-rigid /
    per-view local inconsistency) instead of being pushed towards a blurred average."""
    _, H, W = image.shape
    yy, xx = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32), torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij")
    offs = torch.linspace(-tol, tol, n, device="cuda")
    errs = []
    for dy in offs:
        for dx in offs:
            gx = (xx + dx) / (W - 1) * 2 - 1; gy = (yy + dy) / (H - 1) * 2 - 1
            w = F.grid_sample(image[None], torch.stack([gx, gy], -1)[None], mode="bicubic", padding_mode="border", align_corners=True)[0]
            errs.append(F.avg_pool2d((w - gt).abs().mean(0, keepdim=True)[None], patch, ceil_mode=True)[0, 0])
    E = torch.stack(errs)                       # S x Hp x Wp
    return E.min(0).values.mean()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    global PHASE
    if PHASE_ARGS is not None and PHASE is None:   # [phase] RAFT-free learnable fields for all train views
        PHASE = PhaseWarp(PHASE_ARGS["path"], PHASE_ARGS["scale"], learn=True, lr=PHASE_ARGS["lr"], init_from_fields=False,
                          names=[c.image_name for c in scene.getTrainCameras()])
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss  ([phase] compare the render displaced by the view's phase field with the untouched photo)
        gt_image = viewpoint_cam.original_image.cuda()
        image_l = PHASE(image, viewpoint_cam.image_name) if PHASE is not None else image
        if SHIFT_TOL is not None and iteration >= SHIFT_TOL[3]:
            Ll1 = shift_tolerant_l1(image_l, gt_image, *SHIFT_TOL[:3])
        else:
            Ll1 = l1_loss(image_l, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image_l, gt_image))
        if PHASE is not None and PHASE.learn:
            loss = loss + PHASE_REG * PHASE.reg()

        loss = loss + args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(gaussians.get_scaling).mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (gaussians.get_opacity <= 0.005).squeeze(-1)
                gaussians.relocate_gs(dead_mask=dead_mask)
                gaussians.add_new_gs(cap_max=args.cap_max)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if PHASE is not None and PHASE.opt is not None and iteration >= PHASE_START:
                    PHASE.opt.step()
                if PHASE is not None and PHASE.opt is not None:
                    PHASE.opt.zero_grad(set_to_none=True)

                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                noise = torch.randn_like(gaussians._xyz) * (op_sigmoid(1- gaussians.get_opacity))*args.noise_lr*xyz_lr
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--view_flow", type=str, default=None)      # [phase] json from p4_fields.py
    parser.add_argument("--flow_scale", type=float, default=1.0)    # [phase] 0 = identity (code-path control)
    parser.add_argument("--learn_phase", action="store_true")       # [phase] per-view coefficients as parameters
    parser.add_argument("--phase_lr", type=float, default=1e-3)
    parser.add_argument("--phase_reg", type=float, default=1e-4)    # weight on mean squared coefficient (px^2)
    parser.add_argument("--phase_start", type=int, default=1000)    # iterations before the field starts moving
    parser.add_argument("--phase_init_zero", action="store_true")   # ignore the measured fields, start at identity
    parser.add_argument("--shift_tol", type=float, default=0.0)     # [M2] > 0: per-patch min over sub-pixel shifts (px)
    parser.add_argument("--shift_grid", type=int, default=3)
    parser.add_argument("--shift_patch", type=int, default=32)
    parser.add_argument("--shift_start", type=int, default=7000)   # apply once geometry has settled
    args = parser.parse_args(sys.argv[1:])
    if args.shift_tol > 0:
        SHIFT_TOL = (args.shift_tol, args.shift_grid, args.shift_patch, args.shift_start)
        print(f"[phase-M2] shift-tolerant L1: tol={args.shift_tol}px grid={args.shift_grid} patch={args.shift_patch} from iter {args.shift_start}")
    if args.view_flow and os.path.exists(args.view_flow):
        PHASE = PhaseWarp(args.view_flow, args.flow_scale, learn=args.learn_phase, lr=args.phase_lr, init_from_fields=not args.phase_init_zero)
        print(f"[phase] view_flow={args.view_flow} scale={args.flow_scale} views={len(PHASE.views)} learn={args.learn_phase}")
    elif args.learn_phase:   # no fields file: build learnable fields once the scene (train view names) is known
        PHASE_ARGS = {"path": None, "scale": args.flow_scale, "lr": args.phase_lr}
        print(f"[phase] learnable fields from zero for all train views (no measured fields), lr={args.phase_lr}")
    PHASE_REG = args.phase_reg; PHASE_START = args.phase_start
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    if PHASE is not None:
        print(f"[phase] warped {PHASE.hits} renders, {PHASE.misses} views without field")
        if PHASE.learn:
            A = torch.stack(list(PHASE.params.values())).detach()
            mag = [float(((PHASE.basis(64, 96) @ a).norm(dim=-1)).mean()) for a in A]   # mean field magnitude per view (px, coarse grid)
            print(f"[phase] learned field magnitude px: median {float(torch.tensor(mag).median()):.3f} max {max(mag):.3f}")
            torch.save({k: v.detach().cpu() for k, v in PHASE.params.items()}, os.path.join(args.model_path, "phase_fields.pt"))
    print("\nTraining complete.")

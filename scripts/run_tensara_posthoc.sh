#!/usr/bin/env bash
# Tensara post-hoc refiner (round-3 report recipe) ported to a Mip-NeRF 360 scene at eval resolution.
# Stages: base (gsplat MCMC, 1/6 views held out) -> dump holdout (render/warps/masks/depth) -> UNet -> apply on test -> score.
#   G=2 setsid nohup bash scripts/run_tensara_posthoc.sh bicycle > logs/tensara/bicycle.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$1"; CAP="${CAP:-5900000}"; STEPS="${STEPS:-30000}"; RIT="${RIT:-40000}"
T=$ROOT/third_party/tensara/src
PY=$HOME/projects/ares-gs/.venv_gpu/bin/python          # read-only use of a sibling project's gsplat venv
export PYTHONPATH=$T:$T/gs:$T/refine TORCH_HOME=$ROOT/.torch_hub TORCH_EXTENSIONS_DIR=$ROOT/.torch_ext
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${G:?}" OMP_NUM_THREADS=4
S=$ROOT/data/tensara/$SCENE; R=$ROOT/outputs/tensara/$SCENE; mkdir -p $R $ROOT/logs/tensara
DF48="--depth_mode quant --depth_q 0.5 --depth_levels 48 --depth_smooth 40 --depth_smooth_sig 0.04"
GS="--strategy mcmc --cap_max $CAP --max_steps $STEPS --batch_views 4 --lr_scale 2 --depth_weight 0.05 --wd_weight 0.10 --lpips_weight 0 --lpips_start $((STEPS/2)) --lpips_crop 0 --lpips_crops 8 --wd_sigma 1.0 --geo_consist_weight 0.5 --opacity_reg 0 --scale_reg 0.01 --antialiased 1 --sh_degree 3 --bilagrid 0 --test_use_train_K 1 --save_ckpt 1 --skip_test_render 1"
echo "scene=$SCENE cap=$CAP steps=$STEPS refiner_iters=$RIT utc=$(date -u +%FT%TZ)"
# 1. base with holdout (every 6th train view, offset 0)
[ -f $R/gs/ckpt.pt ] || $PY -u $T/gs/trainer.py --scene_dir $S --result_dir $R/gs $GS --seed 42 --holdout_every 6 --holdout_offset 0 || { echo "base failed"; exit 1; }
echo "STAGE_DONE base"
# 2. dump holdout views
[ -f $R/dump/meta.json ] || $PY $T/refine/refiner_data.py --result_dir $R/gs --scene_dir $S --dump $R/dump --targets holdout --holdout_every 6 --K 4 --depth_fix 1 $DF48 || { echo "dump failed"; exit 1; }
echo "STAGE_DONE dump $(ls -d $R/dump/*/ 2>/dev/null | wc -l) views"
# 3. refiner
[ -f $R/refiner.pt ] || $PY -u $T/refine/refiner_train.py train --data $R/dump --out $R/refiner.pt --amp 1 --ch 48,96,192,384 --extra 1 --K 4 --iters $RIT --bs 4 --crop 512 --lr 2e-4 --w_lpips 0.6 --src_drop 0.3 --max_views 600 --val_frac 0.12 --eval_every 4000 --gpu_data 1 || { echo "refiner failed"; exit 1; }
echo "STAGE_DONE refiner"
# 4. apply on test poses (pinhole, no padding) -> final; raw render of the base for comparison
[ -d $R/final ] || $PY $T/refine/refiner_data.py --result_dir $R/gs --scene_dir $S --targets test --dump $R/dump_test --K 4 --depth_fix 1 $DF48 --render_aa 1 --apply_ckpt $R/refiner.pt --apply_out $R/final --apply_ch 48,96,192,384 --apply_fp16 1 || { echo "apply failed"; exit 1; }
echo "STAGE_DONE apply"
# 5. score raw (dump_test/<view>/render.png) and final against the shared test GT
$PY - "$S/test/gt" "$R/dump_test" "$R/final" <<'EOF'
import sys, os, glob, json, torch, numpy as np, lpips
from PIL import Image
gt_dir, dump, final = sys.argv[1:4]; dev = "cuda"
L = lpips.LPIPS(net="vgg").to(dev)
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(dev)
def gauss(x, k=11, s=1.5):
    t = torch.arange(k, device=x.device) - k // 2; g = torch.exp(-t.float() ** 2 / (2 * s * s)); g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    return torch.nn.functional.conv2d(x, w, padding=k // 2, groups=3)
def ssim(a, b):
    C1, C2 = 0.01 ** 2, 0.03 ** 2; ma, mb = gauss(a), gauss(b)
    va, vb, vab = gauss(a * a) - ma ** 2, gauss(b * b) - mb ** 2, gauss(a * b) - ma * mb
    return float((((2 * ma * mb + C1) * (2 * vab + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).mean())
res = {}
for name, getp in [("raw", lambda s: os.path.join(dump, s, "render.png")), ("final", lambda s: os.path.join(final, s + ".png"))]:
    ps, ss, ls = [], [], []
    for g in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        s = os.path.splitext(os.path.basename(g))[0]; p = getp(s)
        if not os.path.exists(p): continue
        a, b = load(p), load(g)
        if a.shape != b.shape: a = torch.nn.functional.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
        ps.append(float(-10 * torch.log10(((a - b) ** 2).mean()))); ss.append(ssim(a, b)); ls.append(float(L(a * 2 - 1, b * 2 - 1)))
    res[name] = {"n": len(ps), "PSNR": float(np.mean(ps)) if ps else None, "SSIM": float(np.mean(ss)) if ss else None, "LPIPS": float(np.mean(ls)) if ls else None}
print("SCORE", json.dumps(res)); json.dump(res, open(os.path.join(os.path.dirname(final), "score.json"), "w"), indent=1)
EOF
echo JOB_DONE

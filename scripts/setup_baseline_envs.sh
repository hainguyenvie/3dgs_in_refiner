#!/usr/bin/env bash
# Baseline environments for Protocol R (reproduce each author's numbers).
#
#   setsid nohup bash scripts/setup_baseline_envs.sh [ibgs|3dgs|mcmc|all] > logs/setup_baseline_envs.log 2>&1 < /dev/null &
#
# One shared stack for all three baselines = the one IBGS pins:
#   python 3.8, torch 2.1.2+cu121, torchvision 0.16.2, numpy 1.24.4
# Deviations from the authors' environments (record in protocol/ENV.md):
#   - 3DGS pins torch 1.12.1/cu116 and 3DGS-MCMC torch 1.13.1/cu117; neither CUDA supports H200
#     (sm_90 needs CUDA >= 11.8), so both run on the IBGS stack instead.
#   - pytorch3d: IBGS installs git HEAD; HEAD no longer builds with torch 2.1, we pin v0.7.8.
#     IBGS only uses pytorch3d.transforms.quaternion_to_matrix (pure torch).
#   - opencv-python -> opencv-python-headless (same cv2, no libGL on the server).
# Host nvcc is 13.0 and host gcc is 13; torch cu121 needs nvcc 12.1, which rejects gcc 13.
# So nvcc 12.1 + gcc 12 are installed into .cuda121 from conda and used for every build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WHAT="${1:-all}"
log() { echo "[env $(date -u +%H:%M:%S)] $*"; }

export UV_CACHE_DIR="$ROOT/.uv-cache"
export MAMBA_ROOT_PREFIX="$ROOT/.mamba"
UV="$HOME/.local/bin/uv"
MM="$HOME/.local/bin/micromamba"
C="$ROOT/.cuda121"
TP="$ROOT/third_party"

# ---- 1. toolchain: nvcc 12.1 + gcc/g++ 12 ------------------------------------
if [ ! -x "$C/bin/nvcc" ]; then
  log "installing CUDA 12.1.1 nvcc + gcc 12 into $C"
  "$MM" create -y -p "$C" -c nvidia/label/cuda-12.1.1 -c conda-forge \
      cuda-nvcc cuda-cudart-dev cuda-cccl "gxx_linux-64=12" "gcc_linux-64=12"
fi
# torch 2.1.2+cu121 from the pytorch index is the "fat" wheel (CUDA libs bundled, no nvidia-* deps),
# so the cuBLAS/cuSPARSE/cuSOLVER/cuRAND headers that ATen/cuda/CUDAContext.h includes must come from conda.
if [ ! -f "$C/include/cublas_v2.h" ] && [ ! -f "$C/targets/x86_64-linux/include/cublas_v2.h" ]; then
  log "adding CUDA 12.1 dev headers"
  "$MM" install -y -p "$C" -c nvidia/label/cuda-12.1.1 -c conda-forge \
      libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev cuda-profiler-api
fi
"$C/bin/nvcc" --version | tail -2
export CC="$C/bin/x86_64-conda-linux-gnu-gcc" CXX="$C/bin/x86_64-conda-linux-gnu-g++"
"$CXX" --version | head -1
export CUDA_HOME="$C" PATH="$C/bin:$PATH"
T="$C/targets/x86_64-linux"; [ -d "$T" ] || T="$C"   # 12.1 from the nvidia channel is flat: include/, lib/
[ -e "$C/lib64" ] || { rm -f "$C/lib64"; ln -s "$T/lib" "$C/lib64"; }
export TORCH_CUDA_ARCH_LIST="9.0"
export NVCC_PREPEND_FLAGS="-include cstdint"
export CXXFLAGS="-include cstdint"
export MAX_JOBS="${MAX_JOBS:-16}"

# ---- 2. venv with the shared torch stack --------------------------------------
mkvenv() {  # mkvenv <dir>
  local V="$1" PY="$1/bin/python"
  "$UV" venv --python 3.8 --allow-existing "$V"
  "$PY" -c "import torch; assert torch.__version__ == '2.1.2+cu121'" 2>/dev/null || \
    "$UV" pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.1.2 torchvision==0.16.2
  "$UV" pip install --python "$PY" "numpy==1.24.4" setuptools wheel ninja
}
cuda_env() {  # headers of torch's nvidia-* wheels (cuBLAS/cuSPARSE/...) for the builds
  local V="$1"
  local WI; WI=$(ls -d "$V"/lib/python3.8/site-packages/nvidia/*/include 2>/dev/null | tr '\n' ':' || true)
  export CPATH="$T/include:$C/include:${WI}"
  export LIBRARY_PATH="$T/lib:$C/lib"
}
build() {  # build <venv> <import-name> <path-or-url>
  local PY="$1/bin/python"
  if "$PY" -c "import torch, $2" 2>/dev/null; then log "[skip] $2"; return 0; fi
  log "building $2 in $(basename "$1")"
  "$UV" pip install --python "$PY" --no-deps --no-build-isolation "$3"
  "$PY" -c "import torch, $2" && log "[ok] $2"
}
smoke() {  # smoke <venv> <imports...>
  local PY="$1/bin/python"; shift
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${SMOKE_CARD:-7}" "$PY" - "$@" <<'PY'
import sys, importlib, torch, numpy
for m in sys.argv[1:]:
    importlib.import_module(m)
x = torch.randn(512, 512, device="cuda")
print("torch", torch.__version__, "numpy", numpy.__version__, "cuda", torch.version.cuda,
      "|", torch.cuda.get_device_name(0), "| matmul", round(float((x @ x).abs().sum()), 1),
      "| imports ok:", ", ".join(sys.argv[1:]))
PY
}

# ---- 3. IBGS -------------------------------------------------------------------
if [[ "$WHAT" == ibgs || "$WHAT" == all ]]; then
  V="$ROOT/.venv_ibgs"; mkvenv "$V"; cuda_env "$V"
  "$UV" pip install --python "$V/bin/python" open3d plyfile opencv-python-headless lpips trimesh \
      tensorboard tqdm iopath
  build "$V" diff_plane_rasterization "$TP/ibgs/submodules/diff-plane-rasterization"
  build "$V" simple_knn._C "$TP/ibgs/submodules/simple-knn"
  build "$V" pytorch3d._C "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
  smoke "$V" diff_plane_rasterization simple_knn._C pytorch3d.transforms open3d trimesh lpips
  log "IBGS_ENV_DONE"
fi

# ---- 4. 3DGS (graphdeco-inria) --------------------------------------------------
if [[ "$WHAT" == 3dgs || "$WHAT" == all ]]; then
  V="$ROOT/.venv_3dgs"; mkvenv "$V"; cuda_env "$V"
  "$UV" pip install --python "$V/bin/python" plyfile tqdm opencv-python-headless joblib lpips
  build "$V" diff_gaussian_rasterization "$TP/gaussian-splatting/submodules/diff-gaussian-rasterization"
  build "$V" simple_knn._C "$TP/gaussian-splatting/submodules/simple-knn"
  build "$V" fused_ssim "$TP/gaussian-splatting/submodules/fused-ssim"
  smoke "$V" diff_gaussian_rasterization simple_knn._C fused_ssim
  log "3DGS_ENV_DONE"
fi

# ---- 5. 3DGS-MCMC (ubc-vision) ---------------------------------------------------
if [[ "$WHAT" == mcmc || "$WHAT" == all ]]; then
  V="$ROOT/.venv_mcmc"; mkvenv "$V"; cuda_env "$V"
  "$UV" pip install --python "$V/bin/python" plyfile tqdm opencv-python-headless lpips
  build "$V" diff_gaussian_rasterization "$TP/3dgs-mcmc/submodules/diff-gaussian-rasterization"
  build "$V" simple_knn._C "$TP/3dgs-mcmc/submodules/simple-knn"
  smoke "$V" diff_gaussian_rasterization simple_knn._C
  log "MCMC_ENV_DONE"
fi
log "ALL_DONE"

# Máy chủ H200 — hướng dẫn dùng cho repo **3dgs_in_refiner** (refiner đưa vào lúc train 3DGS)

> **File DUY NHẤT về máy chủ của repo này.** Viết 28/09/2026, dựa trên guide của project AReS-GS
> (cùng máy, cùng operator) và probe trực tiếp hôm nay. Chỉ giữ các bài học về **máy**; mọi thứ riêng
> của AReS-GS (week24, LightGBM, parquet, fleet…) đã bỏ.
> Alias ssh, host, đường dẫn tuyệt đối **không** nằm trong file này — chúng ở `infra.env` (gitignored,
> mẫu ở `infra.env.example`) và `~/.ssh/config` của operator. Repo có thể public.

**Mục lục** — [0 Hiện trạng](#0-hiện-trạng-2809) · [1 Truy cập](#1-truy-cập-và-đồng-bộ-code) ·
[2 GPU](#2-gpu--luật-dùng-card) · [3 Giới hạn](#3-ba-giới-hạn-dễ-chết-người) ·
[4 Thư mục](#4-bố-cục-thư-mục) · [5 Môi trường](#5-môi-trường-python--cuda) ·
[6 Data](#6-dataset) · [7 Chạy job](#7-chạy-một-job) · [8 Bẫy](#8-bẫy-đã-dẫm-phải-kế-thừa-từ-ares-gs) ·
[9 Dọn dẹp](#9-dọn-dẹp) · [10 Kỷ luật](#10-kỷ-luật-thí-nghiệm) · [11 Backup](#11-backup) ·
[12 Checklist](#12-checklist-mở-đầu-mỗi-phiên)

## 0. Hiện trạng (28/09)

| Thứ | Trạng thái |
|---|---|
| Workspace | `$RIT_REMOTE_ROOT` (= `~/projects/3dgs-refiner-it`) — tạo 28/09, **chỉ project này ghi vào** |
| GPU | 8 × H200 141 GB, driver 590.48.01; project được dùng **cả 8 card** (operator xác nhận 28/09) |
| CUDA trên dev box | chạy thẳng được, không cần pod |
| nvcc hệ thống | **CUDA 13.0** — không build được extension cho torch cu128 (§5) |
| Toolchain riêng | ⏳ chưa dựng `.cuda128` / `.venv_gpu` cho project này (§5) |
| Data | ⏳ đang copy/tải + giải nén bằng `fetch_data.sh` (§6) |
| k8s / pod | còn sống nhưng project này **chưa** cần; chỉ dùng làm lối thoát CPU (§7.3) |

## 1. Truy cập và đồng bộ code

```bash
. ./infra.env
ssh $RIT_SSH                  # alias trong ~/.ssh/config của operator, ProxyJump qua jump host
```

> **Tuyệt đối không** chép nội dung `~/.ssh/config`, IP, port, tên node, namespace hay đường dẫn
> tuyệt đối của home vào file nào trong repo, commit message hay output. Không mang token/SSH key
> lên server. Quét **trước** khi push (git history không xoá được):

```bash
git grep -niE "hgx[0-9]|mnt/registry|/home/[a-z]+/projects|([0-9]{1,3}\.){3}[0-9]{1,3}|BEGIN .*PRIVATE"
```

Server chạy **UTC** (VN = UTC+7). Mọi timestamp trong log là UTC.

**Local là nguồn sự thật, server chỉ là nơi chạy.** Không có `rsync` → tar qua ssh pipe:

```bash
. ./infra.env
# đẩy code
tar czf - --exclude .git --exclude __pycache__ -C . src scripts configs plan protocol tests \
  | ssh $RIT_SSH "tar xzf - -C $RIT_REMOTE_ROOT/"
# kéo kết quả nhẹ về (bảng metric, per-view JSON/CSV, vài ảnh minh hoạ — không kéo checkpoint)
ssh $RIT_SSH "tar czf - -C $RIT_REMOTE_ROOT outputs/week1/tables" | tar xzf - -C ./
```

Sau mỗi lần đẩy, kiểm `md5sum` file quan trọng ở hai đầu (bài học vòng 3: một file trên VM từng bị
ghi đè bằng nội dung file khác).

Có sẵn trên server: `git`, `tmux`, `wget`, `curl`, `unzip`, `kubectl`, `uv` (`~/.local/bin/uv`),
`micromamba` (`~/.local/bin/micromamba`), `python3` 3.12.3, `nvcc` 13.0 ở `/usr/local/cuda-13.0/bin`
(không nằm trong `PATH`). **Không có**: `rsync`, `conda`, `docker`.

## 2. GPU — luật dùng card

8 × H200 141 GB, **dùng chung với các project khác trong cùng một home** (cùng user, cùng PID
namespace — thấy process của nhau).

- **Phân bổ: cả 8 card (0–7)**, `RIT_CARDS` trong `infra.env`. Nhưng vẫn **chỉ vào card đang
  0 MiB** hoặc card mình đang giữ — job của project khác có thể xuất hiện bất cứ lúc nào.
- Xem ai đang giữ card (đường dẫn process cho biết project chủ):

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
```

- **Không kill process của project khác**, không `nvidia-smi -r`, không đổi MIG (MIG tắt).
- Chọn card bằng `CUDA_VISIBLE_DEVICES=<i>`, trong code dùng `cuda:0`. **Luôn** đặt
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` để index trùng với `nvidia-smi` — mặc định hai thứ tự này khác nhau.
- Card có thể bị chiếm giữa chừng. Job dài phải có cổng chờ VRAM:

```bash
G=7
vgate() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -le 1000 ]; do sleep 120; done; }
```

- 3DGS/MCMC ở độ phân giải benchmark ăn vài–vài chục GB; refiner train với `gpu_data` ~16 GB.
  141 GB/card đủ xếp 2–4 job 3DGS cùng card, nhưng **đo đạc latency/FPS phải chạy một mình trên
  card** (measurement contract của plan §4) — ghi rõ trong log card nào, có job khác không.

## 3. Ba giới hạn dễ chết người

**CPU: `nproc` nói dối.** Dev box thấy 192 core / 2 TB RAM của node, nhưng cgroup là
`cpu.max = 3200000 100000` (**32 CPU**) và `memory.max` **512 GiB**, chia với mọi tenant khác.

- Luôn set `OMP_NUM_THREADS` tường minh (≤ 16 khi máy đông), và `num_workers` của DataLoader nhỏ.
- Kiểm `uptime` trước khi phóng lô lớn: `load average` gần 32 = hết quota.
- Nhiều job nạp ảnh full-res song song dễ OOM theo cgroup 512 GiB — **bị giết không có traceback**.

**Đĩa dùng chung.** `$HOME` 3,5 TB, 28/09 còn **~887 GB (74%)**. Checkpoint 3DGS vài trăm MB–vài
GB mỗi scene; dump refiner (render/warp/mask PNG) cỡ GB mỗi scene.

```bash
df -h $HOME                                          # kiểm TRƯỚC mỗi lô mới
du -sh $RIT_REMOTE_ROOT/outputs/* | sort -rh | head
```

**Không replication, không snapshot.** Ổ local trên một node. Code, plan, protocol, bảng metric
phải nằm trong git có remote (§11).

## 4. Bố cục thư mục

```text
$RIT_REMOTE_ROOT/                 (= ~/projects/3dgs-refiner-it)
├── src/ scripts/ configs/ tests/  code — bản gốc ở local, server chỉ là nơi chạy
├── plan/ reports/ protocol/       plan đóng băng, báo cáo, split/manifest đóng băng → git
├── data/
│   ├── zips/                      zip public nguyên bản + SHA256SUMS
│   ├── mipnerf360/<scene>/        giải nén từ 360_v2.zip + 360_extra_scenes.zip
│   └── tandt_db/{tandt,db}/       giải nén từ tandt_db.zip (bản của repo 3DGS)
├── third_party/                   repo baseline (IBGS, gsplat, 3DGS gốc…), ghim commit
├── outputs/<week>/<method>/<scene>/  checkpoint, render raw/final, per-view metric — KHÔNG git
├── logs/
├── .cuda128/  .venv_gpu/          toolchain riêng của project (§5)
└── scripts/fetch_data.sh          stage 0 data (§6)
```

`~/projects/` còn các project khác (kể cả `ares-gs`) — **không ghi vào**. Chỉ được **đọc** khi
cần (ví dụ copy zip public đã kiểm byte, §6).

## 5. Môi trường Python + CUDA

Nguyên tắc: **một venv chính** `.venv_gpu` (python 3.12, torch 2.8.0+cu128, numpy 1.26.4); baseline
nào xung đột phụ thuộc (ví dụ IBGS ghim phiên bản khác) thì venv riêng `.venv_<baseline>`, không
vá chung. Gọi thẳng `$V/bin/python`, **không `activate`**.

Vấn đề cần biết: rasterizer 3DGS/gsplat build CUDA extension, nhưng `nvcc` hệ thống là **13.0**
trong khi torch là cu128 → `torch.utils.cpp_extension` từ chối (*"CUDA 13.0 ≠ 12.8"*). Wheel
`nvidia-cuda-nvcc-cu12` **không** có binary `nvcc`. Cách đã kiểm ở AReS-GS (22/09, tái tạo số chính
xác): cài nvcc 12.8 bằng micromamba vào repo và dùng làm `CUDA_HOME`.

```bash
R=$RIT_REMOTE_ROOT; V=$R/.venv_gpu; C=$R/.cuda128
export UV_CACHE_DIR=$R/.uv-cache MAMBA_ROOT_PREFIX=$R/.mamba   # xoá .uv-cache sau khi dựng xong
~/.local/bin/uv venv --python 3.12 --allow-existing $V
~/.local/bin/uv pip install --python $V/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
~/.local/bin/uv pip install --python $V/bin/python numpy==1.26.4 ninja setuptools wheel
# nvcc 12.8: chỉ compiler + cudart + cccl (bộ runtime 1 GB từng timeout trên đường mạng này)
~/.local/bin/micromamba create -y -p $C -c nvidia/label/cuda-12.8.1 -c conda-forge \
    cuda-nvcc-impl cuda-cudart-dev cuda-cccl
```

Biến môi trường khi build extension (thiếu cái nào là build hỏng):

```bash
export CUDA_HOME=$C PATH=$C/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16
T=$C/targets/x86_64-linux; [ -e $C/lib64 ] || ln -s $T/lib $C/lib64
export CPATH="$T/include:$C/include:$(ls -d $V/lib/python3.12/site-packages/nvidia/*/include | tr '\n' ':')"
export LIBRARY_PATH="$T/lib:$C/lib"
export NVCC_PREPEND_FLAGS="-include cstdint" CXXFLAGS="-include cstdint"   # gcc 13 thiếu <cstdint> transitively
~/.local/bin/uv pip install --python $V/bin/python --no-deps --no-build-isolation <path-hoặc-url-extension>
```

Script mẫu đầy đủ (gồm smoke test render gsplat): `scripts/setup_devbox_gpu.sh` của AReS-GS — port
sang `scripts/setup_gpu_env.sh` của repo này, bỏ các gói không dùng. Sau khi dựng, ghi phiên bản thật
(torch, gsplat, rasterizer, commit baseline) vào mục §0.

## 6. Dataset

Chỉ dùng **bản public nguyên gốc**, kiểm được bằng byte.

| Zip | Nguồn | Kích thước (byte) | Ghi chú |
|---|---|---|---|
| `360_v2.zip` | `storage.googleapis.com/gresearch/refraw360/` | 12 535 427 936 | Mip-NeRF 360: bicycle, bonsai, counter, garden, kitchen, room, stump |
| `360_extra_scenes.zip` | cùng host | 4 488 140 217 | **flowers, treehill** — không có trong `360_v2.zip` |
| `tandt_db.zip` | `repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/` | 682 628 995 | T&T train/truck + Deep Blending drjohnson/playroom (bản của repo 3DGS) |

28/09: zip `360_v2` và `tandt_db` trong `ares-gs/data/` **trùng từng byte** với Content-Length của
host chính thức → copy zip (không copy thư mục đã giải nén của ares-gs vì có file dẫn xuất như
`images_4_png`), giải nén lại. `360_extra_scenes.zip` tải thẳng. SHA-256 lưu ở `data/zips/SHA256SUMS`.

```bash
setsid nohup bash scripts/fetch_data.sh > logs/fetch_data.log 2>&1 < /dev/null &   # idempotent, guard từng bước
```

Shiny (NeX) chưa tải — thêm khi tới pha mở rộng. Split/resolution **không** tự chọn: theo đúng
protocol của từng baseline (plan §2, Protocol R) và đóng băng vào `protocol/` trước khi chạy.

## 7. Chạy một job

### 7.1 Ba điều bắt buộc

```bash
setsid nohup bash scripts/<job>.sh > logs/<job>.log 2>&1 < /dev/null &
ssh -n $RIT_SSH "tail -5 $RIT_REMOTE_ROOT/logs/<job>.log"
```

1. **`setsid`**, không phải `nohup … & disown` — ssh bị `timeout` giết thì job con chết theo nếu chỉ `nohup`.
2. **`< /dev/null`** — không thì ssh giữ channel. Kèm theo: **ssh không tự thoát** dù job đã detach →
   bọc `timeout 20`, coi `Terminated`/exit 124 là bình thường. Tạo thư mục log **trước** khi redirect.
3. **`ssh -n`** khi không cần stdin.

### 7.2 Khung tối thiểu của một script

```bash
#!/bin/bash
set -euo pipefail
ROOT=$HOME/projects/3dgs-refiner-it
PY=$ROOT/.venv_gpu/bin/python
export PYTHONPATH=$ROOT/src
export OMP_NUM_THREADS=16                # cgroup chỉ 32 CPU, chia với người khác
export CUDA_DEVICE_ORDER=PCI_BUS_ID
G=${G:-7}; vgate                         # chờ card trống (§2)
OUT=$ROOT/outputs/week1/mcmc/bicycle
[ -f $OUT/metrics_test.json ] || \
  CUDA_VISIBLE_DEVICES=$G $PY -u src/... --out $OUT >> $ROOT/logs/x.log 2>&1
echo JOB_DONE
```

Ba quy ước: **guard tái chạy** trên **file cuối cùng** của chuỗi (`[ -f <output cuối> ] || …`),
**cổng rẻ trước bước đắt** (1 scene/1 seed đạt gate rồi mới chạy đủ), **`python -u`**.

### 7.3 Pod k8s (chỉ khi cần)

Project này mặc định **không** dùng pod. Chỉ cân nhắc khi stage CPU nặng mà dev box đông (pod có
cgroup riêng). Khi đó port `k8s/mkpod.sh` từ AReS-GS; luật đã học: `pods/exec` bị cấm (mọi lệnh nằm
trong `command:` lúc tạo), dùng `pin:` thay vì xin `nvidia.com/gpu` (tránh `Pending` vĩnh viễn),
`set -o pipefail` trong lệnh pod, xác nhận xong bằng artefact chứ không bằng `Succeeded`.
⚠ Đừng kéo image pod khi ổ containerd của node sát ngưỡng — từng gây DiskPressure, pod dev bị evict,
mất SSH ~20 phút. Kiểm `df -B1G /` trước.

## 8. Bẫy đã dẫm phải (kế thừa từ AReS-GS)

| Triệu chứng | Nguyên nhân | Cách xử |
|---|---|---|
| Job nền chết khi ssh đóng | `nohup … & disown` không tách session | `setsid nohup … < /dev/null &` |
| ssh không return dù job đã nền | ssh chờ EOF | `timeout 20`, coi `Terminated` là bình thường |
| `logs/x.log: No such file or directory` lúc phóng | redirect chạy trước `mkdir` trong script | `mkdir -p logs` trước dòng `setsid` |
| Bash exit 255/144 khi dừng job | `pkill -f`/`pgrep -f` khớp chính dòng ssh → tự sát | kill **theo PID** |
| Build extension báo *CUDA 13.0 ≠ 12.8* | nvcc hệ thống 13.0 | `CUDA_HOME=.cuda128` (§5) |
| Build lỗi `uint32_t`/`uintptr_t` undeclared | gcc 13 không kéo `<cstdint>` | `NVCC_PREPEND_FLAGS`/`CXXFLAGS="-include cstdint"` |
| Run hỏng tưởng hoàn tất | guard chỉ kiểm file trung gian | kiểm **file cuối**; thiếu thì xoá hẳn thư mục rồi chạy lại |
| Job nền không có log tiến độ | `cmd \| tail -N` gom đầu vào | ghi thẳng ra file; đừng pipe qua `grep` (nuốt traceback) |
| Traceback in **trước** dòng stdout trước nó | stderr không buffer | quy tội bằng thư mục output nào thiếu file |
| `git clone` treo rồi tự xoá thư mục | clone hỏng phiên trước dọn muộn | `pgrep -af "git clone"` trước khi clone lại |
| `OverflowError` lúc import gsplat/pycolmap | `np.uint64(-1)` với numpy 2 | **`numpy==1.26.4`** ghim cứng |
| `np.trapezoid` không tồn tại | chỉ có từ numpy 2 | `np.trapz`; lỗi kiểu này nổ ở **bước cuối** sau nhiều giờ GPU |
| `ValueError: X.__spec__ is None` | stub module thiếu `__spec__` | `ModuleSpec(name, loader=None)`; import torch trước khi stub |
| `pip install … \| tail` báo OK nhưng thiếu module | pipe nuốt lỗi build | không pipe output `pip`/`uv` |
| `uv venv` báo *already exists* | thiếu cờ | `--allow-existing` |
| `uv pip install` exit 1 kèm *hint: sympy … torch depends on* | **lỗi mạng**, không phải xung đột | chạy lại |
| `$VAR` rỗng trong script sinh bằng heredoc | nội suy sớm | heredoc `<<'EOF'` |
| Số lệch ~1e-4 dù phép tính tất định | TF32 bật mặc định | tắt TF32 khi so fp32 |

Bẫy riêng của refiner (từ vòng 3, cần tránh khi port post-hoc baseline): dump và suy luận phải cùng
base, cùng khung camera, cùng cờ depth (`depth_levels`…); truyền đúng `holdout_offset` (từng dump
nhầm view train); GT khử méo vs ảnh ra ở khung méo khi chấm.

## 9. Dọn dẹp

**Đi qua thùng rác, đừng `rm -rf` thẳng:**

```bash
T=$HOME/_trash_$(date +%Y%m%d); mkdir -p $T
mv <thứ-cần-bỏ> $T/
#  … kiểm vẫn chạy …
rm -rf $T
```

| | dựng lại được | **không** dựng lại được |
|---|---|---|
| **nhẹ** | log | **code, `protocol/`, bảng metric, `plan/`, `reports/`** → git |
| **nặng** | checkpoint, render, dataset, venv | (không nên tồn tại) |

Bảng metric per-view đừng để lẫn trong `outputs/` — copy về `reports/`/git ngay khi đóng một run.
`.uv-cache/` xoá sau khi dựng venv. Đừng suy ra "không cần trọng số này" từ giá trị mặc định trong
script — kiểm bằng log/config của chính run đó.

## 10. Kỷ luật thí nghiệm

1. **Đóng băng quyết định TRƯỚC khi chạy.** `plan/weekN_plan.md` ghi gate + nhánh; `reports/weekN_report.md`
   đối chiếu. Không đạt thì ghi FAIL, không nới gate, không đổi metric sau khi thấy số.
2. **Xác nhận job xong bằng artefact**, không bằng exit code của wrapper.
3. **Kiểm giả định trước khi đốt GPU.** Commit sửa file training → `git diff` để chắc không đổi
   training math; nếu đổi, run cũ không còn so được với run mới.
4. **Test split chỉ để đánh giá cuối** (plan §2): không chọn tham số, checkpoint hay pseudo-target bằng test.
5. **Cảnh giác kết quả ở mép lưới** — chưa tìm thấy đỉnh, hoặc lưới trỏ ra khỏi phương pháp.
6. **"Chưa đủ power" ≠ "không có hiệu ứng"** — CI chứa 0 với vài seed không có nghĩa hiệu ứng bằng 0.
7. **Nghi artifact của harness** (split, mask, resolution, scorer) trước khi kết luận hiện tượng khoa học.
8. **Đo đúng đại lượng** — mức view vs mức vùng/patch có thể lật kết luận.
9. **Vài scene không đại diện cho một dataset.**

## 11. Backup

Server là working copy. Nhẹ-mà-quý (code, `plan/`, `protocol/`, `reports/`, bảng metric per-view)
phải về local và vào git remote sau mỗi phiên. Checkpoint/render lớn: thống kê kích thước rồi quyết,
không tự ý bỏ. Khi sao, lưu `git status`/HEAD của server kèm theo, so SHA-256 hai đầu; **không xoá
nguồn trong cùng lượt với backup**. So hai cây bằng:

```bash
find . -type f -printf "%s %P\n" | sort | md5sum
```

## 12. Checklist mở đầu mỗi phiên

```bash
. ./infra.env
ssh -n $RIT_SSH 'hostname; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'   # card nào trống
ssh -n $RIT_SSH 'uptime; cat /sys/fs/cgroup/cpu.max'                                          # quota CPU
ssh -n $RIT_SSH "pgrep -af 3dgs-refiner-[i]t | head"                                          # job cũ của mình
ssh -n $RIT_SSH 'df -h $HOME | tail -1'                                                       # đĩa
```

Bốn câu hỏi trước lệnh đầu tiên: gate tuần này **đã đóng băng** chưa; split/protocol của run này
**đã ghi vào `protocol/`** chưa; stage này **cần GPU thật** hay chỉ CPU; nếu chạy 6 tiếng rồi chết thì
**guard tái chạy** có cứu được không.

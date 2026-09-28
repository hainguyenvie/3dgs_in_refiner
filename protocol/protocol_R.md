# Protocol R — tái hiện số của tác giả (đóng băng 28/09/2026, trước khi chạy)

Mục đích: kiểm tra triển khai. Mỗi baseline chạy **đúng** split/resolution/cờ của repo tác giả; báo
`paper / reproduced / Δ`. Đây **không** phải bảng so sánh công bằng giữa các phương pháp (xem Protocol C).

## 1. Dữ liệu (chung)

| Dataset | Nguồn | Scene | Split |
|---|---|---|---|
| Mip-NeRF 360 | `360_v2.zip` + `360_extra_scenes.zip` (Google, byte-verified) | 9 | mọi repo: sắp tên ảnh, `idx % 8 == 0` → test |
| T&T, Deep Blending | `tandt_db.zip` của repo 3DGS (byte-verified) | train, truck, drjohnson, playroom | như trên |
| Shiny | bản đã xử lý của IBGS (Google Drive trong README IBGS) | guitars, lab, cd | như trên |

## 2. Môi trường (chung, sai lệch có chủ đích)

python 3.8.20, torch 2.1.2+cu121, torchvision 0.16.2, numpy 1.24.4, nvcc 12.1.105 + gcc 11 (conda),
`TORCH_CUDA_ARCH_LIST=9.0`, GPU H200. Script: `scripts/setup_baseline_envs.sh`.

- IBGS: đúng stack tác giả khai (torch 2.1.2/cu121; python 3.8). pytorch3d ghim v0.7.8 (HEAD không build với torch 2.1).
- 3DGS (torch 1.12/cu116) và MCMC (torch 1.13/cu117): CUDA đó không hỗ trợ sm_90 → chạy trên stack IBGS.
  Không đổi code/hyper-parameter nào.
- Tác giả chạy RTX 4090 → **thời gian train/FPS không so trực tiếp** với paper.

## 3. Cờ từng baseline

### IBGS — `third_party/ibgs` @ `977e96c` (script `scripts/run_ibgs.sh`, cờ chép từ `exp_script.py`)

| Nhóm | Cờ |
|---|---|
| Mip360 outdoor (bicycle, flowers, garden, stump, treehill) | `-r 4 --eval` (resize từ `images/` full-res trong code) |
| Mip360 indoor (bonsai, counter, kitchen, room) | `-r 2 --eval` |
| Deep Blending | `-r 1 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5` |
| T&T | `-r 2 --eval --exposure_compensation --enable_exposure_correction` |
| Shiny | `-r 1008 --eval --multi_view_max_angle 50 --multi_view_max_dis 4.5` |

Seed cố định 24 trong `train.py`. `render.py` thêm `--skip_train` (không ảnh hưởng metric test).
Hành vi giữ nguyên của tác giả: lúc test, ảnh nguồn (train) được **ghi lại JPEG rồi đọc lại** trước khi
render để "mô phỏng deploy"; FPS đo trước bước đó, 5 vòng × test views sau 1 vòng warm-up.
Output: `renders/` = **raw** (base Gaussian), `renders_aggregate/` = **final**. Metric: `metrics.py` của tác giả
(SSIM Gauss-11, PSNR, LPIPS-VGG của `lpipsPyTorch`).

### 3DGS — `third_party/gaussian-splatting` @ `54c035f` (theo `full_eval.py`, không bật cờ tuỳ chọn mới)

| Nhóm | Cờ |
|---|---|
| Mip360 outdoor | `-i images_4 --eval` |
| Mip360 indoor | `-i images_2 --eval` |
| T&T, DB | `--eval` |

Chung: `--disable_viewer --quiet --eval --test_iterations -1`. Số paper = 3DGS paper, Ours-30k (bảng 4–9).
⚠ Lưu ý: paper 3DGS (2023) chạy code bản gốc; commit hiện tại có thay đổi mặc định nhỏ — ghi Δ, không chỉnh.

### 3DGS-MCMC — `third_party/3dgs-mcmc` @ `7b4fc9f`

`--config configs/<scene>.json --eval --init_type sfm` (config tác giả: `resolution` 4/2/1, `cap_max` per scene;
mặc định `scale_reg 0.01, opacity_reg 0.01` (DB drjohnson 0.001 theo config), `noise_lr 5e5`).
⚠ Mặc định repo là `init_type random`; ta so với cột **Ours (SfM)** của paper. Paper không có flowers/treehill
(không có config `cap_max`) → Protocol R của MCMC = 7 scene Mip360 + T&T + DB. Paper báo trung bình 3 lần chạy.

### Kiểm chứng dữ liệu/độ phân giải (28/09, từ ảnh GT GADA công bố — cùng codebase IBGS)

- IBGS/GADA lấy kích thước gốc từ **intrinsics COLMAP**, không từ file ảnh (`loadCam`: `round(cam.width / r)`).
  `tandt_db` có COLMAP 1959×1090 (train), 1957×1091 (truck) nhưng ảnh 980×545 / 979×546 → `-r 2` cho
  **980×545 / 978×546**. GT T&T của GADA trùng **từng pixel** với ảnh `tandt_db` (train) → cùng dữ liệu với ta;
  độ phân giải đánh giá T&T tương đương 3DGS (truck lệch 1 px do resample 979→978).
- Mip360: GT 1236×822 (bicycle) = `round(4946/4)` từ `images/` full-res; `images_4` của 3DGS là 1237×822 → hai
  protocol **khác nhau về ảnh GT** (resize trong code vs JPEG downsample sẵn). Protocol C phải chọn một.
- GADA công bố renders + GT + `results_renders*.json` cho Mip360 (9, tên `flower`), T&T (2), Shiny (3);
  **không có Deep Blending**. Lưu ở `outputs/published/gada/`.

## 4. Số mục tiêu (paper)

### IBGS (NeurIPS'25 Tab. 1–2; chỉ có trung bình dataset)

| Dataset | PSNR | SSIM | LPIPS | #Gauss (M) | Mem (MB) |
|---|---|---|---|---|---|
| Mip-NeRF 360 (9) | 28.33 | 0.837 | 0.186 | 1.59 | 291 |
| T&T (2) | 24.84 | 0.869 | 0.148 | 0.75 | 143 |
| Deep Blending (2) | 30.12 | 0.912 | 0.237 | 1.11 | 197 |
| Shiny guitars / lab / cd (PSNR) | 35.65 / 35.06 / 35.23 | 0.953 / 0.966 / 0.955 | 0.105 / 0.056 / 0.060 | | |

Train (RTX 4090): 44 / 21 / 39 phút (Mip360 / T&T / DB). VRAM inference: 6.12 / 2.97 / 5.36 GB.

### 3DGS (paper, Ours-30k, PSNR / SSIM / LPIPS)

| bicycle | flowers | garden | stump | treehill | room | counter | kitchen | bonsai |
|---|---|---|---|---|---|---|---|---|
| 25.246 / .771 / .205 | 21.520 / .605 / .336 | 27.410 / .868 / .103 | 26.550 / .775 / .210 | 22.490 / .638 / .317 | 30.632 / .914 / .220 | 28.700 / .905 / .204 | 30.317 / .922 / .129 | 31.980 / .938 / .205 |

| truck | train | drjohnson | playroom |
|---|---|---|---|
| 25.187 / .879 / .148 | 21.097 / .802 / .218 | 28.766 / .899 / .244 | 30.044 / .906 / .241 |

### 3DGS-MCMC (paper Tab. 5, Ours (SfM), PSNR / SSIM / LPIPS)

| counter | stump | kitchen | bicycle | bonsai | room | garden | **avg7** |
|---|---|---|---|---|---|---|---|
| 29.51/.92/.22 | 27.80/.82/.19 | 32.27/.94/.14 | 26.15/.81/.18 | 32.88/.95/.22 | 32.48/.94/.25 | 28.16/.89/.10 | 29.89/.90/.19 |

| train | truck | **T&T avg** | drjohnson | playroom | **DB avg** |
|---|---|---|---|---|---|
| 22.47/.83/.24 | 26.11/.89/.14 | 24.29/.86/.19 | 29.00/.89/.33 | 30.33/.90/.31 | 29.67/.89/.32 |

⚠ LPIPS của MCMC cao bất thường ở DB (0.32 vs 3DGS 0.24): khả năng khác cài đặt LPIPS (paper nói có
"correct LPIPS as reported by [4]"). Khi so, đọc LPIPS theo script của chính repo.

### Tham chiếu phụ: GADA (arXiv 2607.00595, code chưa phát hành)

| | Mip360 (9) PSNR/SSIM/LPIPS | T&T | DB |
|---|---|---|---|
| GADA paper Tab. 1 — Ours | 28.62 / 0.840 / 0.179 | 24.92 / 0.871 / 0.144 | 30.22 / 0.911 / 0.235 |
| GADA JSON công bố — final (ta tính trung bình) | 28.63 / 0.841 / 0.178 | 24.93 / 0.871 / 0.145 | — |
| GADA JSON công bố — **raw** | 27.33 / 0.826 / 0.207 | 23.13 / 0.843 / 0.192 | — |
| GADA chạy lại **IBGS** (Tab. 1) | 28.29 / 0.831 / 0.191 | 24.75 / 0.861 / 0.154 | 29.94 / 0.899 / 0.237 |
| GADA chạy lại **3DGS-MCMC** (9 scene!) | 27.98 / 0.835 / 0.224 | 24.29 / 0.860 / 0.190 | 29.67 / 0.895 / 0.320 |

→ Một lần tái hiện IBGS độc lập đã lệch paper −0.04 / −0.09 / −0.18 dB: dùng làm mốc dung sai thực tế.
→ Raw của GADA (27.33) thấp hơn MCMC (27.98): gain nằm ở nhánh warp, không ở Gaussian.
Shiny (final) GADA: 35.57 trung bình 3 scene. Chấm lại ảnh GADA bằng scorer của ta trước khi dùng các số này.

## 5. Gate chấp nhận một lần tái hiện

Chưa đo run variance trên máy này → **không** đặt ngưỡng cứng trước. Quy tắc: báo Δ từng dataset; nếu
|ΔPSNR| của trung bình dataset > 0.3 dB (≈ 5–10× std MCMC báo) thì **truy nguồn** (split, resize, cờ,
phiên bản) trước khi dùng baseline đó cho Protocol C. Mọi Δ, kể cả nhỏ, ghi vào `reports/`.

## 6. Nhật ký truy nguồn (ghi khi phát sinh)

- **28/09 — MCMC playroom −1.10 dB** (29.23 vs paper 30.33; drjohnson +0.50). `configs/playroom.json` của repo
  **thiếu `opacity_reg: 0.001`** (drjohnson có), trong khi paper nói Deep Blending dùng λo = 0.001 → run r1 (đúng
  theo repo) chạy λo = 0.01. Run chẩn đoán `playroom_oreg001` (`EXTRA_TRAIN='--opacity_reg 0.001'`). r1 giữ nguyên là
  "theo repo"; nếu oreg001 về sát paper thì Protocol C dùng λo = 0.001 cho cả hai scene DB và ghi rõ.
  **Kết quả:** oreg001 = 30.03 / 0.909 / 0.229 (vs r1 29.23; paper 30.33) → lỗi config giải thích ~0.8/1.1 dB.
  DB avg với λo = 0.001: 29.77 vs paper 29.66 (+0.11). **Quyết định: Protocol C dùng λo = 0.001 cho DB.**
- LPIPS của MCMC paper cao hơn hệ thống so với `metrics.py` của repo (vd. room 0.25 vs 0.171) → không so LPIPS
  MCMC với paper; chỉ PSNR/SSIM.

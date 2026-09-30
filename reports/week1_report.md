# Week 1 — Báo cáo tổng hợp (cập nhật 29/09/2026, 01:55 UTC — thí nghiệm GPU tạm dừng, card đã nhả)

Một file duy nhất cho toàn bộ tuần 1: mục tiêu, baseline, chuỗi chẩn đoán, method, kết quả 13 scene, các thí nghiệm
đã đóng (kể cả null), định vị so với prior art, và việc đang chạy. Các file `reports/*.md/json` khác chỉ là log số
liệu thô do script sinh ra; mọi thứ cần đọc nằm ở đây.

---

## 0. Tóm tắt một trang

- **Mục tiêu**: từ intuition của Tensara (refiner post-hoc có điều kiện láng giềng) → đưa tín hiệu đó *vào training*
  để inference **một stage** (không cần ảnh nguồn lúc test), và vượt IBGS/GADA trên Mip-NeRF 360 / T&T / DB.
- **Phát hiện chính**: render 3DGS mờ vì **các camera COLMAP lệch pha nhau dưới pixel** (không phải hết năng lực);
  nhánh warp/IBR không chuyển được tần số cao vì **toán tử resample** ở pha lẻ đã phá HF (không phải depth).
  Field pha per-view mà mình học được hoá ra **chính là camera refinement**: phần per-view = pose 6-DoF (72–97%),
  phần chung = intrinsics (principal point/focal).
- **Method** (`src/phase/train_mcmc_phase.py`): 3DGS-MCMC + field pha bậc 3 học được cho từng view train, áp lên
  render *trước* loss; anchor 1/8 view giữ gauge; field chung (intrinsics) áp cho camera test. Không cần gradient pose
  từ rasterizer, chi phí train ≈ +0, inference = 3DGS thuần.
- **Kết quả (1 seed, 13 scene)** — `m1w [sh]` so với MCMC: Mip-360 **+0.03** (outdoor 5/5 lên, indoor 4/4 xuống nhẹ),
  T&T **+0.07**, DB **+0.20**. So với GADA final: **thắng cả 5 scene outdoor Mip-360 (+0.10…+0.63) và truck (+0.19)**,
  thua indoor rất xa (bonsai −2.65, counter −1.45) và train (−0.98). Trên Mip-360 trung bình: SSIM/LPIPS hơn IBGS final
  (0.849/0.170 vs 0.841/0.181), PSNR kém −0.20. Scene tốt nhất: stump **+0.27**, bicycle **+0.20**, playroom **+0.20** (số +0.49 trong bảng là do baseline dính một view hỏng — xem 4.4).
- **Vì sao gain metric chuẩn bị chặn**: camera *test* cũng lệch pose sub-pixel như camera train (đo được, là thuộc tính
  dữ liệu), nên model càng sắc càng bị PSNR phạt. CamP (Zip-NeRF) được +0.2…+0.6 dB cùng cơ chế nhưng phải tối ưu lại
  camera test bằng ảnh test (protocol BARF) — protocol 3DGS chuẩn không cho phép. Đo theo protocol đó (phase-aligned),
  mình +0.27 dB trung bình (12/13 scene).
- **Seed 2 — xong (01:50 UTC), tất cả card đã nhả, không có job nào chạy.** stump 27.97 vs 27.95 (MCMC r1/r2 27.69/27.65) → **+0.27…+0.32**; bicycle 26.31 vs 26.33 (MCMC 26.13/26.165) → **+0.15…+0.20**; playroom trên 25 view sạch 30.62/30.63 vs MCMC 30.43/30.36 → **+0.20…+0.26** (số all-view vô nghĩa vì view hỏng, 4.4). Ba gain đầu bảng đều tái lập ở seed 2 trong ±0.02 dB.

---

## 1. Hạ tầng và protocol

- Server H200: `~/projects/3dgs-refiner-it`; **chỉ dùng card 2,3,4,7** (0,1,5,6 giao cho dự án khác từ 28/09 11:25 UTC).
- Ba venv py3.8 / torch 2.1.2+cu121 (`.venv_ibgs`, `.venv_3dgs`, `.venv_mcmc`), CUDA 12.1 + gcc 11 từ conda.
- Dữ liệu: bản public chính thức (Mip-NeRF 360, Tanks&Temples, Deep Blending), kiểm tra checksum trước khi dùng lại.
- **Protocol R** (đóng băng, `protocol/protocol_R.md`): 30k iter, train/test 1/8, độ phân giải như tác giả, không train
  trên test, không chọn tham số bằng test. Phân biệt **raw** (rasterize) và **final** (sau nhánh warp/aggregation).

### Baseline tái hiện (Protocol R)

| | Mip-360 (9) | T&T (2) | DB (2) | Ghi chú |
|---|---|---|---|---|
| 3DGS (mình chạy) | khớp tác giả | | | rerun độc lập trùng số |
| 3DGS-MCMC (mình chạy) | 28.30 / 0.846 / 0.175 | 24.46 | 29.77 | playroom cần `opacity_reg 0.001` (bug config) |
| IBGS raw (ckpt tác giả) | 27.35 | 23.13 | 29.83 | kém MCMC 1 dB |
| IBGS final (ckpt tác giả) | 28.53 / 0.841 / 0.181 | 24.89 | 29.92 | re-render khớp 1e-4 dB |
| GADA final (số công bố) | 28.63 | 24.93 | — | không có code |

MCMC thuần **thắng GADA final ở cả 5 scene outdoor** ngay từ đầu — SOTA IBR chỉ mạnh indoor.

---

## 2. Chuỗi chẩn đoán (E-chain) — cái gì làm 3DGS mờ, và vì sao IBR không cứu

| Bước | Câu hỏi | Kết quả | Kết luận |
|---|---|---|---|
| E4 | Gain của nhánh warp nằm băng tần nào? | Outdoor: 91–96% gain ở tần thấp, HF ≈ 0; affine màu 12 tham số/view ≥ final | Ở outdoor warp **không chuyển texture** |
| E1/E3 | Lỗi raw là lệch hình học? | Lệch raw↔GT 0.1–0.4 px; phần trơn/view 0.04–0.15 px | Lỗi là **mất HF**, không phải pose cứng |
| E5 | Render mờ bao nhiêu? | σ_eff 0.38–0.71 px | Mọi render = GT mờ |
| **E7** | Mờ do năng lực hay do dữ liệu? (train lại trên GT nhân tạo nhất quán, tiêm nhiễu pose) | σ_eff **0.15 px** khi nhất quán → 0.39 ở σ_pose=0.6 px; thật 0.44. **−0.7 dB mỗi 0.1 px lệch** | **Do dữ liệu không nhất quán dưới pixel**, không do năng lực |
| E1b/E1c | Warp nguồn tại test có tốt không? | Outdoor: warp kém raw 2–2.6 dB dù lệch chỉ 0.22 px; căn flow/sửa màu không cứu | Warp outdoor **tệ hơn ảnh mờ** |
| **E8** | Warp trong thế giới nhân tạo tĩnh, depth đúng | Vẫn chỉ 27.5 / 29.6 dB (raw 39 / 44) | Sàn lỗi **nội tại** của warp |
| **E10** | Chỉ resample nửa pixel tốn bao nhiêu? | trên render tổng hợp: bilinear 28.65 / 28.48 / 36.0 dB; **trên ảnh thật (B1, §8.1): 37 / 39.7 / 45 dB** | Sàn warp = sàn lấy mẫu **chỉ đúng với render alias**; với ảnh thật nút thắt là lệch hình học |
| D1 | Nhánh residual làm base "lười"? MCMC densify cứu IBGS? | noagg ≈ raw (H2 bác bỏ); MCMC densify trong IBGS +0.23 final (garden/train vượt GADA) | densify giúp; loss hình học IBGS hại |
| N1/N3 | Sàn nhiễu ảnh & ngân sách photometric/view | nhiễu ảnh không giải thích được gap | Vấn đề là pha, không phải nhiễu |
| Tensara post-hoc | Chạy thử refiner của report | bicycle −0.65, garden −0.90 | Không dùng |

**Câu chuyện**: cả độ mờ của 3DGS lẫn thất bại của IBR ở outdoor đều là **pha dưới pixel** — ở phía dữ liệu (camera)
và ở phía toán tử (resample). Hướng đúng là sửa dữ liệu lúc train và bỏ nhánh warp lúc test.

---

## 3. Method — "Phase-consistent Gaussian Splatting"

`src/phase/train_mcmc_phase.py` (copy của MCMC train.py + `PhaseWarp`):

- Mỗi view train có bảng hệ số đa thức bậc 3 (20 hệ số × 2), khởi tạo 0, học cùng Gaussians (Adam, lr 1e-3, bắt đầu
  iter 1000, L2 reg). Render được warp bởi field này (bicubic) **trước** khi tính loss ⇒ Gaussians hội tụ về một
  hình học nhất quán, chính ảnh train được "kéo về pha chung".
- **Gauge**: 1/8 view (anchor) giữ field = 0 → scene không trôi khỏi hệ camera gốc. Zero-mean 2D là sai về nguyên lý
  (xoá luôn thành phần intrinsics thật) — đã đo, bỏ.
- **Field chung** (`--phase_shared`): một bảng hệ số áp cho mọi view kể cả anchor; đây là lỗi **intrinsics** của camera
  dùng chung, nên hợp lệ áp cho camera test (`[sh]`).
- Tuỳ chọn: cap độ lớn field/view (`--phase_cap`, chống view lệch pose lớn), field affine-only, colour affine (M3),
  loss dung sai dịch (M2).
- Chi phí: train +≈0 (một grid_sample/iter), inference = 3DGS thuần (không ảnh nguồn, không mạng).

### Field học được là gì (đo không dùng GT — `gauge3d_fit.py`, `gauge3d_stats.py`, `shared_field_decompose.py`)

| | Kết quả | Ý nghĩa |
|---|---|---|
| Per-view part | 72–97% năng lượng field = **một pose correction 6-DoF/view** (kitchen 95%, flowers 97%, bicycle 77%) | Mình đang làm pose refinement mà không cần gradient pose |
| Độ lớn | quay 0.001–0.006°, dịch 1e-5…1e-4 khoảng cách scene → 0.05–0.2 px pha | Sai số COLMAP nhỏ hơn mọi thang đo pose nhưng đáng +0.1…+0.5 dB |
| Shared part | flowers 98% affine, principal point lệch ≈1 px @1/4 res; kitchen 71%, bicycle 65%, playroom 41% affine | **Intrinsics** của camera dùng chung; không có field chung, flowers −1.49, room −2.75 |
| Gauge 3D | sim3 chung trong per-view part chỉ 2.4% (kitchen); áp sim3 lên Gaussians: không đổi | Anchor **giữ được** gauge; resample không phải nút thắt (render "exact" = resample tới 0.01 dB) |
| Phần dư | garden/truck/train còn 0.06–0.08 px không phải pose | rolling shutter / distortion / lệch cục bộ |

---

## 4. Kết quả

### 4.1 Bảng chính — `m1w [sh]` trên 13 scene (1 seed; PSNR / SSIM / LPIPS)

| scene | MCMC | **ours (one-stage)** | Δ MCMC | IBGS final | GADA final | Δ GADA |
|---|---|---|---|---|---|---|
| bicycle | 26.13 / 0.812 / 0.161 | **26.33 / 0.822 / 0.153** | +0.20 | 26.06 | 26.16 | **+0.17** |
| flowers | 22.41 / 0.658 / 0.286 | **22.39 / 0.660 / 0.279** | −0.02 | 22.34 | 22.29 | **+0.10** |
| garden | 28.19 / 0.885 / 0.090 | **28.29 / 0.888 / 0.084** | +0.10 | 27.57 | 27.74 | **+0.55** |
| stump | 27.69 / 0.822 / 0.164 | **27.96 / 0.835 / 0.155** | **+0.27** | 27.31 | 27.33 | **+0.63** |
| treehill | 23.33 / 0.678 / 0.267 | **23.45 / 0.686 / 0.262** | +0.12 | 23.06 | 23.16 | **+0.29** |
| bonsai | 32.78 / 0.953 / 0.161 | 32.72 / 0.953 / 0.158 | −0.07 | 34.98 | 35.37 | −2.65 |
| counter | 29.43 / 0.924 / 0.164 | 29.39 / 0.923 / 0.163 | −0.05 | 30.65 | 30.84 | −1.45 |
| kitchen | 32.21 / 0.939 / 0.107 | 32.03 / 0.937 / 0.107 | −0.18 | 32.10 | 32.09 | −0.06 |
| room | 32.48 / 0.938 / 0.171 | 32.42 / 0.937 / 0.171 | −0.06 | 32.68 | 32.67 | −0.25 |
| **Mip-360 (9)** | 28.30 / 0.846 / 0.175 | **28.33 / 0.849 / 0.170** | +0.03 | 28.53 / 0.841 / 0.181 | 28.63 | −0.30 |
| train | 22.61 / 0.841 / 0.182 | **22.69 / 0.841 / 0.178** | +0.08 | 23.69 | 23.67 | −0.98 |
| truck | 26.31 / 0.901 / 0.104 | **26.38 / 0.901 / 0.096** | +0.07 | 26.10 | 26.19 | **+0.19** |
| **T&T (2)** | 24.46 / 0.871 / 0.143 | **24.53 / 0.871 / 0.137** | +0.07 | 24.89 | 24.93 | −0.40 |
| drjohnson | 29.50 / 0.903 / 0.235 | 29.42 / 0.904 / 0.231 | −0.08 | 29.51 | — | — |
| playroom | 30.03 / 0.909 / 0.229 | **30.52 / 0.912 / 0.226** | **+0.49** | 30.34 | — | — |
| **DB (2)** | 29.77 / 0.906 / 0.232 | **29.97 / 0.908 / 0.228** | +0.20 | 29.92 | — | — |

Đọc: outdoor 5/5 lên, thắng GADA 6 scene; indoor xuống nhẹ và thua GADA xa vì gain indoor của IBR là colour
aggregation nhiều nguồn — thứ one-stage không có (và cũng không nhắm tới). Nhiễu run-to-run ≈ 0.035 dB (MCMC bicycle
r1/r2) ở Mip-360; **playroom là ngoại lệ (4.4)**: +0.49 thực chất là +0.20. +0.07…+0.12 cần seed 2 để khẳng định.

### 4.2 Ablation các biến thể (Δ PSNR trung bình so với MCMC; chi tiết `ablation_variants.md`)

| biến thể | cấu hình | Mip-360 | T&T | DB | ghi chú |
|---|---|---|---|---|---|
| m1 | reg 1e-4, không gauge | outdoor lên, indoor xuống | | | trôi gauge; phase-aligned +0.27 (12/13 scene tốt hơn) |
| m1s | reg 1e-3 + zero-mean 2D | −0.04 (mean) | | | zero-mean xoá intrinsics — sai nguyên lý |
| **m1n** | reg 1e-3 + anchor 1/8 | +0.01 | +0.03 | +0.04 | thắng GADA 5 outdoor + truck; aligned +0.07 |
| m1n_cap | m1n + cap 0.5 px | kitchen −0.28 → **−0.03** | | | một view lệch lớn (DSCF0931) là thủ phạm |
| m1sh | m1n + shared | bonsai +0.07, kitchen −0.30, playroom −0.24 | | | |
| **m1w** | reg 1e-4 + anchor + shared, `[sh]` lúc test | **+0.03** | **+0.07** | **+0.20** | bảng 4.1; không `[sh]`: Mip-360 −0.6 |
| m1wc | m1w + cap | kitchen −0.24 | | | cap không cứu recipe m1w ở kitchen |
| m1a | affine-only | ≈ m1n | | | bậc 3 cần cho phần dư |
| M2 | loss dung sai dịch | âm | | | bác bỏ |
| M3 | colour affine/view | ≈0 / âm | | | trôi gauge màu; không dùng |

### 4.3 Phase-aligned PSNR (protocol CamP/BARF cho 3DGS) — vì sao metric chuẩn chặn gain

`scripts/analysis/phase_aligned_psnr.py`: fit field trơn bậc 3 (RAFT) giữa render và ảnh test rồi chấm — tương đương
"tối ưu lại camera test" của CamP. **Chỉ báo cạnh metric chuẩn, không thay thế.**

- m1 (không gauge): aligned **+0.27** trung bình, tốt hơn MCMC 12/13 scene, trong khi standard −0.05.
- m1n: aligned +0.07; m1w flowers: standard −0.02 nhưng aligned **+0.42**; playroom aligned +0.53.
- N2b: **view test nào lệch, lệch bao nhiêu là thuộc tính dữ liệu** (corr 0.8–0.95 giữa MCMC và các model pha; median
  0.1–0.5 px). Model pha sắc hơn 3–4% (HF rms) và **trả giá nhiều hơn cho cùng sai số pose test**: flowers penalty
  +0.38 (MCMC) → +0.82 dB (m1w). PSNR chuẩn với camera test cố định đang "đánh thuế" độ sắc nét.

### 4.4 Playroom: MCMC không ổn định — view test hỏng làm số dao động ±0.5 dB

Mọi run MCMC trên playroom (baseline lẫn method) thỉnh thoảng có 1–2 view test bị **một Gaussian mờ màu vàng che kín**
(PSNR 9–19 dB): MCMC r1 view 25 = 9.6 dB, baseline dùng trong bảng (oreg001) view 22 = 23.4, m1n view 14 = 19.3,
m1s/m1sh view 25 = 9.4/15.8, m1w seed 1 sạch, m1w seed 2 view 18/22 = 15.8/15.3. Trên 25 view sạch (loại 4 view từng
hỏng ở bất kỳ run nào):

| run | PSNR tất cả view | PSNR 25 view sạch |
|---|---|---|
| MCMC oreg001 (baseline bảng) | 30.03 | 30.43 |
| MCMC r1 | 29.23 | 30.03 |
| MCMC r2 | 29.96 | 30.36 |
| m1n | 30.17 | 30.57 |
| m1w seed 1 | 30.52 | **30.63** |
| m1w seed 2 | 29.56 | **30.62** |

⇒ gain thật của m1w trên playroom là **+0.20** (hai seed trùng nhau tới 0.01), không phải +0.49; và số "all views" của
scene này không dùng được để so sánh nếu không kiểm tra view hỏng. Lỗi haze là của MCMC (relocation/noise trên scene
này), không phải của field pha; cần ghi chú trong paper hoặc bỏ playroom khỏi claim.

---

## 5. Các thí nghiệm đã đóng (null / bác bỏ) — để không làm lại

| Thí nghiệm | Kết quả | Kết luận |
|---|---|---|
| Tensara post-hoc refiner (port, apply_tile 512) | bicycle −0.65, garden −0.90 | refiner 2D không vượt sàn resample |
| H2 (nhánh residual làm base lười) | noagg ≈ raw ±0.05 | bác bỏ |
| P1 pose 6-DoF từ warp láng giềng | hại cả hai dấu | phép đo qua warp là nhiễu; pose từ render-signal +0.14 |
| M2 loss dung sai dịch | âm | bác bỏ |
| M3 colour affine | ≈0/âm | trôi gauge màu |
| Zero-mean 2D (m1s) | −0.04 | sai nguyên lý (xoá intrinsics) |
| Render "exact" field chung qua ma trận chiếu | = resample tới 0.01 dB | resample không phải nút thắt |
| Áp sim3 fit được lên Gaussians (g3d/g3dn) | không đổi | anchor đã giữ gauge 3D |
| Supersampled application (sh2) | −0.9…−1.8 | bỏ |
| m1wc (cap + shared) kitchen | −0.24 | cap chỉ hiệu quả với m1n |

---

## 6. Định vị so với prior art

- **Joint camera refinement**: BARF, SCNeRF, **CamP** (Zip-NeRF; Mip-360 pose COLMAP: 28.27 → 28.48 (SE3) … 28.86
  (Focal+intrinsics+CamP), nhưng chấm theo BARF: tối ưu camera test bằng ảnh test), Robust-GS (ScanNet++/Deblur-NeRF),
  3R-GS/JOGS (pose nhiễu/MASt3R). **Chưa ai báo gain trên benchmark 3DGS chuẩn với camera test cố định** — mình đo được
  vì sao (§4.3).
- **Khác biệt của mình**: (i) tham số hoá trong không gian ảnh — không cần gradient pose từ rasterizer, dùng với mọi
  renderer, tự phân rã thành intrinsics (shared) + extrinsics (per-view) + phần dư bậc cao; (ii) chuỗi chẩn đoán:
  blur = pha camera (E7), IBR không chuyển được HF vì resample (E8/E10) — giải thích vì sao IBGS/GADA chỉ mạnh indoor;
  (iii) gauge fixing bằng anchor thay cho test-time optimisation; (iv) one-stage inference vượt GADA ở outdoor.
- **Hạn chế thẳng thắn**: gain metric chuẩn nhỏ (+0.0…+0.3/scene, tái lập ở seed 2 cho stump/bicycle/playroom, các scene khác 1 seed); indoor thua IBR xa; playroom cần loại view hỏng (4.4); ngân sách còn lại nằm
  ở pose camera test — không hợp lệ chạm tới trong protocol chuẩn.

---

## 7. Trạng thái và việc tiếp theo

- **Trạng thái (29/09, 01:55 UTC): thí nghiệm GPU tạm dừng theo yêu cầu; không job nào chạy; cả 8 card trống.**
  Chờ quyết định hướng đi mới trước khi chạy thêm.
- Việc còn dở nếu tiếp tục hướng này (không cần GPU nhiều): đo thời gian train m1w vs MCMC; viết bảng 4.1 + 4.3 song
  song, hình σ_eff(σ_pose) (E7), phân rã field → pose/intrinsics (§3), so sánh CamP. Một lần duy nhất: ghép cap vào
  recipe m1w cho indoor (kitchen m1n_cap −0.03 vs m1w −0.18).
- Không làm nữa: mọi thứ trong §5.
- Ba điều rút ra cho việc chọn hướng: (1) trần của "sửa camera train" ≈ +0.3 dB/scene trên protocol chuẩn vì camera test
  cũng sai pose và không được chạm — đẩy thêm là tuning; (2) gap lớn thật (1.5–2.7 dB) ở indoor là colour aggregation
  nhiều nguồn của IBR — one-stage muốn lấy phải học được appearance/LF theo view, không phải hình học; (3) E7 cho thấy
  trần thật của Gaussian cao hơn hiện tại vài dB nếu dữ liệu nhất quán — hoặc làm dữ liệu nhất quán bằng cách khác
  (render "mờ theo pha đúng" thay vì mờ trung bình), hoặc đề xuất protocol/benchmark với camera test được cân chỉnh.

### Vị trí code
`src/phase/train_mcmc_phase.py` (trainer), `scripts/run_p4.sh` / `launch_p6.sh` / `collect_p6.py` (chạy & bảng),
`scripts/analysis/` (E-chain, `phase_aligned_psnr.py`, `gauge3d_*.py`, `shared_field_decompose.py`,
`n2b_test_misalignment.py`), `configs/mcmc/`, `protocol/protocol_R.md`. Log số liệu thô theo thứ tự thời gian:
`reports/week1_diagnostics_notes.md`.

---

## 8. Câu hỏi 30/09: dùng matching kiểu Tensara (SuperPoint + LightGlue) để ràng buộc depth/warp — có khả thi?

**Tensara thực ra dùng matching để làm gì**: match dày ảnh train (SuperPoint 8192 kp + LightGlue + MAGSAC++) → phát
hiện camera model sai (Sampson error tăng từ tâm ra góc, 13–40 px giữa các dải bay) → **chạy lại SfM có tự hiệu chỉnh
distortion**, chuyển pose test sang hệ mới bằng PnP. Matching là công cụ *chẩn đoán + dựng lại SfM*, không phải loss
trong training. Sàn nhiễu của matcher trong report là **2.5 px**; nó dùng được vì lỗi của họ lớn gấp 5–15 lần sàn đó.

**Trên benchmark của mình, dùng làm ràng buộc warp/depth: không khả thi.**
- Lỗi cần sửa: 0.05–0.35 px (sai số tái chiếu COLMAP trung vị 0.23–0.35 px, E6; pose correction học được 0.05–0.2 px).
  Nhiễu của SuperPoint/LightGlue: median epipolar 1.0–1.7 px trên MegaDepth, chỉ 27–33% match đúng trong 1 px
  (HPatches) — **lớn hơn lỗi cần sửa 5–30 lần**. SIFT (thứ COLMAP đang dùng) còn chính xác sub-pixel hơn SuperPoint.
- Warp bị chặn bởi resample chứ không bởi depth: E8 với depth hoàn hảo vẫn chỉ 27.5 dB outdoor. Depth tốt hơn từ
  matching không nâng được phần HF.
- Các paper dùng matching prior cho 3DGS (SCGaussian, MCGS, GeoTrack-GS, TWINGS…) đều ở **sparse-view**, nơi hình
  học thiếu ràng buộc. Dense-view (Mip-360 đủ ảnh) thì loss photometric đã ràng buộc depth rồi.

**Phần có thể chuyển giao là cách chẩn đoán + dựng lại SfM của Tensara**, không phải matcher. Dấu hiệu ở dữ liệu của
mình: mọi scene dùng **một camera PINHOLE, principal point đặt đúng tâm ảnh** (bicycle 2473.0 = 4946/2; COLMAP mặc
định không tinh chỉnh principal point), trong khi field chung học được nói flowers lệch principal point ≈1 px @1/4 res
(≈4 px full res). Đây là "lỗi calibration bị BA hấp thụ", giống Tensara nhưng nhỏ hơn nhiều. Ảnh test nằm chung trong
SfM gốc, nên dựng lại SfM tốt hơn sẽ sửa luôn pose test mà không dùng pixel test để fit model.
- Probe rẻ, chỉ CPU: chẩn đoán Sampson theo bán kính và theo số điểm chung trên COLMAP gốc của Mip-360/T&T/DB. Matcher
  cần sub-pixel: SIFT + featuremetric refinement (Pixel-Perfect SfM), hoặc so pose với field pha học được.
- Rủi ro: đổi pose là đổi benchmark, phải chạy lại mọi baseline trên pose mới và viết rõ trong protocol; lợi ích có thể
  nhỏ nếu Sampson không có profile theo bán kính.

### 8.1 Làm theo tư duy Tensara trên dữ liệu của mình — kết quả (30/09, chỉ CPU)

**C1 — audit COLMAP gốc từ chính track của nó** (`scripts/analysis/c1_calib_audit.py`, 13 scene): mọi scene một camera
PINHOLE, principal point đúng tâm. Sai số tái chiếu trung vị 0.5–1.2 px ở full res (0.23–0.35 px ở độ phân giải đánh
giá, khớp E6). Residual **tăng từ tâm ra góc** ở các scene outdoor/T&T (flowers 1.07 → 1.42, garden 1.06 → 1.60,
truck 0.56 → 0.90, bonsai 0.64 → 1.12). Trên chính track của nó thì field hệ thống ≈ 0, vì BA đã hấp thụ hết, đúng như
Tensara mô tả.

**C2 — BA lại với camera model khác nhau** (cùng track, tất cả ảnh train+test như bản gốc):
- Chỉ pose: không đổi gì. Model phát hành đã hội tụ.
- Thêm principal point / distortion OPENCV: sai số giảm < 1–6%, principal point trượt 3–40 px dọc theo mode suy biến
  với phép xoay camera. **Không phải lỗi distortion kiểu Tensara.**
- **Focal riêng từng ảnh** (focus breathing): sai số giảm rõ và profile theo bán kính phẳng lại.

**C4 — holdout (điểm 3D không tham gia BA, triangulate lại bằng camera mới)** (`c4_holdout_calib.py`):

| scene | shared camera (bản phát hành) | **focal riêng từng ảnh** | Δ |
|---|---|---|---|
| flowers | 1.159 px | **0.986** | −15% (focal + pp riêng: 0.964, −17%) |
| bicycle | 1.135 | 1.045 | −8% |
| garden | 1.189 | 1.127 | −5% |
| stump | 1.076 | 0.995 | −8% |
| treehill | 1.002 | **0.853** | −15% |
| bonsai | 0.726 | 0.688 | −5% |
| counter | 0.584 | 0.582 | 0% |
| kitchen | 0.530 | 0.514 | −3% |
| room | 0.675 | 0.670 | −1% |
| train | 0.669 | **0.577** | −14% |
| truck | 0.671 | 0.630 | −6% (focal + pp riêng: **0.514, −23%**) |
| drjohnson | 0.527 | **0.412** | −22% |
| playroom | 0.493 | 0.473 | −4% |

Principal point *chung* không giúp; principal point *riêng từng ảnh* giúp thêm mạnh ở truck (video — nghi chống rung /
crop). 11/13 scene cải thiện trên điểm holdout, cả 3 dataset. EXIF đã bị xoá khỏi mọi ảnh phát hành nên không xác minh
trực tiếp nguyên nhân vật lý (focus breathing, chống rung, zoom không khoá). Pipeline chuẩn `convert.py` của 3DGS/MCMC
chạy COLMAP với `--ImageReader.single_camera 1`, tức là ép một camera chung — một lựa chọn mặc định của cộng đồng.

Focal lệch giữa các ảnh std 0.13–0.22% (range 0.6–1.3%) ≈ 1.3–3.7 px ở mép ảnh full res ≈ **0.3–0.9 px ở độ phân giải
train** — cùng bậc với ngân sách sub-pixel của cả dự án. Đây là một lỗi camera model thật (generalise trên điểm
holdout), bị BA của bản phát hành hấp thụ vào pose, và nó ảnh hưởng cả camera **test** (test nằm trong SfM).
Tương quan với field học được chỉ yếu (+0.1…+0.2 kitchen/truck, sau khi căn theo điểm 3D) — field học được nhỏ hơn
(0.06–0.16 px) và phần lớn bù pose; phép thử quyết định là train lại trên calibration mới (cần GPU).

**B1 — sửa một kết luận cũ về warp** (`b1_supersampled_resampling.py`, ảnh thật, không model): resample 0.5 px bằng
bicubic trên **ảnh chụp thật** ở độ phân giải benchmark chỉ mất tới mức 37 dB (bicycle), 39.7 (garden), 45 (bonsai);
sàn 28.5 dB của E10 là do render 3DGS lấy mẫu điểm bị alias, không đúng cho ảnh chụp. Lệch vị trí không bù mới tốn:
0.25 px → 34–39 dB, 0.5 px → 28–33 dB. ⇒ Với ảnh thật, thứ giết warp là **lệch hình học/pose sub-pixel** và
visibility/appearance, không phải resample. Nguồn oversampled (full-res, 4×) xoá hẳn phần resample (54–66 dB) nhưng
phần đó vốn không phải nút thắt. Mọi thứ quy về cùng một biến: **độ chính xác hình học sub-pixel của camera**.

### 8.2 Train lại trên calibration đã sửa (focal riêng từng ảnh) — lượt 1 (30/09, card 0,1,5,6)

MCMC thuần, cùng config/cap_max, cùng tập ảnh test; **chỉ đổi calibration** (`data/calib/<scene>_pf`, dựng bằng
`scripts/analysis/c5_make_pf.py`: BA trên track của bản phát hành, một focal/ảnh, ảnh lệch >3% đưa về trung vị).

| scene | MCMC (calib phát hành) | **MCMC (calib sửa)** | Δ PSNR | SSIM | LPIPS | so sánh |
|---|---|---|---|---|---|---|
| flowers | 22.41 | **23.01** | **+0.60** | 0.658 → 0.693 | 0.286 → 0.269 | GADA 22.29, IBGS 22.34, m1w 22.39 |
| bicycle | 26.13 / 26.17 (r1/r2) | **26.57** | **+0.40…+0.44** | 0.812 → 0.831 | 0.161 → 0.154 | GADA 26.16, m1w 26.33 |
| truck | 26.31 | **26.58** | +0.27 | 0.901 → 0.903 | 0.104 → **0.110** | GADA 26.19, m1w 26.38 |
| drjohnson | 29.50 | **30.22** | **+0.71** | 0.903 → 0.918 | 0.235 → 0.220 | IBGS 29.51; 28/33 view tốt lên, median +0.66 |

Đọc: chỉ sửa camera model (không đổi method) cho +0.3…+0.7 dB — lớn hơn mọi thứ tuần này (field pha: +0.0…+0.3), và
vượt IBGS/GADA final (IBR hai stage) bằng MCMC một stage thuần. Khớp với dự đoán từ phase-aligned (§4.3: camera test
lệch làm mất ~0.4–0.8 dB). Truck LPIPS xấu đi nhẹ — cần xem. Lưu ý so sánh công bằng: IBGS/GADA là số trên calib phát
hành; muốn claim phải chạy lại các baseline trên calib sửa (và ghi rõ protocol: calib dùng keypoint của mọi ảnh như
SfM gốc, không dùng render).

Patch principal point riêng từng ảnh đã kiểm trên GPU: dịch cx +8 px @COLMAP → ảnh dịch 4.001 px (kỳ vọng 4.002).

**Tách nguồn gốc gain (D2, `d2_refit_test_cams.py` + `src/phase/render_cams_json.py`)**: chỉ fit lại camera test từ
keypoint của chúng, scene và camera train cố định. Đối chứng (fit lại với focal chung trên model cũ) giữ nguyên PSNR
(22.41 / 26.131 / 29.506) → quy trình trung tính.

| scene | model cũ | cũ + camera test focal riêng | mới + camera test focal chung | **mới (đầy đủ)** |
|---|---|---|---|---|
| flowers | 22.41 | 22.49 (+0.08) | 22.66 (+0.25) | **23.01 (+0.60)** |
| bicycle | 26.13 | 26.21 (+0.08) | 26.30 (+0.17) | **26.57 (+0.44)** |
| drjohnson | 29.50 | 29.64 (+0.14) | 29.81 (+0.30) | **30.22 (+0.71)** |

Phía train (model sắc hơn): +0.17…+0.30; phía test trên model cũ: +0.08…+0.14; **cộng hưởng**: model sắc cần camera
test đúng (mất 0.27–0.41 nếu test dùng focal chung) — đúng như N2b. (Đổi chéo nguyên bộ camera giữa hai calibration thì
vô nghĩa: focal và pose bù trừ nhau, lệch > 0.5 px — 19.8 / 18.4 dB.)

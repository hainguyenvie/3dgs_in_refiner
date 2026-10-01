# Week 2 — reference routing → phân xử theo evidence (theo băng tần) giữa Gaussians và ảnh thật

> File báo cáo **duy nhất** của tuần 2 (bám `plan/week2_plan.md`). Số liệu cập nhật dần; mục nào chưa chạy ghi ⏳.
> Server mới (1×H200). Dữ liệu tải lại từ nguồn chính thức (360_v2, 360_extra_scenes, tandt_db — khớp byte), checkpoint
> IBGS tải lại từ link Drive gốc của tác giả, MCMC train lại (khớp tuần 1: train 22.73 vs 22.61, bonsai 32.84 vs 32.78).

## 0. Tóm tắt (cập nhật liên tục)

1. **R0 — routing *tập nguồn* có ít dư địa (gate R0 của plan: dừng router lớn).** Cùng Gaussians/warper/mạng IBGS, chỉ
   đổi tập nguồn: oracle cấp ảnh chỉ **+0.12–0.13 dB** (bonsai, counter); coverage-K **kém** nearest-K. Ngoại lệ T&T train:
   oracle +1.1–1.4 dB, nhưng phần lớn là "đoán đúng phơi sáng của ảnh test" (camera quay vòng → nguồn gần về không gian
   nhưng xa 120–300 frame về thời gian, khác phơi sáng) — thông tin phụ thuộc GT; phần lấy được hợp lệ chỉ là bỏ bớt
   nguồn (nearest-2: +0.21). Rule "chọn nguồn khớp màu với render Gaussian" **hại** (−2.6 dB).
2. **Đòn bẩy thật: phân xử explicit ↔ image-based theo vùng *và theo tần số*.** MCMC thắng IBGS ở 5/5 outdoor + truck +
   kitchen; IBGS thắng indoor/train. Oracle theo pixel giữa hai bên: +2.0 dB (bonsai), +1.6 (counter), +2.1 (train).
3. **Intuition (mô hình lệch đăng ký):** ảnh IBR ≈ T(x+δ) với δ lệch sub-pixel (pose/calib/depth) ⇒ phổ sai số
   S_E(ω) ≈ 2|T̂(ω)|²(1−e^{−ω²σ²/2}): ~0 ở tần thấp, ≈2|T̂|² ở tần cao (tệ hơn đoán 0). Gaussians thì ngược lại: sai ở
   phần phụ thuộc góc nhìn/phơi sáng (tần thấp–trung). Lấy trung bình nhiều nguồn chỉ giảm phương sai, không cứu tần cao
   ⇒ *chọn nguồn nào* ít quan trọng (giải thích R0), *tin evidence đến tần số nào, ở vùng nào* mới quan trọng. Trọng số
   tối ưu kiểu Wiener w_E(ω) = S_I/(S_I+S_E); vùng không có nguồn (u0) ⇒ w_E = 0 (mở rộng L2R: support không chỉ quyết
   định *có tin* external evidence không mà *tin đến băng tần nào*).
4. **Kết quả LOSO đầu tiên (3 scene, gate không thấy GT của scene được chấm):** gate theo băng tần thắng IBGS
   +0.38–0.43 và GADA +0.17–0.55 trên counter/train, hoà GADA ở bonsai. Trọng số học được tự tái hiện intuition: tỉ trọng
   MCMC cao nhất ở băng tần mịn nhất (0.37–0.41) và giảm dần ở băng thô (§3.3).
5. **Phát hiện phụ:** mạng residual của IBGS (train cho base của IBGS) đặt lên render MCMC — *không train lại* — đã hơn IBGS
   (bonsai 35.38 vs 34.92, counter 30.87 vs 30.63, train 23.94 vs 23.79): residual nhận "warp − render" nên sửa lỗi của
   *bất kỳ* base nào; base mạnh hơn ⇒ kết quả tốt hơn. Nhưng warp bằng geometry MCMC thì kém (bonsai 34.42; 12% pixel không
   qua depth test vs 3% với geometry IBGS): **chất lượng evidence do geometry quyết định, chất lượng màu do base quyết định.**

## 1. Setup

| Thứ | Giá trị |
|---|---|
| Base explicit | 3DGS-MCMC, config tác giả, SfM init, calib phát hành, 30k it |
| IBR | IBGS (NeurIPS'25) checkpoint tác giả; warper trong kernel (per pixel: tối đa 5 nguồn, giữ nguồn hợp lệ đầu tiên theo thứ tự, depth test tương đối 0.01), mạng aggregation mean-pool + CNN residual |
| Split / res | LLFF 1/8; Mip outdoor r4, indoor r2; T&T và DB độ phân giải gốc (IBGS dù cờ `-r 2` vẫn đánh giá T&T ở 980×545 — đã kiểm GT khớp 1e-15) |
| Metric | PSNR/SSIM/LPIPS(vgg) trên ảnh 8-bit, trung bình theo view (như `metrics.py`) |
| Giao thức gate | **LOSO**: train trên view test của các scene *khác*, chấm trên scene giữ lại; không GT nào của scene được chấm được dùng (không chọn checkpoint/ngưỡng trên nó) |

## 2. R0 — có cơ hội cho routing tập nguồn không?

Probe `src/route/r0_probe.py`: cùng Gaussians, cùng warper, cùng mạng; pool 8 ứng viên theo luật IBGS; render **mọi**
tập con kích thước 1–3 (+ nearest-5, IBGS mặc định, K=0). PSNR trung bình theo view (vùng crop bội 32):

| scene | IBGS mặc định | nearest-2 | coverage-3 | oracle ảnh K≤3 | oracle patch (chọn bằng SSIM, chấm MSE) | oracle patch (chọn = chấm, lạc quan) |
|---|---|---|---|---|---|---|
| bonsai | 34.92 | +0.00 | −0.16 | **+0.12** | +0.31 | +0.83 |
| counter | 30.63 | −0.05 | −0.09 | **+0.12** | +0.57 | +1.04 |
| train | 23.79 | +0.21 | +0.01 | +1.11 | +1.33 | +2.45 |

- random-K kém xa (−3.6…−7.7 dB): đưa nguồn tồi vào mạng residual còn **kém hơn raw** — mạng tin warp mù quáng.
- Train: nguồn oracle thường là frame kề thời gian (|Δt|=1); pool mặc định chứa frame quay lại (Δt≈120–300) khác phơi sáng;
  rule hợp lệ dựa vào độ khớp màu với render đều thua (render Gaussian mang màu "trung bình", không phải phơi sáng của target).
- ⇒ Theo bảng quyết định §6 của plan ("oracle gần best fixed-K → dừng router lớn"): không xây router chọn nguồn; chuyển
  ngân sách sang hành động K=0/fallback và phân xử nguồn render (R1 "rule có stop" mở rộng thành gate học được).
- (Outdoor R0 tạm dừng để nhường GPU; sẽ chạy lại bicycle/garden/stump cho đủ bảng.)

## 3. Phân xử explicit ↔ image-based theo evidence

### 3.1 Ứng viên và evidence (mỗi pixel, có sẵn lúc test)
Ứng viên: MCMC raw `I`, IBGS final `E`, residual IBGS trên MCMC `I+r`. Evidence: số warp hợp lệ (0–5), biên depth test,
|warp − render|, độ phân tán giữa các warp, |residual|, |MCMC − IBGS-base| (bất đồng giữa hai mô hình Gaussian độc lập),
log-depth. (`src/route/dump_hybrid.py`)

### 3.2 Bản closed-form (Wiener theo băng tần, không học) — `src/route/wiener_band.py`
Train: chỉ gate theo support (c=0: dùng E nơi có nguồn, I nơi không) +0.11 dB so với IBGS; ước lượng S_E từ phân tán
giữa các warp (c>0) **làm giảm** — giả định sai số độc lập không khớp với output của mạng IBGS. ⇒ cần gate học được.

### 3.3 Gate học được (CNN nhỏ, 5 tầng dilated, softmax trên ứng viên) — `src/route/gate_loso.py`
Chế độ `band`: pyramid Laplacian 5 tầng, mỗi tầng một bộ trọng số (tái tạo chính xác, kiểm 6e-8).

**LOSO, 3 scene** (PSNR/SSIM/LPIPS):

| scene | MCMC | IBGS final | I+r | **gate (band)** | GADA (paper) |
|---|---|---|---|---|---|
| bonsai | 32.84/0.953/0.212 | 34.92/0.957/0.194 | 35.38/0.960/0.185 | 35.32/0.961/0.191 | 35.37 |
| counter | 29.48/0.924/0.219 | 30.63/0.926/0.199 | 30.87/0.929/0.190 | **31.01/0.933/0.192** | 30.84 |
| train | 22.73/0.838/0.221 | 23.79/0.844/0.208 | 23.94/0.853/0.194 | **24.22/0.860/0.191** | 23.67 |

(LPIPS ở đây là `lpips` vgg của gói pip trên ảnh full — cột IBGS/GADA paper dùng LPIPS của chính họ; chỉ so PSNR với paper.)

Trọng số trung bình theo tầng (tầng 0 = mịn nhất → 4 = thô nhất), phần dành cho **MCMC**:
bonsai 0.37/0.24/0.25/0.01/0.33 · counter 0.37/0.14/0.12/0.06/0.12 · train 0.41/0.17/0.13/0.09/0.05 — tần cao tin Gaussians,
tần thấp tin ảnh thật, đúng dự đoán §0.3.

Theo lớp support (PSNR gộp; u0 = không nguồn nào thấy pixel):

| scene | lớp (tỉ lệ) | MCMC | IBGS | I+r | gate |
|---|---|---|---|---|---|
| counter | u0 (4.4%) | 24.65 | 24.19 | 24.91 | **25.21** |
| | 1–2 nguồn (22.5%) | 28.48 | 29.13 | 29.43 | **29.52** |
| | ≥3 nguồn (73%) | 29.83 | 31.32 | 31.37 | **31.50** |
| train | u0 (9.0%) | 18.59 | 18.50 | 18.98 | **19.40** |
| | 1–2 nguồn (30%) | 20.76 | 21.76 | 21.72 | **22.03** |
| | ≥3 nguồn (61%) | 23.36 | 24.74 | 24.77 | **24.93** |

Ở u0, IBGS kém cả MCMC (residual "bịa" khi không có evidence) — đúng khung L2R: không có external evidence thì phải tin
mô hình nội tại.

## 4. Kết quả chính ⏳ (chờ MCMC 13 scene → LOSO đủ 13 scene, và leave-one-dataset-out)

## 5. Việc tiếp
- Chạy đủ 13 scene; leave-one-dataset-out (train Mip → test T&T/DB và ngược lại).
- Ablation: pixel vs band; bỏ từng nhóm evidence (support, disagreement); bỏ ứng viên I+r / E.
- Chi phí: FPS (MCMC + IBGS + mạng + gate), bộ nhớ (2 mô hình Gaussian + ảnh) — một mô hình cần geometry nhất quán.
- Giao thức dev-target sạch (giấu 1/8 view train, train lại MCMC/IBGS) cho bản cuối nếu LOSO giữ được.

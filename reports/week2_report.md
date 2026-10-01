# Week 2 — Tin ảnh thật đến tần số nào? Phân xử Gaussians ↔ IBR theo vùng và theo băng tần

> File báo cáo **duy nhất** của tuần 2, bám `plan/week2_plan.md`. Cập nhật: 01/10/2026 16:30 UTC, đang chạy tiếp.
> ⏳ = đang chạy. Máy: 1×H200 mới. Dữ liệu tải lại từ nguồn chính thức (khớp byte). Checkpoint IBGS lấy từ link
> Drive của tác giả. MCMC train lại, khớp tuần 1 (train 22.73 vs 22.61, bonsai 32.84 vs 32.78, bicycle 26.18 vs 26.13).
> Số IBGS tái tạo bằng `metrics.py` của tác giả trùng với số trong pipeline của mình (bonsai 34.92, counter 30.63,
> train 23.79, bicycle 26.08, garden 27.59). MCMC khớp tuần 1 ở cả garden 28.20 (28.19), stump 27.69 (27.69), kitchen 32.31
> (32.21), truck 26.42 (26.31).

---

## 0. Tóm tắt

| | Kết luận | Bằng chứng |
|---|---|---|
| **Hướng 1 gốc: chọn ảnh nguồn nào** | **Ít dư địa** → dừng theo gate R0 của plan | Oracle chọn tập nguồn trên toàn ảnh chỉ +0.12 dB (bonsai, counter). Coverage-K **kém** nearest-K (§2). |
| **Đòn bẩy thật: tin ảnh thật đến tần số nào, ở vùng nào** | Mức tin ảnh thật phụ thuộc **băng tần × support × scene**. Ở 5/8 scene ảnh thật đúng ở tần thấp, sai ở tần cao (lệch đăng ký sub-pixel). Ở garden/stump ảnh thật thua ở mọi băng; ở truck thua ở băng thô nhất (phơi sáng). **Đúng ở 8/8:** không có nguồn thì ảnh thật thua ở mọi băng. ⇒ cần gate học được, không phải tách băng cố định. | Phổ sai số theo băng tần trên 8 scene (§3.2). |
| **Phương pháp** | Gate học được, trộn MCMC / IBGS / residual-trên-MCMC **theo băng Laplacian**, có điều kiện trên evidence (support, disagreement, biên depth test) | LOSO: gate không thấy GT của scene được chấm. |
| **Kết quả gate (LOSO-8)** | **Thắng GADA ở cả 8/8 scene đã có**, +0.13 … +0.89 dB; thắng IBGS +0.38 … +1.02. Trung bình 6 scene Mip-360 30.49 vs GADA 29.92 (+0.57); T&T 25.61 vs 24.93 (+0.68). ⏳ 13 scene + Shiny zero-shot đang chạy. | §4.1 |
| **Baseline không học (8 scene)** | "Băng mịn nhất từ MCMC, băng thô từ ảnh thật nơi có nguồn" **không thua cả MCMC lẫn IBGS ở scene nào trong 8**; hơn GADA (paper) ở 7/8 (thua ở bonsai: 35.23 vs 35.37). Gain nhỏ ở garden (+0.02 so với MCMC), lớn ở kitchen (+0.63). | §4.2 |

**Intuition một câu:** ảnh warp ≈ ảnh đích bị dịch δ sub-pixel (do pose, calib, depth), nên sai số của nó ở tần số ω tăng
theo ω²σ². Ảnh thật vì thế đáng tin cho màu, phơi sáng và shading phụ thuộc góc nhìn (tần thấp), nhưng phá chi tiết (tần
cao). Lấy trung bình nhiều nguồn không cứu được tần cao, nên *chọn nguồn nào* gần như không quan trọng. Điều quan trọng
là *tin evidence đến băng tần nào*, và điều đó phụ thuộc vào support tại từng vùng.

Đây là chỗ mở rộng L2R-GS. L2R cho thấy external evidence không xếp hạng được vùng không có nguồn. Ở đây support quyết
định không chỉ *có* tin evidence ngoài không, mà còn tin *đến tần số nào*.

---

## 1. Setup

| Thứ | Giá trị |
|---|---|
| Base explicit | 3DGS-MCMC, config tác giả, SfM init, calib phát hành, 30k it |
| IBR | IBGS (NeurIPS'25), checkpoint tác giả. Warper nằm trong kernel: mỗi pixel giữ ≤5 nguồn hợp lệ đầu tiên theo thứ tự, depth test tương đối 0.01. Mạng aggregation dùng mean-pool + CNN residual. |
| Ứng viên cho gate | `I` = MCMC raw; `E` = IBGS final; `I+r` = mạng residual của IBGS đặt lên render MCMC (không train lại) |
| Evidence mỗi pixel (có lúc test) | số warp hợp lệ (0–5), biên depth test, \|warp − render\|, độ phân tán giữa các warp, \|residual\|, \|MCMC − IBGS-base\|, log-depth |
| Split / độ phân giải | LLFF 1/8. Mip outdoor r4, indoor r2. T&T, DB và Shiny theo đúng độ phân giải eval của IBGS (T&T thực ra là 980×545 dù cờ là `-r 2`; đã kiểm GT khớp 1e-15). |
| Metric | PSNR / SSIM / LPIPS(vgg) trên ảnh 8-bit, trung bình theo view |
| Giao thức gate | **LOSO**: train trên view test của các scene *khác*, chấm trên scene giữ lại. ⏳ Thêm 2 giao thức: **LODO** (giữ lại cả dataset) và **Shiny-only** (train gate trên Shiny, zero-shot cho cả 13 scene benchmark, không view benchmark nào được dùng để học). |

---

## 2. R0 — routing *tập nguồn* có đáng làm không?

Probe `src/route/r0_probe.py` giữ nguyên Gaussians, warper và mạng IBGS, chỉ đổi tập nguồn đưa vào kernel. Pool gồm 8
ứng viên theo luật IBGS, và **mọi** tập con kích thước 1–3 đều được render. PSNR trung bình theo view:

| scene | IBGS mặc định | nearest-2 | coverage-3 | oracle ảnh K≤3 | oracle patch (chọn bằng SSIM, chấm MSE) |
|---|---|---|---|---|---|
| bonsai | 34.92 | +0.00 | −0.16 | **+0.12** | +0.31 |
| counter | 30.63 | −0.05 | −0.09 | **+0.12** | +0.57 |
| train | 23.79 | +0.21 | +0.01 | +1.11 | +1.33 |

- **Random-K kém xa** (−3.6…−7.7 dB). Đưa nguồn tồi vào mạng residual còn tệ hơn không dùng warp: mạng tin warp một
  cách mù quáng.
- **Train là ngoại lệ, nhưng không khai thác được.** Camera quay vòng, nên pool chứa các frame quay lại cùng chỗ sau
  120–300 frame, khác phơi sáng. Oracle lớn chủ yếu vì "đoán đúng phơi sáng của ảnh test", một thông tin phụ thuộc GT.
  Rule hợp lệ "chọn nguồn khớp màu với render" còn **hại** (−2.6 dB), vì render Gaussian mang màu trung bình chứ không
  phải màu của target.
- **Quyết định** theo bảng §6 của plan ("oracle gần best fixed-K → dừng router lớn"): không xây router chọn nguồn, dồn
  sang hành động "dừng / K=0", rồi mở rộng thành phân xử theo vùng và theo băng tần.

---

## 3. Intuition và bằng chứng

### 3.1 Mô hình lệch đăng ký
Gọi T là ảnh đích. Có hai nguồn ước lượng:
- **Ảnh IBR:** E = T(x+δ) + sai lệch appearance, với δ ~ N(0, σ²). Khi đó
  E|e^{iωδ} − 1|² = 2(1 − e^{−ω²σ²/2}), nên phổ sai số là S_E(ω) ≈ 2|T̂(ω)|²(1 − e^{−ω²σ²/2}). Nó gần 0 ở tần thấp và
  tiến tới 2|T̂|² ở tần cao, tức là còn tệ hơn đoán bằng 0.
- **Gaussians:** I = T + e_I, với e_I là phần mô hình không biểu diễn được (phụ thuộc góc nhìn, phơi sáng, giới hạn
  dung lượng).

Trộn tối ưu theo từng băng (sai số độc lập) cho trọng số Wiener w_E(ω) = S_I / (S_I + S_E). Có ba dự đoán kiểm chứng được:
1. Lợi thế của ảnh thật giảm dần khi tần số tăng.
2. Tần số giao cắt thấp hơn ở nơi σ·|∇T| lớn (outdoor, lá cây).
3. Nhiều nguồn hợp lệ hơn → phần phương sai nhỏ hơn → tin hơn. Không có nguồn → w_E = 0.

### 3.2 Phổ sai số đo được (`src/route/band_spectrum.py`)
Bảng dưới là tỉ số MSE IBGS / MCMC theo băng Laplacian, từ thô nhất đến mịn nhất. Tỉ số < 1 nghĩa là ảnh thật tốt hơn
Gaussians.

| scene | vùng ≥3 nguồn | vùng 1–2 nguồn | vùng **0 nguồn** |
|---|---|---|---|
| bonsai | 0.18 → 0.47 → 0.61 → 0.75 → **0.95** | 0.53 → … → 1.04 | 2.70 → 1.47 → 1.31 → 1.23 → 1.19 |
| counter | 0.66 → 0.65 → 0.71 → 0.80 → **0.90** | 0.81 → … → 1.00 | 1.14 → … → 1.17 |
| train | 0.68 → 0.69 → 0.80 → 0.82 → **0.88** | 0.70 → … → 0.98 | 0.96 → 1.18 → … → 1.10 |
| bicycle | 0.82 → **1.05 → 1.04 → 1.04 → 1.03** | 0.77 → **1.08 → … → 1.06** | 1.02 → 1.40 → … → 1.12 |
| kitchen | 0.66 → … → **1.17** | 1.26 → … → 1.15 | 2.04 → … → 1.37 |
| garden | **1.05 → 1.03 → 1.09 → 1.11 → 1.13** | 1.07 → … → 1.16 | 1.35 → … → 1.40 |
| stump | **1.20 → 1.06 → 1.10 → 1.10 → 1.09** | 1.05 → … → 1.05 | 1.05 → … → 1.12 |
| truck | **1.27** → 0.97 → 0.95 → 0.91 → 0.96 | 0.90 → 0.99 → 0.99 → 0.97 → 1.05 | 1.52 → … → 1.19 |

Đọc bảng (8 scene) — **dự đoán đúng một phần, phải nói rõ:**
- **Đúng ở 5/8** (bonsai, counter, kitchen, train, bicycle): lợi thế của ảnh thật lớn nhất ở băng thô và mất dần ở băng
  mịn. Riêng 4 scene này cho thấy điều **ngược với cách IBGS/GADA tự giải thích** ("ảnh nguồn đem lại chi tiết tần cao").
- **Garden, stump:** ảnh thật (qua IBGS) thua MCMC ở **mọi** băng → ở đây không băng nào nên tin ảnh thật.
- **Truck:** ngang hoặc hơn ở băng mịn–trung nhưng **thua ở băng thô nhất** (1.27). Nhiều khả năng do hiệu chỉnh phơi sáng
  của IBGS (affine theo nguồn đầu tiên) làm lệch màu tổng thể — scene video có phơi sáng thay đổi.
- **Đúng ở 8/8:** vùng không có nguồn → ảnh thật thua ở mọi băng (1.05–2.70). Mạng residual "bịa" khi không có evidence.
- Nhiều nguồn thì tỉ số thường thấp hơn (bonsai, counter, train), nhưng không đều ở mọi scene (kitchen 1–2 nguồn tệ hơn).
- ⇒ **Hệ quả cho phương pháp:** mức tin ảnh thật thay đổi theo băng × support × scene và không có một quy tắc cố định
  nào đúng cho mọi scene. Đây là lý do cần gate *học được, có điều kiện evidence*; tách băng cố định chỉ là xấp xỉ.

### 3.2b Cái gì quyết định tần số giao cắt? (`src/route/misreg_sigma.py`, chỉ để phân tích — có dùng GT)
Đo trực tiếp độ lệch giữa warp và GT bằng phase correlation sub-pixel (patch 32×32 có texture, ≥95% hợp lệ):

| scene | σ RMS (px, độ phân giải eval) | median \|shift\| | giao cắt đo được (§3.2, vùng ≥3 nguồn) |
|---|---|---|---|
| bonsai (r2) | 1.44 | 0.77 | băng mịn nhất (0.95) |
| counter (r2) | 1.31 | 0.60 | băng mịn nhất (0.90) |
| train (r1) | 0.97 | 0.38 | băng mịn nhất (0.88) |
| bicycle (r4) | 0.86 | 0.32 | ngay sau băng thô nhất |
| garden (r4) | 0.62 | 0.21 | ⏳ |

**σ một mình không dự đoán được giao cắt.** Indoor có σ lớn hơn, nhưng lại giao cắt muộn hơn. Thứ quyết định là **tỉ số
hai phổ sai số** S_I/(S_I+S_E), tức đúng công thức Wiener ban đầu chứ không chỉ vế lệch đăng ký. Ở băng thô nhất (vùng
≥3 nguồn):
- **Indoor:** MCMC sai **gấp 5.5 lần** IBGS ở bonsai (shading theo góc nhìn, phản xạ — thứ ảnh thật mang đúng), nên ảnh
  thật thắng đến tận băng mịn.
- **Outdoor:** MCMC vốn tốt (bicycle chỉ gấp 1.2 lần), nên lệch đăng ký làm ảnh thật thua ngay từ băng thứ hai.

Câu chuyện đúng là "tin ảnh thật theo **tỉ số sai số của hai bên** × băng tần × support", không phải "σ quyết định tần
số cắt".

### 3.3 Gate học được tự tái hiện cấu trúc này
Trọng số trung bình mà gate theo băng (LOSO) dành cho **MCMC**, theo tầng từ mịn nhất đến thô nhất:

| scene | trọng số cho MCMC |
|---|---|
| bonsai | 0.40 / 0.31 / 0.22 / 0.12 / 0.05 |
| counter | 0.42 / 0.20 / 0.16 / 0.11 / 0.05 |
| train | 0.41 / 0.17 / 0.13 / 0.09 / 0.05 |

Không có ràng buộc nào ép gate theo hướng này. Nó tự học được rằng tần cao nên tin Gaussians, tần thấp nên tin ảnh thật.

---

## 4. Kết quả

### 4.1 Gate theo băng tần, LOSO (PSNR / SSIM / LPIPS)

| scene | MCMC | IBGS final | I+r (không train lại) | **gate band, LOSO-3** | **gate band, LOSO-4** | GADA (paper) |
|---|---|---|---|---|---|---|
| bonsai | 32.84 / .953 / .212 | 34.92 / .957 / .194 | 35.38 / .960 / .185 | 35.32 / .961 / .191 | **35.51** / .961 / .191 | 35.37 |
| counter | 29.48 / .924 / .219 | 30.63 / .926 / .199 | 30.87 / .929 / .190 | 31.01 / .933 / .192 | **31.02** / .933 / .193 | 30.84 |
| train | 22.73 / .838 / .221 | 23.79 / .844 / .208 | 23.94 / .853 / .194 | **24.22** / .860 / .191 | **24.22** / .859 / .192 | 23.67 |
| bicycle | 26.18 / .809 / .184 | 26.08 / .803 / .194 | 26.30 / .814 / .171 | — | **26.75** / .826 / .170 | 26.16 |

- LOSO-3 train trên {bonsai, counter, train} trừ scene được chấm. LOSO-4 thêm bicycle vào tập train.
- Gate train **chỉ trên 2 scene indoor Mip-360** vẫn khái quát sang T&T train: 24.22, tức +0.43 so với IBGS, +0.55 so
  với GADA.
- Thêm một scene (bicycle) vào tập train làm bonsai tăng từ 35.32 lên 35.51. Kỳ vọng tiếp tục tốt lên khi đủ 13 scene.
- **Bicycle (outdoor) đạt 26.75 dù gate chỉ học trên 3 scene không có outdoor**: +0.57 so với MCMC, +0.67 so với IBGS,
  +0.59 so với GADA. Ở outdoor, riêng MCMC hay riêng IBGS đều không vượt được GADA; phần trộn theo băng tần thì vượt.
- Cột LPIPS dùng gói `lpips`; chỉ so PSNR với số paper của GADA. Số của GADA là số công bố, không phải mình chạy lại.

Phân rã theo lớp support (LOSO-4, PSNR gộp theo pixel):

| scene | lớp (tỉ lệ) | MCMC | IBGS | I+r | gate |
|---|---|---|---|---|---|
| bonsai | 0 nguồn (3%) | 27.29 | 25.63 | 27.91 | **27.92** |
| | 1–2 (21%) | 30.77 | 32.28 | 32.79 | **32.84** |
| | ≥3 (76%) | 32.36 | 36.39 | 36.34 | **36.59** |
| counter | 0 nguồn (4%) | 24.65 | 24.19 | 24.91 | **25.31** |
| | 1–2 (23%) | 28.48 | 29.13 | 29.43 | **29.56** |
| | ≥3 (73%) | 29.83 | 31.32 | 31.37 | **31.50** |

### 4.1a LOSO-8 (8 scene benchmark đã có; gate train trên 7 scene còn lại)

| scene | MCMC | IBGS final | I+r | **gate band** | GADA (paper) | Δ GADA | Δ IBGS |
|---|---|---|---|---|---|---|---|
| bicycle | 26.18 | 26.08 | 26.30 | **26.79** / .826 / .171 | 26.16 | +0.63 | +0.71 |
| garden | 28.20 | 27.59 | 28.10 | **28.49** / .886 / .112 | 27.74 | +0.75 | +0.90 |
| stump | 27.69 | 27.29 | 27.57 | **28.17** / .837 / .181 | 27.33 | +0.84 | +0.88 |
| bonsai | 32.84 | 34.92 | 35.38 | **35.50** / .962 / .190 | 35.37 | +0.13 | +0.58 |
| counter | 29.48 | 30.63 | 30.87 | **31.01** / .933 / .194 | 30.84 | +0.17 | +0.38 |
| kitchen | 32.31 | 31.96 | 32.73 | **32.98** / .938 / .129 | 32.09 | +0.89 | +1.02 |
| train | 22.73 | 23.79 | 23.94 | **24.18** / .859 / .193 | 23.67 | +0.51 | +0.39 |
| truck | 26.42 | 26.18 | 26.72 | **27.03** / .907 / .126 | 26.19 | +0.84 | +0.85 |
| **TB 6 Mip** | 29.45 | 29.75 | 30.16 | **30.49** | 29.92 | **+0.57** | +0.74 |
| **TB T&T** | 24.58 | 24.99 | 25.33 | **25.61** | 24.93 | **+0.68** | +0.62 |

- Lần chạy đầu của fold garden bị **phân kỳ** (logit nổ, gate chọn 100% IBGS → 27.59). Đã thêm clip gradient + kiểm tra
  "loss train phải thấp hơn ứng viên đơn tốt nhất, không thì train lại với lr nhỏ hơn"; fold garden chạy lại → 28.49.
  Các fold khác không bị ảnh hưởng (gate ≠ một ứng viên đơn). Chuỗi 13 scene dùng bản đã có cơ chế này.
- Ở outdoor, nơi riêng IBGS thua MCMC, gate vẫn cộng thêm +0.29…+0.61 so với MCMC.

### 4.1b Ablation sớm (LOSO-4)

| biến thể | bonsai | counter | train | bicycle |
|---|---|---|---|---|
| gate band (mặc định) | 35.51 | 31.02 | 24.22 | 26.75 |
| + evidence theo băng (\|E−I\| và độ phân tán warp mỗi tầng = hai vế Wiener) | 35.57 | 31.03 | 24.19 | 26.74 |
| + ứng viên MCMC + affine phơi sáng | 35.43 | 31.03 | 24.21 | 26.67 |

Evidence theo băng và ứng viên affine phơi sáng đều không giúp thêm (±0.08, trong nhiễu): gate đã tự rút được các thông tin
đó từ các ứng viên hiện có.

### 4.2 Baseline không học (`src/route/baselines_band.py`, 8 scene)
Tầng cắt k chọn theo LOSO (ra k = 1 ở mọi scene). "E" là nguồn tần thấp: IBGS final hoặc I+r.

| scene | MCMC | IBGS | I+r | MCMC + affine phơi sáng | tách băng cố định [E=IBGS] | **tách băng theo support [E=IBGS]** | tách băng theo support [E=I+r] | gate (LOSO-4) | GADA (paper) |
|---|---|---|---|---|---|---|---|---|---|
| bicycle | 26.18 | 26.08 | 26.30 | 25.27 | 26.42 | **26.49** | 26.38 | 26.75 | 26.16 |
| garden | 28.20 | 27.59 | 28.10 | 27.90 | 28.05 | 28.22 | **28.25** | ⏳ | 27.74 |
| stump | 27.69 | 27.29 | 27.57 | 27.57 | 27.62 | **27.82** | 27.64 | ⏳ | 27.33 |
| bonsai | 32.84 | 34.92 | 35.38 | 33.04 | 34.97 | 35.23 | 35.27 | 35.51 | 35.37 |
| counter | 29.48 | 30.63 | 30.87 | 29.44 | 30.73 | **30.87** | 30.86 | 31.02 | 30.84 |
| kitchen | 32.31 | 31.96 | 32.73 | 32.23 | 32.29 | 32.66 | **32.94** | ⏳ | 32.09 |
| train | 22.73 | 23.79 | 23.94 | 23.10 | 23.84 | **23.93** | 23.79 | 24.22 | 23.67 |
| truck | 26.42 | 26.18 | 26.72 | 26.30 | 26.45 | 26.64 | **26.68** | ⏳ | 26.19 |

Định nghĩa các baseline:
- **Tách băng cố định:** các băng mịn lấy từ MCMC, các băng thô lấy từ IBGS.
- **Tách băng theo support:** giống trên, nhưng chỉ lấy IBGS ở nơi có ≥1 nguồn hợp lệ.
- **MCMC + affine phơi sáng:** dùng đúng mô hình phơi sáng của IBGS, áp lên MCMC.

Đọc bảng:
- **Gain không chỉ đến từ phơi sáng.** Affine chỉ giúp train (+0.37) và hại bicycle (−0.91).
- **Quy tắc không học (tách theo support) không thua cả MCMC lẫn IBGS ở scene nào trong 8** và hơn GADA (paper) ở 7/8 (bonsai 35.23 < 35.37; counter chỉ hơn 0.03).
  Ở outdoor yếu (garden) gain chỉ +0.02–0.05 so với MCMC; ở kitchen +0.63; ở bicycle thắng cả hai thành phần.
- Nguồn tần thấp tốt nhất đổi theo scene (IBGS ở bicycle/stump/train, I+r ở garden/kitchen/truck) — thêm một lý do để
  gate tự chọn.
- **Gate học được cộng thêm +0.09…+0.29 dB** ở 4 scene đã có LOSO; ⏳ 4 scene còn lại đang chạy.

### 4.3 Phát hiện phụ: base mạnh + residual từ ảnh thật
Mạng residual của IBGS đặt lên render MCMC, không train lại, hơn IBGS ở **cả 8 scene** (bonsai 35.38 vs 34.92, kitchen
32.73 vs 31.96, truck 26.72 vs 26.18), nhưng ở garden/stump vẫn kém MCMC một chút (28.10 vs 28.20; 27.57 vs 27.69). Mạng nhận
"warp − render" làm input nên sửa lỗi của bất kỳ base nào, và base mạnh hơn cho kết quả tốt hơn.

Nhưng warp bằng geometry của MCMC thì kém: bonsai 34.42, với 12% pixel trượt depth test so với 3% khi dùng geometry IBGS.
Nới ngưỡng depth không cứu được (34.16 ở ngưỡng 0.03). ⇒ **Chất lượng evidence do geometry quyết định, chất lượng màu do
base quyết định.** Hệ hiện tại vì vậy dùng 2 mô hình Gaussian. Muốn còn 1 mô hình thì cần Gaussians có geometry nhất quán.

---

## 5. Định vị so với related work (rà soát 01/10)

| Đã có | Khác với mình |
|---|---|
| Deep Blending (Hedman'18): softmax theo pixel giữa render mesh và các warp, có fallback | Không theo băng tần, không dùng evidence support/disagreement |
| BlendedMVS / Baumberg'02: tần thấp từ ảnh chụp, tần cao từ render, bộ lọc cố định | Cố định; mình học được, theo vùng, có điều kiện evidence; và dùng cho NVS |
| HDR+ (Wiener merge theo tile × tần số, fallback về frame tham chiếu) | Gần nhất về cấu trúc. Mình là bản học được, với "frame tham chiếu" là render 3DGS |
| IBGS / GADA: residual cộng thêm; GADA căn warp bằng deformable offset | Không có băng tần; tự nhận ảnh nguồn đem lại tần cao, ngược với đo đạc ở §3.2 |

**Mới (chưa tìm thấy):**
- Gate học được theo băng Laplacian giữa render 3DGS và evidence warp, có điều kiện trên support và disagreement.
- Bằng chứng định lượng (phổ sai số theo lớp support) rằng ảnh nguồn đóng góp chủ yếu ở tần thấp.
- Quy tắc closed-form rút từ mô hình lệch đăng ký, đủ để thắng IBGS.

**Việc cần làm thêm:**
1. ~~Đo σ lệch đăng ký để kiểm dự đoán tần số giao cắt~~ — đã đo (§3.2b): σ một mình không dự đoán được; phải xét tỉ số
   hai phổ sai số.
2. Thử gate chồng lên alignment kiểu GADA để xem hai cơ chế bổ trợ nhau không.
3. Đọc kỹ "image-based view-dependent appearance for 3DGS" (Displays, 2026) — abstract nói có tách tần cao/thấp.

---

## 6. Đang chạy và việc tiếp

- **Tiến độ (16:30 UTC):** MCMC + dump hybrid xong **10/16** (bicycle, bonsai, counter, garden, kitchen, stump, train,
  truck, cd, guitars); room đã train xong, đang dump. Đang train: lab (80%), drjohnson (46%), playroom (34%), flowers;
  treehill còn trong hàng đợi. Ước tính đủ 16/16 sau ~2–3 giờ.
- ⏳ Đang chạy riêng: gate LOSO-8 (8 scene benchmark đã có), ablation thêm ứng viên MCMC + affine phơi sáng (LOSO-4).
- ⏳ Chuỗi đánh giá tự chạy khi đủ dữ liệu (`scripts/after_all_week2.sh`):
  - gate band LOSO 13 scene;
  - Shiny-only zero-shot;
  - LODO;
  - gate theo pixel (kiểu Deep Blending);
  - 6 ablation: bỏ support / bỏ disagreement / bỏ bất đồng giữa hai mô hình / bỏ màu / chỉ {MCMC, IBGS} / chỉ {MCMC, I+r};
  - baseline không học và phổ sai số cho 13 scene;
  - R0 cho các scene outdoor.
- Chi phí cần báo: FPS đo lại khi GPU rảnh; bộ nhớ = 2 mô hình Gaussian + ảnh nguồn + mạng IBGS + gate (~0.3 MB).
- Giao thức dev-target sạch (giấu 1/8 view train, train lại MCMC/IBGS) cho bản cuối, nếu LOSO và Shiny-only giữ được kết quả.
- Nhánh 2 (Difix): mới dựng sẵn env, **chưa chạy**; chỉ mở khi hướng 1 đã chốt.

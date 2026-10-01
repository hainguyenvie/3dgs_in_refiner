# Week 2 — Tin ảnh thật đến tần số nào? Phân xử Gaussians ↔ IBR theo vùng và theo băng tần

> File báo cáo **duy nhất** của tuần 2, bám `plan/week2_plan.md`. Cập nhật: 01/10/2026, đang chạy tiếp.
> ⏳ = đang chạy. Máy: 1×H200 mới. Dữ liệu tải lại từ nguồn chính thức (khớp byte). Checkpoint IBGS lấy từ link
> Drive của tác giả. MCMC train lại, khớp tuần 1 (train 22.73 vs 22.61, bonsai 32.84 vs 32.78, bicycle 26.18 vs 26.13).
> Số IBGS tái tạo bằng `metrics.py` của tác giả trùng với số trong pipeline của mình (bonsai 34.92, counter 30.63,
> train 23.79, bicycle 26.08, garden 27.59).

---

## 0. Tóm tắt

| | Kết luận | Bằng chứng |
|---|---|---|
| **Hướng 1 gốc: chọn ảnh nguồn nào** | **Ít dư địa** → dừng theo gate R0 của plan | Oracle chọn tập nguồn trên toàn ảnh chỉ +0.12 dB (bonsai, counter). Coverage-K **kém** nearest-K (§2). |
| **Đòn bẩy thật: tin ảnh thật đến tần số nào, ở vùng nào** | Lệch đăng ký sub-pixel giữ cho ảnh warp **đúng ở tần thấp, sai ở tần cao**. Gaussians ngược lại. Không có nguồn thì không tin ảnh. | Phổ sai số theo băng tần trên 4 scene (§3.2). |
| **Phương pháp** | Gate học được, trộn MCMC / IBGS / residual-trên-MCMC **theo băng Laplacian**, có điều kiện trên evidence (support, disagreement, biên depth test) | LOSO: gate không thấy GT của scene được chấm. |
| **Kết quả hiện có** | Thắng IBGS ở **mọi** scene đã chạy. Thắng GADA ở bonsai (35.51 vs 35.37), counter (31.02 vs 30.84), train (24.22 vs 23.67) | §4 |
| **Baseline không học, rút thẳng từ lý thuyết** | "Băng mịn nhất từ MCMC, băng thô từ IBGS nơi có nguồn" đã thắng IBGS ở cả 4 scene. Ở bicycle nó thắng cả MCMC lẫn IBGS. | §4.2 |

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

Cả ba dự đoán đều đúng:
- Lợi thế giảm đơn điệu theo tần số.
- Outdoor giao cắt ngay sau băng thô nhất, indoor đến băng mịn nhất mới giao cắt.
- Nhiều nguồn thì tỉ số thấp hơn ở mọi băng. Không có nguồn thì IBGS thua ở **mọi** băng: mạng residual "bịa" ra nội dung.
- Điều này **đảo ngược cách IBGS và GADA tự giải thích** ("ảnh nguồn đem lại chi tiết tần cao").

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
| train | 22.73 / .838 / .221 | 23.79 / .844 / .208 | 23.94 / .853 / .194 | **24.22** / .860 / .191 | ⏳ | 23.67 |
| bicycle | 26.18 | 26.08 | 26.30 | — | ⏳ | 26.16 |

- LOSO-3 train trên {bonsai, counter, train} trừ scene được chấm. LOSO-4 thêm bicycle vào tập train.
- Gate train **chỉ trên 2 scene indoor Mip-360** vẫn khái quát sang T&T train: 24.22, tức +0.43 so với IBGS, +0.55 so
  với GADA.
- Thêm một scene (bicycle) vào tập train làm bonsai tăng từ 35.32 lên 35.51. Kỳ vọng tiếp tục tốt lên khi đủ 13 scene.
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

### 4.2 Baseline không học (`src/route/baselines_band.py`)
Tầng cắt k của các baseline tách băng được chọn theo LOSO; ở cả 4 scene đều ra k = 1.

| scene | MCMC | IBGS | MCMC + affine phơi sáng | tách băng cố định | **tách băng theo support** | gate học được |
|---|---|---|---|---|---|---|
| bonsai | 32.84 | 34.92 | 33.04 | 34.97 | **35.23** | 35.51 |
| counter | 29.48 | 30.63 | 29.44 | 30.73 | **30.87** | 31.02 |
| train | 22.73 | 23.79 | 23.10 | 23.84 | **23.93** | 24.22 (LOSO-3) |
| bicycle | 26.18 | 26.08 | 25.27 | 26.42 | **26.49** | ⏳ |

Định nghĩa các baseline:
- **Tách băng cố định:** các băng mịn lấy từ MCMC, các băng thô lấy từ IBGS.
- **Tách băng theo support:** giống trên, nhưng chỉ lấy IBGS ở nơi có ≥1 nguồn hợp lệ.
- **MCMC + affine phơi sáng:** dùng đúng mô hình phơi sáng của IBGS, áp lên MCMC.

Đọc bảng:
- **Gain không chỉ đến từ phơi sáng.** Affine chỉ giúp train (+0.37) và hại bicycle (−0.91).
- **Quy tắc không học, rút từ lý thuyết, đã thắng IBGS ở cả 4 scene.** Ở bicycle nó thắng cả hai thành phần (26.49 so với
  26.18 và 26.08).
- **Gate học được cộng thêm +0.09…+0.29 dB.**

### 4.3 Phát hiện phụ: base mạnh + residual từ ảnh thật
Mạng residual của IBGS đặt lên render MCMC, không train lại, hơn IBGS ở cả 4 scene (bonsai 35.38 vs 34.92). Mạng nhận
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

**Hai việc cần làm thêm:**
1. Đo σ lệch đăng ký (flow giữa warp và GT) để kiểm dự đoán tần số giao cắt ≈ 1/(2πσ) theo scene và vùng.
2. Thử gate chồng lên alignment kiểu GADA để xem hai cơ chế bổ trợ nhau không.

---

## 6. Đang chạy và việc tiếp

- ⏳ MCMC cho 13 scene + 3 scene Shiny. Đã xong: bicycle, bonsai, counter, garden, stump, train, cd. Đang chạy: kitchen,
  room, guitars; sau đó truck, drjohnson, playroom, flowers, treehill, lab.
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

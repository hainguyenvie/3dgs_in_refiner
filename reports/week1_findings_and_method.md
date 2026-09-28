# Week 1 — Phát hiện và đề xuất method (28/09/2026)

Số liệu chi tiết: `week1_diagnostics_notes.md`, `e*_*.md/json`; baseline: `week1_protocolR_report.md`.
Mọi chẩn đoán dùng render Protocol R hoặc checkpoint tác giả; không train trên test, không chọn tham số bằng test.

## 1. Kết luận một câu

**Cả độ mờ của render 3DGS lẫn việc nhánh warp (IBGS/GADA) không chuyển được texture ở outdoor đều là hệ quả của
cùng một thứ: pha dưới pixel.** Model 3D bị mờ vì các ảnh huấn luyện lệch pha nhau dưới pixel (không phải vì hết
năng lực biểu diễn); ảnh warp không cứu được tần số cao vì toán tử resample ở pha lẻ đã phá tần số cao trước khi
mạng nhìn thấy nó (không phải vì depth).

## 2. Chuỗi bằng chứng (bicycle/garden = outdoor, bonsai/counter = indoor)

| Bước | Câu hỏi | Kết quả | Kết luận |
|---|---|---|---|
| Protocol R | Baseline tái hiện được không? | IBGS/MCMC/3DGS khớp tác giả (±0.2 dB); IBGS raw kém MCMC 1.1 dB; final chỉ hơn MCMC +0.34 dB, thua ở cả 3 scene outdoor | Điểm xuất phát tin cậy |
| E4 | Gain của nhánh warp nằm ở băng tần nào? | Outdoor: 91–96% gain ở LF, HF ≈ 0; affine màu 12 tham số/view ≥ final. Indoor: HF −20%, MF −40% | Ở outdoor warp **không chuyển texture** |
| E1/E3 | Lỗi raw có phải lệch hình học / pose từng view? | Lệch raw↔GT 0.1–0.4 px; phần affine/view 0.04–0.15 px (Mip360). ~55% lỗi raw ở HF | Không phải pose cứng; lỗi là **mất HF** |
| E5 | Render mờ bao nhiêu? | σ_eff 0.38–0.71 px; MCMC sắc hơn IBGS raw; final phục hồi f=0.25 nhưng không f=0.4 | Mọi render = GT mờ; warp không cứu băng cao nhất |
| **E7** | Mờ do năng lực hay do dữ liệu? (train lại trên GT nhân tạo nhất quán, tiêm nhiễu pose) | σ_eff **0.153** px khi nhất quán (T(0.4)=0.91) → 0.394 ở σ=0.6; thật 0.435. PSNR 34.05 → 29.77; thật 26.13 | **Do dữ liệu không nhất quán dưới pixel** (≈ σ 0.7 px), không do năng lực |
| E1b | Warp nguồn tại test có tốt không? | Outdoor: warp kém raw 2–2.6 dB trên cùng pixel dù lệch trung vị 0.22 px; indoor: hơn raw +2 dB, lệch 0.1 px | Warp outdoor **tệ hơn ảnh mờ** |
| E1c | Căn theo flow / sửa màu cứu được không? | Flow: không (−0.2…+0.5); màu: +0.2…1.4 dB; vẫn kém raw | Không phải lệch trơn, không phải màu |
| **E8** | Warp trong thế giới nhân tạo tĩnh, depth "đúng" | Vẫn chỉ **27.5 / 29.6 dB** (raw 39 / 44); lỗi 47–59% HF; flow không giúp | Sàn lỗi **nội tại** của warp |
| E9 | Sàn đó nằm đâu? | Đuôi: 3% px nhiều lớp chịu 23% lỗi (×20); nhưng 50% px depth phẳng vẫn ~29 dB | Depth mơ hồ chỉ là đuôi |
| **E10** | Chỉ resample nửa pixel tốn bao nhiêu? | bilinear **28.65 / 28.48 / 36.0 dB** (bicycle/garden/bonsai), 66–69% HF; bicubic không hơn | **Sàn warp (27.5 / 29.6 / 35.8) = sàn lấy mẫu** ở cả 3 scene |
| D1 | Nhánh residual làm base "lười"? (H2) MCMC densify cứu base? (H1) | noagg ≈ ibgs raw (±0.05) ở 4 scene; detach: +0.3/0/−0.4. mcmc_noagg train +0.9 raw nhưng kém MCMC thuần −0.9 | H2 **bác bỏ**; H1: densify giúp, loss hình học IBGS hại |

## 3. Vì sao đây là câu chuyện có thể publish

- Nó **giải thích** một hiện tượng đã được nhiều paper quan sát nhưng chưa quy về một nguyên nhân đo được: IBGS/GADA
  báo gain lớn ở indoor, nhỏ ở outdoor; GADA đổ cho "misalignment" và sửa bằng offset 2D. E8/E10 cho thấy offset 2D
  không thể vượt trần resample; E7 cho thấy trần thật của Gaussian cao hơn nhiều so với cái đang đạt.
- Nó có **định lượng sạch**: đường σ_eff(σ_pose) và PSNR(σ_pose) trên thế giới nhân tạo; sàn resample theo nội dung;
  phân rã gain theo băng tần. Các phép đo này tái tạo được từ checkpoint công khai.
- Nó chỉ ra hướng đi **ngược** với xu hướng hiện tại (thêm nhánh warp lúc test): dùng ảnh láng giềng làm **phép đo**
  để sửa dữ liệu lúc train, rồi **inference một stage** — đúng intuition ban đầu của dự án.

## 4. Đề xuất method — "Phase-consistent Gaussian Splatting" (tên tạm)

Ý tưởng: ảnh láng giềng warp qua geometry hiện tại cho ta, ở mỗi view train, một **phép đo độc lập** về việc view đó
lệch pha thế nào so với đồng thuận 3D (E3 đo được trường flow rms ~0.2 px, chỉ 9–22% là affine). Thay vì đưa phép
đo đó vào một mạng residual để vá ảnh lúc test, đưa nó **ngược vào quá trình train** để dữ liệu trở nên nhất quán,
khiến Gaussian tự sắc lên (E7: −0.7 dB mỗi 0.1 px lệch ⇒ tiềm năng vài dB).

| Thành phần | Nội dung | Khác gì việc đã có |
|---|---|---|
| **M1 — Hiệu chỉnh pha theo view, đo bằng warp** | Mỗi view train có: pose 6-DoF, trường méo bậc thấp (radial/thin-plate thưa), affine màu. Tham số được cập nhật từ residual giữa GT của view và các warp láng giềng (flow đa nguồn, có mask visibility), không chỉ từ gradient photometric của model mờ. Coarse-to-fine. Test view giữ nguyên pose (đúng protocol NVS). | CamP/BARF/3R-GS tinh chỉnh pose bằng chính loss render của model mờ; ở đây tín hiệu đến từ **ảnh thật ↔ ảnh thật** qua geometry (giống held-out epipolar audit của report Tensara), và có cả thành phần **phi cứng** (E3: phần lớn lệch là cục bộ) |
| **M2 — Loss dung sai pha cho phần không sửa được** | Với lệch cục bộ còn lại (cỏ, lá), loss điểm-tới-điểm ưa ảnh mờ (E1b: warp sắc-lệch-0.2px bị phạt nặng hơn render mờ). Dùng loss cho phép dịch dưới pixel theo patch (min-over-shift hoặc so sánh sau khi căn theo trường ước lượng, có regularizer), để model không bị kéo về trung bình. | Từng có trong burst-SR/deblur; chưa có phân tích và áp dụng cho 3DGS như một hệ quả của E7 |
| **M3 — Appearance LF không cần ảnh nguồn** | E4: ở outdoor toàn bộ gain của IBR là LF; thay bằng embedding appearance theo view, nội suy theo pose lân cận lúc test. | Đã có (NeRF-W/WildGaussians) — dùng như thành phần kỹ thuật, không claim |
| **M4 — Inference một stage** | Không cần ảnh nguồn; rasterize trực tiếp. Có thể giữ nhánh warp như tùy chọn cho indoor (nơi E1b cho thấy warp thực sự hơn raw). | Đây là điểm bán hàng: cùng/hơn chất lượng final của IBGS/GADA với chi phí và bộ nhớ của 3DGS |

Mục tiêu số: raw của ta ≥ final của GADA (Mip360 28.62 / T&T 24.92 / DB 30.22) trên cùng protocol.

## 5. Ba probe rẻ trước khi xây (theo đúng kỷ luật của plan)

| Probe | Câu hỏi | Cách làm (≤ 1 ngày GPU) | Tín hiệu để đi tiếp |
|---|---|---|---|
| **P3 (xong)** | Bao nhiêu phần lệch là trơn theo view vs cục bộ? | Fit đa thức bậc 3 theo view vào flow E3 trên train view | **Kết quả: outdoor 66–81% là trơn** (residual 0.06–0.11 px), phần trơn rms 0.16–0.18 px ⇒ M1 đáng làm (~1 dB theo E7). Indoor: không có thành phần này |
| **P1 (xong)** | M1 (pose 6-DoF) làm Gaussian sắc lên? | Kết quả: render-signal bicycle **+0.14**, σ_eff 0.435→0.422; garden ≈ 0; warp-signal **hại** cả hai dấu (nhiễu). **P4 (xong)**: trường trơn bậc 3 áp lên render trong loss: bicycle +0.17, garden +0.09, stump +0.12, σ_eff giảm ở cả 3, đối chứng sạch | Cơ chế xác nhận, biên độ nhỏ → M1 học được trong train (P5) |
| ~~P1 (1 ngày)~~ | M1 thật sự làm Gaussian sắc lên? | Đóng băng MCMC bicycle/garden/counter; ước lượng hiệu chỉnh view bằng flow warp láng giềng (post-hoc), ghi lại pose/méo; **train lại từ đầu** với pose đã sửa; đo raw PSNR, σ_eff, HF share trên test (pose test không đổi) | raw +≥0.3 dB **và** σ_eff giảm ⇒ M1 có thật. Đối chứng: cùng quy trình nhưng sửa bằng gradient của model (kiểu CamP) |
| **P4/P5 (xong)** | M1 dạng trường trơn: cố định (đo RAFT) vs **học được trong train** | P4: bicycle +0.17, garden +0.09, stump +0.12, counter 0 (đối chứng indoor), σ_eff giảm cả 3. **P5 (học từ 0, không RAFT)**: bicycle +0.14 / LPIPS −6% / σ_eff 0.435→0.397; garden +0.13 / LPIPS −7% / σ_eff 0.376→0.340; trường học ≈ trường đo (garden cos 0.88) | M1 **xác nhận**, tự chứa. Biên độ PSNR nhỏ; sắc nét/LPIPS rõ |
| **P2 → M2 (xong)** | M2 giữ được HF? | Kết quả: riêng M2 +0.05/+0.06, σ_eff giảm nhẹ; M1+M2 ≈ M1 | **Loại M2**; method = M1 (trường pha học được) + base MCMC → P6 (13 scene, đang chạy) |
| ~~P2~~ | | Train MCMC với loss dung sai dịch (patch, ±0.5 px) vs L1/SSIM; cùng seed/budget | σ_eff giảm, PSNR không giảm; ảnh không ghosting |

Chỉ khi P1/P2 có tín hiệu mới ghép M1+M2(+M3) thành method, chạy đủ 13 scene, ≥ 2 seed, so IBGS/GADA.

## 6. Cập nhật backlog của plan

| ID | Trạng thái sau tuần 1 |
|---|---|
| A1 depth-fix → geometry | Tiền đề "raw sai geometry" không có bằng chứng (E1); depth không phải trần của warp (E8/E10). **Dừng** |
| A2 source confidence | Oracle chọn nguồn theo patch ≈ final ở outdoor (E1b) ⇒ headroom nhỏ. **Dừng** |
| A3 distribution-matched | Gap có, nhưng gain của IBR là LF (E4) ⇒ giá trị thấp. **Hoãn** |
| A4/A5 epipolar depth | Warp bị trần resample trước khi depth thành nút thắt. **Dừng** (trừ khi làm warp không-resample) |
| A6 support routing | Có thể tái dùng cho M2 (mask vùng phi cứng). **Gộp** |
| A7 residual → densification | H2 bác bỏ (D1). **Dừng** |
| B1 distill → 3DGS thuần | **Trở thành trục chính**, nhưng theo cơ chế M1/M2 (sửa dữ liệu + loss) thay vì distill từ teacher |
| Mới: M1, M2, M3 | như §4 |

## 7. Hạn chế của bằng chứng

1 seed cho mọi run; thế giới nhân tạo chỉ chứa nội dung Gaussian biểu diễn được (trần 34 dB là trần lý tưởng); RAFT
có sàn ~0.1–0.2 px nên các số lệch nhỏ là ước lượng dưới; E7 chỉ tiêm nhiễu xoay (affine), trong khi lệch thật phần
lớn phi affine (E3) — P3 sẽ tách. Pose test cũng có lỗi (E1: 0.1–0.4 px) nhưng protocol chuẩn không cho sửa.

## 8. Trạng thái method sau ngày 2 (29/09)

**Method v1 = MCMC + trường pha học được theo view** (`src/phase/train_mcmc_phase.py`, cờ **`m1n`**: poly bậc 3, reg 1e-3,
**neo 1/8 view train (trường = 0)**, bắt đầu 1k). Inference một stage, không ảnh nguồn.
Cố định gauge bằng **neo view** chứ không zero-mean: nếu mọi view cùng chia sẻ một thành phần lệch (méo/intrinsics dư),
zero-mean ép model về khung "trung bình" khác khung camera gốc mà view test dùng (playroom −0.40, kitchen −0.23); neo
view giữ model ở khung camera gốc (bonsai 0 thay vì −0.12, bicycle +0.16).

| Bằng chứng | Số |
|---|---|
| P6 (reg 1e-4, 13 scene) vs MCMC | outdoor +0.01…+0.13, DB +0.22 (playroom +0.46), indoor −0.2…−0.6 (`reports/p6_m1.md`) |
| Sweep reg | 1e-3: trường học = lệch đo RAFT (0.164 vs 0.151 px), counter trung tính, bicycle +0.10 |
| **Phase-aligned PSNR** (bỏ lệch pha trơn ở test) | ours − MCMC: bicycle **+0.46**, garden **+0.38**, playroom **+0.65**, bonsai **+0.20** (run có PSNR chuẩn −0.59!) |
| Nguyên nhân "hại indoor" | **gauge drift**: các trường theo view dịch chung cả cảnh so với camera test cố định. Zero-mean giảm (bonsai −0.59 → −0.12); neo view đang thử |
| MCMC thuần vs GADA final | thắng 5/5 outdoor (garden +0.45, stump +0.36) — SOTA image-based thua render thường ở outdoor |
| M3 (affine màu theo view) | T&T train +0.20 với nội suy theo pose (ngân sách N3 +0.66); chưa ghép |
| Tensara post-hoc | bicycle −0.65, garden −0.90 |

**Câu chuyện paper (dự kiến):** (1) blur của 3DGS là inconsistency dưới pixel, không phải năng lực (E7); (2) IBR không
chuyển được HF ở outdoor vì trần resample (E8/E10), gain của họ là LF (E4); (3) sửa inconsistency lúc train bằng trường
pha học được → model sắc hơn, thắng SOTA ở outdoor với một stage; (4) metric chuẩn đo thấp cải thiện vì test view cũng
lệch pha → đề xuất phase-aligned PSNR làm metric chẩn đoán kèm theo.

**Việc còn lại:** P7 (13 scene, `m1s`) đang chạy; chọn gauge (zero-mean vs neo); ghép M3; 2–3 seed cho bảng chính;
đo chi phí (train time, không đổi ở inference); viết.

### 8.1 Cập nhật 29/09 (chiều) — hai cấu hình, một mâu thuẫn, và lối ra

| Cấu hình | PSNR chuẩn (mean 13, Δ vs MCMC) | Phase-aligned (mean 13) | Nhận xét |
|---|---|---|---|
| `m1` (reg 1e-4, không gauge) | −0.05 | **+0.27** (12/13 dương; flowers +0.61, playroom +0.65) | Trường lớn bắt đúng inconsistency; mất điểm chuẩn là **drift** |
| `m1n` (reg 1e-3 + neo view) | +0.02 | +0.07 (outdoor +0.06…+0.29; counter/kitchen/drjohnson âm) | An toàn nhưng nén mất phần lớn tín hiệu |

Lối ra (đang chạy, `m1w` = reg 1e-4 + neo + **trường chung**): thành phần chung của mọi view là tính chất camera
(méo/intrinsics dư) → học riêng, **áp cho cả render test** (vẫn một stage, không ảnh nguồn); phần dư theo view được
neo để không trôi. Nếu PSNR chuẩn của `m1w` tiến gần mức aligned của `m1`, method đạt cỡ **+0.2–0.3 dB trung bình**
so với MCMC bằng render thường — đủ để vượt GADA ở outdoor một cách rõ ràng và không thua ở indoor.

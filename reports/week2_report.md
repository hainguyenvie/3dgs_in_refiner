# Week 2 — Ảnh thật đáng tin ở đâu và ở tần số nào? Kết luận và các hướng đã thử

> File báo cáo **duy nhất** của tuần 2 (bám `plan/week2_plan.md`). Cập nhật cuối: 02/10/2026. **Không còn job nào chạy.**
> Máy: 1×H200. Dữ liệu tải lại từ nguồn chính thức (khớp byte). Checkpoint IBGS lấy từ link Drive của tác giả; số IBGS
> re-render bằng `metrics.py` của tác giả trùng khít giữa server cũ và mới. MCMC train lại khớp tuần 1 (±0.1 dB).
> Code: `src/route/`, script: `scripts/`. Kết quả số gốc: `outputs/route/*.json`, log trên server `logs/`.

---

## 0. Kết luận tuần 2 (đọc phần này trước)

**Chưa có phương pháp nào đứng được ở setup công bằng.** Tuần này cho ra một bộ phát hiện phân tích có giá trị và loại
bỏ được nhiều hướng bằng số liệu, nhưng claim "vượt SOTA" trước đây **không còn giữ được** sau đối chứng ensemble.

| Hướng | Kết quả | Kết luận |
|---|---|---|
| R0 — chọn tập ảnh nguồn (routing) | oracle cấp ảnh chỉ +0.12 dB so với IBGS (indoor) | Dừng (đúng gate R0 của plan) |
| Gate theo băng tần × support, trộn MCMC / IBGS | LOSO: Mip 29.28 / T&T 25.59 / DB 30.39 — "vượt GADA" | **Không hợp lệ:** gần hết gain là ensemble 2 mô hình (trung bình MCMC+IBGS đã được Mip 29.11). Hệ 2 mô hình so với baseline 1 mô hình |
| Cùng gate, setup công bằng (1 mô hình IBGS: raw + final) | +0.00 … +0.14 dB | Cơ chế gần như không cộng thêm gì |
| Phương pháp riêng không dùng IBGS (BandFuse; band-limited warping với depth MCMC) | tốt nhất ngang IBGS ở indoor; outdoor ≈ 0 | Chưa đạt; nút thắt là geometry |
| Prior sinh ảnh (Difix, hậu xử lý) — split chuẩn | giảm PSNR ở mọi lớp support | Loại |
| Split giữ một cung góc (garden) | 3DGS tụt 3.3 dB; IBGS và Difix đều không giúp | Không có dư địa cho IBR/sinh ảnh |
| Nerfbusters (view xa thật, 1 scene: aloe) | IBGS sụp (−2.5…−3.4 dB); Difix +0.24; chỉ lấy tần thấp của ảnh sinh tốt hơn lấy nguyên ảnh | Gain quá nhỏ |
| Chưng cất ảnh sinh vào 3DGS ("Difix3D có gate") | không làm | **Không có motivation:** teacher (Difix) không tốt hơn student; lợi ích duy nhất là tốc độ inference |

**Những gì đã chứng minh được (dùng được cho hướng tiếp theo):**
1. Ở split chuẩn, **dư địa của mọi cách trộn ảnh thật rất nhỏ**: oracle chỉ hơn IBGS +0.5…1.0 dB ở indoor/T&T, ≈ +0.6 so
   với render ở outdoor kể cả khi căn chỉnh hoàn hảo; 10–24% lỗi nằm ở vùng không ảnh nguồn nào nhìn thấy (§5).
2. **Ảnh thật chủ yếu có ích ở tần thấp**: lợi thế của IBGS so với Gaussians giảm đơn điệu từ băng thô sang băng mịn ở 5/8
   scene; **ở vùng không có nguồn ảnh thật luôn thua ở mọi băng (8/8)** (§3).
3. Tần số "giao cắt" do **tỉ số sai số hai bên** quyết định, không do độ lệch đăng ký σ (§3.3).
4. **IBR (IBGS) không bền khi rời quỹ đạo train** (Nerfbusters, −2.5…−3.4 dB so với 3DGS) (§7).
5. Ở nơi prior sinh ảnh có ích, **chỉ giữ tần thấp của ảnh sinh tốt hơn dùng nguyên ảnh** (§7).

**Quyết định của người dùng trong tuần:** bỏ MCMC khi chưa có motivation và so cùng setup với baseline; hướng
calibration (tuần 1) để dành cho bài khác; dừng hướng sinh ảnh/chưng cất. Hướng tiếp theo: chưa chốt.

---

## 1. Setup và mức độ công bằng

| Thứ | Giá trị |
|---|---|
| Dataset | Mip-NeRF 360 (9), T&T (train, truck), DB (drjohnson, playroom); split LLFF 1/8 |
| Độ phân giải | Mip outdoor r4, indoor r2; T&T/DB theo độ phân giải eval của IBGS (T&T là 980×545 dù cờ `-r 2`; GT khớp 1e-15) |
| IBR | IBGS (NeurIPS'25), checkpoint tác giả. Warper trong kernel: mỗi pixel ≤5 nguồn hợp lệ đầu tiên, depth test tương đối 0.01; mạng mean-pool + CNN residual |
| SOTA đối chiếu | IBGS checkpoint re-render: Mip 28.47 / T&T 24.98 / DB 29.94 (paper 28.53 / 24.89 / 29.92); GADA (paper): 28.63 / 24.93 / 30.22 |
| Metric | PSNR / SSIM / LPIPS(vgg) trên ảnh 8-bit, trung bình theo view |

**Công bằng — tự đánh giá:**
- Gate với ứng viên MCMC: **không công bằng** (2 mô hình, base mạnh hơn baseline) → chỉ giữ làm bằng chứng phân tích.
- Gate 1 mô hình IBGS: công bằng (cùng mô hình, cùng thông tin).
- Split cung góc: 3DGS đọc `images_4` có sẵn còn IBGS tự resize (chênh GT rất nhỏ, nên thống nhất); split tự dựng theo luật
  của "Mind the Gap" (arXiv 07/2026, chưa phản biện), chưa có baseline công bố.
- Nerfbusters: **chưa kiểm chứng protocol**: 3DGS của graphdeco (paper dùng gsplat), mask tự cài lại (pseudo-GT 3DGS thay
  nerfacto, độ phủ 0.76 so với ~0.9 trong FlowR), mới 1/12 scene nên chưa đối chiếu được với 3DGS 17.66 trung bình.

---

## 2. R0 — routing tập ảnh nguồn

Probe `src/route/r0_probe.py`: giữ nguyên Gaussians, warper và mạng IBGS, chỉ đổi tập nguồn; render **mọi** tập con 1–3
trong pool 8 ứng viên.

| scene | IBGS mặc định | nearest-2 | coverage-3 | oracle ảnh K≤3 | oracle patch (chọn bằng SSIM, chấm MSE) |
|---|---|---|---|---|---|
| bonsai | 34.92 | +0.00 | −0.16 | **+0.12** | +0.31 |
| counter | 30.63 | −0.05 | −0.09 | **+0.12** | +0.57 |
| train | 23.79 | +0.21 | +0.01 | +1.11 | +1.33 |

- Random-K kém xa (−3.6…−7.7 dB): mạng residual tin warp mù quáng.
- Train: oracle lớn chủ yếu do "đoán đúng phơi sáng của ảnh test" (camera quay vòng, frame quay lại khác phơi sáng) — phụ
  thuộc GT, không khai thác hợp lệ được; rule "chọn nguồn khớp màu với render" còn hại (−2.6 dB).
- ⇒ Dừng router chọn nguồn (bảng quyết định §6 của plan).

---

## 3. Phát hiện cơ chế: ảnh thật đáng tin ở tần thấp và nơi có support

### 3.1 Mô hình
Ảnh warp E = T(x+δ) + sai lệch appearance, δ ~ N(0, σ²) ⇒ phổ sai số S_E(ω) ≈ 2|T̂(ω)|²(1 − e^{−ω²σ²/2}) (≈0 ở tần thấp,
→2|T̂|² ở tần cao). Gaussians I = T + e_I. Trộn tối ưu theo băng (sai số độc lập): w_E(ω) = S_I / (S_I + S_E); không nguồn
⇒ w_E = 0. Đây là bản theo băng tần của khung support của L2R-GS: support không chỉ quyết định *có* tin evidence ngoài,
mà *tin đến tần số nào*.

### 3.2 Phổ sai số đo được (`src/route/band_spectrum.py`) — tỉ số MSE IBGS / MCMC theo băng, thô nhất → mịn nhất

| scene | vùng ≥3 nguồn | vùng 1–2 nguồn | vùng **0 nguồn** |
|---|---|---|---|
| bonsai | 0.18 → 0.47 → 0.61 → 0.75 → **0.95** | 0.53 → … → 1.04 | 2.70 → 1.47 → 1.31 → 1.23 → 1.19 |
| counter | 0.66 → 0.65 → 0.71 → 0.80 → **0.90** | 0.81 → … → 1.00 | 1.14 → … → 1.17 |
| kitchen | 0.66 → … → **1.17** | 1.26 → … → 1.15 | 2.04 → … → 1.37 |
| train | 0.68 → 0.69 → 0.80 → 0.82 → **0.88** | 0.70 → … → 0.98 | 0.96 → 1.18 → … → 1.10 |
| bicycle | 0.82 → **1.05 → 1.04 → 1.04 → 1.03** | 0.77 → **1.08 → … → 1.06** | 1.02 → 1.40 → … → 1.12 |
| garden | **1.05 → 1.03 → 1.09 → 1.11 → 1.13** | 1.07 → … → 1.16 | 1.35 → … → 1.40 |
| stump | **1.20 → 1.06 → 1.10 → 1.10 → 1.09** | 1.05 → … → 1.05 | 1.05 → … → 1.12 |
| truck | **1.27** → 0.97 → 0.95 → 0.91 → 0.96 | 0.90 → 0.99 → 0.99 → 0.97 → 1.05 | 1.52 → … → 1.19 |

- Đúng ở 5/8 (bonsai, counter, kitchen, train, bicycle): lợi thế lớn nhất ở băng thô, mất dần ở băng mịn — ngược với cách
  IBGS/GADA tự giải thích ("ảnh nguồn đem lại chi tiết tần cao").
- Garden/stump: ảnh thật thua ở mọi băng; truck: thua ở băng thô nhất (nhiều khả năng do hiệu chỉnh phơi sáng của IBGS).
- **Vùng 0 nguồn: ảnh thật thua ở mọi băng ở cả 8/8 scene** (mạng residual "bịa" khi không có evidence).

### 3.3 Cái gì quyết định tần số giao cắt (`src/route/misreg_sigma.py`, phân tích có GT)

| scene | σ RMS (px, độ phân giải eval) | giao cắt (vùng ≥3 nguồn) |
|---|---|---|
| bonsai (r2) / counter (r2) / train | 1.44 / 1.31 / 0.97 | băng mịn nhất |
| bicycle (r4) / garden (r4) | 0.86 / 0.62 | ngay sau băng thô nhất / không giao cắt |

σ một mình **không** dự đoán được giao cắt (indoor σ lớn hơn nhưng giao cắt muộn hơn). Thứ quyết định là tỉ số S_I/S_E:
indoor MCMC sai gấp 5.5× IBGS ở băng thô (bonsai — shading/phản xạ mà ảnh thật mang đúng); outdoor MCMC vốn đã tốt.

### 3.4 Gate học được tự tái hiện cấu trúc này
Trọng số gate (LOSO) dành cho Gaussians theo tầng mịn → thô: bonsai 0.40/0.31/0.22/0.12/0.05, counter
0.42/0.20/0.16/0.11/0.05, train 0.41/0.17/0.13/0.09/0.05 — không ràng buộc nào ép theo hướng này.

---

## 4. Gate phân xử theo băng tần — kết quả và vì sao không còn hợp lệ

### 4.1 Kết quả với ứng viên MCMC (2 mô hình — chỉ để tham khảo)
Ứng viên: MCMC raw, IBGS final, residual IBGS trên MCMC; evidence: số warp hợp lệ, biên depth test, bất đồng warp, độ lớn
residual, bất đồng giữa hai mô hình, depth. LOSO đủ 13 scene (playroom dùng config MCMC tuần 1, `opacity_reg 0.001`):

| dataset | MCMC | IBGS (ckpt) | GADA (paper) | gate băng (LOSO) | gate pixel | gate Shiny zero-shot | LODO |
|---|---|---|---|---|---|---|---|
| Mip-360 | 28.32 | 28.47 | 28.63 | 29.28 | 29.22 | 29.15 | ≈ LOSO |
| T&T | 24.57 | 24.98 | 24.93 | 25.59 | 25.59 | 25.50 | ≈ LOSO |
| DB | 29.69 | 29.94 | 30.22 | 30.39 | — | 30.14* | — |

\* Shiny zero-shot dùng playroom với config repo MCMC (thiếu `opacity_reg`).
Cross-fitting trên chính scene (giấu 1/8 view train, train lại MCMC_dev + IBGS_dev): bonsai 35.56, counter 31.10.

### 4.2 Đối chứng ensemble — tại sao các số trên không phải đóng góp của cơ chế

| scene | MCMC | IBGS | avg(MCMC, IBGS) | avg 3 ứng viên | gate (LOSO) |
|---|---|---|---|---|---|
| bicycle | 26.18 | 26.08 | 26.76 | 26.77 | 26.80 |
| flowers | 22.43 | 22.29 | 23.13 | 23.12 | 23.16 |
| garden | 28.20 | 27.59 | 28.50 | 28.51 | 28.51 |
| stump | 27.69 | 27.29 | 28.21 | 28.17 | 28.20 |
| treehill | 23.36 | 22.93 | 24.04 | 24.04 | 24.09 |
| bonsai | 32.84 | 34.92 | 34.71 | 35.18 | 35.46 |
| counter | 29.48 | 30.63 | 30.69 | 30.90 | 31.00 |
| kitchen | 32.31 | 31.96 | 32.85 | 32.99 | 32.99 |
| room | 32.38 | 32.50 | 33.09 | 33.24 | 33.27 |
| train | 22.73 | 23.79 | 23.96 | 24.10 | 24.20 |
| truck | 26.42 | 26.18 | 27.02 | 27.08 | 26.99 |
| drjohnson | 29.32 | 29.74 | 30.18 | 30.08 | 30.14 |
| playroom | 30.07 | 30.15 | 30.63 | 30.58 | 30.63 |
| **Mip-360** | 28.32 | 28.47 | 29.11 | 29.21 | 29.28 |

Ở outdoor và DB, gate ≈ trung bình hai mô hình độc lập (giảm phương sai), không phải phân xử theo evidence; gate chỉ
đóng góp thật ở indoor (bonsai +0.28, counter +0.10 so với trung bình 3). Đối chứng ensemble 2 seed MCMC đã bắt đầu nhưng
dừng khi quyết định bỏ MCMC.

### 4.3 Setup công bằng — 1 mô hình IBGS (ứng viên IBGS raw + IBGS final, evidence từ IBGS)

| | bicycle | flowers | garden | stump | treehill | bonsai | counter | kitchen | room | train | truck | drjohnson | playroom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IBGS final | 26.08 | 22.29 | 27.59 | 27.29 | 22.93 | 34.92 | 30.63 | 31.96 | 32.50 | 23.79 | 26.18 | 29.74 | 30.15 |
| gate 1 mô hình | 26.13 | 22.34 | 27.66 | 27.34 | 23.07 | 34.89 | 30.64 | 32.06 | 32.48 | 23.78 | 26.19 | 29.74 | 30.15 |

⇒ +0.00 … +0.14 dB: mạng residual của IBGS vốn đã học được nơi nên tin warp.

### 4.4 Các thử nghiệm phụ của gate (với MCMC, tham khảo)
- Baseline không học "băng mịn từ MCMC, băng thô từ IBGS nơi có nguồn": không thua cả MCMC lẫn IBGS ở scene nào trong 8.
- Thêm evidence theo băng (hai vế Wiener) hay ứng viên affine phơi sáng: không giúp (±0.08).
- Một lần train gate bị phân kỳ (fold garden) → đã thêm clip gradient + kiểm tra "loss phải thấp hơn ứng viên đơn tốt nhất".
- Mạng residual IBGS đặt lên render MCMC (không train lại) hơn IBGS ở 8/8 scene đã đo — base mạnh + residual từ ảnh thật;
  nhưng warp bằng geometry MCMC thì kém (12–25% pixel trượt depth test so với 3% với geometry IBGS).

---

## 5. Dư địa của việc trộn ảnh thật ở split chuẩn (`src/route/headroom.py`, oracle ô 8×8, có GT)

| scene | render (MCMC) | oracle(render, warp — geometry IBGS) | oracle(render, TB warp) | oracle(render, evidence của mình — depth MCMC) | oracle căn chỉnh hoàn hảo (RAFT theo GT) | % lỗi ở vùng 0 nguồn | IBGS |
|---|---|---|---|---|---|---|---|
| bonsai | 32.83 | 35.43 | 34.94 | 34.17 | 35.57 | 10% | 34.92 |
| counter | 29.47 | 31.61 | 31.04 | 30.66 | 32.00 | 11% | 30.63 |
| garden | 28.18 | 29.01 | 28.72 | 28.38 | 29.00 | 14% | 27.59 |
| bicycle | 26.17 | 26.80 | 26.59 | 26.36 | 26.92 | 24% | 26.08 |
| train | 22.66 | 25.03 | 24.00 | 23.15 | 25.07 | 18% | 23.79 |

- Indoor/T&T: oracle hơn IBGS +0.5…+1.0; outdoor: chỉ +0.6…+0.8 so với render kể cả căn chỉnh hoàn hảo.
- Căn chỉnh hoàn hảo hầu như không thêm so với geometry IBGS (≤0.4) ⇒ lệch đăng ký không phải giới hạn chính.
- Geometry của mình (depth MCMC) mất 0.3…0.85 dB dư địa so với geometry IBGS.

### Failure analysis outdoor (`src/route/failure_outdoor.py`)

| | indoor | outdoor | T&T / DB |
|---|---|---|---|
| support, geometry IBGS | 0.95–0.98 | 0.76–0.96 | 0.71–0.93 |
| support, depth MCMC | 0.78–0.84 | **0.41–0.72** | 0.45–0.58 |
| MSE tần thấp warp ÷ MCMC, ở nơi geometry IBGS hợp lệ | 0.63–1.21 | **1.22–2.17** | 0.77–1.86 |
| lệch màu theo ảnh (gain affine, dB) | 0.1–0.4 | 0.2–1.0 | 0.2–0.9 |
| bất đồng tần thấp giữa 2 ảnh nguồn (×1e4) | 3.7–10.9 | 8.5–19.9 | 7–84 |
| tỉ số theo vùng gần / giữa / xa | ≈ 1 | gần tệ nhất (bicycle 4.2/3.1/2.3) | truck 3.7/2.2/1.7 |

Depth MCMC mất support ở outdoor, nhưng kể cả với geometry tốt warp vẫn kém ở tần thấp; vùng gần camera hỏng nặng nhất
(δ ∝ Δz/z²); ảnh outdoor bất đồng nhau nhiều (lá/cỏ động, ánh sáng); và MCMC outdoor vốn đã đúng ở tần thấp.

---

## 6. Phương pháp riêng không dùng IBGS

**BandFuse** (`src/route/bandfuse.py`) — mạng trộn từng warp nguồn theo băng tần (encoder theo nguồn, attention giữa các
nguồn, softmax theo băng trên {render, nguồn}; tùy chọn residual, căn chỉnh sub-pixel; học LOSO / cross-fitting).

| BandFuse (tốt nhất) | bonsai | counter | train | bicycle | garden |
|---|---|---|---|---|---|
| LOSO | 34.13 | 30.25 | 23.18 | 26.27 | 28.24 |
| cross-fitting (dev views của chính scene) | 34.83 | 30.63 | 23.08 | 26.24 | 28.33 |
| IBGS | 34.92 | 30.63 | 23.79 | 26.08 | 27.59 |

Cross-fitting cho +0.6…0.9 so với LOSO ở indoor; trên bonsai tự học ra quy tắc gần nhị phân (tần cao tin Gaussians, ba băng
thô tin ảnh thật). Vẫn dùng depth của IBGS để warp; hỏng ở train (phơi sáng thay đổi).

**Band-limited warping với depth MCMC** (`src/route/bandwarp.py`, không IBGS): indoor +0.2…+0.6, outdoor ≈ 0. Nới ngưỡng
depth theo tầng (τ_l ∝ 2^l) **bị bác bỏ** (lọt điểm bị che khuất). Prior art: Stewart et al. EGSR 2003, Brédif 2014.

---

## 7. Prior sinh ảnh và setting view xa

**D0 — Difix (trọng số chính thức), split chuẩn** — PSNR toàn ảnh / vùng 0 nguồn:

| | bonsai | garden | bicycle | train |
|---|---|---|---|---|
| render (MCMC) | 32.84 / 27.54 | 28.20 / 21.48 | 26.18 / 23.38 | 22.73 / 19.19 |
| Difix-ref(IBGS) toàn ảnh | 30.88 / 25.58 | 25.05 / 19.65 | 24.26 / 21.51 | 22.89 / 18.95 |
| Difix chỉ ở vùng 0 nguồn | 34.70 | 27.45 | 25.67 | 23.74 |

Không phải lỗi độ phân giải (576×1024 còn tệ hơn); chỉ giữ tần thấp của Difix đỡ hại hơn nhưng vẫn dưới render.

**Split giữ một cung góc** (garden; K = N/8 view liền nhau, ~39°): 3DGS 24.20 (split chuẩn 27.49), IBGS 24.06, Difix-ref
23.12; vùng 0 nguồn tăng ~4% → 25% nhưng IBR và Difix đều không giúp. (bonsai: 3DGS 28.13 so với 32.28.)

**Nerfbusters aloe** (video train/eval riêng; undistort; downscale 2; mask visibility tự cài):

| | PSNR (mask) | PSNR toàn ảnh | LPIPS |
|---|---|---|---|
| 3DGS gốc | 12.08 | 12.06 | 0.602 |
| IBGS | 9.57 | 8.65 | 0.690 |
| Difix-ref(3DGS) | — | 12.30 | 0.541 |
| Difix chỉ ở vùng 0 nguồn | — | 12.33 | 0.559 |

12 view (vùng 0 nguồn 71%): render 14.85, Difix nguyên 14.72, **chỉ tần thấp của Difix 15.04–15.13**. Lưu ý 3DGS aloe thấp
hơn nhiều so với trung bình công bố (17.66), có lệch phơi sáng giữa hai video, và mới 1 scene.

**Về chưng cất (Difix3D-style):** chưng cất chỉ có nghĩa khi teacher tốt hơn student hoặc giải quyết được điều hậu xử lý
không làm được. Ở cả ba setting đã đo, teacher (Difix) không tốt hơn render hoặc chỉ hơn +0.24 trên một scene bất thường;
lợi ích còn lại là inference một stage (tốc độ), không đủ làm động cơ cho một paper về chất lượng ⇒ **không làm**.

---

## 8. Định vị so với related work (để dùng nếu quay lại hướng này)

| Đã có | Liên hệ |
|---|---|
| Deep Blending (2018): softmax theo pixel giữa render mesh và warp, có fallback | gate theo pixel của mình gần như tương đương về PSNR |
| BlendedMVS / Baumberg 2002 / Stewart 2003: tần thấp từ ảnh, tần cao từ mô hình, bộ lọc cố định | cùng ý tưởng tách băng |
| HDR+ (Wiener theo tile × tần số, fallback về frame tham chiếu) | gần nhất về cấu trúc |
| IBGS / GADA: residual cộng thêm, tự nhận ảnh nguồn đem lại tần cao | đo đạc §3.2 cho thấy ngược lại ở 5/8 scene |
| ArtiFixer, ConFixGS, 3DGS-Enhancer: gate ảnh sinh theo opacity/confidence | chưa ai gate theo băng × support |

---

## 9. Câu hỏi mở cho hướng tiếp theo
- Ở split chuẩn, dư địa còn lại nằm ở vùng không có nguồn (10–24% lỗi) và ở chất lượng mô hình explicit — cần **thông tin
  mới** chứ không phải cách trộn tốt hơn.
- Nếu đi hướng view xa: phải dựng đúng protocol Nerfbusters (mask chính thức với nerfacto, gsplat), tái tạo 3DGS ≈ 17.66 trên
  12 scene trước khi so sánh; và cần một nguồn thông tin tốt hơn cả render lẫn Difix ở vùng không quan sát.
- Các phát hiện §3 (ảnh thật chỉ có ích ở tần thấp, vùng không nguồn luôn hại, IBR sụp khi rời quỹ đạo) có thể là phần
  phân tích/motivation của một bài khác, nhưng tự chúng chưa phải phương pháp.

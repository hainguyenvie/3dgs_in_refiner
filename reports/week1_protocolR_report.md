# Week 1 — Protocol R report (28/09/2026)

Bảng đầy đủ per-scene: [`protocolR_table.md`](protocolR_table.md) (+ `.csv`). Protocol đóng băng:
[`../protocol/protocol_R.md`](../protocol/protocol_R.md). Mọi run: 1 seed, H200, stack chung
(py3.8 / torch 2.1.2+cu121 / nvcc 12.1 / gcc 11), 2–3 job/card nên **thời gian/FPS bị tranh chấp**.

## 1. Kết luận gate

| Baseline | Kiểm tra | Kết quả | Gate |
|---|---|---|---|
| IBGS | Render lại 13 checkpoint tác giả phát hành trên env của ta | max \|Δ\| PSNR 1e-4, SSIM 3e-6, LPIPS 8e-6 | ✅ env + dữ liệu + metric đúng |
| IBGS | Tự train 13 scene vs số tác giả phát hành | final: Mip360 +0.06, T&T −0.09, DB −0.02 dB (per-scene −0.23…+0.18) | ✅ |
| 3DGS | Tự train 13 scene vs paper 2023 | Mip360 +0.36, T&T +0.79, DB +0.27 | ⚠ lệch, **truy được**: khớp chạy lại độc lập (MCMC Tab. 5) ±0.04 ở indoor → bảng 2023 cũ |
| 3DGS-MCMC | Tự train 11 scene vs paper (SfM) | Mip360(7) −0.05, T&T +0.17; playroom −1.10 | ⚠ playroom: config repo thiếu `opacity_reg 0.001`; chạy lại đúng paper → 30.03 (−0.30), DB avg +0.11 |

→ Cả ba baseline dùng được cho Protocol C. Hai quyết định mang sang: (1) MCMC DB dùng λo = 0.001;
(2) 3DGS so bằng số tự chạy, không dùng bảng 2023.

## 2. Quan sát định hướng nghiên cứu (sơ bộ — Protocol R, chưa phải so sánh có kiểm soát)

Mip360 7 scene chung (IBGS `-r` và MCMC `resolution` đều resize trong code → cùng GT):

| | bicycle | garden | stump | bonsai | counter | kitchen | room | avg7 |
|---|---|---|---|---|---|---|---|---|
| MCMC (raw = final) | **26.13** | **28.19** | **27.69** | 32.78 | 29.43 | 32.21 | 32.48 | 29.85 |
| IBGS raw | 25.71 | 27.26 | 27.22 | 30.99 | 28.29 | 30.52 | 31.30 | 28.76 |
| IBGS final | 26.06 | 27.57 | 27.31 | **34.98** | **30.65** | 32.10 | **32.68** | **30.19** |
| IBGS final − raw | +0.35 | +0.31 | +0.09 | +3.99 | +2.36 | +1.58 | +1.38 | +1.43 |

- **Gaussian của IBGS yếu hơn MCMC 1.1 dB** (raw); nhánh warp bù lại và vượt MCMC trung bình +0.34 dB,
  nhưng **thua MCMC ở cả 3 scene outdoor** — nơi gain warp chỉ +0.1…+0.35 dB.
- Gain warp tập trung ở indoor (vùng nhiều nguồn, baseline nhỏ): +1.4…+4.0 dB. Cùng pattern ở GADA công bố
  (raw Mip360 27.33 → final 28.63).
- T&T: IBGS 24.89 vs MCMC 24.46; DB: IBGS 29.92 vs MCMC (λo 0.001) 29.77.
- Hàm ý cho plan: (a) base mạnh (MCMC) + nhánh warp là điểm xuất phát tự nhiên, cần đối chứng "MCMC + post-hoc
  refiner" và "IBGS trên base MCMC"; (b) chẩn đoán §5 nên tách outdoor/indoor — câu hỏi "refiner sửa lỗi ở đâu"
  có tín hiệu rõ theo overlap/baseline; (c) A1 (warp → cập nhật geometry/raw) nhắm đúng chỗ IBGS yếu (raw).
- ⚠ Cảnh báo: khác ngân sách Gaussian (MCMC bicycle 5.9M vs IBGS 3.6M), 1 seed, chưa cùng protocol → **không
  claim**; đây là giả thuyết để Protocol C kiểm.

## 3. Chi phí (tranh chấp, chỉ tham khảo thứ tự lớn)

| | train (phút, median) | #Gauss Mip360 outdoor | FPS test (IBGS render.py) |
|---|---|---|---|
| 3DGS | 17 | 2.9–4.9M | — |
| MCMC | 50 | = cap_max (4.75–5.9M) | — |
| IBGS | 90 | 2.1–3.6M | 5–30 (tranh chấp; tác giả 13.6–16.3 trên 4090) |

Đo lại cô lập (1 job/card) trong Protocol C theo measurement contract.

## 4. Việc tiếp theo

1. Protocol C: chốt GT chung cho Mip360 (resize-in-code vs `images_4`), chạy lại cả ba baseline dưới protocol chung,
   đo chi phí cô lập.
2. Port refiner post-hoc (report Tensara, `round3_official/src/refine/`) sang benchmark công khai trên base MCMC.
3. Chấm lại ảnh GADA công bố bằng scorer của ta; thêm Shiny (dữ liệu đã tải) cho IBGS.
4. Seed thứ 2–3 cho các biến thể quan trọng trước khi đọc chênh lệch < 0.2 dB.

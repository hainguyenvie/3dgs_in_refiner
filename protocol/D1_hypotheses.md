# D1 — chẩn đoán "base yếu" trong image-based GS (đóng băng 28/09/2026, trước khi chạy)

## Quan sát khởi phát (Protocol R, 1 seed)
Mip360 7 scene chung: MCMC 29.85 · IBGS raw 28.76 · IBGS final 30.19. IBGS thua MCMC ở cả 3 scene outdoor
(gain warp chỉ +0.09…+0.35 dB), thắng indoor (+1.4…+4.0 dB).

## Giả thuyết
- **H1 (densification):** base IBGS dưới-khớp vì densify PGSR (cull opacity 0.05, ít Gaussian). Thay bằng MCMC
  relocation (cap_max như MCMC) → raw tăng về gần MCMC, và **gain warp cộng thêm gần như nguyên vẹn** → final tăng.
- **H2 (co-adaptation):** sau khi bật residual (10k), loss raw bị chia đôi và gradient final đi vào base qua mạng
  residual → base "lười". Tắt đường gradient (detach, raw weight 1) → raw tăng; nếu final không giảm thì H2 đúng
  và là một thành phần method.

## Ma trận (src/irgs, `scripts/run_irgs.sh`) — 6 scene: bicycle garden stump (outdoor) · bonsai counter (indoor) · train
| variant | đo cái gì |
|---|---|
| `ibgs` (bicycle, counter) | fork = IBGS? (kỳ vọng |Δ| ≲ 0.1 dB so r1 Protocol R) |
| `noagg` | raw của base IBGS khi không có residual → so raw `ibgs`: co-adaptation |
| `detach` | H2 |
| `mcmc_noagg` | base MCMC + loss hình học IBGS → so MCMC Protocol R (loss hình học có làm giảm raw?) |
| `mcmc_agg` | H1 |
| `mcmc_detach` | H1 + H2 |

## Cách đọc (quy tắc cố định)
- 1 seed → chênh < 0.15 dB/scene là **chưa phân biệt**; đọc theo trung bình 6 scene và theo dấu nhất quán.
- H1 ủng hộ nếu `mcmc_agg` final > `ibgs` final ở ≥ 5/6 scene và trung bình ≥ +0.2 dB, **và** outdoor tăng.
- H2 ủng hộ nếu raw(`detach`) − raw(`ibgs`) ≥ +0.2 dB trung bình mà final(`detach`) ≥ final(`ibgs`) − 0.05.
- Biến thể thắng được chạy đủ 13 scene (+ seed 2) rồi mới so SOTA (IBGS released 28.47/24.98/29.94, GADA
  28.62/24.92/30.22). Không chọn biến thể theo test của scene ngoài 6 scene chẩn đoán rồi báo trên chính chúng
  như kết quả sạch — báo rõ 6 scene là tập chọn.
- Chi phí (số Gaussian, train time, FPS) ghi kèm: MCMC tăng số Gaussian → FPS giảm; phải báo trade-off.

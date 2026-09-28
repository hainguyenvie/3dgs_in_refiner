# Week 1 — nhật ký chẩn đoán (cập nhật liên tục; số liệu ở `e*_*.md/json` cùng thư mục)

Mục tiêu của chuỗi này: trả lời **"nhánh warp/IBR đang sửa cái gì, và cái gì chặn nó"** trước khi xây method.
Mọi phân tích dùng render Protocol R (test view, không train thêm, không chọn tham số trên test).

## Chuỗi phát hiện

| # | Câu hỏi | Kết quả | Hệ quả |
|---|---|---|---|
| E1 | Lỗi raw có phải do lệch hình học? (RAFT GT↔raw) | Lệch trung vị 0.08–0.39 px; chỉ ~5% lỗi raw giải thích được bằng lệch; ~55% năng lượng lỗi ở tần số cao (HF) | Tiền đề "warp sửa geometry sai" (A1 nguyên bản) **không có bằng chứng** ở test view. ⚠ Nhưng flow≈0 **không loại trừ** inconsistency: dưới sai lệch sub-px giữa các view, model tối ưu bị **mờ chứ không lệch** (suy giảm exp(−2π²σ²f²)) |
| E6 | Sai số tái chiếu COLMAP ở độ phân giải đánh giá | Trung vị 0.23–0.35 px (Mip360, T&T), 0.46 px (DB) | Đúng bậc để gây mất HF đáng kể (σ=0.3 → −36% biên độ tại Nyquist). DB: σ lớn nhất **và** gain warp nhỏ nhất |
| E4 | Gain của nhánh warp nằm ở băng tần nào? Oracle affine màu theo view làm được bao nhiêu? | **Outdoor: 91–96% gain ở LF (σ>16px), HF giảm ≈0**; affine màu oracle (12 tham số/view) **≥ final** ở bicycle/stump/treehill. **Indoor: HF giảm 17–21%, MF 33–43%** (texture thật sự được chuyển). T&T train: raw+affine 23.61 ≈ final 23.69 → gain ≈ sửa exposure | Ở outdoor nhánh warp **không chuyển texture**; chỉ indoor mới có. Giá trị "không thể thay thế" của warp = phần HF/MF, hiện chỉ xuất hiện khi baseline nhỏ + depth chính xác |
| E5 | Phổ truyền T(f) render→GT, σ_eff | Mọi render ≈ GT mờ với σ_eff 0.38–0.71 px. MCMC **sắc hơn** IBGS raw (bicycle 0.435 vs 0.475; bonsai 0.570 vs 0.671). Final phục hồi tần số giữa (T(0.25): 0.57→0.69 bonsai) nhưng **không** tần số gần Nyquist (T(0.4): 0.238→0.212) | Base IBGS bị mờ thêm ~0.05–0.1 px σ so với MCMC (co-adaptation đo được bằng phổ). Warp bilinear + lệch sub-px giữa nguồn → không cứu được băng cao nhất |
| E3 | (đang chạy) flow trên **train view**, tách phần affine/view | | Nếu phần affine theo view rõ → pose/intrinsics từng view sai |
| E7 | (đang chạy) **oracle tự nhất quán**: GT nhân tạo render từ MCMC bicycle, train lại với nhiễu pose σ = 0 / 0.15 / 0.3 / 0.6 px (xoay camera train, test giữ nguyên) | | σ=0 cho **sàn năng lực** của biểu diễn; đường cong theo σ cho biết bao nhiêu blur thật là do inconsistency |
| E1b | Warp từng nguồn tại test view (checkpoint tác giả, 5 scene) | **Outdoor: warp của ảnh thật kém hơn raw 2–2.6 dB trên cùng pixel** (bicycle 22.84 vs 25.45) dù lệch trung vị chỉ 0.21–0.23 px (15–20% px > 0.5 px); lỗi warp 40–53% ở HF. Oracle chọn nguồn theo patch 32² ≈ final. **Indoor: warp tốt hơn raw +2 dB**, lệch 0.07–0.12 px; final > oracle patch. T&T train: lệch 0.36 px nhưng gain lớn vì lỗi raw 75% LF (exposure) | Nút thắt outdoor = **lệch dưới pixel của nội dung sắc + loss L1/SSIM phạt lệch hơn phạt mờ** ⇒ mạng học "không tin" warp ⇒ chỉ còn LF. Không phải thiếu thông tin |
| D1 | (đang chạy) `ibgs`/`noagg`/`detach`/`mcmc_*` trên 6 scene | | H1 (densify), H2 (co-adaptation) — xem `protocol/D1_hypotheses.md` |

## Khung lập luận hiện tại (chưa phải kết luận)

1. Gaussian = **đồng thuận đa view**. Giám sát không nhất quán (pose/intrinsics sub-px, exposure) ⇒ đồng thuận tối ưu bị mờ ⇒ lỗi HF (E1, E5, E6). Phần LF của lỗi = appearance theo view mà model đồng thuận không thể biểu diễn (E4).
2. Nhánh warp có hai vai trò **khác bản chất**: (a) sao chép appearance của view lân cận (LF — rẻ, có thể thay bằng mô hình appearance, không cần ảnh nguồn lúc test); (b) sao chép texture thật (HF/MF — giá trị thật, chỉ xảy ra khi warp khớp sub-px).
3. Ở outdoor (b) thất bại. Nghi vấn: depth không xác định ở foliage/mixed pixel + baseline lớn ⇒ warp lệch ⇒ mạng học "không tin" ⇒ chỉ còn (a). E1b sẽ kiểm.
4. Hướng method (đang cân nhắc, chờ E1b/E7): làm cho (b) hoạt động ở outdoor **và** đưa bằng chứng đa view ngược về biểu diễn (giảm inconsistency lúc train: pose/appearance theo view được ước lượng từ chính residual warp; distill HF nhất quán về Gaussian ⇒ inference một stage). Không tune lặt vặt.

## Bài học vận hành
- `ssh -n` + heredoc ⇒ file rỗng; `cd X && cmd &` ⇒ `cd` chạy trong subshell nền. Đã giẫm cả hai.
- 3DGS `render.py` đặt tên render theo **chỉ số** trong danh sách camera sắp theo tên, không theo tên ảnh.
- MCMC + loss hình học IBGS: NaN rời rạc quanh iter 7000 (lúc bật normal/photometric loss) ở ~50% run; guard bỏ qua bước NaN (`nan_skips.txt`).

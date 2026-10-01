# Week 2 plan — chọn reference view theo giá trị bổ sung và sửa vùng thiếu evidence

## 1. Câu hỏi trung tâm

**Mục tiêu nghiên cứu:** cải thiện novel-view synthesis (NVS) trên các benchmark công khai bằng cách sử dụng ảnh nguồn đúng nơi, đúng số lượng và đúng vai trò. Chất lượng ảnh phải được so với 3DGS-MCMC, IBGS, GADA và các phương pháp diffusion liên quan; mọi gain đều đi kèm chi phí training/inference. Ưu tiên một cơ chế có failure mode, phép đo kiểm chứng và ablation rõ, thay vì thêm module chỉ vì keyword đang phổ biến.

Hai hướng trong tuần:

1. **Nhánh chính — marginal-utility reference routing:** dự đoán ảnh nguồn nào *bổ sung* thông tin cho tập đã chọn, tại vùng nào, và khi nào nên dừng hoặc không dùng warp.
2. **Nhánh mở rộng — evidence-constrained diffusion:** chỉ thử diffusion ở nơi evidence ảnh thật không đủ hoặc warp lỗi; kiểm tra liệu nó có cải thiện chất lượng, tính nhất quán và/hoặc có thể chuyển gain vào 3DGS để inference gọn hay không.

**Không tự nhận hai nhánh là một-stage:** router dùng ảnh nguồn lúc test là hybrid inference; Difix3D+ dùng post-render diffusion cũng vậy. Nếu yêu cầu cuối cùng là chỉ render từ representation đã học, phải có thí nghiệm *distillation sang 3DGS* và báo riêng chất lượng student. Tuần này trước hết kiểm tra có signal đáng distill hay không.

Không dùng data aerial của cuộc thi vì không sẵn cho benchmark. Dữ liệu ưu tiên: **Mip-NeRF 360, Tanks and Temples (T&T), Deep Blending (DB)** với split/preprocessing đã chốt ở tuần 1.

## 2. Điều kế thừa và giới hạn suy luận từ hai báo cáo

- L2R-GS, bản thảo ICLR 2027 của nhóm, phân rã **lỗi xếp hạng độ tin cậy vùng ảnh** theo SS/SU/UU: cả hai vùng có nguồn hợp lệ, chỉ một vùng có nguồn, hoặc cả hai không có. Khi tỉ lệ vùng không được hỗ trợ là `u`, tỉ trọng cặp tương ứng thay đổi theo `(1-u)^2`, `2u(1-u)` và `u^2` trong điều kiện của proposition. Nó còn cho thấy external evidence một mình không đủ để xếp hạng vùng thiếu source.
- **Giả thuyết mới, chưa được theorem chứng minh:** độ hữu ích của một reference view đối với chất lượng ảnh *phụ thuộc những source đã có*. Hai ảnh overlap nhiều vẫn có thể bổ trợ nhau về occlusion, góc nhìn, màu hoặc texture; hai ảnh che phủ nhiều nhưng warp lỗi cũng có thể làm chất lượng giảm. Chứng minh cho *ranking error/AURC* không được dùng thay chứng minh cho *render error/PSNR/LPIPS*.
- Chẩn đoán tuần 1 của nhánh NVS: MCMC đã mạnh hơn GADA ở nhiều outdoor scene; warp outdoor thậm chí có thể kém raw 3DGS; Tensara post-hoc port cho gain âm trên bicycle/garden. Vì thế **`K=0` (giữ raw render) là một action hợp lệ và bắt buộc**. Không ép router luôn lấy nhiều ảnh, cũng không giả định thêm source sẽ cải thiện đơn điệu.
- Ở indoor, IBGS/GADA có thể hơn MCMC nhiều nhờ tổng hợp appearance từ ảnh thật. Đây là miền có dư địa thực tế cho source routing; performance toàn bộ benchmark vẫn phải báo để tránh chỉ chọn các scene thuận lợi.

## 3. Protocol và baseline cố định

### Scene và split

| Mức | Scene | Câu hỏi |
| --- | --- | --- |
| Pilot | `bonsai`, `counter`, `train` (IBR mạnh); `bicycle`, `garden`, `stump` (kiểm tra trường hợp warp có hại) | Có tồn tại oracle gain và vùng source thực sự bổ trợ nhau không? |
| Mở rộng | 9 scene Mip-NeRF 360, `train/truck` của T&T và `drjohnson/playroom` của DB | Gain có giữ trên cả indoor/outdoor và giữa các dataset không? |
| Ngoài miền nếu đủ nguồn lực | Shiny hoặc dataset có phản xạ/ánh sáng thay đổi đã được baseline tương ứng dùng | Kiểm tra generalization; không đưa vào claim benchmark chính nếu protocol không tương thích. |

Giữ train/test 1/8, resolution, camera, mask và cách tính PSNR/SSIM/LPIPS theo protocol đã đóng băng. Tạo **development targets từ training pool** bằng cross-fitting: target tạm thời được giấu khỏi tập source của chính nó; train router bằng ground truth của *chỉ* các target development này. Official held-out test **không dùng để học router, chọn checkpoint, đặt ngưỡng hoặc sinh pseudo-label**. Nếu model nền cũng được retrain, mọi baseline phải có cùng ngân sách ảnh train và số bước.

### Baseline phải có trong bảng

- 3DGS-MCMC raw; IBGS raw và final; nearest-`K` với `K=0,1,2,4,8` và cùng renderer/warp; chiến lược maximum coverage/covisibility với cùng `K`; simple disagreement/visibility weighting; refiner post-hoc đã port (kể cả khi số âm).
- GADA final **số paper** trên cùng scene/protocol; chỉ thêm GADA vào so sánh FPS local và ablation trực tiếp khi có implementation tương thích. Không gọi số paper là run tái hiện.
- Nhánh diffusion: Difix pretrained có/không reference, Difix3D cập nhật reconstruction trong training, Difix3D+ có post-render step nếu chạy được đúng split. So **teacher hybrid**, **student render thuần** và **postprocess** trong các dòng riêng.

Lưu per-scene/per-view và per-region `raw/final`, số warp thực dùng, support mask, PSNR/SSIM/LPIPS, crop và lỗi lớn. End-to-end latency gồm source ranking, tải/cache ảnh, warp, aggregation và diffusion nếu có; thêm train time, peak VRAM, footprint model + ảnh nguồn, số Gaussian. Dùng cùng GPU/resolution khi vẽ quality–latency–storage Pareto. Ở `playroom`, báo toàn bộ view đúng protocol, tách riêng phân tích lỗi MCMC; không âm thầm loại view lỗi rồi báo là metric chuẩn.

## 4. Hướng 1 — reference routing dựa trên đóng góp bổ sung

### 4.1 Định nghĩa objective cần đo

Gọi `r` là một region target, `S` là tập source đang dùng, `j` là source ứng viên, `F(r,S)` là ảnh tổng hợp tại region bằng đúng một aggregation cố định và `L` là loss đã tuyên bố. Trên development targets có ground truth, đo:

`Δ(j | S, r) = L(F(r,S), GT_r) − L(F(r,S ∪ {j}), GT_r) − λ·[cost(S ∪ {j}) − cost(S)]`.

Giá trị này **không quan sát được lúc test**: lúc đó router chỉ được dự đoán nó từ geometry, visibility, baseline/góc nhìn, texture, độ lệch giữa các warp, render/source disagreement, uncertainty nội bộ và source đã chọn. Đo **giảm loss thực sau fusion**, không gán nhãn từ overlap đơn thuần. Dùng nhiều target loss tách biệt (PSNR-related pixel loss và LPIPS/structural loss); không gộp chúng vào một oracle rồi đổi mục tiêu khi báo kết quả.

Một source nhìn thấy thêm pixel chưa chắc giảm error; có thể hại do occlusion, sai depth, resampling hoặc khác màu. Vì vậy `Δ` có thể âm và `K` là biến thích nghi theo region. Chỉ đếm chi phí source trước khi warp nếu policy có thể quyết định bằng tín hiệu rẻ; các tín hiệu cần warp phải được tính đủ vào runtime.

### 4.2 Probe quyết định: có cơ hội thật hay không?

Trên development targets, dùng **cùng checkpoint 3DGS, candidate pool, warper và aggregation** để tách giá trị của việc chọn source khỏi chất lượng của bản thân warper. Chọn khoảng 6–10 source ứng viên gần target; khảo sát `K=0,1,2,4` và một phần `K=8` nếu đủ. Với candidate pool nhỏ, enumerate subset khi khả thi; nếu chỉ beam/sample, ghi là **oracle xấp xỉ**, không gọi là upper bound tuyệt đối.

| Probe | Đối chứng | Điều muốn biết |
| --- | --- | --- |
| Oracle subset theo loss sau fusion | nearest-`K`; visibility/coverage-`K`; random-`K`; cùng `K` | Có gap đủ lớn cho source selection? |
| Lợi ích của source thứ `j` sau tập `S` | lợi ích độc lập của từng source | Có tác động bổ sung/redundancy thật hay chỉ cần chọn source tốt nhất? |
| Policy chọn `K` thích nghi, kể cả `K=0` | best fixed-`K`; raw MCMC | Tiết kiệm latency và tránh warp làm hại outdoor? |
| Chia theo supported/unsupported, depth edge, high frequency, specular và indoor/outdoor | cùng mức lỗi raw; cùng K | Gain tới từ cơ chế đề xuất hay vì vùng vốn dễ/khó? |

**Gate R0:** nếu oracle sau fusion gần best fixed-`K` trên nhiều scene, hoặc oracle cũng thường thua `K=0`, dừng việc xây router lớn. Xem lại warper/aggregation hoặc chuyển ngân sách sang nhánh 2. Không lấy oracle dùng GT trên official test làm số method; mọi oracle test (nếu vẽ) chỉ là chẩn đoán được đánh dấu riêng.

### 4.3 Các method theo thứ tự tăng chi phí

| ID | Ý tưởng và intuition | Thử nghiệm đối chứng | Dấu hiệu nên đầu tư |
| --- | --- | --- | --- |
| **R1 — rule có stop** | Gain đến từ việc *không warp* vùng unsupported và chọn source có coverage hợp lệ, góc phù hợp, ít disagreement; chưa cần network. | Nearest/coverage với cùng K, rule không stop, MCMC raw. | Nhiều vùng `K=0` đúng; quality tăng hoặc giữ nguyên mà số warp/latency giảm. |
| **R2 — học marginal utility** | Dự đoán lợi ích có điều kiện `Δ(j\|S,r)`, lần lượt thêm source khi lợi ích dự kiến dương và đủ bù cost. | Học score từng source độc lập; confidence weighting của GADA nếu có thể chạy; fixed-K; cùng candidate pool. | Learned policy thu hẹp rõ khoảng oracle–heuristic, cải thiện held-out PSNR/LPIPS vượt run variance. |
| **R3 — routing cùng aggregation** | Khi nhiều nguồn bất đồng, quyết định subset và trọng số phải xét chung; visibility và agreement là prior, không dùng softmax thuần. | R2 + mean/weighted fusion; confidence-only; thử bỏ geometry, source–source agreement, cost penalty, adaptive K. | Hơn R2 trên occlusion/specular mà không tăng đáng kể p90 latency. |
| **R4 — training-time routing rồi distill** | Nếu teacher R2/R3 thật sự mạnh, dùng nó làm supervision tại pseudo views hợp lệ để học appearance/geometry trong 3DGS. | Student chỉ train view thật; student distill từ nearest-K teacher; student distill từ routed teacher; mask vs không mask. | Student **render không ảnh nguồn** giữ phần đáng kể gain và có Pareto tốt hơn hybrid; không được suy từ gain teacher. |

R1/R2 là ưu tiên của tuần. R3 chỉ mở nếu R2 chứng minh được gap; R4 chỉ mở nếu teacher thắng thuyết phục và có thời gian. L2R-GS có thể gợi ý nhóm signal/support, nhưng **không tái sử dụng trực tiếp score AURC như utility tái tạo ảnh**: nhãn và objective khác nhau.

## 5. Hướng 2 — diffusion: nhiều intuition, một phép thử giá trị chung

### 5.1 Câu hỏi gốc và đối chứng tối thiểu

Ảnh reference gốc là ảnh chụp thật, không cần diffusion “làm đẹp” đại trà. Cái cần sửa là **candidate sau khi warp tới target** hoặc **vùng target chưa được ảnh thật giải thích**. Dùng Difix pretrained + optional reference để đo ba trường hợp: vùng source hợp lệ/đồng thuận, vùng có source nhưng warp bất đồng, và vùng không có source. Bắt đầu trên vài scene indoor và outdoor đã chọn; so raw MCMC, IBGS, Difix không ref, Difix có ref với cùng source và chi phí.

**Gate D0:** nếu diffusion không vượt một baseline warp/3DGS mạnh trên metric đã chốt, hoặc “đẹp” hơn nhưng làm PSNR/geometry/cross-view consistency xấu đáng kể, không fine-tune diffusion. Việc port refiner Tensara bị âm ngoài trời là lý do để thử có mask/cơ chế dừng, không phải lý do khẳng định mọi diffusion sẽ âm.

### 5.2 Idea được xếp theo giá trị intuition và khả năng có kết quả

| Ưu tiên | Idea | Failure mode và trực giác | Phép thử để tách đóng góp | Rủi ro và điều kiện dừng |
| --- | --- | --- | --- | --- |
| **D1 — cao nhất trong nhánh 2** | **Evidence-gated Difix:** dự đoán vùng nào giữ raw GS, vùng nào giữ pixel reference được warp đáng tin, vùng nào cho diffusion sửa. | Diffusion hữu ích khi không có dữ liệu thật; áp lên mọi pixel có thể sửa sai cả chi tiết đã đúng. | Difix toàn ảnh vs mask theo source support vs mask học từ error trên dev; ablate raw fallback, soft/hard mask; kiểm edge/texture/cross-view. | Nếu chỉ cải thiện FID/ảnh nhìn mà giảm PSNR/LPIPS hoặc lỗi 3D tăng, không tiếp tục. |
| **D2 — nếu D1 có gain** | **Warp-consistent diffusion:** cho Difix tín hiệu nhiều warp *đồng thuận/bất đồng* cùng visibility, ràng buộc giữ pixel có evidence mạnh. | Một ảnh reference đơn lẻ có thể bảo tồn texture sai bề mặt; nhiều warp đồng thuận đáng tin hơn, nhưng cùng sai vẫn có thể xảy ra. | 1 ref vs routed set; có/không disagreement; có/không geometric reprojection loss; source dropout; số ref/latency. | Nếu gain chỉ do thêm ảnh nguồn hoặc biến mất ở target novel, không coi là đóng góp method. |
| **D3 — khi teacher D1/D2 thắng** | **Selective teacher-to-3DGS distillation:** pseudo-view chỉ cho gradient ở vùng diffusion có evidence/consistency đáng tin, vùng khác chỉ dùng ảnh thật hoặc raw. | Difix3D đã distill pseudo-view; đóng góp mới chỉ có nếu *lựa chọn nơi và khi nào distill* ngăn hallucination và tăng student quality. | Student vanilla Difix3D, teacher không mask, mask visibility-only, mask learned/consistency, student chỉ train ảnh thật; eval one-stage riêng. | Nếu student không giữ gain teacher hoặc inference vẫn cần diffusion/source, không claim one-stage. |
| **D4 — thăm dò cuối** | **Uncertainty-aware diffusion budget:** chỉ chạy một-step diffusion ở region/view mà expected gain đủ bù latency. | Tốc độ cũng là objective; nhiều view đã tốt nên bỏ diffusion. | Fixed full-frame Difix, fixed budget theo view, learned decision, same compute; p90 latency và chất lượng. | Nếu policy cost cao tương đương diffusion hoặc Pareto không tốt hơn, dừng. |

**Thứ tự thực tế:** D0 → D1; chỉ khi D1 cho gain mới chạy D2/D3. D4 dùng sau khi có teacher tốt, không phải cách cứu một teacher kém. Difix3D+ đã có cả train-time update lẫn post-render enhancement; một bài “thêm Difix rồi distill” không đủ novelty. Nhấn vào **phân bổ can thiệp theo evidence, mức tổn hại do hallucination, và chất lượng student một-stage**.

### 5.3 Metric cần thêm cho diffusion

Ngoài PSNR/SSIM/LPIPS trên official held-out test, đo: lỗi edge/depth tại vùng có source, source–target warp consistency, mức thay đổi ở pixel có evidence cao, độ nhất quán khi render các target lân cận, và latency/VRAM. Có thể báo thêm perceptual metric nếu xác định rõ tập ảnh và giới hạn thống kê. Một ảnh đẹp riêng lẻ hoặc FID tốt không thay cho reconstruction accuracy trên cùng protocol. **Teacher được phép dùng train source; không đưa ảnh test vào reference, pseudo-target hoặc mask học.**

## 6. Quy tắc chọn hướng sau các probe

| Quan sát | Quyết định |
| --- | --- |
| Oracle routing hơn nearest/coverage đáng kể ở nhiều scene, learned R1/R2 bắt đầu thu hẹp gap | Dồn tài nguyên vào marginal-utility routing. Viết câu chuyện “complementarity + adaptive K/stop” với số image-quality và chi phí; xem R4 để đạt one-stage. |
| Oracle routing có gain chủ yếu indoor, outdoor thường `K=0` | Giữ router kiểu support-aware có fallback; báo cả indoor/outdoor, không claim warp cải thiện mọi nơi. |
| Oracle routing không có gain nhưng D1 sửa được vùng unsupported và không gây hallucination | Chuyển nhánh chính sang evidence-gated diffusion, thử D2/D3 để có method rõ. |
| Cả hai teacher đều mạnh nhưng student distillation yếu | Có thể nghiên cứu hybrid chất lượng–FPS; chưa claim one-stage. Xác định bottleneck truyền tín hiệu vào Gaussian trước khi mở kiến trúc mới. |
| Các gain nằm trong dao động seed hoặc đến từ thay baseline/source count không công bằng | Dừng tuning; giữ chẩn đoán và chọn giả thuyết khác. |

**Tiêu chuẩn “intuition hay + kết quả hơn”:** (i) gap oracle hoặc failure mode đo được trước khi thiết kế method; (ii) ablation cho thấy *đúng cơ chế dự đoán* tạo gain; (iii) hơn baseline mạnh trên test chuẩn, không chỉ trên pilot, với độ lớn vượt dao động run-to-run; (iv) bảng Pareto cho biết phải trả bao nhiêu latency/train/VRAM/storage; (v) claim một-stage chỉ khi student thật sự render không cần source và diffusion. Chưa đặt ngưỡng dB tùy ý trước khi nhìn phương sai trên từng scene.

## 7. Deliverable cuối tuần

- Một bảng oracle/source utility theo scene và support bin: `K=0/1/2/4/8`, nearest, coverage, confidence và oracle xấp xỉ, kèm source budget, runtime và ví dụ region bổ trợ/làm hại.
- Một router R1 tối giản; R2 nếu R0 có gap: PSNR/SSIM/LPIPS per-scene, quality–latency Pareto, ablation marginal-vs-independent, support/availability, stop, và generalization từ development targets sang held-out test.
- Một bảng Difix D0 (có/không ref) và D1 nếu có tín hiệu: so trên cùng target, có masked metrics, cross-view consistency, FPS, VRAM; quyết định rõ mở hay dừng D2/D3.
- Một trang failure analysis gồm ít nhất: outdoor warp gây hại, indoor nhiều source có ích, vùng không có correspondence, depth edge, và các scene có appearance/view dependence. Lưu ảnh/crop cùng manifest và ID view, không cherry-pick.
- Một quyết định nhánh chính dựa trên R0/D0 và test thực, tách rõ **điều đã đo**, **giả thuyết còn mở** và **mức độ novelty so với GADA/Difix3D+**.

## 8. Nguồn đối chiếu

- Bản thảo `L2R-GS_iclr2027_conference.pdf` do nhóm cung cấp, §3.2–3.5 và §4.3–4.4: theorem về ranking dưới partial support, không phải theorem về source selection/fusion PSNR.
- [IBGS, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/c2ad28981782bb62f025d2893791b629-Abstract-Conference.html) — residual từ các ảnh train láng giềng; [code](https://github.com/HoangChuongNguyen/ibgs).
- [GADA, ICML 2026](https://proceedings.mlr.press/v306/lim26c.html) — deformable alignment và confidence weighting giữa nhiều warp.
- [Difix3D+, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_DIFIX3D_Improving_3D_Reconstructions_with_Single-Step_Diffusion_Models_CVPR_2025_paper.html) và [code/model](https://github.com/nv-tlabs/Difix3D) — diffusion một bước, pseudo-view update và post-render enhancement.
- [Generative Sparse-View Gaussian Splatting, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_Generative_Sparse-View_Gaussian_Splatting_CVPR_2025_paper.html) — pseudo-view từ diffusion và consistency.
- [Visibility-guided view selection, Aalto thesis](https://aaltodoc.aalto.fi/items/eb4782f5-85c1-42ea-9774-0e100894887a) và [Coverage Optimization for Camera View Selection, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Coverage_Optimization_for_Camera_View_Selection_CVPR_2026_paper.html): coverage/complementary selection đã có; cần làm rõ khác biệt về marginal utility cho *rendered target region* và adaptive stop.

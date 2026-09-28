# Week 1 plan — refiner đưa vào quá trình train 3DGS

## 1. Mục tiêu và phạm vi

**Câu hỏi nghiên cứu:** Tín hiệu từ ảnh nguồn được warp có thể cải thiện trực tiếp geometry và/hoặc appearance của 3DGS trong lúc train, đồng thời tạo ảnh tốt hơn các baseline image-based hiện có hay không? Sau khi có baseline và phân tích lỗi đáng tin, chọn một cơ chế chính để đầu tư.

Tuần này tập trung vào: **tái hiện baseline → cố định protocol → đo chất lượng và chi phí → chẩn đoán theo vùng → chạy các probe ngắn → chọn giả thuyết có bằng chứng**. Các ý tưởng distill sang 3DGS render thuần được ghi trong backlog để thử sau; không trộn chúng vào kết luận của nhánh train chung.

Không dùng bộ aerial của technical report: bộ dữ liệu đó không available cho nghiên cứu và đối chiếu công khai. Các số leaderboard trong report chỉ là động lực đặt giả thuyết, **không là số baseline của thí nghiệm mới**.

**Định nghĩa output phải ghi rõ trong mọi bảng:** `raw` = render trực tiếp từ Gaussian; `final` = ảnh sau nhánh warp/refinement. Với nhánh train chung, `final` thường vẫn cần ảnh nguồn ở inference. “Train chung” **không đồng nghĩa** với “inference chỉ rasterize một lần”.

## 2. Dataset và hai protocol đánh giá

IBGS và GADA dùng các benchmark NVS phổ biến: Mip-NeRF 360, Tanks and Temples, Deep Blending; GADA và IBGS còn xem xét Shiny. Theo mã và trang dự án của tác giả, chọn:

| Pha | Dataset/scene | Vai trò |
| --- | --- | --- |
| Pilot tối thiểu | Mip-NeRF 360 **bicycle, flowers**; Tanks and Temples **train** | Có ảnh test công khai; gồm cảnh ngoài trời có cấu trúc mảnh/texture, cảnh nhiều hoa lá, và cảnh cấu trúc lớn. Chạy xong mới chọn method. |
| Mở rộng sau pilot | Các scene còn lại của Mip-NeRF 360 và Tanks and Temples; Deep Blending **Dr Johnson, playroom**; Shiny **CD** và scene có specularity | Kiểm tra độ ổn định qua indoor/outdoor, nhiều mức overlap, vật mảnh và hiệu ứng phụ thuộc góc nhìn. Giữ đúng split/preprocessing mà bài được đối chiếu sử dụng. |

**Protocol R — reproduction:** Chạy từng baseline theo split, resolution, camera preprocessing, exposure correction, số bước và script metric của chính tác giả. Báo `paper number / reproduced number / chênh lệch`; lưu commit, cấu hình và lý do chênh lệch. Đây là kiểm tra triển khai, chưa phải bảng xếp hạng công bằng giữa các phương pháp.

**Protocol C — controlled comparison:** Một danh sách ảnh train/val/test, camera, resolution, định nghĩa mask, metric và phần cứng chung. Giữ checkpoint Gaussian và số ảnh train giống nhau trong phép thử module khi có thể. Nếu đổi ngân sách Gaussian/iterations để tối ưu từng phương pháp, báo thêm một cột riêng cho **best-per-method**; không trộn hai kiểu so sánh. Không dùng ảnh test hoặc camera test để chọn tham số hay sinh pseudo target khi train.

**Tách split cẩn thận:** official test chỉ dùng để đánh giá cuối; có validation riêng từ train để chọn checkpoint. Nếu refiner cần ảnh “held-out khỏi 3DGS” để học lỗi thật, chỉ lấy chúng trong training pool, và mọi baseline đối chứng cùng một lượng ảnh được dùng để fit Gaussian. Báo rõ số ảnh dùng để fit Gaussian, học refiner và làm nguồn warp. Target train không được làm source của chính nó. Nếu cuối cùng train lại Gaussian trên toàn bộ training pool, phải chạy đối chứng tương ứng.

## 3. Baseline cần có trước khi claim cải thiện

| Mức | Phương pháp | Làm gì | Trạng thái đối chứng |
| --- | --- | --- | --- |
| Bắt buộc | 3DGS chuẩn và **3DGS-MCMC** | Cùng dữ liệu/camera và cấu hình có log; xuất raw RGB/depth/alpha. MCMC là nền phù hợp để so với recipe của report. | Chạy local, đo chất lượng và toàn bộ chi phí. |
| Bắt buộc | **IBGS** | Dùng repo tác giả: reproduce trước rồi chạy Protocol C. Lưu cả raw base và final. | Chạy local; baseline trực tiếp nhất cho train chung với ảnh nguồn. |
| Bắt buộc | **Post-hoc refiner** dựa trên report | Port sang benchmark công khai; train trên held-out renders của frozen base, cùng source/mask/geometry và split hợp lệ. Ghi mọi thay đổi so với report. | Chạy local; đối chứng cho claim “joint training hơn post-hoc”. Nếu port chưa xong, không claim vượt nó. |
| Đối chiếu có giới hạn | **GADA** | Đối chiếu paper metrics và, nếu tải được, ảnh render tác giả trên **đúng scene/split/resolution**. | Repo tác giả hiện chưa có mã train/eval. Không ghi “reproduced GADA”, không đặt FPS công bố vào cùng cột FPS local. Khi code ra mắt, bổ sung run local. |
| Tùy thời gian, sau các mục trên | PGSR/3DGS variant dùng chung base; Difix3D+ | Bổ sung khi có lý do từ failure analysis và protocol so được. | Không để việc chạy nhiều baseline phụ chiếm thời gian chẩn đoán. |

**Gate cho một run hợp lệ:** render ra đủ ảnh đúng tên/kích thước; camera và mask khớp GT; scorer kiểm tra tay trên vài view; train/eval không dùng nhầm ảnh; không có NaN; tái chạy cùng checkpoint cho cùng output; có log config, commit, seed, thời gian và GPU. Nếu paper number không reproduce, truy nguồn preprocessing/split/metric trước khi chỉnh mô hình mới.

## 4. Measurement contract — số nào phải lưu cho mọi run

- **Chất lượng:** per-scene và per-view PSNR, SSIM, LPIPS (ghi backbone, resolution, màu sRGB/linear, mask); điểm trung bình và độ phân tán qua view. Lưu `raw`, `final`, và gain `final − raw` trên **cùng test view**. Nếu thêm metric perceptual khác, báo riêng, không tự đổi tiêu chí giữa các run.
- **Inference:** GPU/model và resolution; batch size; số ảnh nguồn; cách cache ảnh; warm-up; đồng bộ GPU trước/sau đo; latency mỗi view (median và p90), FPS suy từ thời gian đo, cả **compute-only** và **end-to-end** gồm chọn/đọc nguồn, depth, warp, flow/depth-fix, mạng refinement và ghi output. GADA paper FPS chỉ làm mốc tham chiếu, không là phép đo trên máy mình.
- **Training và footprint:** wall-clock train từng pha và tổng, peak VRAM train/inference, số Gaussian, dung lượng checkpoint, dung lượng ảnh nguồn bắt buộc giữ khi deploy, số tham số của refiner. Báo trade-off chất lượng–latency cùng storage và train cost; không chỉ một con số FPS.
- **Tái hiện:** dataset/split manifest, source selector, intrinsics/extrinsics, image scale, background/mask, exposure correction, seed, số steps, loss weights, commit và command. Lưu JSON/CSV per-view và ảnh output để có thể phân tích lại mà không retrain.
- **Thống kê:** pilot dùng để tìm tín hiệu; quyết định paper-level cần nhiều scene và ít nhất vài seed ở biến thể quan trọng. Không đặt ngưỡng PSNR cứng khi chưa đo run variance và chất lượng reproduce.

## 5. Chẩn đoán: kiểm tra intuition bằng số liệu

Mỗi chẩn đoán phải có **phân bố vùng**, vài ví dụ tốt/xấu từ test view, và phép so tương ứng giữa `raw`/`final`; tránh rút kết luận từ vài crop đẹp.

| Câu hỏi | Biến đo / cách chia vùng | Nếu quan sát đúng, ủng hộ hướng nào? | Cách loại trừ giải thích khác |
| --- | --- | --- | --- |
| Refiner sửa lỗi ở đâu? | Gain và loss theo overlap/source count, source baseline, độ lệch góc, occlusion mask, pixel gần depth edge, texture/gradient. | Gain ở vùng nhiều source hợp lệ → fusion/alignment; gain ở biên có lệch warp → geometry. | So với error của raw tại cùng vùng để tránh nhầm “vùng khó có gain lớn” với hiệu quả thực. |
| Warp sai do depth hay pose/appearance? | Disagreement giữa nhiều warp, Jacobian theo depth, reprojection/flow residual, source depth consistency. | Sai lệch tương ứng hướng epipolar và nhất quán qua nhiều nguồn → depth update. | Hai warp đồng ý vẫn có thể cùng sai; dùng GT của held-out view để kiểm chứng màu và edge alignment; tách exposure/specularity. |
| Attention/chọn source có ích hơn rule không? | Per-source photometric error tại view có GT; oracle chọn source trên **test chỉ để chẩn đoán**; so rule nearest/top-K và learned weights. | Oracle–heuristic gap lớn → học source confidence có tiềm năng. | Oracle dùng GT test **không** được đưa vào training hay báo là method có thể deploy. |
| IBGS có train/test gap không? | Residual gain trên train, val, official test; confidence/gate theo vùng và source visibility. | Gap còn rõ sau khi ghép mức khó tương đương → nghiên cứu distribution-matched training. | Tránh target làm source của chính nó; đối sánh coverage và baseline giữa các tập view. |
| Base hay post-hoc là nút thắt? | Lỗi raw vs final theo high-frequency, thin structure, bóng/specular, ít nguồn, và entropy/độ tin cậy của attention. | Raw/final cùng sai geometry → sửa Gaussian; raw đúng cấu trúc nhưng thiếu texture → refine/fusion. | Xem ảnh depth và view lân cận để loại trừ shortcut kéo texture sai bề mặt. |
| Lợi ích đổi lấy chi phí nào? | PSNR/LPIPS theo latency, VRAM, storage và train time ở từng scene. | Xác định điểm Pareto và mục tiêu tối ưu đáng theo. | Dùng đúng một GPU/protocol cho đường Pareto local. |

## 6. Backlog idea — xếp theo giá trị thông tin và tiềm năng

**Trạng thái:** P0 = probe sau baseline/diagnosis; P1 = đầu tư nếu P0 ủng hộ; P2 = giữ cho nhánh sau. Một idea có thể dừng mà không đồng nghĩa cả hướng nghiên cứu thất bại.

| ID | Ưu tiên | Idea và intuition | Phép thử ngắn / ablation cần có | Bằng chứng để đầu tư tiếp |
| --- | --- | --- | --- | --- |
| A1 | **P0** | **Depth-fix giữa hai warp đưa về loss geometry**: refiner sửa warp rồi bỏ, còn 3DGS tiếp tục render sai depth; residual hai warp có thể cập nhật base. | Trên cùng checkpoint/base: không fix → depth-fix chỉ ở inference → depth-fix dùng làm supervision cho rendered depth/geometry lúc train. Mask vùng flow không tin, occlusion và Jacobian quá nhỏ. | Raw depth/edge và raw RGB tốt hơn trên held-out view; final không giảm; warp disagreement giảm mà không gây ghosting. |
| A2 | **P0** | **Source confidence có cấu trúc**: nearest-K không luôn là source tốt; nguồn nào tốt phụ thuộc visibility, baseline và disagreement. | Thử weighting nhỏ theo hand-crafted confidence, rồi learned source weighting; so nearest-K và rule selection với **cùng K/cùng checkpoint**. | Gain ở vùng oracle–heuristic gap lớn và không làm chậm đáng kể; tránh claim novelty chỉ từ softmax vì GADA đã học confidence weights. |
| A3 | **P0** | **Distribution-matched joint training**: train refiner trên lỗi mà frozen base chưa fit có thể học đúng distribution novel view hơn train trên view Gaussian đã thấy. | IBGS gốc vs refiner học trên held-out base rồi fine-tune joint có kiểm soát; so train/val/test gap với cùng ảnh và budget. | Test gain còn sau khi loại self-source và ghép coverage; raw và final đều được ghi rõ. Gain trong report khi chuyển geometry không chứng minh giả thuyết này. |
| A4 | **P1** | **Epipolar depth attention thay offset 2D tự do**: thử vài depth quanh depth render, warp nhiều nguồn và phân phối trọng số trên depth candidate; dùng depth đề xuất để cập nhật Gaussian. | Baseline A1; A1 + depth candidates không attention; A1 + attention; so output chỉ sửa final với output cập nhật raw. | Giữ được chi tiết tại thin structures/depth edges và giảm lỗi geometry; so sát IDESplat (epipolar depth attention) và GADA (offset/confidence). |
| A5 | **P1** | **Joint depth–source attention**: chọn cặp `(depth, source)` cùng lúc để tránh nguồn bị occlusion thắng chỉ vì texture gần giống. | Factorized attention vs joint attention; thêm visibility và cross-source agreement; kiểm tra entropy/calibration của confidence. | Hơn A2/A4 ở vùng occlusion và baseline lớn; cost/memory vẫn hợp lý. |
| A6 | **P1** | **Support-adaptive routing**: vùng nhiều nguồn đồng thuận để geometry-driven correction; vùng ít/không nguồn giữ raw render hoặc dùng residual thận trọng. | Gate theo support/confidence so với gate học thuần và dropout/mask; đánh giá chất lượng theo support bin. | Giảm lỗi nghiêm trọng ở vùng ít overlap mà không mất gain vùng nhiều evidence; câu chuyện failure mode rõ. |
| A7 | **P1** | **Residual attention vào Gaussian/local patch trong lúc train**: dùng attention cục bộ để đặt gradient/densification vào vùng vừa sai raw vừa có warp đáng tin. | So với loss reweighting đơn giản và densification chuẩn; đo số Gaussian, chi tiết và train time. | Tăng raw quality với footprint hợp lý. Tránh global self-attention trên hàng triệu Gaussian. |
| B1 | **P2** | **Warp-refiner teacher → 3DGS thuần** ở novel/pseudo views; inference chỉ rasterize, chấp nhận tăng train cost. | Sau khi có teacher A tốt, thử student có/không confidence mask; kiểm tra student lấy được bao nhiêu phần gain teacher. | Cải thiện raw render nhiều scene, giảm latency/storage so teacher, không có leakage/ghosting. Đây là một nhánh bài toán riêng. |
| B2 | **P2** | **Difix3D+ teacher** là đối chứng generative cho B1, không mặc định là method chính. | So trên cùng train/test và báo thêm consistency/hallucination bên cạnh perceptual metric. | Chỉ theo nếu teacher thực sự tốt và không phá chi tiết/consistency. |

**Thứ tự chạy được đề xuất:** baselines → chẩn đoán → A1/A2/A3 là probe nhỏ → chọn **một** trong A4/A5/A6 để xây method. Chỉ mở B1 sau khi đã hiểu chất lượng và lỗi của teacher A. Kết quả âm cũng được lưu với nguyên nhân và vùng ảnh bị ảnh hưởng.

## 7. Điều kiện quyết định cuối giai đoạn

1. Baseline **3DGS-MCMC, IBGS, post-hoc refiner** đã có run tái hiện hợp lệ và Protocol C trên các scene pilot; GADA được đánh dấu riêng là published outputs/numbers nếu vẫn thiếu code.
2. Có bảng metric theo scene/view cho raw và final; bảng chi phí đầy đủ; tập failure cases và đồ thị gain theo coverage/warp disagreement/depth edge. Chênh lệch giữa kết quả và paper được giải thích theo split/preprocessing/compute, không âm thầm thay protocol.
3. Ít nhất một intuition có **probe với control hợp lệ**: A1 chứng minh lợi ích trên raw, A2 chỉ ra lợi ích thực của source selection, hoặc A3 xác nhận gap và cách giảm gap. Nếu chưa có tín hiệu, chọn hypothesis khác dựa trên failure analysis, không mở rộng model theo cảm tính.
4. Claim “hơn SOTA” chỉ dùng khi **cùng benchmark, cùng split/metric**, so với IBGS và những output GADA có thể đối chiếu; báo cả latency/resource và giới hạn tái hiện. Kết quả trên vài scene pilot là cơ sở chọn phương pháp, chưa là bằng chứng SOTA tổng quát.

## 8. Tài liệu và code đối chứng

- Technical report của Tensara do nhóm cung cấp: `Track01_Tensara_TechnicalReport.pdf`, đặc biệt §4.6, Eq. (3), bảng ablation refiner và các chi phí inference. Dùng để port baseline, không dùng điểm aerial làm benchmark công khai.
- [IBGS — NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/c2ad28981782bb62f025d2893791b629-Abstract-Conference.html); [code và cách train/eval](https://github.com/HoangChuongNguyen/ibgs).
- [GADA — trang tác giả](https://siw00-lim.github.io/GADA-Project-Page/); [repo và trạng thái phát hành code](https://github.com/siw00-lim/GADA).
- [3D Gaussian Splatting — code tác giả](https://github.com/graphdeco-inria/gaussian-splatting); [3DGS-MCMC — paper](https://arxiv.org/abs/2404.09591).
- [IDESplat — epipolar depth attention](https://arxiv.org/abs/2601.03824); [Difix3D+ — train-time distillation và post-render enhancer](https://arxiv.org/abs/2503.01774).

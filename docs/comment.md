# Nhận xét architecture.md: thiết kế nghiên cứu và các lỗ hổng học thuật

## Review lần 2 — bản architecture.md đã chỉnh sửa

**Kết luận:** bản mới đã giải quyết phần lớn vấn đề của review đầu. Các nhận xét bên dưới mục “Ngày đánh giá” được giữ lại làm lịch sử, không nên hiểu là tất cả vẫn còn đúng với bản hiện tại. Chưa có triển khai hoặc kết quả để xác nhận protocol đã được thực hiện đúng.

### Những điểm đã được xử lý tốt

- Bỏ V-blending, giữ V cho chọn cấu hình/early stopping và phân biệt screening với confirmation.
- Đổi NEW thành EXT, giới hạn claim novelty và bổ sung câu hỏi nghiên cứu chính.
- Bổ sung temperature, reduction theo query, soft targets và yêu cầu giữ nguyên list khi tính ListNet.
- Tách B3-repro/B3-ctrl; sửa mô tả Identity, max_len và dev size của baseline gốc.
- Bổ sung audit constraint extractor, random-removal control và audit suspected label issues.
- Nhận diện leakage khi tái tạo OOF sau cleaning và yêu cầu xử lý bên trong outer fold.
- Chạy reference/finalists cùng ba seed; không coi kết quả trên V là xác nhận độc lập.
- Có priority table và cho phép cắt cả hướng mở rộng.

Đã đối chiếu code tại commit `7916cdf`: [train.py](https://github.com/amazon-science/esci-data/blob/7916cdf/ranking/train.py) khớp mô tả cấu hình B3-repro. [prepare_trec_eval_files.py](https://github.com/amazon-science/esci-data/blob/7916cdf/ranking/prepare_trec_eval_files.py) có mapping E=4, S=2, C=3, I=1 và câu lệnh gain mapping trong comment như tài liệu mô tả. Điều này xác nhận nội dung code, chưa chứng minh đã tái lập được điểm công bố.

### Các điểm cần sửa tiếp

**1. Lịch xem Test vẫn mâu thuẫn với quy tắc khóa Test.**

§3.2 và §3.6 yêu cầu khóa danh sách so sánh ở G3 trước mọi Test score, nhưng G2 đã yêu cầu so B3-repro với điểm công bố. Nếu thực hiện trên official Test ở W2 thì vi phạm quy tắc đang viết. Nên chuyển so sánh điểm công bố sang W4; G2 kiểm tra cấu hình, metric và kết quả dev. Nếu muốn ngoại lệ cho reproduction thì phải ghi rõ ngoại lệ và cấm dùng kết quả đó để chọn các biến thể.

**2. Calibration “cho mọi A3 variant” chưa được định nghĩa hợp lệ.**

MSE xuất một scalar, không có vector P(E,S,C,I), nên không thể trực tiếp tính multiclass NLL/Brier/ECE. ListNet-only cũng cần chốt head: scalar head thì không có class probabilities; four-class head thì loss chỉ ràng buộc expected gain, không xác định duy nhất phân bố class.

Sửa thành: calibration chỉ áp dụng cho các cấu hình có head phân loại bốn lớp được định nghĩa rõ; MSE đánh dấu N/A. Không tự chuyển scalar score thành probability bốn lớp nếu chưa định nghĩa thêm phương pháp. Quy định ECE dùng confidence/classwise, số bins và cách weighting.

**3. A7 cần tách ảnh hưởng của backbone, training size và ensemble.**

A4 dùng DeBERTa train toàn T, còn Stage 2 cuối dùng hai model train trên nửa T và average predictions. Nếu dùng full-T score cho hàng trước LightGBM và average-fold score cho hàng sau, Δ không chỉ đo đóng góp của LightGBM.

Cần có hàng “2-fold DeBERTa ensemble, chưa LightGBM”, rồi dùng chính bộ predictions đó cho các hàng có/không LightGBM và constraint features. Cố định noise policy trong từng so sánh. OOF train features từ một fold model nhưng V/Test features từ trung bình hai model cũng có distribution shift; không phải leakage, nhưng nên đo và báo cáo. Full-T model so với fold ensemble phải là một so sánh riêng nếu cả hai được sử dụng.

**4. Thiết kế hiện tại chưa đo được interaction λ × τ.**

EXT 1 nói thay từng yếu tố từ (τ=0.1, λ=1), đồng thời khẳng định sẽ báo cáo interaction. Cách này chỉ có ba tổ hợp; muốn ước lượng interaction 2×2 cần thêm (τ=1, λ=0.5). Hoặc bỏ claim đo interaction và chỉ báo cáo sensitivity có điều kiện.

Không nên kết luận τ=0.1 chắc chắn tốt hơn: nó làm E chiếm ưu thế rất mạnh khi list có Exact, có thể giảm tín hiệu phân biệt các lớp còn lại. Đây vẫn là giả thuyết thực nghiệm.

**5. Fallback giảm max_len theo query có thể làm ablation mất công bằng.**

Nếu cấu hình tốn bộ nhớ hơn tự giảm max_len cho một số query, loss/backbone và lượng văn bản đầu vào cùng thay đổi. Nên chốt truncation policy trước, dùng chung giữa các variant được so sánh và log tỷ lệ bị cắt. Ưu tiên giảm số query trong microbatch hoặc dùng checkpointing trước khi thay đổi input. Nếu buộc đổi max_len thì ghi rõ đó là cấu hình khác.

**6. Primary hypothesis chưa gắn đủ chặt với metric và độ lớn mẫu.**

Main question nói về E-vs-S nhưng primary metric vẫn là nDCG. Cần ghi rõ primary claim được quyết định bằng nDCG hay E-before-S accuracy; metric còn lại là secondary. E-before-S cần định nghĩa macro theo query hay micro theo pair, ties và query không có cả E lẫn S.

Ngưỡng “<10% không thể làm kết quả chính” thiếu cơ sở: subgroup nhỏ vẫn có thể là đối tượng nghiên cứu chính nếu đủ số mẫu và power. Nên dùng số query thực tế, effect size mục tiêu, alpha và power. MDE từ chênh lệch giữa các seed baseline chỉ là proxy; chưa thay thế phân bố paired differences giữa phương pháp và baseline.

**7. Bảng ưu tiên chưa bảo vệ các dependency của giả thuyết chính.**

Nếu constraint features là primary hypothesis, LightGBM và OOF trở thành Must, không thể để Stage 2 ở Extend rồi cắt độc lập. Nếu noise handling là primary thì OOF probabilities và audit cũng phải là Must. Cần ghi quy tắc đưa toàn bộ dependency của primary hypothesis vào Must.

Một số đoạn còn lệch với priority table: phần mapping nói noise handling là core, còn cuối tài liệu cho phép nó là Stretch; checklist vẫn yêu cầu audit/calibration cho thành phần có thể đã bị cắt. Nên thêm điều kiện “nếu chạy thành phần này”.

**8. Quy tắc gộp ba seed để tính CI/p-value chưa rõ.**

Nên định nghĩa trước: với mỗi query, lấy trung bình paired Δ trên ba seed; bootstrap query trên các Δ đã gộp; báo cáo thêm từng seed và độ biến thiên. Đây vẫn là uncertainty theo query, có điều kiện trên ba lần train, không phải CI bao phủ toàn bộ training randomness. Không gộp 3×N quan sát như các mẫu độc lập.

### Điểm nhỏ còn tồn tại

- §1.2 vẫn nói mỗi baseline “isolates one capability”, trong khi input, pretraining và supervision cùng khác nhau. Nên đổi thành so sánh các nhóm phương pháp, không suy diễn nhân quả từng capability.
- Rule negation vẫn nói gặp negated term là violated, dù đoạn sau đã thừa nhận phản ví dụ. Cần sửa đặc tả xử lý; thêm cảnh báo chưa sửa được rule.
- Candidate-list robustness chưa thành thí nghiệm cụ thể. Có thể để exploratory nếu thiếu thời gian, nhưng không nên claim đã chứng minh khả năng tổng quát hóa của list-context features.
- B3-ctrl dùng group sampler nên cần chốt batch theo query/pair, effective batch size và warmup theo optimizer steps. Sao chép warmup 5.000 steps từ baseline gốc không bảo đảm lịch học tương đương.

**Ưu tiên trước khi code:** sửa Test schedule, định nghĩa output head/metric của A3, khóa nguồn predictions cho từng hàng A7 và bảo vệ dependency của primary hypothesis. Các vấn đề còn lại chủ yếu là hoàn thiện protocol; không cần thiết kế lại toàn bộ project.

---

## Review lần 1 — lưu để đối chiếu lịch sử

Ngày đánh giá: 10/11/2026  
Tài liệu được đánh giá: [architecture.md](./architecture.md)

## 1. Đánh giá tổng thể

Plan có nền tảng tốt cho đồ án môn học, nhưng hiện tham vọng về số kỹ thuật lớn hơn độ chặt của thiết kế thực nghiệm. Điểm dễ bị phản biện nhất là: **kết quả tốt lên có thực sự chứng minh giả thuyết nghiên cứu, hay chỉ là hệ quả của chọn cấu hình, thêm dữ liệu và khai thác đặc điểm benchmark?**

Tại thời điểm đánh giá, repo chỉ có tài liệu kế hoạch và README, chưa có code, dữ liệu hay kết quả để kiểm chứng. Proposal và hướng dẫn môn học được nhắc đến cũng chưa có trong repo. Vì vậy, đây là đánh giá thiết kế, không phải kết luận về chất lượng triển khai hoặc mức độ đáp ứng đầy đủ rubric môn học.

Các điểm tốt nên giữ:

- Phạm vi Task 1 và reranking candidate lists được xác định rõ.
- Có baseline, chia dữ liệu theo query và dự kiến dùng OOF.
- Có ablation, lưu config/predictions và kế hoạch tái lập kết quả.
- Có error analysis và chấp nhận kết quả âm hoặc không cải thiện.

Các vấn đề dưới đây được sắp theo mức độ cần ưu tiên sửa.

## 2. Validation đang kiêm quá nhiều vai trò

**Vị trí:** §3.2–3.3, NEW 4, A6.

V được dùng để chọn input, loss, backbone và cách xử lý noise; tune và early-stop LightGBM; sau đó trở thành dữ liệu train LightGBM trong V-blending.

Câu “V is then no longer used for other decisions” không xóa được việc V đã ảnh hưởng đến các quyết định trước đó. GroupKFold trên V cũng không tự loại bỏ ảnh hưởng của việc Stage 1 đã được chọn dựa trên toàn bộ V.

**Hệ quả học thuật:** kết quả CV trên V có thể lạc quan. Test vẫn có thể đánh giá cuối cùng hợp lệ nếu được giữ kín, nhưng các bảng trên V không phải bằng chứng xác nhận độc lập sau quá trình chọn cấu hình.

**Đề xuất sửa:**

- Tách rõ `train / dev chọn cấu hình / blend train Stage 2 / test` nếu chọn blending.
- Nếu thiếu dữ liệu hoặc compute, chọn một scheme stacking làm chính thay vì duy trì cả hai trong core scope.
- Phân biệt kết quả screening trên dev với kết quả xác nhận trên test.
- Chốt trước các cấu hình và phép so sánh được phép chạy trên test. Leave-one-group-out trên test chỉ nên là phân tích đã định trước, không dùng để chọn lại hệ thống.

Ngoài ra, OOF stacking train Stage 2 trên khoảng 90% training queries, còn blending chỉ trên khoảng 10%. So sánh này đo đánh đổi của hai pipeline, chưa tách riêng tác động của scheme khỏi lượng dữ liệu train Stage 2. Nếu muốn kết luận riêng về scheme, cần kiểm soát hoặc phân tích thêm yếu tố kích thước dữ liệu.

## 3. Hybrid loss chưa được định nghĩa đủ chặt

**Vị trí:** NEW 1, A3, phần noise-aware labels.

Target ListNet được định nghĩa là `softmax(gain / τ)`, nhưng plan chưa chốt hoặc thiết kế thí nghiệm cho `τ`.

Ví dụ, nếu `τ = 1`, một list có 1 Exact và 19 Irrelevant sẽ cho xác suất target của Exact:

```text
exp(1) / (exp(1) + 19 × exp(0)) ≈ 0.125
```

Khoảng 87,5% tổng probability mass vẫn nằm trên các sản phẩm Irrelevant. Điều này không tự làm loss sai, nhưng cho thấy tín hiệu ưu tiên Exact khá loãng. Predicted expected gain cũng bị chặn trong `[0, 1]`, nên temperature là yếu tố quan trọng; chỉ thử λ chưa đủ để đánh giá thiết kế này.

Các điểm chưa rõ khác:

- Gradient accumulation thông thường không tương đương ListNet toàn query nếu chia một list thành nhiều microbatch rồi tính softmax riêng.
- CE được lấy trung bình theo pair hay theo query? ListNet được lấy trung bình thế nào? Hai cách reduction có thể làm trọng số hiệu dụng thay đổi theo độ dài list.
- Khi dùng soft labels, target gain của ListNet lấy từ hard label hay expected gain của soft label?
- Khi xóa noisy pairs, list và normalization thay đổi ra sao? Xử lý query mất hết hoặc gần hết tín hiệu relevance như thế nào?

**Đề xuất sửa:** định nghĩa temperature, loss reduction, soft targets và cách tính loss trên toàn list trước khi chạy. Kiểm tra loss và gradient bằng các ví dụ nhỏ tính được bằng tay.

Khẳng định “CE keeps calibrated probabilities” cũng quá mạnh. Train bằng CE không tự bảo đảm probability đã calibrated. Nên đổi thành giả thuyết cần kiểm tra; nếu calibration là lập luận quan trọng cho pipeline, bổ sung phép đo thích hợp như NLL/Brier score và phân tích calibration.

## 4. Claim về bốn điểm NEW mạnh hơn bằng chứng về novelty

**Vị trí:** §1.3, §4.4.

Lập luận rằng những bài đã đọc chưa báo cáo một biến thể chưa đủ để khẳng định phương pháp mới trong literature.

- Validation blending nên được trình bày là lựa chọn tiết kiệm compute.
- Remove vs soften là câu hỏi thực nghiệm có giá trị.
- Hybrid loss và constraint features có thể là đóng góp triển khai hoặc ứng dụng, nhưng plan chưa chứng minh novelty phương pháp.

Đối với đồ án môn học, không cần cố chứng minh bốn thuật toán mới. Một định vị đáng tin hơn là:

> Nghiên cứu có kiểm soát về tác động của tín hiệu ràng buộc thuộc tính đối với reranking ESCI, dưới ngân sách tính toán giới hạn.

**Đề xuất sửa:** chọn một câu hỏi nghiên cứu chính, chẳng hạn:

> Constraint features có cải thiện E–S discrimination trên một cross-encoder mạnh không, và cải thiện ở nhóm query nào?

Các kỹ thuật còn lại phục vụ câu hỏi này. Phân biệt rõ đóng góp phương pháp, đóng góp thực nghiệm và lựa chọn kỹ thuật để giảm chi phí. Nếu giữ claim novelty, cần mở rộng literature review trước khi kết luận.

## 5. Baseline reproduction đang khác cấu hình gốc

**Vị trí:** method card B3, A1–A2, Gate G2.

Method card B3 mô tả MSE trên sigmoid output. Trong khi [train.py của Amazon](https://github.com/amazon-science/esci-data/blob/main/ranking/train.py) cấu hình `Identity`, `max_length=512`, và mặc định giữ lại 200 dev queries. Plan dùng V = 10% và dự kiến thử độ dài 128/256.

Cần phân biệt hai loại baseline:

| Loại | Mục đích |
|---|---|
| Reproduction | Tái lập đúng cấu hình và protocol gốc để kiểm tra khả năng đối chiếu số công bố |
| Controlled baseline | Chạy baseline theo protocol chung của nhóm để so sánh công bằng với các biến thể |

Không nên dùng điều kiện “phải gần 0.857” làm gate cứng khi protocol chưa tương đương. Con số khác không tự chứng minh code sai. Nên pin commit của code gốc và ghi rõ các khác biệt.

B1/B2/B3 cũng khác nhau đồng thời ở input fields, pretraining và supervised training. Chúng là baseline thực dụng tốt, nhưng chưa đủ để kết luận riêng rằng token interaction hoặc semantic matching gây ra toàn bộ mức chênh lệch.

Phần metric cần chốt công thức DCG, gain mapping, cách xử lý ties và query có IDCG bằng 0. Claim về gain mapping bị đảo trong evaluation script cần có đường dẫn và phiên bản code cụ thể; không nên chỉ ghi nhận định trong tài liệu.

## 6. Statistical testing chưa xử lý đúng nguồn bất định chính

**Vị trí:** §3.5, §5.

Chạy ba seed cho reference nhưng có thể chỉ một seed cho variant, rồi so Δ với `2 × reference std`, chưa phải cách đánh giá chắc chắn biến thể tốt hơn.

- Variant cũng có phương sai riêng.
- Paired bootstrap theo query phản ánh bất định theo query với các model đã train; không bao quát toàn bộ bất định của quá trình training.
- Chọn winner qua nhiều vòng A2 → A3 → A5 trên cùng V tạo selection bias. Holm trong từng family không xử lý hết quá trình chọn thích nghi này.
- Nhiều p-value từ các phép kiểm định trên cùng dữ liệu không phải nhiều bằng chứng độc lập.

**Đề xuất sửa:**

1. Dùng một seed cho screening.
2. Chạy lại reference và các finalist bằng cùng bộ ba seed, báo cáo Δ theo từng seed.
3. Chọn một phép kiểm định chính, đồng thời báo cáo effect size và CI.
4. Xác nhận các so sánh chính trên test sau khi chốt cấu hình.
5. Với subgroup như negation, luôn báo cáo số query và CI riêng.

“Không significant” chưa đủ để kết luận hai scheme tương đương. Nếu muốn chứng minh blending gần tương đương OOF, cần xác định trước mức suy giảm có thể chấp nhận và thiết kế đánh giá tương ứng.

## 7. Constraint extractor vừa tạo feature vừa xác định nhóm đánh giá

**Vị trí:** NEW 2, E5.

Cùng một bộ rule được dùng để tạo feature cải thiện ranking và gắn tag xác định nhóm negation/number queries nhằm chứng minh cải thiện.

Đây không nhất thiết là label leakage, nhưng có nguy cơ chỉ đánh giá trên những mẫu rule nhận ra, bỏ qua các trường hợp rule thất bại.

Ví dụ lỗi cần xem xét:

- Query `without front cover`, product ghi `no front cover`: có chuỗi `front cover` nhưng không vi phạm.
- Brand trong title có thể là brand thiết bị tương thích, không phải hãng sản xuất phụ kiện.
- Một product có thể liệt kê nhiều model hoặc nhiều kích thước tương thích.
- Không tìm thấy một thuộc tính không đồng nghĩa thuộc tính đó bị vi phạm.

**Đề xuất sửa:** tạo một tập audit nhỏ, lấy mẫu trước khi xem kết quả, có người gán nhãn độc lập về constraint và match/conflict/unknown. Báo cáo coverage và chất lượng extractor. Khi đó mới phân biệt được lợi ích của rule với lỗi của rule.

Tag tự động vẫn có thể dùng cho phân tích diện rộng, nhưng cần ghi rõ đó là subgroup do detector xác định, không phải toàn bộ query thuộc hiện tượng ngôn ngữ đó.

## 8. Nhãn bị model phản đối chưa chắc là nhãn sai

**Vị trí:** NEW 3, A5, §3.3.

OOF tránh việc model dự đoán lại mẫu đã học, nhưng không chứng minh model đúng hơn annotator. Đặc biệt E/S là ranh giới khó: pruning có thể xóa những ví dụ quan trọng nhất.

**Đề xuất bổ sung:**

- Audit thủ công một mẫu flagged và một mẫu không flagged.
- Control xóa ngẫu nhiên cùng tỷ lệ; cân bằng theo class nếu cần để tách ảnh hưởng của thay đổi phân bố class.
- Thống kê class, query type và số query bị ảnh hưởng bởi pruning.
- Gọi các mẫu flagged là “suspected label issues” trước khi xác minh.

Cần làm rõ thêm: sau khi retrain bằng cleaned labels, Stage 2 nhận OOF predictions từ model trước hay sau cleaning? Nếu khác pipeline sinh score lúc inference, có distribution mismatch; nếu tạo lại OOF thì chi phí tăng.

Nếu tạo lại OOF sau cleaning, mọi bước sử dụng nhãn để tạo cleaned/soft labels cần được xem xét trong phạm vi fold. Không nên mặc định rằng một vòng OOF ban đầu làm mọi lần tái sử dụng tiếp theo tự động không có leakage.

## 9. Loại bỏ vài artifact feature chưa chứng minh được khả năng tổng quát hóa

**Vị trí:** Stage 2 method card, excluded-feature policy, A6.

Loại product-ID artifacts là tốt, nhưng pipeline vẫn dùng list size, brand majority và group statistics. Các feature này có thể hữu ích thật, cũng có thể tận dụng cách candidate lists được tạo.

Khẳng định “labels depend on the list” cần thận trọng: list context giúp dự đoán nhãn không đồng nghĩa định nghĩa relevance phụ thuộc list. Khẳng định mỗi list có ít nhất một Exact cũng cần kiểm tra bằng EDA trước khi dùng làm cơ sở lập luận.

**Đề xuất kiểm tra robustness:**

- Giữ nguyên query–product, thay đổi một phần distractor candidates.
- Đo thứ hạng tương đối của các sản phẩm được giữ lại thay đổi bao nhiêu.
- Báo cáo overlap sản phẩm và query gần trùng giữa các split.

Overlap sản phẩm không tự động là leakage, nhưng giới hạn claim về khả năng xử lý sản phẩm mới.

Kết luận phải giới hạn ở reranking các candidate lists được cung cấp. [Dataset chính thức](https://github.com/amazon-science/esci-data) cũng định nghĩa Task 1 theo phạm vi đó; kết quả không trực tiếp chứng minh chất lượng tìm kiếm toàn catalog.

## 10. Scope năm tuần chưa có phương án thu gọn thực sự

**Vị trí:** §4.3, §6, Risks and fallbacks.

Plan dự kiến khoảng 30 MiniLM runs, nhiều DeBERTa runs, OOF, rule engine, statistics, demo và báo cáo; nhưng lại quy định core A1–A7 “never cut”. Điều này khiến fallback kém hiệu lực.

A7 cũng không đương nhiên chỉ là ghép predictions: mỗi cấu hình tích lũy phải có checkpoint/predictions tương ứng, và mỗi ablation Stage 2 cần fit lại model theo feature set đó.

Nên tính ngân sách bằng GPU-hours, thời gian inference và số lượt retrain thực tế, thay vì chỉ đếm runs. Phân biệt thí nghiệm tái sử dụng predictions với thí nghiệm bắt buộc sinh predictions mới.

**Scope đề xuất:**

| Mức ưu tiên | Nội dung |
|---|---|
| Bắt buộc | Metric đúng, split sạch, B1–B3, structured input, một đóng góp chính, error analysis |
| Xác nhận | Baseline và finalist cùng ba seed, đánh giá test sau khi chốt |
| Mở rộng | Một backbone DeBERTa để kiểm tra kết quả có chuyển sang model mạnh hơn không |
| Stretch goal | Noise cleaning, blending-vs-OOF, hybrid loss nếu không chọn nó làm đóng góp chính |

Final ladder nên phản ánh kết quả thực tế. Không nên mặc định mọi bước đều cải thiện hoặc bắt buộc đưa mọi thành phần đề xuất vào hệ thống cuối cùng.

## 11. Các chỉnh sửa nên làm trước khi triển khai

1. Chốt một câu hỏi nghiên cứu chính và cách đo để trả lời nó.
2. Viết lại protocol, tách vai trò dev/blend/test và xác định rõ đâu là screening, đâu là xác nhận.
3. Chuẩn hóa reproduction baseline, metric và nguồn tham chiếu theo phiên bản cụ thể.
4. Hoàn thiện định nghĩa hybrid loss nếu giữ nó trong scope.
5. Thiết kế audit độc lập cho constraint extractor và suspected label issues.
6. Dành ngân sách nhiều seed cho các so sánh chính, thay vì trải rộng trên quá nhiều biến thể.
7. Thêm kiểm tra robustness với candidate-list context.
8. Điều chỉnh lịch và fallback để có thể bỏ cả một hướng mở rộng khi thiếu thời gian.

## 12. Kết luận

Plan đủ tiềm năng thành đồ án tốt, nhưng nên sửa protocol trước khi viết training pipeline. Ba việc có giá trị nhất là **tách vai trò validation, thu hẹp câu hỏi nghiên cứu, và thiết kế so sánh công bằng cho đóng góp chính**.

Làm ít thí nghiệm hơn nhưng trả lời được “tại sao tốt hơn, chắc đến đâu, và đúng trong phạm vi nào” sẽ thuyết phục hơn một pipeline nhiều thành phần.

## Nguồn đã đối chiếu

- [Amazon ESCI dataset và định nghĩa các task](https://github.com/amazon-science/esci-data).
- [Code training baseline Task 1 của Amazon](https://github.com/amazon-science/esci-data/blob/main/ranking/train.py).

Các nguồn trên được dùng để đối chiếu phạm vi task và cấu hình baseline. Nhận xét này chưa phải một literature review đầy đủ, cũng chưa xác minh toàn bộ các con số và claim novelty của những bài được dẫn trong architecture.md.

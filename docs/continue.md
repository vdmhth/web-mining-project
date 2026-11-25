# Continue: thí nghiệm bổ sung trước Gate G4

Ngày lập: 22/11/2026 · Hạn chót: **G4 6/12** (đóng băng kết quả), thuyết trình tuần 7/12

> **Cập nhật tối 22/11:** đã có DeBERTa-large 5-fold OOF và hệ thống final mới (LightGBM lambdarank trên base + large CE, test 0.898). Các mục mới (R1–R6: độ chặt chẽ; N1–N6: mở rộng) và trạng thái của S1–S11 nằm ở **Phần 2** cuối file. Thứ tự ưu tiên hiện tại theo Phần 2.

## Bối cảnh

Đã xong D1 (input, backbone), D2 (loss trên MiniLM title, ce4 trên DeBERTa), D3 (LightGBM trên OOF 5-fold DeBERTa-base) và các thí nghiệm Extra (DeBERTa-large, bge-reranker, blend). Hệ thống final hiện tại là `d3_lgbm_multiclass`: test 0.8921, val 0.8947.

Đối chiếu với `docs/architecture.md` cho thấy năm khoảng trống:

1. **D2 bị lẫn yếu tố cách tạo batch.** MSE và ce4 dùng batch ngẫu nhiên theo cặp, còn ListNet dùng batch theo query và cắt mỗi query thành các list ≤ 16 sản phẩm. Plan yêu cầu mọi loss dùng cùng sampler và **không cắt list**.
2. **Hybrid loss (EXT 1) và noise handling (EXT 3) chưa chạy.**
3. **Constraint features nằm chung nhóm `lexical`**, nên ablation chưa trả lời được câu hỏi "constraint features có giúp không".
4. **Train và val/test của Stage 2 khác nguồn score.** Feature CE của train lấy từ một fold model, còn val/test lấy từ trung bình 5 fold model.
5. **Còn thiếu calibration, MDE, seed thứ 3, và bảng ladder A7.**

File này liệt kê các việc bổ sung, xếp theo chi phí. Không làm lại những gì đã ổn.

## Nguyên tắc giữ nguyên

- **Mọi quyết định đều dựa trên val.** Test của các run mới được predict và lưu blind. Test chỉ được chấm trong `10_report` sau khi danh sách run bổ sung đã chốt (**29/11**).
- Trong `10_report`, các run mới thuộc stage **`Supplement`**. Test của các run cũ đã bị xem trước đó, nên phần Supplement được ghi rõ là bổ sung sau và không thay hệ thống final, trừ khi quy tắc ở mục "Quy tắc quyết định" cho phép.
- **Seed: 42, 43, 44.** Nhiễu seed đã đo được là DeBERTa main MSE s43 so với s42: test −0.0014, CI [−0.0025, −0.0001]. Vì vậy mọi Δ ≤ 0.002 chỉ chạy 1 seed sẽ được gọi là "chưa phân biệt được với nhiễu seed".
- Mỗi run ghi `<run>.json` và `scores/<run>_{val,test}.parquet` qua `RunLog` như hiện tại.

## Bảng tổng hợp

| ID | Việc | Ưu tiên | Tài nguyên | Ước lượng | Phụ trách | Hạn |
|---|---|---|---|---|---|---|
| S1 | Tách nhóm `constraint` trong ablation 08/09 | 1 | CPU | 2 h | Cong | 25/11 |
| S2 | Stage 2 theo từng fold rồi lấy trung bình | 1 | CPU | 2 h | Cong | 26/11 |
| S3 | Calibration (NLL, Brier, ECE) | 1 | CPU | 1 h | An | 25/11 |
| S4 | MDE, bảng nhiễu seed, bảng ladder A7 | 1 | CPU | 2 h | An | 27/11 |
| S5 | Audit nhãn negation khoảng 120 cặp | 1 | người | 3 h | An + Thao | 27/11 |
| S6 | Sửa D2: cùng sampler, không cắt list, 3 seed | 1 | GPU MiniLM | khoảng 2 h | Thao | 26/11 |
| S7 | Hybrid ce4 + λ·ListNet (EXT 1) | 2 | GPU MiniLM | khoảng 1.5 h | Thao | 27/11 |
| S8 | Noise handling trên MiniLM (EXT 3) | 2 | GPU MiniLM | khoảng 3 h | Duy | 29/11 |
| S9 | Chuyển loss thắng sang DeBERTa main | 3 | GPU DeBERTa | 2.5 h | Duy | 2/12 |
| S10 | DeBERTa main ce4 seed 43 | 3 | GPU DeBERTa | 2.5 h | Duy | 2/12 |
| S11 | Cập nhật architecture.md và chương phương pháp | 1 | không | 2 h | All | 4/12 |

Tổng GPU khoảng 6.5 giờ cho MiniLM và tối đa 5 giờ cho DeBERTa, nằm trong quota một tuần của 1–2 tài khoản Kaggle.

---

## S1. Tách nhóm `constraint` trong ablation (CPU)

**Vì sao.** Đây là câu hỏi chính của đề tài về E-vs-S và constraint. Hiện `num_match, unit_match, color_match, brand_in_query, neg_violation` nằm chung nhóm `lexical` với BM25. Ablation `no_lexical` (val −0.0006, không significant) vì thế không cho biết riêng phần đóng góp của constraint.

**Làm gì.**
- `08_features`: trong `GROUPS`, chuyển 5 feature trên sang nhóm mới `'constraint'`. Thêm luôn `q_has_num, q_has_unit, q_has_color, q_has_neg` từ nhóm `query`, vì chúng là "query có ràng buộc gì".
- `09_lgbm`: không cần sửa vòng lặp. Ablation tự sinh `no_constraint`.
- `10_report` hoặc một cell mới trong 09: báo cáo Δ của `no_constraint` so với full **trên toàn bộ val và riêng từng `source`** (đặc biệt `negations`, `parse_pattern`). Mỗi dòng có số query và CI.

**Xong khi.** Có `d3_ablation.csv` mới với dòng `no_constraint`, và bảng Δ theo source.

**Cách đọc.** Nếu Δ ≈ 0 kể cả trên `negations`, kết luận là "rule constraint không thêm thông tin so với cross-encoder". S5 sẽ giải thích vì sao.

## S2. Stage 2 theo từng fold rồi lấy trung bình (CPU)

**Vì sao.** Mỗi dòng train của LightGBM có feature CE từ **một** model. Còn val/test dùng **trung bình 5 model**, và điểm trung bình thì mượt hơn, ít cực trị hơn. Các feature `ce_score_z`, `p_E_z` (chiếm 69% importance) nhạy với khác biệt này. Cell 8 của `08_features` đã in mean/std của train và val. Cần đo và sửa.

**Làm gì.**
1. Predictions val/test của từng fold đã có trên Kaggle: `scores/oof_deberta_main_ce4_f{k}_{val,test}.parquet` (output của 07 `full`).
2. Với mỗi k = 0..4: dựng feature val/test từ predictions của fold model k, gồm cả các feature trong list (`_rank, _z, _gap, q_ce_*`). Chấm bằng 5 LightGBM fold model rồi lấy trung bình.
3. Lấy trung bình 5 kết quả theo k, tức trung bình của 25 lần dự đoán.
4. So với `d3_lgbm_multiclass` hiện tại trên val bằng paired bootstrap. Báo cáo thêm bảng mean/std của `ce_score_z`, `p_E` trên train, val (trung bình 5 model) và val (một model).

**Xong khi.** Có run `d3_lgbm_multiclass_perfold` và Δ có CI trên val.

**Quy tắc.** Nếu Δ trên val > 0 với CI không chứa 0, thay final bằng bản perfold. Lý do được ghi trước ở đây là sửa protocol, không phải chọn theo điểm. Ngược lại thì giữ bản cũ và ghi shift là một giới hạn.

## S3. Calibration (CPU)

**Vì sao.** LightGBM dùng `p_E..p_I`. Báo cáo không nên viết "xác suất calibrated" khi chưa đo.

**Làm gì.**
- Thêm vào `src/evaluation.py` hàm `calibration(probs, labels)` trả về:
  - NLL (multiclass);
  - Brier score (multiclass);
  - **ECE classwise**: với mỗi lớp c, chia các cặp thành 15 bin đều mẫu theo `P(c)`. ECE_c là trung bình có trọng số theo cỡ bin của `|mean P(c) − tần suất c|`. Báo cáo trung bình 4 lớp, cộng ECE_E và ECE_S riêng.
- Thêm unit test trong `tests/test_evaluation.py`: dự đoán đúng với xác suất 1 cho ECE = 0; một ví dụ nhỏ tính tay.
- Chạy trên **val** cho các model có head 4 lớp: `d2_deberta_main_ce4_s42`, CE 5-fold mean, `d1_debertalarge_main_ce4_s42`, `d2_minilm_title_ce4_s42/s43`. MSE và ListNet ghi **N/A**.
- Vẽ thêm reliability diagram cho P(E) của DeBERTa ce4 (1 hình cho báo cáo).

## S4. MDE, bảng nhiễu seed, bảng ladder A7 (CPU)

**MDE.** `MDE ≈ 2.8 · sd(Δ_q) / √n` (α = 0.05, power 80%). Tính `sd(Δ_q)` từ per-query Δ đã có. Lấy **giá trị lớn hơn** của hai proxy: seed-vs-seed (B3 s42/s43, DeBERTa MSE s42/s43) và một method-vs-baseline (MiniLM main so với title). Báo cáo MDE cho val (2.089 query), test (8.956), và nhóm `negations` (689 query test).

**Bảng nhiễu seed.** Mỗi so sánh trong `paired_tests.csv` có thêm cột "số seed", "Δ có vượt 0.0014 không", và "dấu có giống nhau giữa các seed không".

**Bảng ladder A7.** Dựng từ các run đã có, mỗi hàng chỉ đổi đúng một yếu tố:

| Hàng | Run | Δ đo cái gì |
|---|---|---|
| R0 | B3 MiniLM title MSE (mean s42, s43) | reference |
| R1 | MiniLM main MSE | input |
| R2 | DeBERTa main MSE | backbone |
| R3 | DeBERTa main ce4 (1 model) | loss (≈ 0) |
| R4 | CE 5-fold mean | ensemble 5 model train trên 80% dữ liệu (không phải "OOF") |
| R5 | LightGBM multiclass (hoặc bản perfold từ S2) | Stage 2 |
| R6 | *(nếu S1 cho Δ > 0)* phần đóng góp riêng của constraint | constraint |

DeBERTa-large và các blend để ở bảng **Extra**, không đưa vào ladder.

## S5. Audit nhãn negation (người, khoảng 3 giờ)

**Vì sao.** Mẫu hiện tại (`eda/tables/rule_check_negation_titles.csv`) có 22 sản phẩm thật sự vi phạm điều kiện loại trừ của query, nhưng **7 cái (32%) vẫn được gán E**. Ví dụ "100cc syringe without needle" lại khớp sản phẩm "with Needle". Nếu tỷ lệ này đứng vững, nó giải thích vì sao rule negation không giúp được, và là một finding chính cho chương error analysis.

**Làm gì.**
1. **Chọn mẫu trước khi xem S1:** từ train+val, `source == 'negations'`, các cặp có `neg_violation == 1`. Lấy ngẫu nhiên khoảng 120 cặp, phân tầng theo nhãn (E/S/C/I) để mỗi nhãn có ít nhất 20 cặp, seed 42.
2. An gán nhãn cả 120 cặp, Thao gán độc lập 40 cặp ngẫu nhiên trong số đó. Các cột:
   - `is_real_exclusion` (query có thật sự loại trừ không, ví dụ "gods not dead" thì không);
   - `product_violates_exclusion`;
   - `rule_correct`.
3. Báo cáo:
   - P(nhãn E | vi phạm thật), kèm Wilson CI 95%;
   - precision của rule (`rule_correct`);
   - tỷ lệ đồng thuận giữa 2 người gán.
4. Lưu ra `eda/tables/audit_negation_labels.csv`.

## S6. Sửa D2: cùng sampler, không cắt list (GPU MiniLM)

**Vì sao.** Hai lỗi thiết kế so với plan (§1.3 EXT 1 của architecture.md):
- MSE/ce4 dùng `DataLoader` theo cặp, còn ListNet dùng `query_batches`, nên đang so cả loss lẫn cách tạo batch.
- `query_batches(list_size=16)` cắt các query có trên 16 sản phẩm thành nhiều list con, nên softmax của ListNet không chạy trên cả list.

**Không sửa đè D2 cũ: giữ hai phiên bản chạy song song.**
- **D2a (đã chạy, giữ nguyên):** mỗi loss dùng cách tạo batch "tự nhiên" của nó. MSE và ce4 dùng batch theo cặp; ListNet dùng batch theo query, cắt list ≤ 16. Các run `b3_*`, `d2_*` không đổi. D3/OOF và hệ thống final vẫn xây trên D2a.
- **D2b (mới, có kiểm soát):** mọi loss dùng cùng sampler theo query, không cắt list. Các run mang tiền tố riêng `d2q_*`, nên không ghi đè file nào.
- **Quy tắc so sánh:** chỉ so các loss **trong cùng một phiên bản**. Baseline của D2b là `d2q_*_mse`, không phải B3. So chéo hai phiên bản chỉ dùng để đo ảnh hưởng của cách tạo batch (bảng bên dưới).

**Sửa code (`src/cross_encoder.py`), chỉ thêm, không đổi hành vi cũ.**
- Thêm key cấu hình `sampler: 'pair' | 'query'`, mặc định `'pair'`. `list_size` giữ mặc định 16. Nhờ vậy cấu hình cũ chạy lại vẫn ra đúng như trước. Khi `'query'`, **mọi loss** đều dùng `query_batches`; MSE và ce4 bỏ qua `seg`.
- `query_batches(..., list_size=None)` nghĩa là giữ nguyên query. Các query được ghép vào batch cho tới khi đạt khoảng 32 cặp; một query dài hơn 32 cặp đứng thành batch riêng. Ghi log effective batch size.
- Ghi `sampler` và `list_size` vào `<run>.json` của mọi run mới, để `10_report` phân biệt được hai phiên bản.
- Warmup đang là `min(warmup_steps, 0.42·total)`. Theo log, B3 và ce4 thực tế warmup 4.962/11.816 = 42%, còn ListNet D2a 5.000/12.883 = 38,8%. D2b đặt `warmup_ratio = 0.42` để mọi loss có cùng tỷ lệ warmup với B3.
- Cập nhật self-check ở cell 4 của `06_d2_losses`: với `list_size=None`, mỗi query nằm trọn trong một list.

**Chạy** (MiniLM title, max_len 128, mọi thiết lập khác như B3), mỗi run khoảng 15 phút:

| Run | loss | sampler | seed |
|---|---|---|---|
| `d2q_minilm_title_mse` | mse | query | 42, 43, 44 |
| `d2q_minilm_title_ce4` | ce4 | query | 42, 43, 44 |
| `d2q_minilm_title_listnet` | listnet, không cắt list | query | 42, 43, 44 |

**Ghi chú cho báo cáo.** ce4 có head 4 lớp mới (khởi tạo ngẫu nhiên), còn MSE và ListNet giữ head MS MARCO. Khác biệt này đi kèm với chính loss, nên chỉ ghi rõ, không sửa.

**Cách đọc D2b.** So mỗi loss với `d2q_minilm_title_mse` cùng seed. Gộp Δ theo query (trung bình trên 3 seed) rồi bootstrap. Một loss "thắng" khi Δ > 0 ở **cả 3 seed** trên val và CI của Δ gộp không chứa 0.

**Ablation phụ: ảnh hưởng của cách tạo batch** (so chéo D2a với D2b, chỉ dùng seed 42 và 43 mà cả hai bản đều có):

| So sánh | Đo cái gì |
|---|---|
| `d2q_*_mse` vs `b3_minilm_title_mse` | batch theo query so với theo cặp, khi loss là MSE |
| `d2q_*_ce4` vs `d2_minilm_title_ce4` | batch theo query so với theo cặp, khi loss là ce4 |
| `d2q_*_listnet` vs `d2_minilm_title_listnet` | cả list so với list cắt ≤ 16 |

Nếu Δ của cách tạo batch xấp xỉ 0 thì kết luận D2a vẫn đứng được, và D2b chỉ xác nhận lại. Nếu Δ khác 0 rõ rệt thì kết luận về loss phải lấy theo D2b.

**Trong báo cáo:** bảng D2 có hai khối, "D2a – như đã chạy" và "D2b – cùng sampler", cộng thêm bảng phụ ở trên.

## S7. Hybrid ce4 + λ·ListNet (EXT 1, GPU MiniLM)

**Định nghĩa** (thêm `loss: 'hybrid'` vào `compute_loss`):
- Head 4 lớp; `s_i = softmax(logits_i) · [1, 0.1, 0.01, 0]` là expected gain, nên `s_i ∈ [0, 1]`.
- `L = CE(logits, y) + λ · ListNet(s / τ)`. **Target của ListNet giữ như code hiện tại**, `gain / sum(gain)` trong list, để so được với ListNet của S6. Tính trên cả query, bỏ qua các list toàn gain 0.
- Cách tính trung bình: CE lấy trung bình theo cặp trong từng query rồi theo query; ListNet lấy trung bình theo query. Cả hai term đều có trọng số theo query.
- **τ chỉ áp lên phía dự đoán.** Vì `s ∈ [0, 1]`, với τ = 1 tỷ lệ xác suất dự đoán giữa tốt nhất và tệ nhất tối đa chỉ là e ≈ 2.7. Trong khi target `gain/sum` rất nhọn: 1 E và 19 I cho E xác suất 1.0. Khi đó ListNet có loss không giảm được và chỉ đẩy s về hai cực. Vì vậy dùng **τ ∈ {0.1, 0.05}**.
- Dùng `sampler: 'query'`, không cắt list (như S6).
- Unit test: list 3 phần tử tính tay, so cả loss lẫn gradient bằng `torch.autograd.gradcheck` hoặc so với bản tính thủ công.

**Chạy:** lưới 2×2 λ ∈ {0.5, 1} × τ ∈ {0.1, 0.05}, seed 42 (4 run, khoảng 1 giờ). Ô tốt nhất trên val chạy thêm seed 43 và 44. Lưới được báo cáo là **bảng sensitivity mô tả**, không claim đã đo được tương tác.

**So sánh chính:** hybrid (ô tốt nhất, 3 seed) với `d2q_minilm_title_ce4` (3 seed), cùng sampler. Đo thêm calibration (S3), vì ListNet có thể làm calibration tệ đi.

## S8. Noise handling trên MiniLM (EXT 3, GPU MiniLM)

**Vì sao.** Đây là extension duy nhất có kết quả công bố trên Task 1 (+0.005). Dữ liệu có nhiễu thật: 322/1301 nhóm cùng title trong train+val có nhãn mâu thuẫn (`data_check.json`). Làm trên MiniLM vì rẻ. Không cần Stage 2, nên tránh được vấn đề rò rỉ nhãn qua OOF.

**Cấu hình:** MiniLM `main`, ce4, `sampler` như S6. Fold = `fold10 // 2` như 07.

**Các bước.**
1. **OOF 5-fold** MiniLM main ce4 trên train để lấy `p_E..p_I` cho mọi cặp train (5 run, khoảng 1–1.5 giờ). Có thể dùng lại notebook 07, chỉ đổi `CFG`.
2. **Flag** bằng `cleanlab.filter.find_label_issues(labels, pred_probs, filter_by='prune_by_noise_rate')` trên CPU. Lưu danh sách cặp bị flag và thống kê theo lớp, theo `source`, và số query bị ảnh hưởng.
3. **Train lại trên toàn bộ train, chấm trên val** (mỗi run khoảng 25 phút):

| Run | Xử lý nhãn |
|---|---|
| `d5_minilm_main_none` | giữ nguyên (reference, ce4 cùng cấu hình) |
| `d5_minilm_main_remove` | bỏ các cặp bị flag |
| `d5_minilm_main_random` | bỏ **ngẫu nhiên** cùng số cặp, phân tầng theo lớp (control) |
| `d5_minilm_main_soft` | thay nhãn cặp bị flag bằng `0.7·one_hot + 0.3·p_OOF` (CE với soft target) |

4. Audit tay khoảng 50 cặp bị flag và 50 cặp không bị flag (lấy mẫu trước khi xem kết quả bước 3). Gọi các cặp bị flag là "suspected label issues".

**Cách đọc.** `remove` phải thắng `random`, không chỉ thắng `none`. Nếu `remove` ≈ `random` thì lợi ích (nếu có) đến từ việc bớt dữ liệu hoặc đổi phân bố lớp, không phải từ làm sạch nhãn. Seed 42 cho tất cả; chạy thêm seed 43 cho biến thể tốt nhất và cho `none`.

## S9. Chuyển loss thắng sang DeBERTa main (GPU, 1 run)

**Chỉ chạy khi** S6 hoặc S7 có một loss thắng ce4 theo đúng tiêu chí của S6 (thắng ở cả 3 seed). Nếu không có loss nào thắng thì bỏ S9 và ghi "không có loss nào tốt hơn ce4 trên MiniLM".

**Run:** `d2q_deberta_main_<loss>_s42`, cấu hình như `d2_deberta_main_ce4_s42`, đổi loss và đặt `sampler: 'query'`. Để so cho công bằng, cần thêm một run DeBERTa ce4 với `sampler: 'query'`. Nếu không đủ quota thì so với `d2_deberta_main_ce4_s42` và ghi rõ hai bên khác sampler.

## S10. DeBERTa main ce4 seed 43 (GPU, 1 run, nếu còn quota)

Backbone của hệ thống final mới có 1 seed. Run này cho biết Δ ce4 so với MSE trên DeBERTa có đứng vững không. Nếu phải chọn giữa S9 và S10 thì ưu tiên S9.

## S11. Cập nhật tài liệu

- `docs/architecture.md`: đối chiếu với những gì thực sự đã chạy:
  - tên D1/D2/D3 thay vì A1–A7;
  - DeBERTa OOF 5-fold, không phải 2-fold;
  - B3 chỉ có bản kiểu ctrl (max_len 128, train trên T);
  - nhóm query lấy từ cột `source` của dataset, không phải detector;
  - test được chấm một lần ở cuối.
- Chương phương pháp nêu rõ các khác biệt sau so với plan và lý do:
  - sampler của D2 cũ;
  - ce4 có head mới;
  - shift giữa OOF và val/test;
  - B3 khác Amazon (max_len 128 so với 512, dev 10% so với 200 query).

---

## Không làm thêm

| Việc | Lý do |
|---|---|
| DeBERTa-large 5-fold OOF rồi LightGBM | khoảng 22 giờ GPU; ước tính chỉ hơn blend `final + 0.5 large` (0.8968) ≤ 0.002; chỉ tăng điểm, không thêm insight |
| Thêm blend hoặc bge-reranker | bge kém DeBERTa 0.021; các blend chỉ cho Δ ≤ 0.002, ngang nhiễu seed |
| Train DeBERTa 2 epoch | đã thử: −0.0008, không có lợi |
| Thêm grid LightGBM | CV phẳng (0.8878–0.8888) |
| Ablation phụ của input (marker, dedup, max_len) | lợi ích nhỏ, tốn GPU DeBERTa; để mục future work |

## Quy tắc quyết định (chốt trước khi chạy)

1. Hệ thống final chỉ đổi khi:
   - có quy tắc đã ghi trước (S2); hoặc
   - một thành phần mới thắng trên val với CI không chứa 0, **và** đã chạy ít nhất 2 seed với cùng dấu Δ.
2. Mọi kết quả null đều được báo cáo. Nếu S1, S7, S8 đều null thì kết luận của đề tài là: *phần lớn lợi ích đến từ input giàu hơn và backbone mạnh hơn; loss, rule và làm sạch nhãn không thêm được gì có ý nghĩa thống kê trên ESCI Task 1. Một nguyên nhân là nhãn không phạt nhất quán việc vi phạm ràng buộc (S5).*
3. **29/11:** chốt danh sách run Supplement được chấm test. **6/12:** chạy lại `10_report` và đóng băng `data/report/`.

---

# Phần 2: sau khi có DeBERTa-large OOF (cập nhật 22/11 tối)

## Tình hình hiện tại

| Hàng | Hệ thống | Val | Test |
|---|---|---:|---:|
| B3 | MiniLM title MSE (2 seed) | 0.8572 | 0.8552 |
| Final "chốt trước test" | LightGBM multiclass trên base CE 5-fold + feature | 0.8947 | 0.8921 |
| Large CE | DeBERTa-large ce4, trung bình 5 fold model | 0.8970 | 0.8965 |
| **Final hiện tại** | LightGBM lambdarank trên base + large CE + feature | 0.8999 | **0.8980** |

Đọc từ các bảng mới:

1. **Final được đổi sau khi đã thấy test.** Large xuất hiện lần đầu ở mục Extra và đã được chấm test, rồi mới được đưa vào final. Δ so với final cũ là +0.0059, CI [0.0047, 0.0072], lớn hơn nhiều mọi sai lệch do chọn lựa, nên kết luận vẫn đứng được. Nhưng điều này phải được ghi rõ (R1).
2. **LightGBM gần như không thêm gì khi đã có large.**
   - Final hơn large CE 5-fold chỉ +0.0015 trên test, CI [0.0004, 0.0024], ngang mức nhiễu seed 0.0014.
   - Ablation trên val: `no_base_ce` −0.0004 (không significant); `ce_only` (bỏ mọi feature ngoài CE) −0.0019; `no_large_ce` −0.0056.
   - Phần lớn chất lượng đến từ score của large cộng với chuẩn hóa trong list (`ce3_score_z`, `ce3_p_E_z` chiếm 69% importance).
3. **Phủ định là điểm yếu duy nhất không được cải thiện, và khoảng cách đang rộng ra.**
   - Query có từ phủ định (731 query test): final 0.8531, query không có: 0.9020.
   - Final hơn B3 chỉ +0.030 trên nhóm này, so với +0.044 trên phần còn lại.
   - Trên nhóm này LightGBM thấp hơn large CE một chút (0.8531 so với 0.8543).
4. **Từ D1 trở đi mọi thứ chỉ 1 seed**, nên mọi Δ ≤ 0.002 chưa phân biệt được với nhiễu.
5. Commit `1b22c2c` đã xóa các file score parquet khỏi git, nên `10_report` giờ phụ thuộc vào output trên Kaggle.

## Trạng thái các mục của Phần 1

| Mục | Trạng thái | Ghi chú |
|---|---|---|
| S1 tách nhóm constraint | chưa | vẫn nên làm, chuyển thành một phần của N2 |
| S2 Stage 2 theo từng fold | chưa | ưu tiên thấp: LightGBM chỉ còn thêm +0.0015 |
| S3 calibration | chưa | đổi thành R4 (thêm large) |
| S4 MDE, nhiễu seed, ladder | chưa | gộp vào R1, R2 |
| S5 audit nhãn negation | chưa | thành một phần của N1 |
| S6 D2b cùng sampler | chưa | đổi thành R5 |
| S7 hybrid, S8 noise | **hạ xuống không làm** | large CE chiếm gần hết chất lượng, lợi ích còn lại quá nhỏ so với chi phí |
| S9, S10 | **bỏ** | thay bằng N3 (distillation) |
| S11 cập nhật tài liệu | chưa | thêm vào đó: final hai mức, large OOF |
| Large 5-fold OOF | **đã làm** | trước đó nằm trong mục "Không làm thêm"; nhóm đã chạy, nên giờ ghi nhận là thí nghiệm bổ sung sau test (R1) |

## Bảng tổng hợp Phần 2

| ID | Việc | Loại | Tài nguyên | Ước lượng | Phụ trách | Hạn |
|---|---|---|---|---|---|---|
| R1 | Final hai mức và nhật ký 3 đợt chấm test | chặt chẽ | không | 1 h | An | 25/11 |
| R2 | LightGBM 5 seed và bảng nhiễu seed | chặt chẽ | CPU | 1 h | Cong | 25/11 |
| R3 | Kiểm tra memorization (query hoặc sản phẩm đã thấy trong train) | chặt chẽ | CPU | 2 h | An | 26/11 |
| R4 | Calibration base và large ce4 | chặt chẽ | CPU | 1 h | An | 26/11 |
| R5 | D2b cùng sampler (= S6) | chặt chẽ | GPU MiniLM | 2 h | Thao | 27/11 |
| R6 | Khả năng tái tạo `10_report` | chặt chẽ | không | 1 h | Duy | 27/11 |
| **N1** | Phủ định: lỗi do model hay do nhãn (audit + kiểm thử hành vi) | mở rộng ⭐ | người + GPU khoảng 1 h | 1–1.5 ngày | Thao + An | 1/12 |
| N2 | E-vs-S theo loại thuộc tính khác biệt (+ S1) | mở rộng | CPU | 0.5 ngày | Cong | 29/11 |
| **N3** | Đánh đổi chi phí–chất lượng và distillation vào MiniLM | mở rộng ⭐ | GPU khoảng 1.5 h | 1 ngày | Duy (+ Cong vẽ Pareto) | 1/12 |
| N4 | Tập xác nhận độc lập: các query "dễ" bản reduced đã loại | mở rộng | GPU khoảng 2 h | 1 ngày | Duy | 3/12, nếu còn quota |
| N5 | LLM reranker trên tập query khó | mở rộng, tùy chọn | GPU khoảng 3–4 h | 1–1.5 ngày | Thao | 3/12, chỉ khi N1–N3 xong |
| N6 | Độ bền với thành phần candidate list | mở rộng | CPU | 0.5 ngày | Cong | 1/12 |

**Thứ tự đề xuất:** làm R1–R4 trước (gần như không tốn GPU), sau đó **N1 và N3** là hai ngách chính, N2 và N6 nếu còn người, N4 nếu còn GPU. N5 là stretch.

---

## R1. Final hai mức và nhật ký các lần chấm test

- Trong `10_report` và chương kết quả, trình bày **hai** hệ thống:
  - **Final chốt trước test:** `d3_lgbm_multiclass` (base CE), test 0.8921, được chọn trên val/CV trước khi chấm test lần đầu.
  - **Final sau khi thấy test:** `d3_lgbm_ce3_lambdarank` (base + large), test 0.8980. Ghi rõ: hướng DeBERTa-large được chọn **sau** khi đã thấy test ở mục Extra; trong hướng đó, objective và setting vẫn được chọn bằng CV/val.
- Thêm bảng "nhật ký test" lấy từ git log của `data/report/`. Đến 22/11, test đã được chấm theo **3 đợt** (mỗi đợt chạy lại `10_report`, in lại điểm của mọi cấu hình):

| Đợt | Commit | Cấu hình có điểm test | Quyết định sau đó |
|---|---|---|---|
| 1 | `e7506f4` (20/11) | 25 | không có: final gốc đã chốt trên val/CV trước đợt này |
| 2 | `e16e4d2` (22/11 sáng) | 34 (+9 Extra, có DeBERTa-large) | **chọn hướng large**: quyết định duy nhất có thông tin từ test |
| 3 | `1b22c2c` (22/11 tối) | 38 (+4, large OOF) | final mới; objective và setting vẫn chọn trên CV/val |

- Trong phần hạn chế, ghi đúng số **đợt** (3) và quyết định nào có thông tin từ test (chỉ quyết định thử large). Con số 38 là số cấu hình có điểm test, không phải số lần nhìn test.

## R2. LightGBM 5 seed và bảng nhiễu seed

- `09_lgbm`: cấu hình final (lambdarank, `small`) chạy lại với `seed` ∈ {0, 1, 2, 3, 4}; thêm `bagging_fraction=0.8, bagging_freq=1` để seed có tác dụng (`feature_fraction` 0.8 đã có). Báo cáo mean ± sd của val và test nDCG, và Δ so với large CE 5-fold cho từng seed.
- Bảng nhiễu seed cho chương kết quả: mỗi so sánh trong `paired_tests.csv` thêm các cột "số seed của CE", "|Δ test| > 0.0014?", "cùng dấu giữa các seed?".
- Phần này chỉ đo **phương sai của LightGBM**, không đo phương sai của CE (chạy lại CE large 5-fold quá đắt). Ghi rõ điều này.

## R3. Kiểm tra memorization

**Câu hỏi:** gain có đến từ việc model đã thấy query hoặc sản phẩm gần giống trong train không?

1. **Mức query:** dùng `eda/tables/near_dup_pairs_train_test.csv` (171 cặp). Gắn cờ test query có bản gần trùng trong train. So final và B3 trên nhóm có và không có gần trùng, kèm số query và CI. Nhóm có gần trùng nhỏ (khoảng 100–170 query), nên chỉ mang tính mô tả.
2. **Mức sản phẩm:** chia các cặp test theo "sản phẩm này đã xuất hiện trong train với một query khác" hay chưa. Đo E-over-S và nDCG theo từng nhóm. Đây là phép đo mô tả, không phải feature (feature `product_prior` vẫn bị cấm).
3. Kết luận mong đợi: nếu gain trên nhóm "chưa thấy" ≈ trên nhóm "đã thấy", thì kết quả không đến từ việc nhớ.

## R4. Calibration (thay S3)

Như S3 (NLL, Brier, ECE classwise 15 bin đều mẫu), nhưng chạy trên val cho: base ce4 (1 model và trung bình 5 fold), **large ce4 (1 model và trung bình 5 fold)**, MiniLM title ce4. Thêm reliability diagram của P(E) cho base và large trên cùng một hình. Câu hỏi phụ: trung bình 5 fold có làm xác suất "mượt" hơn không? Đây là nguyên nhân có thể của shift giữa OOF và val (S2).

## R5. D2b cùng sampler

Giữ nguyên như S6 ở Phần 1 (D2a giữ nguyên, D2b chạy song song, tiền tố `d2q_*`, 3 seed MiniLM title). Không chạy hybrid (S7).

## R6. Khả năng tái tạo `10_report`

- Ghi trong README: tên và **version** của Kaggle dataset hoặc notebook output chứa các file score parquet đã bị xóa khỏi git ở commit `1b22c2c`.
- Chạy lại `10_report` từ đầu trên Kaggle với đúng các input đó và so `results_all.csv` với bản trong git (phải khớp đến 4 chữ số).
- Mục tiêu: mọi số trong báo cáo đều tái tạo được bằng một lần chạy `10_report`.

---

## N1. Phủ định: lỗi do model hay do nhãn? ⭐

**Vì sao.** Đây là nhóm query duy nhất mà model mạnh hơn không kéo lên được, và khoảng cách đang rộng ra. Có hai giả thuyết cạnh tranh:
- (a) model không hiểu phủ định;
- (b) nhãn không phạt nhất quán việc vi phạm phủ định. Mẫu nhỏ hiện tại: 7/22 sản phẩm vi phạm vẫn được gán E.

N1 phân biệt hai giả thuyết này.

**Phần A: audit nhãn** (= S5, An làm chính, Thao gán lại 40 cặp). Khoảng 120 cặp từ train+val, `source == 'negations'`, lấy mẫu phân tầng theo nhãn, chọn trước khi xem kết quả phần B. Báo cáo:
- P(E | sản phẩm vi phạm thật), kèm Wilson CI;
- precision của rule `neg_violation`;
- tỷ lệ đồng thuận giữa 2 người gán.

**Phần B: kiểm thử hành vi kiểu CheckList** (Thao; không cần nhãn).
1. Lấy các query val và test có mẫu `X without|no|w/o Y` mà Y bắt được rõ (dùng regex hiện có, lọc tay bỏ các câu như "gods not dead"). Mục tiêu khoảng 200–300 query.
2. Mỗi query sinh 3 biến thể: **gốc** "X without Y", **bỏ phủ định** "X", **đảo** "X with Y". Thêm một biến thể **diễn đạt lại** "X no Y" hoặc "X w/o Y".
3. Chấm cả list sản phẩm gốc với từng biến thể. Ưu tiên dùng large CE (1 model full-T nếu checkpoint còn trên Kaggle). Nếu không còn checkpoint thì dùng base ce4 hoặc MiniLM main (train lại khoảng 15 phút) và ghi rõ là model nào.
4. Chia sản phẩm thành hai nhóm: **có Y** (Y xuất hiện trong title/bullets mà không bị phủ định) và **không có Y**. Các phép đo:
   - **Directional:** với sản phẩm có Y, score(gốc) < score(đảo)? Tính tỷ lệ cặp đúng hướng.
   - **Directional ngược:** với sản phẩm không có Y, score(gốc) ≥ score(đảo)?
   - **Invariance:** Kendall τ giữa thứ hạng của "gốc" và "diễn đạt lại" (kỳ vọng gần 1).
   - **Độ nhạy:** |Δ score| trung bình khi bỏ phủ định. Nếu ≈ 0 thì model bỏ qua phủ định.
5. So các phép đo này giữa B3 MiniLM title, base và large (nếu chạy được), để xem model càng mạnh thì hiểu phủ định tốt hơn không.

**Cách đọc kết hợp A + B:**

| A: nhãn | B: model | Kết luận |
|---|---|---|
| nhãn nhất quán (P(E \| vi phạm) thấp) | model không đảo hướng | lỗi do model, có chỗ để cải thiện |
| nhãn không nhất quán | model đảo đúng hướng | model đúng, nhãn kéo nDCG xuống: **trần do nhãn** |
| cả hai đều kém | | cả hai cùng góp phần, báo cáo cả hai |

**Đầu ra:** `eda/tables/audit_negation_labels.csv`, `data/report/negation_probe.csv`, 1 hình (tỷ lệ đúng hướng theo model), 1 mục trong chương error analysis.

## N2. E-vs-S theo loại thuộc tính khác biệt (+ S1)

1. Với mỗi query trong val (dùng val để giữ test sạch; test chỉ chạy một lần ở cuối), lấy mọi cặp (sản phẩm E, sản phẩm S).
2. Dùng các hàm có sẵn trong `src/features.py` (`units_of`, `colors_of`, `NEG`, brand) để gắn **loại khác biệt** cho mỗi cặp E/S:
   - số kèm đơn vị khác nhau;
   - màu khác nhau;
   - brand khác nhau;
   - E thỏa phủ định còn S vi phạm;
   - không phát hiện được (khác ở chỗ rule không thấy).
3. Đo **E-before-S accuracy theo từng loại** cho B3, base, large, final. Mỗi loại ghi số cặp, số query và CI (bootstrap theo query).
4. Đồng thời làm S1: tách nhóm `constraint` trong ablation 09, và báo cáo Δ của `no_constraint` theo từng loại khác biệt.

**Đầu ra:** bảng "E-over-S theo loại thuộc tính" và Δ của constraint. Bảng này trả lời trực tiếp câu hỏi nghiên cứu "phân biệt E/S khó ở đâu".

## N3. Đánh đổi chi phí–chất lượng và distillation ⭐

**Vì sao.** Final cần **10 cross-encoder** (base và large, mỗi loại 5 fold) cộng LightGBM cho mỗi lần dự đoán. Đây là hệ thống không thực tế cho search thật, và báo cáo nên có góc nhìn này.

**Phần A: đường Pareto** (Cong, CPU). Với mỗi cấu hình, lấy thời gian inference từ `predict_minutes` và số cặp trong các file `<run>.json`, quy về **ms GPU / 1.000 cặp** (ghi loại GPU). Ghép các model trong ensemble lại theo số model. Vẽ test nDCG theo chi phí (log scale) cho: BM25, bi-encoder, MiniLM title và main, base (1 model, 5 model), large (1, 5), final, và student của phần B. Đánh dấu đường Pareto.

**Phần B: distill vào MiniLM** (Duy, khoảng 15–25 phút mỗi run).
- **Teacher:** xác suất OOF của large 5-fold trên train. Mỗi cặp được chấm bởi fold model chưa thấy nó, nên soft target không bị rò rỉ.
- **Student:** `cross-encoder/ms-marco-MiniLM-L-12-v2`, input `main`, max_len 256, head 4 lớp. Loss `L = α·CE(nhãn cứng) + (1−α)·CE(soft target của teacher)`. Mọi thiết lập khác như MiniLM main.
- **Các run** (seed 42; run tốt nhất thêm seed 43):

| Run | α |
|---|---|
| `d4_minilm_main_ce4` (reference, ce4 nhãn cứng) | 1.0 |
| `d4_minilm_main_distill_a05` | 0.5 |
| `d4_minilm_main_distill_a0` | 0.0 (chỉ soft target) |

- **Chỉ số chính:** tỷ lệ khoảng cách lấy lại được = (student − MiniLM ce4) / (large 5-fold − MiniLM ce4), trên val. Báo cáo kèm chi phí inference so với final.
- Code: thêm `loss: 'ce4_soft'` vào `compute_loss` (CE với target là phân bố), nhận cột `soft_E..soft_I` từ train df. Unit test: với target one-hot, kết quả phải bằng ce4.

## N4. Tập xác nhận độc lập: các query "dễ"

**Vì sao.** Test đã bị chấm nhiều lần (R1). Bản large của ESCI có thêm các query `us` **không thuộc** Task 1 (`small_version == 0`). Đó là các query "dễ" mà bản reduced đã loại, nhưng vẫn có nhãn. Đây là một tập **chưa ai đụng tới**.

**Làm gì.**
1. Từ `examples` (đầy đủ), lấy `product_locale == 'us'`, `small_version == 0`, `split == 'test'`. Lấy mẫu ngẫu nhiên **3.000 query**, seed 42. Ghi danh sách query vào `data/splits/easy_holdout_queries.csv` **trước khi** chạy bất kỳ model nào.
2. Kiểm tra trùng: bỏ query nào trùng hoặc gần trùng (jaccard ≥ 0.8) với query của Task 1.
3. Chấm: BM25 main, bge bi-encoder, MiniLM main, base 5-fold, large 5-fold. Final LightGBM là tùy chọn, vì cần dựng lại toàn bộ feature cho các cặp mới.
4. Báo cáo: nDCG và E-over-S trên tập dễ cho từng phương pháp, và thứ tự các phương pháp có giữ nguyên so với Task 1 không.

**Chỉ chấm một lần**, sau khi mọi quyết định khác đã chốt (29/11), và không dùng để chọn gì.

## N5. LLM reranker trên tập query khó (tùy chọn, stretch)

- Model: Qwen2.5-7B-Instruct hoặc Llama-3.1-8B-Instruct qua vLLM (nhóm đã có notebook vLLM offline). Prompt listwise: đưa query và danh sách candidate (title + brand + color, cắt ngắn), yêu cầu trả về thứ tự. Thêm một biến thể pointwise (hỏi E/S/C/I cho từng sản phẩm) nếu listwise khó parse.
- Tập đánh giá: khoảng 300 query phủ định và khoảng 300 query ngẫu nhiên lấy từ **val**. Chọn trước, seed 42.
- So với large CE trên cùng các query, kèm paired bootstrap. Câu hỏi: LLM zero-shot có xử lý phủ định tốt hơn cross-encoder đã fine-tune không? Bài LLM trên ESCI (arXiv 2408.12779) chỉ làm Task 2.
- Chỉ làm khi N1–N3 đã xong. Đây là phần mở rộng, không thay hệ thống final.

## N6. Độ bền với thành phần candidate list

**Vì sao.** Feature trong list (`_z`, `_rank`, `_gap`, `q_ce_*`, `brand_share`, `n_cand`) chiếm phần lớn importance. Cần biết LightGBM có phụ thuộc vào cách candidate list được tạo ra không.

**Làm gì** (val, CPU, dùng feature table của 08 và 5 LightGBM fold model của final):
1. Với mỗi query có ≥ 10 candidate: **bỏ ngẫu nhiên 30% candidate** (không nhìn nhãn), lặp 5 lần với seed khác nhau.
2. Tính lại mọi feature trong list trên list đã thu nhỏ, chấm lại bằng LightGBM.
3. Đo trên các sản phẩm **còn lại**:
   - Kendall τ giữa thứ hạng mới và thứ hạng cũ (giới hạn trên các sản phẩm còn lại). Large CE có τ = 1 theo định nghĩa, vì score từng cặp không phụ thuộc list.
   - nDCG trên list thu nhỏ của final so với large CE.
4. Biến thể thứ hai: bỏ 30% candidate **có nhãn I**. Biến thể này dùng nhãn, chỉ để mô phỏng một bộ candidate "sạch" hơn.

**Cách đọc.** Nếu τ của final thấp rõ rệt hoặc final mất lợi thế so với large CE trên list thu nhỏ, thì phần +0.0015 của LightGBM phụ thuộc vào cách tạo list. Khi đó kết luận chỉ được giới hạn trong các candidate list do ESCI cung cấp.

---

> **Plan chi tiết cho các mục cần GPU** (R5, N1-B, N3, N4, N5, cộng một run DeBERTa-base có lưu checkpoint): xem `docs/plan_training.md` (các thẻ T1–T7). Lưu ý: không run DeBERTa nào lưu checkpoint, và B3 chỉ có checkpoint nếu notebook 03 được chạy với `SAVE_MODEL = True` (mặc định là False). Vì vậy N1-B và N4 dùng checkpoint của các run mới (notebook 14, 15), không dùng được large hay final. Notebook: 13 (D2b), 14 (distill), 15 (DeBERTa-base s43), 16 (kiểm thử phủ định). Checklist tổng hợp mọi việc còn lại: `docs/todo.md`.

## Quy tắc quyết định (Phần 2)

1. **Hệ thống final không đổi thêm nữa.** Các mục R và N chỉ để phân tích và báo cáo. Student của N3 và LLM của N5 là kết quả riêng, không thay final.
2. Tập xác nhận độc lập của N4 chỉ chấm **một lần**, sau 29/11.
3. Mọi phép đo trên test của Phần 2 (R3, N1-B nếu dùng query test, N4) được liệt kê trước ngày 29/11, giống danh sách Supplement.
4. Kết quả null vẫn được báo cáo. Ví dụ: nếu N1 cho thấy "trần do nhãn" thì đó là một kết luận chính của chương error analysis, không phải một thất bại.

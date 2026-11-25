# Plan chi tiết: các thí nghiệm train và inference bổ sung (nhánh `supplement`)

Ngày lập: 22/11/2026 · Hạn: chạy xong trước **3/12**, đóng băng kết quả **6/12** (G4)
Liên quan: `docs/continue.md` Phần 2 (mã R*, N*). File này chi tiết hóa các mục **cần GPU**: R5, N1-B, N3, N4, N5, cộng thêm một run DeBERTa-base có lưu checkpoint. Mục R2 (LightGBM nhiều seed, CPU) cũng được viết ở đây vì nó là "train".

---

## 0. Tóm tắt

| ID | Thí nghiệm | Mục trong continue.md | GPU | Ước lượng | Phụ thuộc |
|---|---|---|---|---|---|
| T1 | D2b: so sánh loss với cùng sampler, 3 seed | R5 | 1 × T4 | khoảng 2.5 h | C1 |
| T2 | Distill large → MiniLM main (+ checkpoint MiniLM) | N3-B | 1 × T4 | khoảng 4 h | C1, file OOF của large |
| T3 | DeBERTa-base ce4 seed 43, **lưu checkpoint** | (mới, phục vụ N1, N4, nhiễu seed) | 1 × T4 | khoảng 3 h | không |
| T4 | Kiểm thử hành vi với phủ định | N1-B | 1 × T4 | khoảng 0.5 h | C3, checkpoint của T2, T3 |
| T5 | Tập xác nhận độc lập: query "dễ" | N4 | 1 × T4, Internet on | khoảng 1 h | checkpoint của T2, T3 |
| T6 | LightGBM 5 seed (+ N6 thay đổi candidate list) | R2, N6 | CPU | khoảng 0.5 h | output của 08 |
| T7 | LLM reranker (tùy chọn) | N5 | **2 × T4** | khoảng 1.5 h | C3 |

**Tổng GPU:** khoảng 11 giờ trên 1 × T4, cộng khoảng 3 GPU-giờ nếu làm T7. Chia cho 2–3 tài khoản Kaggle (mỗi tài khoản khoảng 30 giờ/tuần), nên vừa trong 1 tuần.

**Chạy song song:** T1, T2, T3 độc lập với nhau, chạy cùng lúc trên 3 tài khoản. T4 và T5 phải chờ checkpoint của T2 và T3.

### Phát hiện ảnh hưởng tới plan: thiếu checkpoint

Chỉ `03_cross_encoder` có code lưu model (`model.save_pretrained`, thư mục `model_<run>`), và **mặc định `SAVE_MODEL = False`**, nên rất có thể B3 cũng không có checkpoint. Muốn dùng B3 trong T4/T5 thì chạy lại 03 với `SAVE_MODEL = True` (khoảng 15 phút); nếu không, notebook 16 tự bỏ qua B3. Các notebook 05, 06, 07, 11, 12 **không lưu weights**, nên không còn DeBERTa base hay large nào để chạy inference. Hệ quả:
- T4 (kiểm thử phủ định) và T5 (query dễ) chỉ chạy được với B3, MiniLM main (từ T2) và DeBERTa-base (từ T3). **Large và final không đánh giá được** nếu không train lại (large 5-fold mất khoảng 25 giờ trên 2 × T4). Điều này được ghi là một hạn chế trong báo cáo.
- **Quy tắc từ giờ:** mọi run GPU trên nhánh `supplement` đều lưu checkpoint ở fp16 vào `/kaggle/working/models/<run>/`. MiniLM khoảng 65 MB, base khoảng 370 MB, vừa giới hạn output 20 GB của Kaggle.

### Việc làm được ngay ở local (không cần data, không cần GPU)

Máy local không có dữ liệu ESCI, không có file score, không có GPU, nên ở local chỉ làm **code, unit test và khung notebook**:

| ID | Việc code | File | Test |
|---|---|---|---|
| C1 | `sampler`, `list_size=None`, `warmup_ratio`, loss `ce4_soft` | `src/cross_encoder.py` | `tests/test_cross_encoder.py` (mới, chạy CPU) |
| C2 | `calibration()` (NLL, Brier, ECE classwise) cho R4 | `src/evaluation.py` | `tests/test_evaluation.py` |
| C3 | Parse phủ định, sinh biến thể query, xác định sản phẩm có chứa term | `src/probes.py` (mới) | `tests/test_probes.py` (mới) |
| C4 | Khung notebook 13–19 (mode `smoke` / `full`, theo style của 05/06/12) | `notebooks/` | smoke trên Kaggle |

---

## 1. Thay đổi code chung (C1–C3)

Nguyên tắc: **chỉ thêm, không đổi hành vi mặc định.** Mọi cấu hình cũ chạy lại vẫn phải ra đúng như trước.

### C1. `src/cross_encoder.py`

1. **`query_batches(query_ids, rng, list_size=16, batch_pairs=32)`**: nhận `list_size=None`, nghĩa là giữ nguyên cả query (không cắt). Khi đóng batch, một query dài hơn `batch_pairs` đứng thành batch riêng. Mặc định 16 giữ nguyên.
2. **`cfg['sampler']`** ∈ {`'pair'`, `'query'`}:
   - mặc định là `'query'` nếu `loss == 'listnet'`, ngược lại là `'pair'` (đúng như hành vi hiện tại);
   - khi `'query'`, mọi loss đều dùng `query_batches`; MSE, ce4 và ce4_soft bỏ qua `seg`;
   - ghi `sampler`, `list_size` và `mean_batch_pairs` vào `info`.
3. **`cfg['warmup_ratio']`** (tùy chọn): nếu có thì `warmup = int(ratio × total)`, ngược lại giữ công thức cũ `min(warmup_steps, 0.42 × total)`. Các run D2b đặt `warmup_ratio = 0.42`, để tỷ lệ warmup bằng đúng B3 dù số bước thay đổi. Theo log: B3, ce4 và MiniLM main thực tế warmup **4.962 / 11.816 bước** (giới hạn 42% có tác dụng), còn ListNet ở D2a warmup 5.000 / 12.883 = 38,8%. Như vậy D2a còn lệch nhỏ cả về tỷ lệ warmup giữa các loss, và D2b sửa luôn điểm này.
4. **Loss `ce4_soft`** (cho distillation):
   - head 4 lớp như ce4 (đi cùng nhánh khởi tạo của ce4);
   - target là ma trận (n, 4): `y = α · one_hot(nhãn) + (1 − α) · soft`, trong đó `soft` lấy từ các cột `soft_E, soft_S, soft_C, soft_I` của `train_df`, và `α = cfg['alpha']`;
   - `compute_loss` gọi `F.cross_entropy(logits, y)`, vì PyTorch nhận target dạng xác suất;
   - `to_score` và `predict` không đổi (vẫn trả expected gain và p_E..p_I).
5. **Unit test** (CPU, model giả có `logits` cố định):
   - `query_batches(list_size=None)`: mỗi query nằm trọn trong một list, mọi dòng xuất hiện đúng 1 lần, batch ≤ 32 trừ khi một query dài hơn 32;
   - `ce4_soft` với `α = 1` cho kết quả bằng `ce4` trên cùng logits;
   - `ce4_soft` với soft target one-hot cũng bằng `ce4`;
   - `warmup_ratio` tính đúng số bước.

### C2. `src/evaluation.py`: `calibration(probs, labels, n_bins=15)`

- Trả về `nll`, `brier` (multiclass, tổng trên 4 lớp), `ece_c` cho từng lớp, và `ece_mean`.
- ECE classwise: với mỗi lớp c, sắp các cặp theo `P(c)` rồi chia thành 15 bin **đều mẫu**. `ECE_c = Σ_b (n_b / N) · |mean P(c)_b − tần suất c_b|`.
- Test: dự đoán one-hot đúng cho NLL ≈ 0 và ECE = 0; dự đoán đều 0.25 trên nhãn cân bằng cho ECE = 0 và NLL = ln 4.

### C3. `src/probes.py` (cho T4 và T7)

- `negation_phrase(query)`: bắt cụm sau từ phủ định (`without`, `w/o`, `no`, `not`, `non`, `-free`), tối đa 3 token, dừng ở liên từ hoặc giới từ (`and`, `for`, `with`, `in`, `,`). Ví dụ: "ipad case 8th generation without front cover" cho ra ("without", "front cover"). Lưu ý rule hiện tại trong `features.py` chỉ lấy 1 token.
- `variants(query)` trả về một dict:
  - `orig`: câu gốc;
  - `drop`: bỏ cả cụm phủ định ("ipad case 8th generation");
  - `flip`: "… with front cover";
  - `para`: đổi từ phủ định ("… no front cover" hoặc "… w/o front cover"), giữ nghĩa.
- `term_status(product_text, term)` ∈ {`has`, `negated`, `absent`}. `negated` khi term đứng sau từ phủ định trong cùng title/bullets, mở rộng từ `title_negates`.
- Test: khoảng 15 ví dụ viết tay, gồm cả các câu bẫy ("gods not dead", "i did not miss the boat" phải trả về `None` khi đã lọc bằng danh sách loại trừ).

---

## 2. Thẻ thí nghiệm

Mọi run mới ghi vào `results/supplement/` (`<run>.json`, `scores/<run>_{val,test}.parquet`) qua `RunLog` như hiện tại. **Test được predict và lưu blind**, chỉ chấm trong `10_report` sau ngày 29/11.

### T1. D2b: so sánh loss với cùng sampler (R5)

**Câu hỏi.** Khi mọi loss dùng cùng cách tạo batch (theo query, không cắt list), thứ tự MSE, ce4, ListNet có còn như kết quả D2a không? Cách tạo batch tự nó có ảnh hưởng không?

**Notebook:** `13_d2b_sampler.ipynb` (copy cấu trúc của `06_d2_losses`). **Kaggle:** 1 × T4, Internet on.

**Cấu hình chung** (bằng B3, trừ sampler):
```python
BASE = {'model': 'cross-encoder/ms-marco-MiniLM-L-12-v2', 'fields': 'title', 'max_len': 128,
        'epochs': 1, 'batch_size': 32, 'lr': 7e-6, 'warmup_steps': 5000, 'warmup_ratio': 0.42,
        'weight_decay': 0.01, 'max_grad_norm': 1.0, 'fp16': True,
        'sampler': 'query', 'list_size': None}
```

| Run | loss | seed | Ước lượng |
|---|---|---|---|
| `d2q_minilm_title_mse_s{42,43,44}` | mse | 42, 43, 44 | khoảng 15 phút mỗi run |
| `d2q_minilm_title_ce4_s{42,43,44}` | ce4 | 42, 43, 44 | khoảng 15 phút mỗi run |
| `d2q_minilm_title_listnet_s{42,43,44}` | listnet | 42, 43, 44 | khoảng 15 phút mỗi run |

9 run, tổng khoảng 2.3 giờ, vừa một session (giới hạn 12 giờ). `eval_every = None` để tiết kiệm thời gian; chỉ chấm val ở cuối.

**Smoke** (5% train, seed 42, chỉ val):
- loss hữu hạn;
- val nDCG > 0.80 (ngẫu nhiên là 0.742);
- `mean_batch_pairs` trong khoảng 20–32;
- không query nào bị chia ra nhiều batch (assert trong notebook).

**Cách đọc:**
1. **Trong D2b:** so mỗi loss với `d2q_*_mse` cùng seed. Gộp Δ theo query (trung bình trên 3 seed) rồi bootstrap. Một loss "thắng" khi Δ > 0 ở cả 3 seed và CI không chứa 0.
2. **Ảnh hưởng của cách tạo batch:** so D2b với D2a ở seed 42 và 43 (`d2q_*_mse` với `b3_*`, `d2q_*_ce4` với `d2_*_ce4`, `d2q_*_listnet` với `d2_*_listnet`).
3. Báo cáo cả hai khối, kèm calibration (C2) cho ce4.

**Rủi ro.** Batch theo query có ít cặp hơn 32 nên số bước tăng (ListNet cũ là 12.883 bước so với 11.816), chậm hơn khoảng 10%. Không đáng kể.

### T2. Distill large → MiniLM main (N3-B)

**Câu hỏi.** Một MiniLM (khoảng 33M tham số; final gồm 5 base khoảng 184M và 5 large khoảng 435M, tổng khoảng 3,1 tỷ, tức gấp khoảng 90 lần) học từ soft target của DeBERTa-large lấy lại được bao nhiêu phần khoảng cách giữa MiniLM (0.866) và large (0.8965)?

**Notebook:** `14_distill.ipynb`. **Kaggle:** 1 × T4, Internet on.

**Input:**
- `01_preprocess` (`pairs.parquet`, `products_clean.parquet`);
- output `assemble` của `12_deberta_large_t4x2`: `oof/ce_debertalarge_main_ce4.parquet`, có `p_E..p_I` cho mọi cặp train, mỗi cặp do fold model chưa thấy nó chấm.

**Soft target.** Ghép `p_E..p_I` của split `train` vào `train_df`, đổi tên thành `soft_E..soft_I`. Assert: mọi cặp train đều có soft target, và tổng mỗi hàng ≈ 1.

**Cấu hình** (bằng `d1_minilm_main_mse_s42`, trừ loss):
```python
BASE = {'model': 'cross-encoder/ms-marco-MiniLM-L-12-v2', 'fields': 'main', 'max_len': 256,
        'epochs': 1, 'batch_size': 32, 'lr': 7e-6, 'warmup_steps': 5000,
        'weight_decay': 0.01, 'max_grad_norm': 1.0, 'fp16': True, 'eval_batch_size': 512}
```

| Run | loss | α | seed | Ghi chú |
|---|---|---|---|---|
| `d4_minilm_main_ce4_s42` | ce4 | (1.0) | 42 | reference nhãn cứng; **lưu checkpoint** (dùng cho T4, T5) |
| `d4_minilm_main_kd05_s42` | ce4_soft | 0.5 | 42 | nửa nhãn cứng, nửa teacher |
| `d4_minilm_main_kd00_s42` | ce4_soft | 0.0 | 42 | chỉ teacher |
| `d4_minilm_main_<best>_s43` | (α tốt nhất trên val) | | 43 | seed thứ hai |
| `d4_minilm_main_ce4_s43` | ce4 | | 43 | seed thứ hai của reference |

Mỗi run khoảng 45 phút (train 34 + predict 7 + eval 4), tổng 5 run khoảng 3.75 giờ. Chia làm 2 session: s42 (3 run) và s43 (2 run, sau khi đã chọn α).

**Tùy chọn** (nếu còn quota): `d4_deberta_main_kd<best>_s42`, distill large vào DeBERTa-base (khoảng 3 giờ). Câu hỏi: base có đuổi kịp large không?

**Smoke:** 5% train; loss `ce4_soft` hữu hạn; val nDCG của `kd00` > 0.80; assert `α=1` cho cùng loss với ce4 trên batch đầu (lặp lại unit test với model thật).

**Chỉ số chính** (trên val, rồi test sau 29/11):
- `recovered = (student − ce4 ref) / (large 5-fold − ce4 ref)`, kèm CI bootstrap theo query (tử số và mẫu số cùng resample);
- chi phí inference (ms GPU / 1.000 cặp) lấy từ `predict_minutes`.

**Đường Pareto (Cong, CPU, notebook 18):** nDCG theo **GPU-giây / 1.000 cặp**, tính bằng `predict_minutes × 60 × số GPU / số cặp × 1000`. Large chạy trên 2 GPU nên phải nhân 2. Ensemble thì cộng chi phí các model thành phần. Các điểm: BM25 (CPU, ghi riêng), bi-encoder, MiniLM title/main, student, base (1 model, 5 model), large (1, 5), final.

**Rủi ro.**
- Soft target từ fold model chỉ được train trên 80% dữ liệu, nên hơi yếu hơn large full. Ghi rõ.
- Head 4 lớp của ce4 được khởi tạo ngẫu nhiên, còn MSE B3 giữ head MS MARCO. So sánh chính là student với `d4_*_ce4` (cùng head), không so với `d1_minilm_main_mse`.

### T3. DeBERTa-base ce4 seed 43, có lưu checkpoint

**Vì sao.** (1) Cần một checkpoint DeBERTa cho T4 và T5. (2) Có được seed thứ hai cho base ce4, để biết Δ ce4 so với MSE và mức nhiễu seed của backbone final.

**Notebook:** `15_deberta_base_s43.ipynb` (copy phần train của `06` với `SETUP='deberta_main'`, thêm cell lưu model). **Kaggle:** 1 × T4.

```python
CFG = {'model': 'microsoft/deberta-v3-base', 'fields': 'main', 'max_len': 256, 'loss': 'ce4',
       'lr': 2e-5, 'warmup_steps': 1200, 'epochs': 1, 'batch_size': 32, 'weight_decay': 0.01,
       'max_grad_norm': 1.0, 'fp16': True, 'eval_batch_size': 64}      # = d2_deberta_main_ce4_s42
SEED = 43
```
Run: `d2_deberta_main_ce4_s43`, khoảng 2.8 giờ (train 143 + predict 22 phút). Đặt `eval_every = None`.

**Lưu checkpoint:** `model.half().save_pretrained(...)` và `tokenizer.save_pretrained(...)` vào `/kaggle/working/models/d2_deberta_main_ce4_s43/`. Lưu **sau** khi predict xong, vì `half()` đổi dtype của model.

**Kiểm tra:** val nDCG nằm trong khoảng 0.8891 ± 0.003 (s42 là 0.8891). Nếu lệch hơn thì kiểm tra lại cấu hình trước khi dùng checkpoint.

**Cách đọc:** báo cáo s42 và s43 của ce4 cạnh MSE s42 và s43, rồi cập nhật bảng nhiễu seed (R2). So sánh `ce4 vs MSE` trên DeBERTa giờ có 2 seed ở mỗi bên.

### T4. Kiểm thử hành vi với phủ định (N1-B)

**Câu hỏi.** Model có đổi score theo đúng hướng khi query đổi giữa "without Y", "with Y" và không nhắc Y không?

**Notebook:** `16_negation_probe.ipynb`. **Kaggle:** 1 × T4, không cần internet (attach các checkpoint).

**Model** (tất cả đều có checkpoint):

| Model | Checkpoint |
|---|---|
| B3 MiniLM title MSE | output của `03_cross_encoder` (`model_b3_minilm_title_mse_s42`) |
| MiniLM main ce4 | output của T2 (`d4_minilm_main_ce4_s42`) |
| MiniLM main distill | output của T2 (run tốt nhất) |
| DeBERTa-base ce4 | output của T3 (`d2_deberta_main_ce4_s43`) |

**Tập query:**
1. Lấy mọi query **val** có `negation_phrase(query) is not None`. Dùng val để không đụng thêm vào test. Nếu val có ít hơn 150 query hợp lệ thì thêm test, và ghi tên vào danh sách chốt trước ngày 29/11.
2. Chạy `16_negation_probe` với `MODE = 'prepare'` (CPU) để sinh `negation_probe_candidates.csv`. An đánh dấu `keep` = 1/0 (khoảng 1 giờ), bỏ các câu không phải loại trừ thật (tên phim, thành ngữ), có thể sửa tay `term`/`drop`/`flip`/`para`. Lưu thành `eda/tables/negation_probe_queries.csv`, commit, và đưa vào `wm-src` **trước khi** chạy model. Notebook tìm file này ở bất kỳ đâu trong input.

**Chạy:** mỗi query có 4 biến thể (`orig`, `drop`, `flip`, `para`), chấm trên đúng list candidate gốc. Khoảng 200 query × 4 × 20 cặp ≈ 16k cặp cho mỗi model, mất vài phút.

**Phép đo** (CI bootstrap theo query cho mọi phép đo):

| Phép đo | Định nghĩa | Kỳ vọng nếu model hiểu phủ định |
|---|---|---|
| Directional (có Y) | tỷ lệ cặp có `term_status == has` mà `s(orig) < s(flip)` | ≫ 0.5 |
| Directional (đã loại Y) | tỷ lệ cặp có `term_status == negated` mà `s(orig) > s(flip)` | ≫ 0.5 |
| Độ nhạy | median của `|s(orig) − s(drop)|` / median của `|s|` | rõ ràng > 0 |
| Invariance | Kendall τ giữa thứ hạng `orig` và `para` | ≈ 1 |
| Δ nDCG nhờ phủ định | nDCG(`orig`) − nDCG(`drop`) với nhãn thật (val) | > 0 |

**Kết hợp với audit nhãn (N1-A):** dùng bảng đọc kết quả trong continue.md. Nếu model đổi hướng đúng nhưng Δ nDCG ≈ 0, đó là dấu hiệu nhãn không phạt vi phạm phủ định.

### T5. Tập xác nhận độc lập: query "dễ" (N4)

**Câu hỏi.** Trên các query `us` mà bản reduced đã loại (`small_version == 0`, chưa ai dùng), thứ tự các phương pháp có giữ nguyên không?

**Notebook:** `17_easy_holdout.ipynb`. **Kaggle:** 1 × T4, **Internet on** (tải dataset đầy đủ).

**Bước 1: chốt mẫu (CPU, làm trước, commit vào git).**
- Tải `shopping_queries_dataset_examples.parquet` và `shopping_queries_dataset_products.parquet` từ `amazon-science/esci-data` (ghi lại commit).
- Lọc `product_locale == 'us'`, `small_version == 0`, `split == 'test'`. Loại query có text chuẩn hóa trùng với bất kỳ query nào của Task 1, hoặc jaccard token ≥ 0.8 với một query Task 1.
- Lấy mẫu **3.000 query** (seed 42). Ghi `data/splits/easy_holdout_queries.csv`, gồm query_id, số cặp và phân bố nhãn. **Commit file này trước bước 2.**

**Bước 2: chấm (GPU).**
- Làm sạch product text bằng `src/text.py` (giống `01_preprocess`), dựng `main` và `title`.
- Các model: B0 random, BM25 `main` (IDF tính trên toàn bộ sản phẩm `us` của bản small như B1, ghi rõ), bge zero-shot `main`, B3 (checkpoint), MiniLM main ce4 (T2), student distill (T2), DeBERTa-base ce4 s43 (T3). Large và final: **không có** (thiếu checkpoint).
- Khoảng 60k cặp. Base predict khoảng 6 phút, MiniLM khoảng 2 phút, bge khoảng 5 phút. Tổng khoảng 1 giờ, tính cả tải và làm sạch.

**Báo cáo:** bảng nDCG, nDCG@10, E-over-S theo phương pháp, Task 1 test cạnh tập dễ, và Spearman giữa thứ tự các phương pháp trên hai tập. **Chỉ chấm một lần**, sau 29/11.

### T6. LightGBM 5 seed và thay đổi candidate list (R2, N6)

**Notebook:** `18_lgbm_seeds_lists.ipynb` (CPU). **Input:** output của `08_features` (bản có `ce3_*`).

**R2:** cấu hình final (lambdarank, `small`) với `seed` ∈ {0..4}, thêm `bagging_fraction=0.8`, `bagging_freq=1`. Mỗi seed chạy một lần 5-fold CV, khoảng 2 phút. Báo cáo mean ± sd của nDCG trên OOF, val và test, cùng Δ so với large CE 5-fold cho từng seed.

**N6** (theo continue.md): bỏ ngẫu nhiên 30% candidate (không nhìn nhãn, 5 lần lặp), tính lại các feature trong list, chấm lại bằng 5 fold model của final. Đo Kendall τ trên các sản phẩm còn lại và nDCG trên list thu nhỏ của final so với large CE. Chỉ dùng val.

### T7. LLM reranker trên tập query khó (N5, tùy chọn)

**Chỉ làm khi T1–T5 đã xong và còn quota.**

**Notebook:** `19_llm_rerank.ipynb`. **Kaggle:** **2 × T4** (chỉ notebook này và notebook 12 dùng 2 GPU), Internet on.
- Model: `Qwen/Qwen2.5-7B-Instruct`, fp16, vLLM `tensor_parallel_size=2`, `max_model_len` 4096, `temperature 0`.
- Prompt listwise: query cùng tối đa 20 candidate, đánh số `[1]..[n]`, mỗi candidate là title + brand + color cắt còn khoảng 40 token. Yêu cầu trả về đúng một dòng gồm các số theo thứ tự liên quan giảm dần.
- Parse: lấy các số hợp lệ, bỏ trùng; candidate bị thiếu xếp cuối theo thứ tự gốc. Báo cáo tỷ lệ parse lỗi.
- Tập đánh giá: các query phủ định của T4, cộng 300 query val ngẫu nhiên (seed 42). Chốt trước khi chạy.
- So sánh với DeBERTa-base s43 và MiniLM main trên cùng query, bằng paired bootstrap. Ghi chú: LLM zero-shot, không fine-tune.

---

## 3. Lịch và cách chia việc

| Ngày | Duy (GPU) | Thao | An | Cong |
|---|---|---|---|---|
| 23/11 | review C1 | **C1** + test; C3 + test | C2 + test; chọn query phủ định (T4) | khung notebook 18 |
| 24/11 | smoke T3, rồi full T3 | smoke T1, rồi full T1 | chốt mẫu T5 bước 1 (CPU) | T6 (R2) |
| 25/11 | smoke T2, rồi full T2 (s42) | đọc T1 | R3, R4 (calibration, dùng C2) | N6, Pareto (chờ T2) |
| 26–27/11 | T2 (s43); T5 bước 2 (predict blind) | T4 (khi có checkpoint của T2, T3) | audit nhãn N1-A | N2 |
| 28/11 | dự phòng / T7 nếu còn quota | viết N1 | viết R1–R4 | viết N2, N3-A |
| **29/11** | **chốt danh sách các phép đo trên test** | | | |
| 30/11–3/12 | `10_report` (Supplement), T5 chấm một lần | | | |

## 4. Checklist trước mỗi run full

- [ ] Smoke pass, kèm kiểm tra cụ thể của từng thẻ.
- [ ] `wm-src` là version build từ nhánh `supplement`, và `src_version` được ghi trong json.
- [ ] Tên run đúng quy ước, output vào `results/supplement/`, không trùng tên run cũ.
- [ ] Đúng loại GPU: 1 × T4 cho mọi notebook trừ notebook 12 và 19.
- [ ] Có cell lưu checkpoint (T2 reference, T3).
- [ ] Test chỉ được predict và lưu, không chấm trong notebook train.
- [ ] Sau khi chạy: tải `results/supplement/` về, commit vào nhánh `supplement` kèm tên version Kaggle trong commit message.

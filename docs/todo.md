# TODO: các việc còn lại (nhánh `supplement`)

Cập nhật: 23/11/2026 · Hạn chạy xong: **3/12** · Chốt danh sách phép đo trên test: **29/11** · Đóng băng (G4): **6/12**
Chi tiết từng mục: `docs/continue.md` (mã R*, N*) và `docs/plan_training.md` (mã T*, C*).

## Đã xong (chưa commit)

- [x] C1 `src/cross_encoder.py`: `sampler`, `list_size=None`, `warmup_ratio`, loss `ce4_soft`
- [x] C2 `src/evaluation.py`: `calibration()` (NLL, Brier, ECE)
- [x] C3 `src/probes.py`: parse phủ định, sinh biến thể, `term_status`
- [x] Test: `tests/test_cross_encoder.py`, `tests/test_probes.py`, thêm vào `tests/test_evaluation.py` (30 test pass)
- [x] Notebook 13–16 (mới compile và dry-run phần phân tích; **chưa chạy trên GPU**)

## 1. Chuẩn bị trước khi lên Kaggle

- [ ] Commit toàn bộ thay đổi vào nhánh `supplement` (`src/`, `tests/`, `notebooks/13–16`, `docs/`)
- [ ] (Tùy chọn) Tag commit final trên `main` là `final-v1`
- [ ] Thêm notebook 13–16 vào README
- [ ] Build lại dataset `wm-src` từ nhánh `supplement` (notebook báo lỗi nếu dùng bản cũ)
- [ ] Kiểm tra output của `12_deberta_large_t4x2` ở mode `assemble` (`ce_debertalarge_main_ce4.parquet`) còn trên Kaggle, vì notebook 14 cần file này
- [ ] (Tùy chọn) Chạy lại `03_cross_encoder` với `SAVE_MODEL = True` (khoảng 15 phút) để có checkpoint B3 cho notebook 16 và 17

## 2. Chạy GPU (1 × T4, trừ khi ghi khác)

| | Notebook | Thẻ | Thứ tự chạy | Phụ thuộc | Thời gian |
|---|---|---|---|---|---|
| [ ] | `13_d2b_sampler` | T1 / R5 | smoke → full | không | ~2.3 h |
| [ ] | `14_distill` | T2 / N3-B | smoke → full `STAGE='s42'` → đặt `KD_BEST` → full `STAGE='s43'` | output 12 assemble | ~2.3 + 1.5 h |
| [ ] | `15_deberta_base_s43` | T3 | smoke → full | không | ~2.8 h |
| [ ] | `16_negation_probe` | T4 / N1-B | `prepare` (CPU) → **lọc tay** ra `negation_probe_queries.csv` → smoke → full | checkpoint của 14, 15 (và B3 nếu có) | ~0.5 h |
| [ ] | `17_easy_holdout` *(chưa viết)* | T5 / N4 | bước 1 chốt mẫu 3.000 query, **commit trước** → bước 2 chấm, **chỉ chấm một lần, sau 29/11** | checkpoint của 14, 15 | ~1 h, Internet on |
| [ ] | `19_llm_rerank` *(chưa viết, tùy chọn)* | T7 / N5 | chỉ làm khi T1–T5 xong và còn quota | query list của 16 | ~1.5 h, **2 × T4** |

13, 14, 15 chạy song song được trên 3 tài khoản. Sau mỗi run: tải `results/supplement/` về và commit kèm tên version Kaggle.

## 3. Phân tích CPU

| | ID | Việc | Ghi chú |
|---|---|---|---|
| [ ] | R1 | Final hai mức (0.8921 chốt trước test, 0.8980 sau khi thấy test) và bảng nhật ký 3 đợt chấm test | trong `10_report` và chương kết quả |
| [ ] | R2 | LightGBM 5 seed (`bagging_fraction=0.8`) và bảng nhiễu seed | notebook `18_lgbm_seeds_lists` *(chưa viết)* |
| [ ] | N6 | Bỏ 30% candidate, tính lại feature trong list, đo Kendall τ và nDCG so với large CE | cùng notebook 18, chỉ dùng val |
| [ ] | R3 | Memorization: query gần trùng (171 cặp) và sản phẩm đã thấy trong train | |
| [ ] | R4 | Calibration base và large ce4 (1 model và trung bình 5 fold), reliability diagram P(E) | dùng `calibration()` từ C2 |
| [ ] | R6 | Ghi version dataset chứa các file score vào README, chạy lại `10_report` và so `results_all.csv` | |
| [ ] | N1-A | Audit nhãn phủ định khoảng 120 cặp (người), 40 cặp gán đôi | lấy mẫu **trước** khi xem kết quả 16 |
| [ ] | N2 | E-over-S theo loại thuộc tính khác biệt, kèm S1 (tách nhóm `constraint` trong ablation) | dùng val |
| [ ] | N3-A | Đường Pareto: nDCG theo ms GPU / 1.000 cặp | cần student từ 14 |
| [ ] | S2 | Stage 2 theo từng fold | ưu tiên thấp |

## 4. Báo cáo và tài liệu

- [ ] **29/11:** chốt danh sách mọi phép đo trên test của phần bổ sung (R3, N1-B nếu dùng query test, N4)
- [ ] Thêm mục Supplement vào `10_report` (D2b, distill, base s43, kiểm thử phủ định, query dễ)
- [ ] Cập nhật `docs/architecture.md` cho khớp với bản đã cài (tên D1/D2/D3, DeBERTa 5-fold, large OOF, final hai mức)
- [ ] Ghi các hạn chế:
  - final được đổi sang large sau khi đã thấy test (R1);
  - không còn weights fine-tuned của DeBERTa base/large, nên kiểm thử phủ định và query dễ chạy trên base s43 và MiniLM, **không** trên large hay final;
  - R2 chỉ đo phương sai của LightGBM, không đo phương sai của CE large;
  - từ D1 trở đi CE chỉ 1 seed (trừ base s43 mới), nên mọi Δ ≤ 0.002 chưa phân biệt được với nhiễu (0.0014).
- [ ] Viết các chương: N1 (lỗi do model hay do nhãn), N2, N3 (chi phí–chất lượng), kết quả D2b
- [ ] **6/12:** đóng băng kết quả (G4)

## Không làm

- S7 (hybrid loss), S8 (noise handling), S9, S10: đã bỏ (xem `continue.md`, Phần 2).
- Train lại DeBERTa-large chỉ để kiểm thử phủ định: quá đắt (~25 giờ trên 2 × T4).
- Đổi hệ thống final thêm lần nữa.

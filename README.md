# Semantic Matching and Ranking for E-commerce Product Search

IT4868 Web Mining project (HUST, SoICT). Query–product ranking on the Amazon Shopping Queries dataset (ESCI), **KDD Cup 2022 Task 1**.

Given a query and its list of candidate products, rank the candidates so that relevant products come first. Each query–product pair is labelled **E**xact, **S**ubstitute, **C**omplement or **I**rrelevant. Rankings are scored with nDCG over the full candidate list, using the official gains E = 1, S = 0.1, C = 0.01, I = 0.

**Scope:** Task 1 only, English (`us`) locale, reduced (small) version. The candidates are given, so there is no retrieval over the full catalogue.

## Approach

| Stage | Model |
| --- | --- |
| Baselines | B1 BM25 inside each candidate list · B2 zero-shot bi-encoder · B3 reproduction of Amazon's MiniLM cross-encoder |
| Direction 1 | Stronger cross-encoder: brand, color and bullet points added to the input; `deberta-v3-base` backbone |
| Direction 2 | Training objectives for graded labels: MSE on gains, 4-class cross-entropy → expected gain, ListNet |
| Direction 3 | LightGBM re-ranker on out-of-fold cross-encoder scores plus list-level and lexical features |

Details: `proposal/proposal.tex`.

## Data

| File | Rows | Content |
| --- | ---: | --- |
| `examples_us_small.parquet` | 601,354 | Query–product pairs with label and official train/test split |
| `products_us_small.parquet` | 482,105 | Title, description, bullet points, brand, color |
| `sources_us_small.csv` | 29,844 | Query type (other, parse_pattern, negations, behavioral, nlqec), used only for analysis |

Train: 20,888 queries / 419,653 pairs. Test: 8,956 queries / 181,701 pairs. Source: [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Apache-2.0).

The data files are **not stored in this repository**. They are kept in a private Kaggle dataset (`esci-raw`) and, locally, in the folder that contains the repository.

## Repository layout

```text
.
├── README.md
├── proposal/proposal.tex          project proposal (LaTeX)
├── eda/                           exploratory analysis, frozen
│   ├── eda_audit.py, common.py    builds the validation split and the tables below
│   ├── manifest.json              SHA-256 of the raw data files
│   ├── figures/                   label shares, list sizes, text length, missing fields, signals
│   └── tables/
│       ├── query_groups_val_proposal.csv   validation split (input of 01_preprocess)
│       ├── near_dup_pairs_train_test.csv   near-duplicate train/test queries (memorization check)
│       ├── single_feature_ranking.csv      nDCG of each feature used alone
│       └── rule_check_*.csv                hand-checked samples of the negation and color rules
├── src/                           shared code, imported by every notebook
│   ├── evaluation.py              nDCG (both gain mappings), nDCG@10, MRR, E-over-S, bootstrap,
│   │                              RunLog (writes <run>.json + scores)
│   ├── text.py                    field cleaning, product text presets
│   ├── cross_encoder.py           training loop, prediction, losses
│   ├── features.py                BM25 (baseline B1); Direction 3 match rules, within-query and brand-list features
│   └── data.py                    find(), load_pairs(fields): outputs of 01_preprocess
├── tests/                         test_evaluation.py (metrics, text), test_features.py (Direction 3 features)
├── notebooks/                     run on Kaggle, one notebook per experiment
│   ├── 01_preprocess.ipynb        clean, join, split
│   ├── 02_bm25.ipynb              B0 random, B1 BM25
│   ├── 03_cross_encoder.ipynb     B3 (Amazon MiniLM reproduction, GPU)
│   ├── 04_biencoder.ipynb         B2 (zero-shot bi-encoders, GPU) + features for Direction 3
│   ├── 05_d1_input_backbone.ipynb Direction 1: title_bc / main input, DeBERTa-v3-base (GPU)
│   ├── 06_d2_losses.ipynb         Direction 2: MSE vs 4-class CE vs ListNet (MiniLM title; ce4 on DeBERTa main)
│   ├── 07_oof.ipynb               Direction 3 input: 5-fold out-of-fold DeBERTa main ce4 scores + p_E..p_I (GPU)
│   ├── 08_features.ipynb          Direction 3 feature table: CE, within-query, lexical, bi-encoder, brand-list (CPU)
│   ├── 09_lgbm.ipynb              Direction 3: LightGBM re-ranker, lambdarank vs multiclass, 5-fold CV, ablation (CPU)
│   ├── 10_report.ipynb            final test scores, paired tests, breakdown by query type, error analysis (CPU)
│   ├── 11_bge_reranker.ipynb      second backbone: bge-reranker-base (XLM-R) main ce4, vs and averaged with DeBERTa (GPU)
│   └── 12_deberta_large_t4x2.ipynb DeBERTa-v3-large main ce4 on 2 x T4 (DDP via torchrun): full run, 5-fold OOF + assemble
└── data/                          small outputs only; large files stay on Kaggle
    ├── splits/                    split_manifest.csv, data_check.json, label_conflicts.csv
    ├── results/                   <run>.json + scores/<run>_{val,test}.parquet
    └── report/                    tables of 10_report (final results, paired tests, by query type, worst queries)
```

## Workflow

Every step runs as a **Kaggle notebook** (CPU for data and BM25, GPU T4 with internet for transformers). Notebooks are chained on Kaggle with *Add Input → Notebook Output*, so large intermediate files never leave Kaggle.

```text
esci-raw ──► 01_preprocess ─┬─► 02_bm25 (B1) ─────────────────────────┐ (08 also reads 01, 02, 04)
                            ├─► 03_cross_encoder (B3)                 │
                            ├─► 04_biencoder (B2) ────────────────────┤
                            └─► 05_d1, 06_d2 ──► 07_oof ──► 08_features ──► 09_lgbm ──► 10_report
```

05 and 06 choose the cross-encoder configuration on val; 07 trains it once per fold; 08 builds the re-ranker's feature table; 09 trains the LightGBM re-ranker; 10 scores every chosen run on test once.

| Output | Where it lives | In git |
| --- | --- | :---: |
| Raw data (225 MB) | Kaggle dataset `esci-raw` | no |
| `pairs.parquet`, `products_clean.parquet` | Output of `01_preprocess` on Kaggle (can be rebuilt) | no |
| Split manifest, data checks | `data/splits/` | yes |
| Run metrics and per-pair scores | `data/results/` | yes |
| Out-of-fold scores of 07 (`oof_*` fold files, `oof/ce_*.parquet`, ~70 MB) | Output of `07_oof` on Kaggle; local copy in `../kaggle_outputs/oof_scores/` (fold json files stay in `data/results/`) | no |
| Feature table of 08 (`features.parquet`, ~70 MB) | Output of `08_features` on Kaggle; local copy in `../kaggle_outputs/features/` | no |
| Report tables of 10 (final results, paired tests, by query type, worst queries) | `data/report/` | yes |

## Reference numbers

| Ordering | nDCG |
| --- | ---: |
| Random order (validation) | 0.742 |
| BM25 on titles, IDF from the full catalogue (validation) | 0.816 |
| Amazon MiniLM cross-encoder (paper, test) | 0.857 |
| All Exact items first (test) | 0.997 |

The BM25 score in the ESCI paper (0.675) is below random order on the candidate lists, so it is not used as a reference.

## Final results (test, `10_report`)

Every configuration was chosen on val (B, D1, D2) or on train CV + val (D3) before test was scored, once, in `10_report`. Test nDCG with official gains; configurations run with two seeds show the mean. Full tables: `data/report/`.

| Stage | Configuration | Val | Test [CI 95%, seed 42] | Test nDCG@10 | Test E-over-S |
| --- | --- | ---: | ---: | ---: | ---: |
| B1 | BM25 `main` | 0.8227 | 0.8244 [0.8209, 0.8275] | 0.701 | 0.619 |
| B2 | bge-base bi-encoder `main`, zero-shot | 0.8429 | 0.8471 [0.8440, 0.8500] | 0.740 | 0.643 |
| B3 | MiniLM cross-encoder `title`, MSE (2 seeds) | 0.8572 | 0.8552 [0.8518, 0.8578] | 0.754 | 0.677 |
| D1 | MiniLM `main`, MSE (2 seeds) | 0.8684 | 0.8660 [0.8625, 0.8685] | 0.772 | 0.695 |
| D1 | DeBERTa-v3-base `main`, MSE (2 seeds) | 0.8877 | 0.8868 [0.8845, 0.8902] | 0.809 | 0.730 |
| D2 | DeBERTa-v3-base `main`, ce4 | 0.8891 | 0.8867 [0.8838, 0.8895] | 0.808 | 0.728 |
| D3 | DeBERTa `main` ce4, mean of 5 fold models | 0.8920 | 0.8885 [0.8855, 0.8912] | 0.811 | 0.731 |
| D3 | LightGBM lambdarank | 0.8936 | 0.8920 [0.8891, 0.8945] | 0.816 | 0.733 |
| **D3** | **LightGBM multiclass (final)** | **0.8947** | **0.8921 [0.8890, 0.8947]** | **0.817** | **0.733** |

Paired differences on test (bootstrap CI over the 8,956 test queries; all Wilcoxon p < 0.005 unless marked n.s.):

| Question | Test difference [CI 95%] | Same conclusion as val? |
| --- | --- | --- |
| Final vs B3 | +0.0372 [0.0348, 0.0395] | yes |
| Final vs BM25 `main` | +0.0677 [0.0649, 0.0704] | yes |
| D1 `main` vs `title` (DeBERTa) | +0.0102 [0.0086, 0.0118] | yes |
| D1 DeBERTa vs MiniLM (`main`, s42 / s43) | +0.022 / +0.020 | yes |
| D2 ce4 vs MSE (MiniLM s42 / s43; DeBERTa) | −0.0005 / +0.0012 / −0.0009 (n.s. except s43) | yes: no real difference |
| D2 ListNet vs MSE (MiniLM s42 / s43) | +0.0032 / +0.0026 (both CI exclude 0) | no: n.s. on val, small gain on test |
| D3 mean of 5 fold models vs 1 model on all train | +0.0019 [0.0007, 0.0029] | yes |
| D3 LightGBM multiclass vs CE (5 folds) | +0.0035 [0.0023, 0.0047] | yes |
| D3 multiclass vs lambdarank | +0.0001 [−0.0007, 0.0009] (n.s.) | yes: a tie |
| Seed: DeBERTa `main` MSE s43 vs s42 | −0.0014 [−0.0025, −0.0001] | seed noise is about 0.001 |

### Additional experiments (after the final system was chosen)

Run after the choice above, so none of them changes it; their test scores are reported, not used to choose. Blends: scores standardized within each query, then weighted. Each weight is the best of 0.1–0.5 on val (bge 0.2 with DeBERTa-base, reused for the MiniLM blend and final + bge; base 0.3 and bge 0.2 with large; large 0.5 with the final system), so val is slightly optimistic for blends and test is the fair comparison. Rows `Extra` in `data/report/`.

| Run | Val | Test [CI 95%] | Paired difference, test [CI 95%] |
| --- | ---: | ---: | --- |
| DeBERTa `main` ce4, 2 epochs | 0.8889 | 0.8858 [0.8828, 0.8887] | vs 1 epoch: −0.0008 [−0.0019, 0.0004] (n.s.) |
| bge-reranker-base `main` ce4 (`11_bge_reranker`) | 0.8671 | 0.8659 [0.8628, 0.8687] | vs DeBERTa: −0.0208 [−0.0229, −0.0185]; vs MiniLM `main`: +0.0003 (n.s.) |
| Blend DeBERTa ce4 + 0.2 bge-reranker | 0.8922 | 0.8888 [0.8860, 0.8916] | vs DeBERTa: +0.0022 [0.0015, 0.0028] |
| Blend DeBERTa ce4 + 0.2 MiniLM `main` | 0.8894 | 0.8874 [0.8844, 0.8902] | vs DeBERTa: +0.0007 [−0.0001, 0.0015] (n.s.) |
| Blend final + 0.2 bge-reranker | 0.8952 | 0.8926 [0.8895, 0.8952] | vs final: +0.0005 [−0.0003, 0.0011] (n.s.) |
| DeBERTa-v3-large `main` ce4 (`12_deberta_large_t4x2`, 2 × T4) | 0.8946 | 0.8936 [0.8908, 0.8964] | vs base: +0.0070 [0.0052, 0.0086]; vs final: +0.0015 [−0.0001, 0.0032] (n.s.) |
| Blend large + 0.3 DeBERTa-base | 0.8967 | 0.8950 [0.8923, 0.8979] | vs large: +0.0014 [0.0007, 0.0022] |
| Blend large + 0.2 bge-reranker | 0.8964 | 0.8946 [0.8917, 0.8973] | vs large: +0.0009 [0.0002, 0.0017] |
| **Blend final + 0.5 large** | **0.8978** | **0.8968 [0.8938, 0.8995]** | vs final: **+0.0047 [0.0037, 0.0059]** |

- **A second epoch does not help.** Val stays at 0.886–0.889 through epoch 2 while the training loss drops at the epoch boundary (memorization), so 1 epoch is kept.
- **A different model family adds, a weaker model of the same kind does not.** bge-reranker and MiniLM `main` score the same alone and agree with DeBERTa about equally, yet only bge improves the blend. Whether a model helps an ensemble has to be measured, not inferred from its own score.
- **Most of that gain is already in the final system.** Added to LightGBM the blend gives +0.0005 (n.s.): the bi-encoder and lexical features already carry similar information. Using bge properly would need 5-fold OOF scores as a feature group (expected +0.001–0.002), not done.
- **DeBERTa-v3-large is the strongest single model.** Same settings as the base ce4 run except lr 1e-5, trained on 2 × T4 (DDP, batch 32 = 2 GPUs × 4 micro-batches of 4 pairs; 4.7 h training + 0.5 h prediction). Alone it matches the LightGBM final on val (0.8946 vs 0.8947) and is slightly above it on test (n.s.); it beats DeBERTa-base by +0.007 on test, and every metric improves (nDCG@10 0.818, E-over-S 0.737).
- **Large adds to the final system.** Averaged with LightGBM (z-scores, weight 0.5) it gives +0.0047 on test (CI excludes 0, 4,150 queries better / 3,208 worse), ten times the bge blend. This blend was chosen after the final system and its weight on val, so it is reported as an extra result; the clean way to use large is 5-fold OOF scores as a LightGBM feature group (~27 GPU hours on 2 × T4).

Error analysis (`10_report`, sections 5–8):

- **Query type.** Negation queries are the hardest (final 0.848 vs 0.897 for `other`); fine-tuned cross-encoders gain most there over BM25 / bi-encoders (+0.07 to +0.09).
- **Labels.** Mean position in the list (0 = top): E 0.381, S 0.568, C 0.623, I 0.638 for the final system; the S − E gap grows from 0.099 (BM25) to 0.187.
- **Per query vs B3.** 6,023 test queries improve, 2,645 get worse; most large drops are lists with a single E and no S, where E falling from rank 1 to 2 costs about half of the nDCG.
- **Worst queries.** Several have a doubtful Exact label (e.g. "fake weed jar" → catnip, "cryptid" → an unrelated book), so label noise caps part of the score.

## Baseline results

nDCG with official gains, ties broken at random (mean of 3 seeds), 95% bootstrap CI over queries. Source: `data/results/*.json`.

| Run | Val | Test [CI 95%] | Test E-over-S |
| --- | ---: | ---: | ---: |
| B0 random order | 0.7422 | 0.7459 [0.7429, 0.7491] | 0.500 |
| B1 BM25 title | 0.8162 | 0.8209 [0.8179, 0.8240] | 0.610 |
| B1 BM25 main | 0.8227 | 0.8244 [0.8209, 0.8275] | 0.619 |
| MiniLM MS MARCO, zero-shot | 0.8330 | 0.8340 [0.8311, 0.8369] | 0.625 |
| B3 MiniLM fine-tuned, title, MSE (seed 42) | 0.8566 | 0.8549 [0.8518, 0.8578] | 0.676 |
| B3 MiniLM fine-tuned, title, MSE (seed 43) | 0.8578 | 0.8555 [0.8523, 0.8586] | 0.679 |
| B2 MiniLM-L6 bi-encoder, title, zero-shot | 0.8202 | 0.8281 [0.8248, 0.8311] | 0.606 |
| B2 MiniLM-L6 bi-encoder, main, zero-shot | 0.8264 | 0.8306 [0.8276, 0.8339] | 0.614 |
| B2 bge-base bi-encoder, title, zero-shot | 0.8418 | 0.8450 [0.8420, 0.8480] | 0.640 |
| B2 bge-base bi-encoder, main, zero-shot | 0.8429 | 0.8471 [0.8440, 0.8500] | 0.643 |

B3 reproduces Amazon's 0.857 (mean 0.8552; their training set was about 10% larger) and beats BM25 main by +0.031 [0.028, 0.034] on test. One run takes 10.5 minutes on a T4.

## Direction 1 and 2 results (validation, used to choose; test in Final results)

Val nDCG per seed (42 / 43). Paired differences are bootstrap over val queries, same seed.

| Run | Val nDCG | Val E-over-S | Paired difference |
| --- | ---: | ---: | --- |
| D1 MiniLM `title_bc`, MSE | 0.8588 / 0.8595 | 0.683 / 0.685 | vs B3: +0.002 / +0.002 (CI contains 0) |
| D1 MiniLM `main`, MSE | 0.8669 / 0.8700 | 0.698 / 0.697 | vs B3: +0.010 / +0.012 |
| D1 DeBERTa `title`, MSE | 0.8801 / – | 0.715 | vs B3: +0.024 |
| D1 DeBERTa `main`, MSE | 0.8878 / 0.8876 | 0.731 / 0.732 | vs MiniLM `main`: +0.021 / +0.018 |
| D2 MiniLM `title`, ce4 | 0.8560 / 0.8579 | 0.680 / 0.682 | vs B3: −0.001 / +0.000 (CI contains 0) |
| D2 MiniLM `title`, ListNet | 0.8593 / 0.8564 | 0.686 / 0.684 | vs B3: +0.003 / −0.001 (CI contains 0) |
| D2 DeBERTa `main`, ce4 | 0.8891 / – | 0.733 | vs DeBERTa `main` MSE: +0.001 (CI contains 0) |

Chosen cross-encoder for Direction 3: DeBERTa-v3-base, `main`, `max_len` 256, ce4 (ties with MSE and gives class probabilities).

## Direction 3 results (train CV + validation, used to choose; test in Final results)

LightGBM on the 48 features of `08_features`, 5-fold CV on train (out-of-fold CE features), val/test = mean of the 5 fold models. Reference: the CE score alone (train out-of-fold / val mean of 5 fold models). CV is slightly optimistic (early stopping on the held-out fold); val is not used for fitting or stopping.

| Run | CV nDCG (train) | Val nDCG [CI 95%] | Val nDCG@10 | Val E-over-S | Val vs CE alone |
| --- | ---: | ---: | ---: | ---: | --- |
| CE alone (DeBERTa `main` ce4, 5 folds) | 0.8848 | 0.8920 [0.8860, 0.8976] | 0.816 | 0.737 | – |
| LightGBM lambdarank (`base`) | 0.8888 | 0.8936 [0.8881, 0.8993] | 0.819 | 0.738 | +0.0016 [−0.0008, 0.0039] |
| LightGBM multiclass (`small`) | 0.8884 | 0.8947 [0.8891, 0.9003] | 0.821 | 0.738 | +0.0027 [0.0005, 0.0048] |

Both objectives add +0.004 on CV (CI excludes 0); on val only multiclass is clearly above the CE alone (also nDCG@10 +0.005, MRR +0.007). Multiclass − lambdarank: −0.0004 on CV, +0.0011 [−0.0004, 0.0025] on val (a tie). Grid settings differ by ≤ 0.001.

Ablation (lambdarank, paired with all features, CV): without any CE feature 0.8612 (−0.028; val 0.8605, above B3); CE features only −0.0035; dropping one other group costs 0.0005–0.0011 (bi-encoder most), except the query group (0). On val every single-group change is inside ±0.002 (CI contains 0). Gain importance: within-query CE features (`ce_score_z`, `p_E_z`, …) 86%, raw CE 5%, bi-encoder 5%.

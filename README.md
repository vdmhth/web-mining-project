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

Details: `proposal/proposal.tex` (proposal), `ke_hoach_du_an.md` (team plan, in Vietnamese).

## Data

| File | Rows | Content |
| --- | ---: | --- |
| `examples_us_small.parquet` | 601,354 | Query–product pairs with label and official train/test split |
| `products_us_small.parquet` | 482,105 | Title, description, bullet points, brand, color |
| `sources_us_small.csv` | 29,844 | Query type (other, parse_pattern, negations, behavioral, nlqec), used only for analysis |

Train: 20,888 queries / 419,653 pairs. Test: 8,956 queries / 181,701 pairs. Source: [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Apache-2.0).

The data files are **not stored in this repository**. They are kept in a private Kaggle dataset (`esci-raw`) and, locally, in the repository root (ignored by git).

## Repository layout

Status: **done** = in the repository; **planned** = to be added as the project progresses.

```text
.
├── README.md
├── proposal/                      project proposal                               done
├── eda/                           exploratory analysis, frozen                               done
│   ├── eda_analysis.py, eda_audit.py, common.py
│   ├── eda_report.pdf, eda_report_en.pdf
│   ├── eda_stats.json, audit_stats.json, manifest.json (SHA-256 of the data files)
│   ├── figures/
│   └── tables/                    single-feature ranking, near-duplicate queries, rule checks,
│                                  query_groups_val_proposal.csv (validation split)
├── src/                           shared code, imported by every notebook                    done
│   ├── evaluation.py              nDCG (both gain mappings), nDCG@10, MRR, E-over-S, bootstrap,
│   │                              RunLog (writes <run>.json + scores)
│   ├── text.py                    field cleaning, product text presets
│   ├── cross_encoder.py           training loop, prediction, losses
│   └── data.py                    find() for Kaggle inputs
├── tests/test_evaluation.py       metric and text checks on hand-made data                   done
├── notebooks/                     run on Kaggle, one notebook per experiment
│   ├── 1-preprocess.ipynb        clean, join, split                                         done
│   ├── 2-bm25.ipynb              B0 random, B1 BM25                                         done
│   ├── 3-cross_encoder.ipynb     B3 (Amazon MiniLM reproduction, GPU)                       done
│   ├── 4-biencoder.ipynb         B2 (GPU)                                                   planned
│   ├── 5-oof_lgbm.ipynb          out-of-fold scores + LightGBM re-ranker (Direction 3)      planned
│   └── 6-report.ipynb            result tables, significance tests, error analysis          planned
└── data/                          small outputs only; large files stay on Kaggle
    ├── splits/                    split_manifest.csv, data_check.json, label_conflicts.csv   done
    ├── results/                   <run>.json + scores/<run>_{val,test}.parquet               done
    └── tables/                    tables for the report (baseline_table.csv, ...)            planned
```

## Workflow

Every step runs as a **Kaggle notebook** (CPU for data and BM25, GPU T4 with internet for transformers). Notebooks are chained on Kaggle with *Add Input → Notebook Output*, so large intermediate files never leave Kaggle.

```text
esci-raw ──► 01_preprocess ──► 02_bm25 ─────────────┐
                 │                                   │
                 ├──► 03_cross_encoder ──► 05_oof_lgbm ──► 06_report
                 └──► 04_biencoder ─────────────────┘
```

| Output | Where it lives | In git |
| --- | --- | :---: |
| Raw data (225 MB) | Kaggle dataset `esci-raw` | no |
| `pairs.parquet`, `products_clean.parquet` | Output of `01_preprocess` on Kaggle (can be rebuilt) | no |
| Split manifest, data checks | `data/splits/` | yes |
| Run metrics and per-pair scores | `data/results/` | yes |
| Report tables | `data/tables/` | yes |

### Shared code (`src/`)

Notebooks hold the experiment (config, calls, tables, comments). Anything used by more than one notebook lives in `src/`: above all the evaluation, so every run is scored by the same code. The first cell of each notebook makes `src/` importable on Kaggle:

1. if a Kaggle dataset containing the repo is attached, it is used;
2. otherwise the repo is cloned into `/tmp` (needs *Internet on*; while the repo is private, add a Kaggle secret `GITHUB_TOKEN`).

The commit of `src/` is saved in every run's json (`src_version`). Rules:

- Do not edit `src/evaluation.py` without re-scoring the saved runs. `tests/test_evaluation.py` must pass (`python tests/test_evaluation.py`).
- A new loss goes into `compute_loss` in `src/cross_encoder.py`; a new experiment gets its own notebook, not new cells in someone else's.
- Push changes to `src/` before running the notebook that needs them.

### Running a notebook on Kaggle

1. Raw data: a private dataset `esci-raw` with the three data files and `eda/tables/query_groups_val_proposal.csv`.
2. *File → Import Notebook*, then add the inputs listed in the notebook's first cell (`esci-raw` or *Notebook Output* of an earlier notebook). CPU for 01 and 02, GPU T4 for 03. *Internet on*.
3. *Run All*, check the printed checks, then *Save Version → Save & Run All*.
4. Download the small outputs into `data/splits/` or `data/results/` and push them.

## Conventions

These rules apply to every experiment so that results stay comparable.

| Topic | Rule |
| --- | --- |
| Validation split | 10% of training queries (2,089), grouped by near-duplicate query (`dup_group`), seed 42. Taken from `eda/tables/query_groups_val_proposal.csv`. |
| Folds | `fold10` (0–9) on the remaining 18,799 training queries, assigned by `dup_group`. 2 folds = `fold10 // 5`, 5 folds = `fold10 // 2`. |
| Model selection | On validation only. The test split is scored once per final configuration. |
| Metrics | nDCG (main), nDCG@10, MRR of the first Exact item, E-over-S pair accuracy. Ties broken at random, averaged over 3 seeds. |
| Gains | Official (E 1, S 0.1, C 0.01, I 0). Amazon's script swaps S and C; `ndcg_amazon` keeps that mapping for reference only. The published 0.857 matches the official gains (B3 below). |
| Significance | Paired bootstrap / Wilcoxon on per-query nDCG. |
| Text cleaning | Light: HTML removed, NFC, whitespace collapsed. Case, digits, units and punctuation kept. |
| Product text | Presets `title`, `title_bc`, `main` (+ bullets), `all` (+ description); format `title [SEP] brand: X [SEP] color: Y [SEP] bullets`; missing fields dropped. |
| Banned features | Catalogue row position, `product_id` patterns, `small_version` / `large_version`, the product's labels in other queries. |
| Run names | `<model>_<fields>[_<loss>]_s<seed>`, e.g. `bm25_title`, `ce-minilm-l12_title_mse_s42`. |

## Reference numbers

| Ordering | nDCG |
| --- | ---: |
| Random order (validation) | 0.742 |
| BM25 on titles, IDF from the full catalogue (validation) | 0.816 |
| Amazon MiniLM cross-encoder (paper, test) | 0.857 |
| All Exact items first (test) | 0.997 |

The BM25 score in the ESCI paper (0.675) is below random order on the candidate lists, so it is not used as a reference. See section 6 of the EDA report.

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

B3 reproduces Amazon's 0.857 (mean 0.8552; their training set was about 10% larger) and beats BM25 main by +0.031 [0.028, 0.034] on test. One run takes 10.5 minutes on a T4.


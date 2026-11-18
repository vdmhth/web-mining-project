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
│   ├── features.py                BM25 (baseline B1, Direction 3 features)
│   └── data.py                    find(), load_pairs(fields): outputs of 01_preprocess
├── tests/test_evaluation.py       metric and text checks on hand-made data
├── notebooks/                     run on Kaggle, one notebook per experiment
│   ├── 01_preprocess.ipynb        clean, join, split
│   ├── 02_bm25.ipynb              B0 random, B1 BM25
│   └── 03_cross_encoder.ipynb     B3 (Amazon MiniLM reproduction, GPU)
└── data/                          small outputs only; large files stay on Kaggle
    ├── splits/                    split_manifest.csv, data_check.json, label_conflicts.csv
    └── results/                   <run>.json + scores/<run>_{val,test}.parquet
```

## Workflow

Every step runs as a **Kaggle notebook** (CPU for data and BM25, GPU T4 with internet for transformers). Notebooks are chained on Kaggle with *Add Input → Notebook Output*, so large intermediate files never leave Kaggle.

```text
esci-raw ──► 01_preprocess ──► 02_bm25 ─────────────┐
                 │                                  │
                 ├──► 03_cross_encoder ──► 05_oof_lgbm ──► 06_report
                 └──► 04_biencoder ─────────────────┘
```

| Output | Where it lives | In git |
| --- | --- | :---: |
| Raw data (225 MB) | Kaggle dataset `esci-raw` | no |
| `pairs.parquet`, `products_clean.parquet` | Output of `01_preprocess` on Kaggle (can be rebuilt) | no |
| Split manifest, data checks | `data/splits/` | yes |
| Run metrics and per-pair scores | `data/results/` | yes |

### Shared code (`src/`)

Notebooks hold the experiment (config, calls, tables, comments). Anything used by more than one notebook lives in `src/`: above all the evaluation, so every run is scored by the same code. `src/` reaches Kaggle as a private dataset **`wm-src`**; the first cell of each notebook finds it under `/kaggle/input` and makes it importable. After changing `src/`, zip it from the repo root:

```powershell
tar -a -c -f ..\wm-src.zip --exclude=__pycache__ src
```

Then upload `wm-src.zip` as a new version of the Kaggle dataset `wm-src`, and in the notebook click *Update* on that input.

Every run's json records `src_version.files_sha1` (a hash of `src/*.py`; two runs with the same value ran the same code) and `env` (Python and library versions). Rules:

- Do not edit `src/evaluation.py` without re-scoring the saved runs. `tests/test_evaluation.py` must pass (`python tests/test_evaluation.py`).
- A new loss goes into `compute_loss` in `src/cross_encoder.py`; a new experiment gets its own notebook, not new cells in someone else's.
- Rebuild and upload `wm-src` before running a notebook that needs a change in `src/`.

### Running a notebook on Kaggle

1. Raw data: a private dataset `esci-raw` with the three data files and `eda/tables/query_groups_val_proposal.csv`.
2. *File → Import Notebook*, then add the inputs listed in the notebook's first cell: always `wm-src`, plus `esci-raw` or the *Notebook Output* of an earlier notebook. CPU for 01 and 02, GPU T4 with *Internet on* for 03 (Hugging Face models).
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
| Run names | `<stage>_<model>[_<fields>][_<loss>][_s<seed>]`, lower case with `_`; stage is `b0`–`b3` or `d1`–`d3`. E.g. `b1_bm25_title`, `b3_minilm_title_zeroshot`, `b3_minilm_title_mse_s42`. |
| Notebook names | `NN_<topic>.ipynb` (two-digit number, `_`), e.g. `03_cross_encoder.ipynb`. |

## Reference numbers

| Ordering | nDCG |
| --- | ---: |
| Random order (validation) | 0.742 |
| BM25 on titles, IDF from the full catalogue (validation) | 0.816 |
| Amazon MiniLM cross-encoder (paper, test) | 0.857 |
| All Exact items first (test) | 0.997 |

The BM25 score in the ESCI paper (0.675) is below random order on the candidate lists, so it is not used as a reference.

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


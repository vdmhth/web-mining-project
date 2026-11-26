# Semantic Matching and Ranking for E-commerce Product Search

IT4868 Web Mining project (HUST, SoICT), on the Amazon Shopping Queries dataset (ESCI), **KDD Cup 2022 Task 1**.

**Task.** Given a query and its candidate products, rank the products. Each pair is labelled **E**xact, **S**ubstitute, **C**omplement or **I**rrelevant; the metric is nDCG over the full candidate list with the official gains E = 1, S = 0.1, C = 0.01, I = 0.

**Scope.** English (`us`) locale, small version: 20,888 train queries (419,653 pairs) and 8,956 test queries (181,701 pairs). The candidates are given, so there is no retrieval step.

**Result.** The final system reaches **test nDCG 0.8980** [0.8953, 0.9006], +0.043 over the Amazon cross-encoder baseline (0.8552 in our reproduction, 0.857 in the paper). The first version, without DeBERTa-large, reaches 0.8921 (see *How the final system was chosen*).

## Approach

| Stage | What is done | Notebook |
| --- | --- | --- |
| Data | Clean text, join, validation split by near-duplicate query groups | `01` |
| B1–B3 | BM25 · zero-shot bi-encoders · Amazon's MiniLM cross-encoder | `02`–`04` |
| D1 | Richer input (`main` = title + brand + color + bullet points) and DeBERTa-v3-base | `05` |
| D2 | Loss for graded labels: MSE, 4-class cross-entropy (ce4), ListNet | `06` |
| D3 | LightGBM re-ranker on out-of-fold scores of two cross-encoders (DeBERTa-v3-base and -large), list-level, lexical and bi-encoder features | `07`–`09`, `12` |
| Report | Test scores, paired tests, error analysis | `10` |
| Extra | 2 epochs (`06` with `epochs` = 2), bge-reranker, score blends | `06`, `11`, `10` |
| Supplement | Validation only: losses on the same batches, distillation into MiniLM, seed variance of DeBERTa-v3-base, negation probe, pooled comparisons | `13`–`16`, `10b` |

Configurations were chosen on validation (2,089 queries) or 5-fold CV on train, and scored on test in `10_report` after the choice. The supplementary runs were never scored on test.

## Results

nDCG with official gains. The 95% bootstrap CI is over the 8,956 test queries. Rows with 2 seeds show the mean, and the CI of seed 42.

| Stage | System | Val | Test [95% CI] |
| --- | --- | ---: | ---: |
| B0 | Random order | 0.7422 | 0.7459 [0.7429, 0.7491] |
| B1 | BM25 `main` | 0.8227 | 0.8244 [0.8209, 0.8275] |
| B2 | bge-base bi-encoder `main`, zero-shot | 0.8429 | 0.8471 [0.8440, 0.8500] |
| B3 | MiniLM cross-encoder `title`, MSE (2 seeds) | 0.8572 | 0.8552 [0.8518, 0.8578] |
| D1 | MiniLM `main`, MSE (2 seeds) | 0.8684 | 0.8660 [0.8625, 0.8685] |
| D1 | DeBERTa-v3-base `main`, MSE (2 seeds) | 0.8877 | 0.8868 [0.8845, 0.8902] |
| D2 | DeBERTa-v3-base `main`, ce4 | 0.8891 | 0.8867 [0.8838, 0.8895] |
| D3 | DeBERTa-v3-base ce4, mean of 5 fold models | 0.8920 | 0.8885 [0.8855, 0.8912] |
| D3 | LightGBM on base scores + features (first version) | 0.8947 | 0.8921 [0.8890, 0.8947] |
| D3 | DeBERTa-v3-large ce4, mean of 5 fold models | 0.8970 | 0.8965 [0.8938, 0.8992] |
| **D3** | **LightGBM lambdarank on base + large scores + features (final)** | **0.8999** | **0.8980 [0.8953, 0.9006]** |

**What each step adds** (paired difference on the same test queries):

| Question | Test difference [95% CI] | Answer |
| --- | --- | --- |
| Final vs B3 | +0.0431 [0.0407, 0.0456] | 6,231 queries better, 2,440 worse |
| D1: `main` vs `title` input (DeBERTa) | +0.0102 [0.0086, 0.0118] | Richer input helps |
| D1: DeBERTa vs MiniLM (seeds 42 / 43) | +0.0220 / +0.0196 | The largest gain among our changes |
| D2: ce4 vs MSE (DeBERTa) | −0.0009 [−0.0022, 0.0005] | No real difference; ce4 kept because it gives class probabilities |
| D2: ListNet vs MSE (MiniLM, seeds 42 / 43) | +0.0032 / +0.0026 | Small gain on test, not significant on val |
| D3: large vs base (5 fold models each) | +0.0080 [0.0065, 0.0095] | Larger backbone helps |
| D3: 5 fold models vs 1 model (base / large) | +0.0019 / +0.0029 | Averaging helps a little |
| D3: final vs large alone (5 fold models) | +0.0015 [0.0004, 0.0024] | The re-ranker still adds, less than in the first version (+0.0035) |
| D3: final vs first version | +0.0059 [0.0047, 0.0072] | Gain from the large model |
| D3: final vs blend of first version and large | +0.0005 [−0.0002, 0.0013] | Not significant on test |

Two seeds of the same DeBERTa model differ by 0.0014 on test; five seeds of DeBERTa-v3-base ce4 (`15`) have a val standard deviation of 0.0008 (range 0.0022). The bootstrap CI covers the sampling of queries, not training randomness, so differences around 0.002 between single runs should not be read as progress.

**How the final system was chosen.** In two steps, each chosen on CV and val:

1. The first version: LightGBM on the base cross-encoder (val 0.8947).
2. DeBERTa-v3-large, first run as an extra experiment (scored with the other extras), clearly improved val over the base model (0.8946 against 0.8891, one model each), so it was trained on 5 folds and added to the re-ranker. The new LightGBM was accepted by a rule written in `09` before the run: keep it if it beats the blend on CV with a CI above 0 and is not worse on val. It does (CV +0.0008 [0.0002, 0.0014], val +0.0006), and lambdarank has the best CV (0.8957, multiclass 0.8952). On test multiclass is slightly higher (0.8985, not significant); the choice was not changed.

On val the final system is +0.0052 above the first version (0.8999 against 0.8947), in line with +0.0059 on test.

**Ablation** (final model, CV nDCG 0.8957): without large scores −0.0071, without base scores −0.0010, cross-encoder scores only −0.0016, no cross-encoder at all −0.0342. Every other single group is worth 0.0006 or less.

**Extra experiments**, not used to choose. The blend averages within-query z-scores with a weight chosen on val.

| Run | Val | Test [95% CI] | Paired difference on test |
| --- | ---: | ---: | --- |
| DeBERTa-v3-base, 2 epochs | 0.8889 | 0.8858 [0.8828, 0.8887] | −0.0008 vs 1 epoch (n.s.) |
| DeBERTa-v3-base + 0.2 bge-reranker | 0.8922 | 0.8888 [0.8860, 0.8916] | +0.0022 [0.0015, 0.0028] vs DeBERTa-v3-base |
| DeBERTa-v3-large, 1 model on all train | 0.8946 | 0.8936 [0.8908, 0.8964] | +0.0070 [0.0052, 0.0086] vs DeBERTa-v3-base |

**Error analysis** (`10_report`):

- **Negation queries are the hardest type:** 0.851 (final), against 0.903 for ordinary queries. Fine-tuned cross-encoders gain most there over BM25 and bi-encoders, but the re-ranker adds nothing over the large model (0.851).
- **By query property:** the final system beats B3 in every group (+0.030 to +0.050, all CIs above 0), least for queries with a negation word (+0.030, against +0.044 without). Final nDCG is lower for queries of 6 or more words (0.887, against 0.899 to 0.901), with a negation word (0.853, against 0.902) or with a number (0.892, against 0.900); random order is lower on these lists too, so part of the gap is list difficulty. Queries that name a candidate's brand score higher (0.913, against 0.895). Details: `data/report/by_query_property.csv`.
- **Exact and Substitute are separated better:** the gap in mean list position grows from 0.099 (BM25) to 0.195 (final).
- **Against B3**, 1,758 queries gain more than 0.1 and 447 lose more than 0.1. The extremes are lists with one or two Exact products (56% of the 50 largest losses, against 14% of all queries), where a single swap changes nDCG a lot.
- **Label noise:** some of the worst queries have a doubtful Exact label (e.g. "fake weed jar" → catnip).

All 38 runs: `data/report/results_all.csv`. All paired tests: `data/report/paired_tests.csv`.

## Supplementary experiments (validation only)

Run after the final system was fixed, to check the conclusions above; not scored on test. Pooled comparisons: per query, the mean over seeds of (b − a), 95% bootstrap CI over the 2,089 val queries. Results: `data/results/supplement/`.

| Notebook | Question | Result (val) |
| --- | --- | --- |
| `13_d2b_sampler` | Do the losses rank the same when all three see the same batches (whole queries), MiniLM `title`, 4 seeds? | ListNet − MSE +0.0036 [0.0014, 0.0057], 4/4 seeds; ce4 − MSE −0.0014 (n.s.). But whole-query batches lower MSE by 0.0027 and ce4 by 0.0032 against random batches, while ListNet is unchanged: most of ListNet's edge comes from the batching |
| `14_distill` | How much of the gap to DeBERTa-large does a MiniLM recover when trained on the large model's out-of-fold probabilities? | kd00 (teacher only) − ce4 +0.0063 [0.0042, 0.0082]; kd00 − MSE +0.0027 [−0.0001, 0.0057] (`10b`). About 10% of the gap between MiniLM MSE and the teacher; ECE worse (0.025 → 0.04) |
| `15_deberta_base_s43` | Seed variance of DeBERTa-v3-base ce4 (seeds 42–46), with saved checkpoints | Val 0.8877–0.8899, standard deviation 0.0008 |
| `16_negation_probe` | Do the models read "without Y"? 129 hand-checked val queries scored as written, without the negated phrase, with "with Y" and with "no Y" | Every model reacts: products containing Y are scored lower in 72–88% of cases (0.5 = ignored); DeBERTa-v3-base raises products that say "no Y" in 92–96% of cases, MiniLM in 71%. Keeping the negation is worth +0.04 to +0.07 nDCG under the real labels, so the labels reward it. DeBERTa-large and the final system have no saved weights and are not probed |
| `10b_supplement_val` | Pooled comparisons over seeds 42 and 43 | DeBERTa-v3-base ce4 − MSE +0.0018 [−0.0004, 0.0039] (both seeds positive); MiniLM `main` ce4 − MSE −0.0036 [−0.0060, −0.0010]; two MiniLM MSE seeds differ by 0.0031 [0.0015, 0.0048] |

**Reading:** the training loss matters little (all differences below 0.004, close to seed variation, and the best loss depends on the model); distillation clearly beats ce4 but not MSE beyond seed noise; models of base size do react to negations, so the weakness on negation queries is not simply the model ignoring them.

## Repository

```text
├── proposal/    project proposal (LaTeX)
├── eda/         exploratory analysis: scripts, figures, tables, validation split proposal
├── src/         shared code: metrics and run log, text, cross-encoder training, features, data loading,
│                parallel GPU jobs (gpu_jobs.py), negation variants (probes.py)
├── tests/       tests of the metrics, features and GPU jobs (pytest)
├── notebooks/   00–16 and 10b, one per step
└── data/
    ├── splits/  train/val/test split, data checks
    ├── results/ one json per run (settings, time, metrics), D3 tables; folds/ = one json per fold model;
    │            supplement/ = outputs of 10b and 13–16 (val only)
    └── report/  final tables written by 10_report
```

**Running.** All notebooks run on Kaggle, with the GPU steps on T4. Each reads the output of earlier ones (*Add Input → Notebook Output*) and imports `src/` from an attached dataset; its inputs are listed in the first cell.

```text
01 Preprocess
 ├─► 02 BM25 (B1) ──────────────────────────────────────────────────┐
 ├─► 04 Bi-encoders (B2) ───────────────────────────────────────────┤
 ├─► 05 Input and backbone (D1) ─┐                                  │
 ├─► 06 Losses (D2) ─────────────┴─► 07 OOF scores, DeBERTa-base ───┤
 ├─► 12 DeBERTa-large: full run, OOF scores ────────────────────────┤
 ├─► 03 MiniLM cross-encoder (B3)                                   │
 └─► 11 bge-reranker (extra)                                        │
 ┌──────────────────────────────────────────────────────────────────┘
 ▼
08 Feature table ─► 09 LightGBM re-ranker (D3) ─► 10 Final report
```

`05` and `06` choose the settings used in `07` and `12`. `10` also reads the scores of every run, including `03` and `11`.

Supplementary notebooks (validation only):

```text
00 Hugging Face models (CPU, Internet on: lets the GPU notebooks run offline)
01 Preprocess
 ├─► 13 Losses on the same batches (MiniLM title, 4 seeds) ◄─ 03, 06 (scores, for D2b vs D2a)
 ├─► 14 Distillation into MiniLM ◄─ 12 (out-of-fold teacher probabilities) ──► checkpoints ─┐
 ├─► 15 DeBERTa-v3-base ce4, seeds 43–46 ◄─ 05, 06 (scores) ───────────────────► checkpoints ─┤
 │                                                                                            ▼
 ├─► 16 Negation probe (hand-checked query list data/results/supplement/16/) ◄────────────────┘
 └─► 10b Pooled comparisons over seeds ◄─ scores of 05, 06, 14, 15
```

`13`–`16` train or score several runs at once, one per GPU (`src/gpu_jobs.py`); each notebook finds its inputs by file name anywhere under `/kaggle/input`. `16` first runs with `MODE = 'prepare'` (CPU) to write the candidate queries, which are checked by hand before the `smoke` and `full` runs.

Not in the repository because of size: the raw data ([amazon-science/esci-data](https://github.com/amazon-science/esci-data), Apache-2.0), per-pair scores of the runs, out-of-fold scores, the feature table and the model checkpoints of `14` and `15`.

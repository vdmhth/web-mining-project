"""Evaluation for ESCI Task 1: nDCG over the full candidate list, helper metrics, and run logging.

Every notebook scores its runs with this module, so all numbers in the result tables are computed the same way.
Do not change a metric here without re-scoring every saved run (`data/results/scores/`).
The functions were moved unchanged from notebooks 02_bm25 and 03_cross_encoder.
"""
import glob
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TIE_SEEDS = (0, 1, 2)   # ties broken at random, metric averaged over 3 seeds (as in eda/eda_audit.py)
GAINS = {'official': {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0},
         'amazon_script': {'E': 1.0, 'S': 0.01, 'C': 0.1, 'I': 0.0}}   # S/C swapped as in Amazon's script
METRICS = ['ndcg', 'ndcg_amazon', 'ndcg10', 'mrr_e']


def _positions(codes):
    # 0-based rank inside each query; `codes` must be sorted so every query is contiguous
    n = len(codes)
    start = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    return np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))


def query_metrics(df, score, seeds=TIE_SEEDS):
    # Per-query metrics of one score vector. df needs query_id and esci_label, in the same row order as score.
    score = np.asarray(score, dtype=float)
    assert len(score) == len(df) and np.isfinite(score).all()
    codes, qids = pd.factorize(df['query_id'].to_numpy())
    nq = len(qids)
    lab = df['esci_label'].to_numpy()
    gain = {k: pd.Series(lab).map(m).to_numpy() for k, m in GAINS.items()}
    is_e = lab == 'E'

    def dcg(order, g, k=None):
        c = codes[order]
        pos = _positions(c)
        w = 1 / np.log2(pos + 2)
        if k is not None:
            w = np.where(pos < k, w, 0.0)
        return np.bincount(c, g[order] * w, minlength=nq)

    ideal = {k: np.lexsort((-g, codes)) for k, g in gain.items()}
    idcg = {k: dcg(ideal[k], g) for k, g in gain.items()}
    idcg10 = dcg(ideal['official'], gain['official'], 10)

    acc = {'ndcg': 0.0, 'ndcg_amazon': 0.0, 'ndcg10': 0.0, 'mrr_e': 0.0}
    for sd in seeds:
        order = np.lexsort((np.random.default_rng(sd).random(len(df)), -score, codes))   # query, score desc, random
        acc['ndcg'] = acc['ndcg'] + dcg(order, gain['official'])
        acc['ndcg_amazon'] = acc['ndcg_amazon'] + dcg(order, gain['amazon_script'])
        acc['ndcg10'] = acc['ndcg10'] + dcg(order, gain['official'], 10)
        c, pos, e = codes[order], _positions(codes[order]), is_e[order]
        first_e = np.full(nq, np.inf)
        np.minimum.at(first_e, c[e], pos[e])
        acc['mrr_e'] = acc['mrr_e'] + 1 / (first_e + 1)

    n = len(seeds)
    with np.errstate(divide='ignore', invalid='ignore'):   # IDCG = 0 (all-I query) -> NaN, skipped in means
        return pd.DataFrame({
            'query_id': qids,
            'n_cand': np.bincount(codes, minlength=nq),
            'ndcg': acc['ndcg'] / n / idcg['official'],
            'ndcg_amazon': acc['ndcg_amazon'] / n / idcg['amazon_script'],
            'ndcg10': acc['ndcg10'] / n / idcg10,
            'mrr_e': np.where(np.bincount(codes, is_e.astype(float), minlength=nq) > 0, acc['mrr_e'] / n, np.nan),
        })


def e_over_s(df, score):
    # Share of (E, S) pairs inside a query where E scores higher; ties count 0.5 (Mann-Whitney U), pooled.
    f = pd.DataFrame({'q': df['query_id'].to_numpy(), 'l': df['esci_label'].to_numpy(),
                      's': np.asarray(score, dtype=float)})
    f = f[f['l'].isin(['E', 'S'])].copy()
    f['rk'] = f.groupby('q')['s'].rank(method='average')
    f['is_e'] = f['l'] == 'E'
    agg = f.groupby('q').agg(n_e=('is_e', 'sum'), n=('is_e', 'size'))
    agg['r_e'] = f[f['is_e']].groupby('q')['rk'].sum().reindex(agg.index).fillna(0.0)
    agg['n_s'] = agg['n'] - agg['n_e']
    agg = agg[(agg['n_e'] > 0) & (agg['n_s'] > 0)]
    u = agg['r_e'] - agg['n_e'] * (agg['n_e'] + 1) / 2
    return float(u.sum() / (agg['n_e'] * agg['n_s']).sum())


def calibration(probs, labels, classes=('E', 'S', 'C', 'I'), n_bins=15):
    """Calibration of 4-class probabilities (supplement R4): multiclass NLL, multiclass Brier score (sum over the
    classes, mean over pairs) and classwise ECE. For each class c the pairs are sorted by P(c) and cut into `n_bins`
    equal-mass bins; ECE_c = sum over bins of (bin size / N) * |mean P(c) - share of pairs labelled c|.
    probs: (n, 4) array or DataFrame in `classes` order (rows are renormalised); labels: E/S/C/I per pair."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum(1, keepdims=True)
    code = pd.Series(np.asarray(labels)).map({c: i for i, c in enumerate(classes)})
    assert code.notna().all(), f'labels must be in {classes}'
    y = np.eye(len(classes))[code.to_numpy(int)]
    out = {'nll': float(-np.log(np.clip((p * y).sum(1), 1e-12, None)).mean()),
           'brier': float(((p - y) ** 2).sum(1).mean())}
    for i, c in enumerate(classes):
        order = np.argsort(p[:, i], kind='stable')
        out[f'ece_{c}'] = float(sum(len(b) * abs(p[b, i].mean() - y[b, i].mean())
                                    for b in np.array_split(order, n_bins) if len(b)) / len(p))
    out['ece_mean'] = float(np.mean([out[f'ece_{c}'] for c in classes]))
    return {k: round(v, 5) for k, v in out.items()}


def bootstrap_ci(x, n_boot=1000, seed=SEED):
    x = np.asarray(x, dtype=float)
    means = x[np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def paired_bootstrap(pq_a, pq_b, metric='ndcg', n_boot=1000, seed=SEED):
    # Mean of (b - a) over the same queries, its 95% CI, and the share of resamples where b <= a.
    d = pq_a[['query_id', metric]].merge(pq_b[['query_id', metric]], on='query_id', suffixes=('_a', '_b')).dropna()
    diff = (d[metric + '_b'] - d[metric + '_a']).to_numpy()
    means = diff[np.random.default_rng(seed).integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return {'diff': round(float(diff.mean()), 4),
            'ci95': [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            'p_b_le_a': float((means <= 0).mean())}


def summarize(df, score):
    pq = query_metrics(df, score)
    s = {m: round(float(pq[m].mean()), 4) for m in METRICS}
    s['ndcg_ci95'] = bootstrap_ci(pq['ndcg'].dropna())
    s['e_over_s'] = round(e_over_s(df, score), 4)
    s['n_queries'] = int(pq['ndcg'].notna().sum())
    s['n_skipped_all_irrelevant'] = int(pq['ndcg'].isna().sum())
    return s, pq


def src_version():
    """Hash of the src/*.py files, saved in every run's json: two runs with the same value ran the same code.

    Line endings are normalised, so the Windows copy and the Kaggle copy of the same code give the same hash.
    """
    src = Path(__file__).resolve().parent
    code = b''.join(f.read_bytes().replace(b'\r\n', b'\n') for f in sorted(src.glob('*.py')))
    return {'files_sha1': hashlib.sha1(code).hexdigest()[:10]}


def env_versions():
    """Python and library versions; torch and transformers only if the notebook imported them."""
    out = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
    for lib in ('torch', 'transformers', 'lightgbm'):
        if lib in sys.modules:
            out[lib] = sys.modules[lib].__version__
    return out


class RunLog:
    """Scores runs and writes `<res_dir>/<run>.json` and `<res_dir>/scores/<run>_<split>.parquet`.

    parts: {split: DataFrame with example_id, query_id, esci_label}. The scores given to `log` for a split must be
    in the row order of parts[split]. Every notebook that produces scores uses this format, so the report
    notebook can read them all. Each json also records `src_version` and `env` (library versions).
    """

    def __init__(self, res_dir, parts):
        self.res = Path(res_dir)
        (self.res / 'scores').mkdir(parents=True, exist_ok=True)
        self.parts = parts
        self.records, self.per_query = {}, {}
        self.version = src_version()

    def log(self, name, scores, info, compare_to=None, blind=(), extra=None):
        # scores: {split: array}; compare_to: an earlier run (logged or loaded) for a paired bootstrap on nDCG.
        # Splits in `blind` (e.g. 'test' while configurations are still being chosen on val) are saved as scores
        # only, with no metrics computed or printed; the report notebook scores them later.
        # extra: {split: DataFrame in the row order of parts[split]} of columns saved next to `score`
        # (e.g. the ce4 class probabilities p_E..p_I, features for the re-ranker).
        extra = extra or {}
        rec = {'run': name, **info, 'tie_seeds': list(TIE_SEEDS), 'bootstrap': 1000,
               'created_utc': time.strftime('%Y-%m-%d %H:%M', time.gmtime()),
               'src_version': self.version, 'env': env_versions(), 'metrics': {}}
        if blind:
            rec['blind_splits'] = sorted(blind)
        self.per_query[name] = {}
        for split, sc in scores.items():
            part = self.parts[split]
            sc = np.asarray(sc, dtype=float)
            if split not in blind:
                rec['metrics'][split], self.per_query[name][split] = summarize(part, sc)
            out = part[['example_id', 'query_id']].assign(score=sc.astype('float32')).reset_index(drop=True)
            if split in extra:
                ex = extra[split].reset_index(drop=True)
                assert len(ex) == len(out) and not set(ex.columns) & set(out.columns), f'bad extra columns for {split}'
                out = pd.concat([out, ex], axis=1)
            out.to_parquet(self.res / 'scores' / f'{name}_{split}.parquet', index=False)
        if compare_to in self.per_query:
            rec['paired_vs'] = {compare_to: {s: paired_bootstrap(self.per_query[compare_to][s], self.per_query[name][s])
                                             for s in self.per_query[name] if s in self.per_query[compare_to]}}
        elif compare_to:
            print(f'{compare_to} not logged or loaded: no paired comparison')
        self.records[name] = rec
        (self.res / f'{name}.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
        print(name, {s: (v['ndcg'], v['ndcg_ci95'], 'E>S', v['e_over_s']) for s, v in rec['metrics'].items()})
        return rec

    def load(self, run, splits=None, root='/kaggle/input'):
        # Per-query metrics of a run saved earlier (e.g. an attached notebook output); False if its scores are missing.
        for split in splits or list(self.parts):
            hits = glob.glob(f'{root}/**/scores/{run}_{split}.parquet', recursive=True)
            if not hits:
                return False
            part = self.parts[split]
            sc = part[['example_id']].merge(pd.read_parquet(hits[0])[['example_id', 'score']], on='example_id', how='left')
            assert sc['score'].notna().all(), f'{run} scores do not cover {split}'
            m, self.per_query.setdefault(run, {})[split] = summarize(part, sc['score'].to_numpy())
            print(f'{run} {split}: nDCG {m["ndcg"]}')
        return True

    def table(self, cols=('ndcg', 'ndcg_ci95', 'ndcg_amazon', 'ndcg10', 'mrr_e', 'e_over_s', 'n_queries')):
        return pd.DataFrame([{'split': s, 'run': n, **{c: r['metrics'][s][c] for c in cols}}
                             for n, r in self.records.items() for s in r['metrics']]).set_index(['split', 'run']).sort_index()

    def print_paired(self):
        for n, r in self.records.items():
            for ref, by_split in r.get('paired_vs', {}).items():
                for s, v in by_split.items():
                    print(f'{s:4s} {n} - {ref}: {v}')

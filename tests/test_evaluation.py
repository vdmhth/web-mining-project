"""Checks for src/evaluation.py and src/text.py on tiny hand-made data.

Run from the repo root: `python -m pytest tests` or, without pytest, `python tests/test_evaluation.py`.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import GAINS, RunLog, calibration, e_over_s, query_metrics, summarize   # noqa: E402
from src.text import clean_bullets, clean_text, product_text   # noqa: E402


def make_df(rows):
    return pd.DataFrame(rows, columns=['query_id', 'esci_label'])


def random_lists(n_queries=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(q, rng.choice(list('ESCI'), p=[0.43, 0.35, 0.05, 0.17]))
            for q in range(n_queries) for _ in range(rng.integers(2, 30))]
    return make_df(rows)


def test_hand_computed_ndcg():
    # one query, S ranked above E
    df = make_df([(1, 'E'), (1, 'S')])
    pq = query_metrics(df, [0.0, 1.0])
    w2 = 1 / math.log2(3)
    assert math.isclose(pq['ndcg'][0], (0.1 + w2) / (1 + 0.1 * w2))
    assert math.isclose(pq['ndcg_amazon'][0], (0.01 + w2) / (1 + 0.01 * w2))
    assert math.isclose(pq['mrr_e'][0], 0.5)


def test_official_gains():
    assert GAINS['official'] == {'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}


def test_oracle_worst_flat():
    df = random_lists()
    gain = df['esci_label'].map(GAINS['official']).to_numpy()
    oracle, _ = summarize(df, gain)
    worst, _ = summarize(df, -gain)
    flat, _ = summarize(df, np.zeros(len(df)))
    assert oracle['ndcg'] == 1.0 and oracle['ndcg10'] == 1.0 and oracle['mrr_e'] == 1.0 and oracle['e_over_s'] == 1.0
    assert worst['ndcg'] < flat['ndcg'] < oracle['ndcg']
    assert flat['e_over_s'] == 0.5


def test_ties_do_not_follow_row_order():
    # a constant score must not keep the input order: E rows first would give nDCG ~1
    df = random_lists().sort_values(['query_id', 'esci_label']).reset_index(drop=True)   # E first in every query
    flat, _ = summarize(df, np.zeros(len(df)))
    assert flat['ndcg'] < 0.9


def test_all_irrelevant_query_skipped():
    df = make_df([(1, 'E'), (1, 'I'), (2, 'I'), (2, 'I')])
    s, pq = summarize(df, [1.0, 0.0, 1.0, 0.0])
    assert s['n_queries'] == 1 and s['n_skipped_all_irrelevant'] == 1 and s['ndcg'] == 1.0


def test_e_over_s_ties_half():
    df = make_df([(1, 'E'), (1, 'S'), (1, 'S')])
    assert e_over_s(df, [1.0, 1.0, 0.0]) == 0.75


def test_runlog_writes_json_and_scores(tmp_path=None):
    import importlib.util
    import json
    import tempfile
    if importlib.util.find_spec('pyarrow') is None and importlib.util.find_spec('fastparquet') is None:
        print('skip: no parquet engine (pyarrow) in this environment; runs on Kaggle')
        return
    tmp = Path(tmp_path or tempfile.mkdtemp())
    df = random_lists(50).assign(example_id=lambda d: range(len(d)))
    log = RunLog(tmp, {'val': df})
    gain = df['esci_label'].map(GAINS['official']).to_numpy()
    log.log('oracle', {'val': gain}, {'model': 'oracle'})
    log.log('flat', {'val': np.zeros(len(df))}, {'model': 'flat'}, compare_to='oracle')
    rec = json.loads((tmp / 'flat.json').read_text(encoding='utf-8'))
    assert len(rec['src_version']['files_sha1']) == 10 and 'pandas' in rec['env']
    assert rec['paired_vs']['oracle']['val']['diff'] < 0
    sc = pd.read_parquet(tmp / 'scores' / 'oracle_val.parquet')
    assert list(sc.columns) == ['example_id', 'query_id', 'score'] and len(sc) == len(df)
    assert log.table().loc[('val', 'oracle'), 'ndcg'] == 1.0
    # a blind split is saved as scores but gets no metrics
    log2 = RunLog(tmp, {'val': df, 'test': df})
    log2.log('blind', {'val': gain, 'test': gain}, {}, blind=('test',))
    rec = json.loads((tmp / 'blind.json').read_text(encoding='utf-8'))
    assert list(rec['metrics']) == ['val'] and rec['blind_splits'] == ['test']
    assert (tmp / 'scores' / 'blind_test.parquet').exists()
    # extra columns (ce4 probabilities) are saved next to the score, row-aligned, even with a non-default index
    sub = df.iloc[10:].copy()
    probs = pd.DataFrame({'p_E': np.arange(len(sub)) / len(sub), 'p_I': 0.5})
    log3 = RunLog(tmp, {'val': sub})
    log3.log('probs', {'val': gain[10:]}, {}, extra={'val': probs})
    sc = pd.read_parquet(tmp / 'scores' / 'probs_val.parquet')
    assert list(sc.columns) == ['example_id', 'query_id', 'score', 'p_E', 'p_I']
    assert (sc['example_id'].to_numpy() == sub['example_id'].to_numpy()).all()
    assert np.allclose(sc['p_E'], probs['p_E'])


def test_calibration():
    # perfect one-hot predictions
    lab = np.array(list('ESCI') * 30)
    one_hot = np.eye(4)[pd.Series(lab).map({'E': 0, 'S': 1, 'C': 2, 'I': 3})]
    c = calibration(one_hot, lab)
    assert c['nll'] < 1e-9 and c['brier'] == 0 and c['ece_mean'] == 0
    # uniform predictions on balanced labels (bins of 4k pairs hold k of each label): calibrated, NLL = ln 4
    lab = np.array(list('ESCI') * 15 * 3)
    c = calibration(np.full((len(lab), 4), 0.25), lab)
    assert abs(c['nll'] - round(math.log(4), 5)) < 1e-5 and abs(c['brier'] - 0.75) < 1e-9 and c['ece_mean'] == 0
    # hand-computed, one bin
    p = np.array([[0.7, 0.1, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]])
    c = calibration(p, ['E', 'S'], n_bins=1)
    assert abs(c['nll'] - round(-(math.log(0.7) + math.log(0.6)) / 2, 5)) < 1e-5
    assert abs(c['brier'] - 0.17) < 1e-9                       # (0.09+0.01+0.01+0.01 + 0.04+0.16+0.01+0.01) / 2
    assert abs(c['ece_E'] - 0.05) < 1e-9 and abs(c['ece_S'] - 0.15) < 1e-9 and abs(c['ece_C'] - 0.1) < 1e-9
    # unnormalised rows are renormalised; unknown labels are rejected
    assert calibration(p * 3, ['E', 'S'], n_bins=1) == c
    try:
        calibration(p, ['E', 'X'])
    except AssertionError:
        pass
    else:
        raise AssertionError('unknown label accepted')


def test_text():
    assert clean_text('a&lt;br&gt;b  <b>c</b>') == 'a b c'
    assert clean_text('size < 5') == 'size < 5'
    assert clean_text('  ') is None
    assert clean_bullets('one\n\n <i>two</i>') == 'one ; two'
    prod = pd.DataFrame({'title': ['T', 'U'], 'brand': ['B', None], 'color': [None, 'red'], 'bullets': ['x', None]})
    assert product_text(prod, 'main').tolist() == ['T [SEP] brand: B [SEP] x', 'U [SEP] color: red']


if __name__ == '__main__':
    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]
    for f in tests:
        f()
        print('ok  ', f.__name__)
    print(f'{len(tests)} tests passed')

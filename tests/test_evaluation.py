"""Checks for src/evaluation.py and src/text.py on tiny hand-made data.

Run from the repo root: `python -m pytest tests` or, without pytest, `python tests/test_evaluation.py`.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import GAINS, e_over_s, query_metrics, summarize   # noqa: E402
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

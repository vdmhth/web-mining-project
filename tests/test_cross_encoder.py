"""Checks for the batching, targets, warm-up and losses of src/cross_encoder.py (CPU only, no model download).

Run from the repo root: `python -m pytest tests` or, without pytest, `python tests/test_cross_encoder.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import (LABELS, compute_loss, make_epochs, make_targets, query_batches,   # noqa: E402
                               sampler_of, warmup_of)


def query_ids(n_queries=60, seed=0, max_len=45):
    rng = np.random.default_rng(seed)
    return np.repeat(np.arange(n_queries), rng.integers(1, max_len, n_queries))   # includes 1-pair queries


def train_df(n_queries=60, seed=0):
    q = query_ids(n_queries, seed)
    rng = np.random.default_rng(seed + 1)
    lab = rng.choice(list(LABELS), len(q))
    soft = rng.dirichlet(np.ones(4), len(q))
    df = pd.DataFrame({'query_id': q, 'esci_label': lab,
                       'gain': pd.Series(lab).map({'E': 1.0, 'S': 0.1, 'C': 0.01, 'I': 0.0}).to_numpy()})
    for i, l in enumerate(LABELS):
        df[f'soft_{l}'] = soft[:, i]
    return df


def test_query_batches_default_unchanged():
    # list_size 16, 1-pair lists dropped: the D2a ListNet batches (same checks as the self-check of 06_d2_losses)
    q = query_ids()
    batches = query_batches(q, np.random.default_rng(1), list_size=16, batch_pairs=32)
    rows = np.concatenate([i for i, _ in batches])
    sizes = pd.Series(q).value_counts()
    assert len(rows) == len(set(rows)) == sizes[sizes > 1].sum()
    for i, s in batches:
        assert len(i) <= 32
        for k in range(s.max() + 1):
            assert len(set(q[i[s == k]])) == 1 and 2 <= (s == k).sum() <= 16


def test_query_batches_whole_queries():
    q = query_ids()
    batches = query_batches(q, np.random.default_rng(1), list_size=None, batch_pairs=32, drop_single=False)
    rows = np.concatenate([i for i, _ in batches])
    assert len(rows) == len(set(rows)) == len(q)                 # every row once, 1-pair queries kept
    sizes = pd.Series(q).value_counts()
    seen = set()
    for i, s in batches:
        lists = [i[s == k] for k in range(s.max() + 1)]
        for rows_k in lists:
            qs = set(q[rows_k])
            assert len(qs) == 1                                  # a list is one query ...
            (qid,) = qs
            assert len(rows_k) == sizes[qid] and qid not in seen  # ... complete, and in one list only
            seen.add(qid)
        assert len(i) <= 32 or len(lists) == 1                   # only a single long query exceeds the batch size
    assert seen == set(q)


def test_query_batches_drop_single_with_whole_queries():
    q = query_ids()
    batches = query_batches(q, np.random.default_rng(1), list_size=None, drop_single=True)
    sizes = pd.Series(q).value_counts()
    assert sum(len(i) for i, _ in batches) == sizes[sizes > 1].sum()


def test_sampler_rules():
    assert sampler_of({'loss': 'mse'}) == 'pair'
    assert sampler_of({'loss': 'ce4'}) == 'pair'
    assert sampler_of({'loss': 'listnet'}) == 'query'
    assert sampler_of({'loss': 'ce4', 'sampler': 'query'}) == 'query'
    for bad in ({'loss': 'listnet', 'sampler': 'pair'}, {'loss': 'mse', 'sampler': 'list'}):
        try:
            sampler_of(bad)
        except AssertionError:
            continue
        raise AssertionError(f'{bad} should be rejected')


def test_make_epochs_default_unchanged():
    df = train_df()
    pair = make_epochs(df, {'loss': 'mse', 'batch_size': 32, 'epochs': 2}, seed=42)
    assert len(pair) == 2 and len(pair[0]) == len(df) // 32      # DataLoader with drop_last, as in B3
    cfg = {'loss': 'listnet', 'batch_size': 32, 'epochs': 1}
    got = make_epochs(df, cfg, seed=42)[0]
    ref = query_batches(df['query_id'].to_numpy(), np.random.default_rng(42), 16, 32)   # D2a call
    assert len(got) == len(ref)
    assert all((a == c).all() and (b == d).all() for (a, b), (c, d) in zip(got, ref))


def test_make_epochs_query_sampler_keeps_every_row_for_pointwise_losses():
    df = train_df()
    for loss in ('mse', 'ce4', 'ce4_soft'):
        cfg = {'loss': loss, 'batch_size': 32, 'epochs': 1, 'sampler': 'query', 'list_size': None}
        rows = np.concatenate([i for i, _ in make_epochs(df, cfg, seed=0)[0]])
        assert sorted(rows) == list(range(len(df)))


def test_make_targets():
    df = train_df()
    assert make_targets(df, {'loss': 'mse'}).dtype == np.float32
    ce4 = make_targets(df, {'loss': 'ce4'})
    assert ce4.dtype == np.int64 and (np.array(LABELS)[ce4] == df['esci_label'].to_numpy()).all()
    hard = make_targets(df, {'loss': 'ce4_soft', 'alpha': 1.0})
    assert (hard.argmax(1) == ce4).all() and np.allclose(hard.sum(1), 1) and np.allclose(hard.max(1), 1)
    soft = make_targets(df, {'loss': 'ce4_soft'})                  # alpha defaults to 0: teacher only
    assert np.allclose(soft, df[[f'soft_{l}' for l in LABELS]].to_numpy(), atol=1e-6)
    mix = make_targets(df, {'loss': 'ce4_soft', 'alpha': 0.5})
    assert np.allclose(mix, 0.5 * hard + 0.5 * soft, atol=1e-6) and np.allclose(mix.sum(1), 1, atol=1e-5)
    bad = df.copy()
    bad['soft_E'] += 0.5                                           # rows no longer sum to 1
    for cfg, frame in (({'loss': 'ce4_soft'}, bad), ({'loss': 'ce4_soft', 'alpha': 1.5}, df)):
        try:
            make_targets(frame, cfg)
        except AssertionError:
            continue
        raise AssertionError(f'{cfg} should be rejected')


def test_ce4_soft_loss_equals_ce4_on_hard_targets():
    torch.manual_seed(0)
    logits = torch.randn(50, 4)
    y = torch.randint(0, 4, (50,))
    ce4 = compute_loss(logits, y, {'loss': 'ce4'})
    soft = compute_loss(logits, F.one_hot(y, 4).float(), {'loss': 'ce4_soft'})
    assert torch.allclose(ce4, soft, atol=1e-6)
    # hand check on one row: -sum(t * log softmax)
    t = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    z = torch.tensor([[1.0, 0.0, 0.0, -1.0]])
    expected = -(t * torch.log_softmax(z, -1)).sum()
    assert torch.allclose(compute_loss(z, t, {'loss': 'ce4_soft'}), expected, atol=1e-6)


def test_warmup():
    # the values recorded in the D1/D2a run files: B3 / ce4 used 4,962 of 11,816 steps (the 42 % cap applies),
    # D2a ListNet 5,000 of 12,883
    assert warmup_of({'warmup_steps': 5000}, 11816) == 4962
    assert warmup_of({'warmup_steps': 5000}, 12883) == 5000
    assert warmup_of({'warmup_steps': 5000}, 1000) == 420              # smoke runs
    assert warmup_of({'warmup_steps': 5000, 'warmup_ratio': 0.42}, 11816) == 4962   # D2b: same share as B3
    assert warmup_of({'warmup_steps': 5000, 'warmup_ratio': 0.42}, 12883) == int(0.42 * 12883)
    assert warmup_of({'warmup_steps': 5000, 'warmup_ratio': None}, 11816) == 4962


if __name__ == '__main__':
    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]
    for f in tests:
        f()
        print('ok  ', f.__name__)
    print(f'{len(tests)} tests passed')

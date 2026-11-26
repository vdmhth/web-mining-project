"""Cross-encoder training and prediction: plain PyTorch loop around a Hugging Face model.

Moved from notebook 03_cross_encoder. New input fields (Direction 1) only need a different `text` column in the
DataFrames. DataFrames passed in need the columns `query`, `text` and, for training, `gain` (plus `esci_label` for
loss 'ce4' and `query_id` for loss 'listnet').
Optional cfg key `eval_batch_size` (default 512): pairs per prediction batch; lower it for large models with long
inputs (DeBERTa at max_len 256 runs out of memory on a T4 at 512).

Losses (cfg['loss'], Direction 2):
- 'mse': one output, MSE against the official gain (Amazon's B3 setup).
- 'ce4': four outputs (E, S, C, I), cross-entropy on the label; the score is the expected official gain
  p_E + 0.1 p_S + 0.01 p_C. The classification head is new (randomly initialised) when the checkpoint has one output.
- 'listnet': one output, batches made of whole query lists (`query_batches`); ListNet cross-entropy between
  softmax(scores) and the target distribution gain / sum(gain) inside each list.
- 'ce4_soft' (supplement, distillation): the 'ce4' head and score, trained with cross-entropy on the soft target
  alpha * one_hot(label) + (1 - alpha) * (soft_E, soft_S, soft_C, soft_I); the four `soft_*` columns of the training
  DataFrame hold the teacher's probabilities, cfg['alpha'] (default 0) the weight of the hard label.

Optional cfg keys added on the supplement branch (defaults keep every earlier run unchanged):
- `sampler`: 'pair' (random batches of pairs; default for mse / ce4 / ce4_soft) or 'query' (batches of whole query
  lists from `query_batches`; always used by listnet). With 'query', every loss sees the same kind of batch.
- `list_size`: maximum list length for 'query' batches (default 16, as in D2); None keeps every query whole.
- `warmup_ratio`: warm-up as a share of all steps; without it, warm-up = min(warmup_steps, 0.42 * steps).

Models are loaded from a local copy in the inputs when one is attached (`model_source`, notebook 00_hf_models), else
from the Hugging Face Hub; cfg['model'] (and so every run's json) keeps the Hub name.
"""
import functools
import glob
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .evaluation import GAINS, summarize

DEVICE = 'cuda'
LABELS = ('E', 'S', 'C', 'I')                      # class order of the 'ce4' head
CLASS_GAIN = [GAINS['official'][l] for l in LABELS]


@functools.lru_cache(maxsize=None)
def model_source(name, root='/kaggle/input'):
    """Folder of a local copy of Hub model `name` (hf_models/<org>__<model>/ in the inputs), else `name` itself."""
    if Path(name).is_dir():
        return name
    hits = sorted(glob.glob(f"{root}/**/hf_models/{name.replace('/', '__')}/config.json", recursive=True))
    return str(Path(hits[0]).parent) if hits else name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def encode(tokenizer, queries, texts, max_len):
    return tokenizer(list(queries), list(texts), truncation=True, max_length=max_len,
                     padding=True, return_tensors='pt').to(DEVICE)


def make_optimizer(model, cfg):
    no_decay = ('bias', 'LayerNorm.weight', 'LayerNorm.bias')
    groups = [
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         'weight_decay': cfg['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg['lr'])


def query_batches(query_ids, rng, list_size=16, batch_pairs=32, drop_single=True):
    """One epoch of ListNet batches. Each query's candidates are shuffled and split into near-equal lists of at most
    `list_size` (None: the whole query is one list); lists of one pair are dropped when `drop_single` (their ListNet
    loss is always 0; keep them for pointwise losses). Lists are shuffled and packed into batches of at most
    `batch_pairs` pairs; a list longer than `batch_pairs` is a batch of its own.
    Returns [(row indices, list id of each row)], every kept row once."""
    order = rng.permutation(len(query_ids))
    lists = []
    for idx in pd.Series(order).groupby(np.asarray(query_ids)[order], sort=False):
        idx = idx[1].to_numpy()
        parts = [idx] if list_size is None else np.array_split(idx, -(-len(idx) // list_size))
        lists += [c for c in parts if len(c) > 1 or not drop_single]
    batches, cur, n = [], [], 0
    for j in rng.permutation(len(lists)):
        if cur and n + len(lists[j]) > batch_pairs:
            batches.append(cur)
            cur, n = [], 0
        cur.append(lists[j])
        n += len(lists[j])
    if cur:
        batches.append(cur)
    return [(np.concatenate(b), np.repeat(np.arange(len(b)), [len(c) for c in b])) for b in batches]


def listnet_loss(pred, gain, seg):
    """Mean over lists of -sum(t * log softmax(pred)), t = gain / sum(gain) within each list (seg = list id per row).
    Lists whose gains are all 0 carry no ordering information and are skipped."""
    pred, n = pred.float(), int(seg.max()) + 1
    mx = torch.full((n,), -torch.inf, device=pred.device).scatter_reduce(0, seg, pred.detach(), 'amax')
    log_z = torch.zeros(n, device=pred.device).index_add(0, seg, torch.exp(pred - mx[seg])).log() + mx
    g_sum = torch.zeros(n, device=pred.device).index_add(0, seg, gain)
    target = gain / g_sum[seg].clamp_min(1e-12)
    per_list = -torch.zeros(n, device=pred.device).index_add(0, seg, target * (pred - log_z[seg]))
    keep = g_sum > 0
    return per_list[keep].mean() if keep.any() else pred.sum() * 0


def compute_loss(logits, target, cfg, seg=None):
    if cfg['loss'] == 'mse':
        return F.mse_loss(logits.squeeze(-1).float(), target)
    if cfg['loss'] in ('ce4', 'ce4_soft'):   # target: class index (ce4) or a probability row per pair (ce4_soft)
        return F.cross_entropy(logits.float(), target)
    if cfg['loss'] == 'listnet':
        return listnet_loss(logits.squeeze(-1), target, seg)
    raise ValueError(f'unknown loss {cfg["loss"]}')


def to_score(logits):
    # one output: the logit itself; four outputs ('ce4'): expected official gain
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    return torch.softmax(logits, -1) @ torch.tensor(CLASS_GAIN, device=logits.device)


@torch.no_grad()
def predict(model, tokenizer, df, max_len, batch_size=512, fp16=True, return_probs=False):
    # Sorts pairs by length to reduce padding, then puts the scores back in the original row order.
    # return_probs (ce4 models only): also return a DataFrame p_E, p_S, p_C, p_I in df's row order.
    model.eval()
    order = np.argsort((df['query'].str.len() + df['text'].str.len()).to_numpy(), kind='stable')
    q, t = df['query'].to_numpy()[order], df['text'].to_numpy()[order]
    out = np.empty(len(df), dtype=np.float32)
    probs = np.empty((len(df), len(LABELS)), dtype=np.float32) if return_probs else None
    for i in range(0, len(df), batch_size):
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
            logits = model(**encode(tokenizer, q[i:i + batch_size], t[i:i + batch_size], max_len)).logits
        logits = logits.float()
        out[i:i + batch_size] = to_score(logits).cpu().numpy()
        if return_probs:
            assert logits.shape[-1] == len(LABELS), 'return_probs needs a ce4 model (4 outputs)'
            probs[i:i + batch_size] = torch.softmax(logits, -1).cpu().numpy()
    model.train()
    scores = np.empty_like(out)
    scores[order] = out
    if not return_probs:
        return scores
    p = np.empty_like(probs)
    p[order] = probs
    return scores, pd.DataFrame(p, columns=[f'p_{l}' for l in LABELS])


def sampler_of(cfg):
    """'query' or 'pair' (see the module docstring); listnet needs whole query lists."""
    sampler = cfg.get('sampler', 'query' if cfg['loss'] == 'listnet' else 'pair')
    assert sampler in ('pair', 'query'), f'unknown sampler {sampler}'
    assert not (cfg['loss'] == 'listnet' and sampler == 'pair'), 'listnet needs sampler "query"'
    return sampler


def make_targets(train_df, cfg):
    """Training target per row: float gain (mse, listnet), class index (ce4) or a probability row (ce4_soft)."""
    if cfg['loss'] == 'ce4':
        return train_df['esci_label'].map({l: i for i, l in enumerate(LABELS)}).to_numpy(np.int64)
    if cfg['loss'] == 'ce4_soft':
        alpha = cfg.get('alpha', 0.0)
        assert 0.0 <= alpha <= 1.0, f'alpha must be in [0, 1], got {alpha}'
        soft = train_df[[f'soft_{l}' for l in LABELS]].to_numpy(np.float64)
        assert np.isfinite(soft).all() and (soft >= 0).all(), 'soft targets must be finite and non-negative'
        assert np.allclose(soft.sum(1), 1.0, atol=1e-3), 'each soft target row must sum to 1'
        hard = np.eye(len(LABELS))[train_df['esci_label'].map({l: i for i, l in enumerate(LABELS)}).to_numpy()]
        return (alpha * hard + (1 - alpha) * soft / soft.sum(1, keepdims=True)).astype(np.float32)
    return train_df['gain'].to_numpy(np.float32)


def make_epochs(train_df, cfg, seed):
    """Batches of every epoch, drawn up front so the schedule knows the number of steps. 'pair': a shuffled
    DataLoader of row indices (as in B3); 'query': `query_batches` lists of (row indices, list id per row)."""
    if sampler_of(cfg) == 'query':
        rng = np.random.default_rng(seed)
        return [query_batches(train_df['query_id'].to_numpy(), rng, cfg.get('list_size', 16), cfg['batch_size'],
                              drop_single=cfg['loss'] == 'listnet')
                for _ in range(cfg['epochs'])]
    loader = DataLoader(np.arange(len(train_df)), batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    return [loader] * cfg['epochs']


def warmup_of(cfg, total):
    """Warm-up steps: cfg['warmup_ratio'] of all steps, else min(warmup_steps, 0.42 * steps).
    Amazon: 5,000 of ~11.8k steps; the 0.42 cap scales it down for smoke runs."""
    if cfg.get('warmup_ratio') is not None:
        return int(cfg['warmup_ratio'] * total)
    return min(cfg['warmup_steps'], int(0.42 * total))


def train_ce(cfg, train_df, seed, eval_df=None, eval_every=None, print_every=1000):
    # Trains for cfg['epochs'] and returns the model after the last step (no early stopping, as in Amazon's script).
    # eval_df is scored every `eval_every` steps only to record the val curve.
    set_seed(seed)
    src = model_source(cfg['model'])
    tokenizer = AutoTokenizer.from_pretrained(src)
    if cfg['loss'] in ('ce4', 'ce4_soft'):
        model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=len(LABELS),
                                                                   ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1)
    model = model.float().to(DEVICE)
    q, t = train_df['query'].to_numpy(), train_df['text'].to_numpy()
    y = make_targets(train_df, cfg)
    sampler = sampler_of(cfg)
    epochs = make_epochs(train_df, cfg, seed)
    total = sum(len(e) for e in epochs)
    warmup = warmup_of(cfg, total)
    opt = make_optimizer(model, cfg)
    sched = get_linear_schedule_with_warmup(opt, warmup, total)
    scaler = make_scaler(cfg['fp16'])

    loss_log, val_curve = [], []
    step, run_loss, t0, eval_sec = 0, torch.zeros((), device=DEVICE), time.time(), 0.0
    model.train()
    for batches in epochs:
        for b in batches:
            idx, seg = b if sampler == 'query' else (b.numpy(), None)
            if cfg['loss'] != 'listnet':
                seg = None   # pointwise losses ignore the list ids of 'query' batches
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['fp16']):
                logits = model(**encode(tokenizer, q[idx], t[idx], cfg['max_len'])).logits
            loss = compute_loss(logits, torch.from_numpy(y[idx]).to(DEVICE), cfg,
                                None if seg is None else torch.from_numpy(seg).to(DEVICE))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.detach()
            if step % 100 == 0 or step == total:
                n = 100 if step % 100 == 0 else step % 100
                avg = run_loss.item() / n
                assert np.isfinite(avg), f'loss is not finite at step {step}'
                loss_log.append({'step': step, 'loss': round(avg, 6), 'lr': sched.get_last_lr()[0]})
                run_loss.zero_()
                if step % print_every == 0 or step == total:
                    el = time.time() - t0 - eval_sec   # training time only, val checks excluded
                    print(f'step {step:>6}/{total} | loss {avg:.5f} | {step / el:.1f} it/s | '
                          f'{el / 60:.1f} min, ~{(total - step) / (step / el) / 60:.0f} min left')
            if eval_df is not None and eval_every and (step % eval_every == 0 or step == total):
                te = time.time()
                torch.cuda.empty_cache()   # return cached training memory before the large prediction batches
                m, _ = summarize(eval_df, predict(model, tokenizer, eval_df, cfg['max_len'],
                                                  batch_size=cfg.get('eval_batch_size', 512), fp16=cfg['fp16']))
                val_curve.append({'step': step, 'val_ndcg': m['ndcg'], 'val_e_over_s': m['e_over_s']})
                print(f'   val @ {step}: nDCG {m["ndcg"]} | E-over-S {m["e_over_s"]}')
                eval_sec += time.time() - te
    info = {'steps': step, 'warmup': warmup, 'train_minutes': round((time.time() - t0 - eval_sec) / 60, 1),
            'eval_minutes': round(eval_sec / 60, 1), 'it_per_s': round(step / (time.time() - t0 - eval_sec), 2),
            'loss_log': loss_log, 'val_curve': val_curve}
    if sampler == 'query':
        info['train_pairs_used'] = int(sum(len(i) for i, _ in epochs[0]))
        info['mean_batch_pairs'] = round(info['train_pairs_used'] / len(epochs[0]), 2)
    info['sampler'] = sampler
    if sampler == 'query':
        info['list_size'] = cfg.get('list_size', 16)
    return model, tokenizer, info

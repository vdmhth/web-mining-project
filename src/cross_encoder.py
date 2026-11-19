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
"""
import random
import time

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


def query_batches(query_ids, rng, list_size=16, batch_pairs=32):
    """One epoch of ListNet batches. Each query's candidates are shuffled and split into near-equal lists of at most
    `list_size`; lists of one pair are dropped (their ListNet loss is always 0). Lists are shuffled and packed into
    batches of at most `batch_pairs` pairs. Returns [(row indices, list id of each row)], every kept row once."""
    order = rng.permutation(len(query_ids))
    lists = []
    for idx in pd.Series(order).groupby(np.asarray(query_ids)[order], sort=False):
        idx = idx[1].to_numpy()
        lists += [c for c in np.array_split(idx, -(-len(idx) // list_size)) if len(c) > 1]
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
    if cfg['loss'] == 'ce4':
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


def train_ce(cfg, train_df, seed, eval_df=None, eval_every=None, print_every=1000):
    # Trains for cfg['epochs'] and returns the model after the last step (no early stopping, as in Amazon's script).
    # eval_df is scored every `eval_every` steps only to record the val curve.
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'])
    if cfg['loss'] == 'ce4':
        model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], num_labels=len(LABELS),
                                                                   ignore_mismatched_sizes=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], num_labels=1)
    model = model.float().to(DEVICE)
    q, t = train_df['query'].to_numpy(), train_df['text'].to_numpy()
    if cfg['loss'] == 'ce4':
        y = train_df['esci_label'].map({l: i for i, l in enumerate(LABELS)}).to_numpy(np.int64)
    else:
        y = train_df['gain'].to_numpy(np.float32)
    if cfg['loss'] == 'listnet':   # batches of whole query lists, drawn up front so the schedule knows the length
        rng = np.random.default_rng(seed)
        epochs = [query_batches(train_df['query_id'].to_numpy(), rng, cfg.get('list_size', 16), cfg['batch_size'])
                  for _ in range(cfg['epochs'])]
    else:
        loader = DataLoader(np.arange(len(train_df)), batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))
        epochs = [loader] * cfg['epochs']
    total = sum(len(e) for e in epochs)
    warmup = min(cfg['warmup_steps'], int(0.42 * total))   # Amazon: 5,000 of ~11.8k steps; scaled down for smoke runs
    opt = make_optimizer(model, cfg)
    sched = get_linear_schedule_with_warmup(opt, warmup, total)
    scaler = make_scaler(cfg['fp16'])

    loss_log, val_curve = [], []
    step, run_loss, t0, eval_sec = 0, torch.zeros((), device=DEVICE), time.time(), 0.0
    model.train()
    for batches in epochs:
        for b in batches:
            idx, seg = b if cfg['loss'] == 'listnet' else (b.numpy(), None)
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
    if cfg['loss'] == 'listnet':
        info['train_pairs_used'] = int(sum(len(i) for i, _ in epochs[0]))
    return model, tokenizer, info

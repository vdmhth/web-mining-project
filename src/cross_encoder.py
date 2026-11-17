"""Cross-encoder training and prediction: plain PyTorch loop around a Hugging Face model (num_labels = 1).

Moved from notebook 03_cross_encoder. A new training objective (Direction 2) only needs a new branch in
`compute_loss`; new input fields (Direction 1) only need a different `text` column in the DataFrames.
DataFrames passed in need the columns `query`, `text` and, for training, `gain`.
"""
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .evaluation import summarize

DEVICE = 'cuda'


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


def compute_loss(pred, target, cfg):
    if cfg['loss'] == 'mse':
        return F.mse_loss(pred.float(), target)
    raise ValueError(f'unknown loss {cfg["loss"]}')


@torch.no_grad()
def predict(model, tokenizer, df, max_len, batch_size=512, fp16=True):
    # Sorts pairs by length to reduce padding, then puts the scores back in the original row order.
    model.eval()
    order = np.argsort((df['query'].str.len() + df['text'].str.len()).to_numpy(), kind='stable')
    q, t = df['query'].to_numpy()[order], df['text'].to_numpy()[order]
    out = np.empty(len(df), dtype=np.float32)
    for i in range(0, len(df), batch_size):
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
            logits = model(**encode(tokenizer, q[i:i + batch_size], t[i:i + batch_size], max_len)).logits
        out[i:i + batch_size] = logits.squeeze(-1).float().cpu().numpy()
    model.train()
    scores = np.empty_like(out)
    scores[order] = out
    return scores


def train_ce(cfg, train_df, seed, eval_df=None, eval_every=None, print_every=1000):
    # Trains for cfg['epochs'] and returns the model after the last step (no early stopping, as in Amazon's script).
    # eval_df is scored every `eval_every` steps only to record the val curve.
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'])
    model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], num_labels=1).float().to(DEVICE)
    q, t = train_df['query'].to_numpy(), train_df['text'].to_numpy()
    y = train_df['gain'].to_numpy(np.float32)
    loader = DataLoader(np.arange(len(train_df)), batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    total = len(loader) * cfg['epochs']
    warmup = min(cfg['warmup_steps'], int(0.42 * total))   # Amazon: 5,000 of ~11.8k steps; scaled down for smoke runs
    opt = make_optimizer(model, cfg)
    sched = get_linear_schedule_with_warmup(opt, warmup, total)
    scaler = make_scaler(cfg['fp16'])

    loss_log, val_curve = [], []
    step, run_loss, t0, eval_sec = 0, torch.zeros((), device=DEVICE), time.time(), 0.0
    model.train()
    for _ in range(cfg['epochs']):
        for idx in loader:
            idx = idx.numpy()
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg['fp16']):
                pred = model(**encode(tokenizer, q[idx], t[idx], cfg['max_len'])).logits.squeeze(-1)
            loss = compute_loss(pred, torch.from_numpy(y[idx]).to(DEVICE), cfg)
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
                m, _ = summarize(eval_df, predict(model, tokenizer, eval_df, cfg['max_len'], fp16=cfg['fp16']))
                val_curve.append({'step': step, 'val_ndcg': m['ndcg'], 'val_e_over_s': m['e_over_s']})
                print(f'   val @ {step}: nDCG {m["ndcg"]} | E-over-S {m["e_over_s"]}')
                eval_sec += time.time() - te
    info = {'steps': step, 'warmup': warmup, 'train_minutes': round((time.time() - t0 - eval_sec) / 60, 1),
            'eval_minutes': round(eval_sec / 60, 1), 'it_per_s': round(step / (time.time() - t0 - eval_sec), 2),
            'loss_log': loss_log, 'val_curve': val_curve}
    return model, tokenizer, info

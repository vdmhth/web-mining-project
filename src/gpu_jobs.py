"""Runs independent training / prediction jobs on every visible GPU: one subprocess per GPU, one job at a time each.

A notebook writes its data once (`save_frame`), lists its jobs, and calls `run_jobs`. Each job runs in
`python -m src.gpu_jobs <job.json>` with `CUDA_VISIBLE_DEVICES` set to one GPU, so the code of `src.cross_encoder`
(device 'cuda') is unchanged; a job trains with the same function, seed and settings as a run in the notebook
itself would, so where it ran does not change what it computes. With one GPU the jobs run one after the other.

Job kinds (dict keys besides 'name' and 'kind'):
- 'train': `cfg`, `seed`, `train` (parquet path), `predict` ({split: parquet path}), optional `eval_every`,
  `print_every`, `curve_split` (default 'val': the split scored every `eval_every` steps), `save_dir` (checkpoint
  in fp16, saved after predicting) and `reload_check` (rows of the first predicted split re-scored by the reloaded
  checkpoint; the largest score difference is returned as `reload_gap`).
- 'predict': `checkpoint` (folder), `data` (parquet with `query`, `text`), `max_len`, optional `batch_size`.
- 'noop' / 'fail': for the CPU tests of the scheduler.

Result of a job (`load_result`): {'info': dict (train_ce info plus gpu, predict_minutes, predict_pairs,
peak_memory_gb, ...), 'scores': {split: array}, 'probs': {split: DataFrame p_E..p_I or None}}.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOB_COLS = ('example_id', 'query_id', 'query', 'text', 'esci_label', 'gain')   # plus soft_* (distillation targets)
HF_IGNORE = ['onnx/*', 'openvino/*', '*.onnx', '*.h5', '*.msgpack', '*.ot', 'tf_model*', 'flax_model*', 'rust_model*']


def n_gpus():
    import torch
    return torch.cuda.device_count()


def prefetch(name):
    """Downloads a Hugging Face model once, before the jobs start (so parallel jobs do not download it together).
    A local copy from the inputs (notebook 00_hf_models) is used when it is attached; returns where it is."""
    from .cross_encoder import model_source
    src = model_source(name)
    if src != name:
        print(f'{name}: local copy {src}')
        return src
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(name, ignore_patterns=HF_IGNORE)
    except Exception as e:   # no Internet and no local copy
        raise RuntimeError(f'{name}: download failed ({type(e).__name__}). Turn Internet on, or attach the output '
                           f'of 00_hf_models (hf_models/{name.replace("/", "__")}/)') from e
    print(f'{name}: in the cache {path}')
    return path


def save_frame(df, path):
    """Writes the columns a job reads (JOB_COLS and soft_*) to parquet; returns the path. Put the data under /tmp,
    not /kaggle/working: it is a temporary copy and should not count against (or end up in) the notebook output."""
    keep = [c for c in df.columns if c in JOB_COLS or str(c).startswith('soft_')]
    df = df[keep] if keep else df
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return str(path)


def _echo(path, offset, name):
    with open(path, 'rb') as f:
        f.seek(offset)
        chunk = f.read()
    text = chunk.decode('utf-8', errors='replace')
    if '\n' not in text:
        return offset
    text = text[:text.rindex('\n') + 1]          # whole lines only; the rest is printed next time
    for line in text.splitlines():
        print(f'[{name}] {line}', flush=True)
    return offset + len(text.encode('utf-8'))


def resources(paths=('/kaggle/working', '/tmp')):
    """CPU count, RAM and free disk, printed before the jobs start (Kaggle: output folder /kaggle/working is limited)."""
    msg = [f'{os.cpu_count()} CPUs']
    try:
        mem = dict(l.split(':', 1) for l in Path('/proc/meminfo').read_text().splitlines())
        msg.append(f"RAM {int(mem['MemTotal'].split()[0]) / 2**20:.0f} GB "
                   f"({int(mem['MemAvailable'].split()[0]) / 2**20:.0f} GB free)")
    except (OSError, KeyError, ValueError):
        pass
    for p in paths:
        if Path(p).exists():
            msg.append(f'{p} {shutil.disk_usage(p).free / 2**30:.0f} GB free')
    return ' | '.join(msg)


def run_jobs(jobs, work, n_gpu=None, poll=10):
    """Runs `jobs` in list order, each on the first free GPU; prints each job's output with a [name] prefix and,
    after each finished job, an estimate of the time left. Returns {name: result} in the order of `jobs`.
    If a job fails, the others still finish, then it raises."""
    work = Path(work)
    names = [j['name'] for j in jobs]
    assert len(set(names)) == len(names), 'job names must be unique'
    n_gpu = n_gpu or n_gpus()
    assert n_gpu >= 1, 'No GPU: Settings -> Accelerator -> GPU'
    threads = max(1, (os.cpu_count() or 4) // n_gpu)
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(ROOT), os.environ.get('PYTHONPATH', '')]),
           'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': str(threads), 'MKL_NUM_THREADS': str(threads),
           'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'TRANSFORMERS_VERBOSITY': 'error',
           'HF_HUB_OFFLINE': '1'}   # prefetch() has put the model on disk: jobs never reach the Hub
    print(f'{len(jobs)} jobs on {n_gpu} GPU(s), {threads} CPU threads each | {resources()}', flush=True)
    pending, running, failed, t_all, took = list(jobs), {}, [], time.time(), []
    try:
        while pending or running:
            for g in range(n_gpu):
                if g in running or not pending:
                    continue
                job = pending.pop(0)
                d = work / job['name']
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
                (d / 'job.json').write_text(json.dumps({**job, 'out': str(d)}), encoding='utf-8')
                log = open(d / 'log.txt', 'wb')
                proc = subprocess.Popen([sys.executable, '-u', '-m', 'src.gpu_jobs', str(d / 'job.json')],
                                        stdout=log, stderr=subprocess.STDOUT, cwd=str(work),
                                        env={**env, 'CUDA_VISIBLE_DEVICES': str(g)})
                running[g] = {'job': job, 'proc': proc, 'log': log, 'offset': 0, 't0': time.time()}
                print(f"[{job['name']}] started on GPU {g}", flush=True)
            time.sleep(poll)
            for g in list(running):
                r = running[g]
                name = r['job']['name']
                r['offset'] = _echo(work / name / 'log.txt', r['offset'], name)
                if r['proc'].poll() is None:
                    continue
                r['log'].close()
                _echo(work / name / 'log.txt', r['offset'], name)
                ok = r['proc'].returncode == 0 and (work / name / 'done.json').exists()
                print(f"[{name}] {'finished' if ok else 'FAILED (exit %s)' % r['proc'].returncode} in "
                      f"{(time.time() - r['t0']) / 60:.1f} min | {(time.time() - t_all) / 60:.1f} min since the start",
                      flush=True)
                if not ok:
                    failed.append(name)
                took.append(time.time() - r['t0'])
                del running[g]
                left = len(pending) + len(running)
                if left:
                    rounds = -(-len(pending) // n_gpu) + (1 if running else 0)
                    print(f'{left} jobs left | about {rounds * np.mean(took) / 60:.0f} min more '
                          f'(mean job {np.mean(took) / 60:.0f} min)', flush=True)
    finally:
        for r in running.values():   # interrupted: do not leave processes on the GPUs
            r['proc'].kill()
            r['log'].close()
    assert not failed, f'jobs failed: {failed} (full output in {work}/<name>/log.txt)'
    return {n: load_result(work / n) for n in names}


def load_result(d):
    d = Path(d)
    info = json.loads((d / 'done.json').read_text(encoding='utf-8'))
    scores, probs = {}, {}
    for f in sorted(d.glob('pred_*.npz')):
        z = np.load(f)
        split = f.stem[len('pred_'):]
        scores[split] = z['score']
        probs[split] = pd.DataFrame(z['probs'], columns=[f'p_{l}' for l in 'ESCI']) if z['probs'].size else None
    return {'info': info, 'scores': scores, 'probs': probs}


def cleanup(work, data=None):
    """Removes the data copies (`data` folder) and prediction files of a finished set of jobs (the logs stay)."""
    work = Path(work)
    shutil.rmtree(Path(data) if data else work / 'data', ignore_errors=True)
    for f in work.glob('*/pred_*.npz'):
        f.unlink()


def _save_pred(out, split, o):
    score, probs = (o[0], o[1].to_numpy(np.float32)) if isinstance(o, tuple) else (o, np.empty((0, 4), np.float32))
    np.savez(out / f'pred_{split}.npz', score=np.asarray(score, np.float32), probs=probs)


def _train(job, out):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .cross_encoder import DEVICE, predict, train_ce

    cfg = job['cfg']
    train = pd.read_parquet(job['train'])
    evals = {s: pd.read_parquet(p) for s, p in job['predict'].items()}
    curve = evals.get(job.get('curve_split', 'val')) if job.get('eval_every') else None
    model, tok, info = train_ce(cfg, train, job['seed'], eval_df=curve, eval_every=job.get('eval_every'),
                                print_every=job.get('print_every', 1000))
    del train
    torch.cuda.empty_cache()
    probs = cfg['loss'] in ('ce4', 'ce4_soft')
    t0, first = time.time(), None
    for s, df in evals.items():
        o = predict(model, tok, df, cfg['max_len'], batch_size=cfg.get('eval_batch_size', 512), fp16=cfg['fp16'],
                    return_probs=probs)
        _save_pred(out, s, o)
        first = first if first is not None else (s, o[0] if probs else o)
    info.update({'predict_minutes': round((time.time() - t0) / 60, 2),
                 'predict_pairs': int(sum(len(df) for df in evals.values())),
                 'gpu': torch.cuda.get_device_name(0),
                 'peak_memory_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)})
    if job.get('save_dir'):
        d = Path(job['save_dir'])
        model.half().save_pretrained(d)
        tok.save_pretrained(d)
        print('checkpoint saved:', d, flush=True)
        if job.get('reload_check'):
            del model
            torch.cuda.empty_cache()
            n = job['reload_check']
            s, ref = first
            m2 = AutoModelForSequenceClassification.from_pretrained(d, torch_dtype=torch.float16).to(DEVICE)
            again = predict(m2, AutoTokenizer.from_pretrained(d), evals[s].iloc[:n], cfg['max_len'],
                            batch_size=cfg.get('eval_batch_size', 512), fp16=True)
            info['reload_gap'] = round(float(np.abs(again - ref[:n]).max()), 5)
            info['checkpoint_mb'] = round(sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 2**20)
    return info


def _predict(job, out):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .cross_encoder import DEVICE, predict

    df = pd.read_parquet(job['data'])
    model = AutoModelForSequenceClassification.from_pretrained(job['checkpoint'], torch_dtype=torch.float16).to(DEVICE)
    tok = AutoTokenizer.from_pretrained(job['checkpoint'])
    t0 = time.time()
    _save_pred(out, 'data', predict(model, tok, df, job['max_len'], batch_size=job.get('batch_size', 256), fp16=True))
    return {'predict_minutes': round((time.time() - t0) / 60, 2), 'predict_pairs': len(df),
            'gpu': torch.cuda.get_device_name(0)}


def main(path):
    job = json.loads(Path(path).read_text(encoding='utf-8'))
    out = Path(job['out'])
    kind = job['kind']
    if kind == 'noop':
        info = {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'value': job.get('value')}
        np.savez(out / 'pred_x.npz', score=np.arange(3, dtype=np.float32), probs=np.empty((0, 4), np.float32))
    elif kind == 'fail':
        raise SystemExit('this job fails on purpose')
    elif kind == 'train':
        info = _train(job, out)
    elif kind == 'predict':
        info = _predict(job, out)
    else:
        raise ValueError(f'unknown job kind {kind}')
    (out / 'done.json').write_text(json.dumps(info), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])

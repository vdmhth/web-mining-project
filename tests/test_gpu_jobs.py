"""Checks for the job scheduler of src/gpu_jobs.py (CPU only: 'noop' and 'fail' jobs, no GPU or model needed).

Run from the repo root: `python -m pytest tests` or, without pytest, `python tests/test_gpu_jobs.py`.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cross_encoder import model_source           # noqa: E402
from src.gpu_jobs import cleanup, run_jobs, save_frame   # noqa: E402


def test_jobs_spread_over_gpus_and_keep_order():
    with tempfile.TemporaryDirectory() as work:
        jobs = [{'name': f'job{i}', 'kind': 'noop', 'value': i} for i in range(5)]
        res = run_jobs(jobs, work, n_gpu=2, poll=0.2)
        assert list(res) == [f'job{i}' for i in range(5)]
        assert [r['info']['value'] for r in res.values()] == list(range(5))
        assert {r['info']['cuda_visible_devices'] for r in res.values()} == {'0', '1'}
        assert all((r['scores']['x'] == np.arange(3)).all() and r['probs']['x'] is None for r in res.values())


def test_failed_job_raises_after_the_others_finish():
    with tempfile.TemporaryDirectory() as work:
        jobs = [{'name': 'bad', 'kind': 'fail'}, {'name': 'good', 'kind': 'noop'}]
        try:
            run_jobs(jobs, work, n_gpu=2, poll=0.2)
        except AssertionError as e:
            assert 'bad' in str(e) and 'good' not in str(e)
            assert (Path(work) / 'good' / 'done.json').exists()
        else:
            raise AssertionError('a failed job should raise')


def test_cleanup_keeps_logs():
    import pandas as pd
    with tempfile.TemporaryDirectory() as work:
        save_frame(pd.DataFrame({'a': [1]}), Path(work) / 'data' / 'x.parquet')
        run_jobs([{'name': 'j', 'kind': 'noop'}], work, n_gpu=1, poll=0.2)
        cleanup(work)
        assert not (Path(work) / 'data').exists() and not list(Path(work).glob('*/pred_*.npz'))
        assert (Path(work) / 'j' / 'log.txt').exists() and (Path(work) / 'j' / 'done.json').exists()


def test_model_source():
    with tempfile.TemporaryDirectory() as root:
        assert model_source('org/some-model', root) == 'org/some-model'          # nothing attached: the Hub name
        d = Path(root) / 'any' / 'hf_models' / 'org__other-model'
        d.mkdir(parents=True)
        (d / 'config.json').write_text('{}')
        assert Path(model_source('org/other-model', root)) == d
        flat = Path(root) / 'uploaded' / 'org__flat-model'                        # dataset uploaded without hf_models/
        flat.mkdir(parents=True)
        (flat / 'config.json').write_text('{}')
        assert Path(model_source('org/flat-model', root)) == flat
        cache = Path(root) / 'cache' / 'models--org--cache-model' / 'snapshots' / 'abc123'   # Hugging Face cache layout
        cache.mkdir(parents=True)
        for f in ('config.json', 'tokenizer_config.json'):
            (cache / f).write_text('{}')
        assert Path(model_source('org/cache-model', root)) == cache
        plain = Path(root) / 'x' / 'Plain-Model'                                   # only the model name, any case
        plain.mkdir(parents=True)
        for f in ('config.json', 'tokenizer_config.json'):
            (plain / f).write_text('{}')
        assert Path(model_source('org/plain-model', root)) == plain
        assert model_source('org/plain-model-large', root) == 'org/plain-model-large'   # no false match


def test_runlog_load_any_folder():
    import numpy as np, pandas as pd
    from src.evaluation import RunLog
    part = pd.DataFrame({'example_id': range(4), 'query_id': [1, 1, 2, 2], 'esci_label': ['E', 'I', 'S', 'E'],
                         'gain': [1.0, 0.0, 0.1, 1.0]})
    with tempfile.TemporaryDirectory() as root:
        log = RunLog(Path(root) / 'res', {'val': part})
        assert not log.load('run_a', splits=['val'], root=root)
        pd.DataFrame({'example_id': range(4), 'score': np.arange(4.0)}).to_parquet(Path(root) / 'run_a_val.parquet')
        assert log.load('run_a', splits=['val'], root=root)                        # no scores/ folder


if __name__ == '__main__':
    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]
    for f in tests:
        f()
        print('ok  ', f.__name__)
    print(f'{len(tests)} tests passed')

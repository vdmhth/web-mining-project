"""Locating inputs on Kaggle."""
import glob
from pathlib import Path


def find(name, hint='', root='/kaggle/input'):
    """Path of the first file called `name` under the attached inputs."""
    hits = glob.glob(f'{root}/**/{name}', recursive=True)
    assert hits, f'{name} not found under {root}. {hint}'
    return Path(hits[0])

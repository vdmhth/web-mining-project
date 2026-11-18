"""Locating inputs on Kaggle and loading the outputs of 01_preprocess."""
import glob
from pathlib import Path

import pandas as pd

from .text import product_text

PREPROCESS_HINT = 'Add Input -> Notebook Output -> 01_preprocess'
STORED_TEXT = {'title': 'title', 'title_bc': 'text_title_bc', 'main': 'text_main'}   # columns of products_clean


def find(name, hint='', root='/kaggle/input'):
    """Path of the first file called `name` under the attached inputs."""
    hits = glob.glob(f'{root}/**/{name}', recursive=True)
    assert hits, f'{name} not found under {root}. {hint}'
    return Path(hits[0])


def product_texts(fields, root='/kaggle/input'):
    """Product text for a preset (title, title_bc, main, all), as a Series indexed by product_id."""
    path = find('products_clean.parquet', PREPROCESS_HINT, root)
    if fields in STORED_TEXT:
        prod = pd.read_parquet(path, columns=['product_id', STORED_TEXT[fields]])
        text = prod[STORED_TEXT[fields]]
    else:   # 'all' is not stored: build it from the cleaned fields
        prod = pd.read_parquet(path, columns=['product_id', 'title', 'brand', 'color', 'bullets', 'description'])
        text = product_text(prod, fields)
    return pd.Series(text.to_numpy(), index=prod['product_id'])


def load_pairs(fields='title', root='/kaggle/input'):
    """pairs.parquet (all splits) plus a `text` column with the product text for the preset `fields`."""
    pairs = pd.read_parquet(find('pairs.parquet', PREPROCESS_HINT, root))
    pairs['text'] = pairs['product_id'].map(product_texts(fields, root))
    assert pairs['text'].notna().all(), 'products missing from products_clean.parquet'
    return pairs

"""Checks for the Direction 3 features in src/features.py on tiny hand-made data.

Run from the repo root: `python -m pytest tests` or, without pytest, `python tests/test_features.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import MATCH_COLS, brand_list_features, match_features, match_row, units_of, within_query   # noqa: E402


def row(query, title, brand=None, color=None, bullets=None):
    return dict(zip(MATCH_COLS, match_row(query, title, brand, color, bullets)))


def test_overlap():
    r = row('red apple ipad case', 'Apple iPad Case', bullets='Fits the red model')
    assert r['overlap_title'] == 0.75 and r['overlap_main'] == 1.0
    assert r['q_words'] == 4 and r['title_words'] == 3 and r['bullet_words'] == 4


def test_numbers_and_units():
    assert units_of('64 GB usb 3.0') == {('64', 'gb')}
    assert units_of('12 pack') == units_of('12 count') == {('12', 'ct')}
    assert units_of('10.2" tablet') == units_of('10.2 inch tablet') == {('10.2', 'in')}
    r = row('ipad 64gb', 'Apple iPad 64 GB Wi-Fi')
    assert r['num_match'] == 1.0 and r['unit_match'] == 1.0
    r = row('ipad 128gb', 'Apple iPad 64 GB', bullets='storage 64GB')
    assert r['num_match'] == 0.0 and r['unit_match'] == 0.0 and r['q_has_unit'] == 1.0
    r = row('ipad case', 'Apple iPad 64 GB')
    assert r['num_match'] == 0.5 and r['unit_match'] == 0.5 and r['q_has_num'] == 0.0


def test_color_brand_negation():
    assert row('black shoes', 'Running Shoe', color='Black')['color_match'] == 1.0
    assert row('black shoes', 'Running Shoe', color='Red')['color_match'] == 0.0
    assert row('black shoes', 'Running Shoe')['color_match'] == 0.5          # product has no color
    assert row('grey hoodie', 'Hoodie gray')['color_match'] == 1.0
    assert row('nike air max', 'Air Max 90', brand='Nike')['brand_in_query'] == 1.0
    assert row('nike air max', 'Air Max 90', brand='Adidas')['brand_in_query'] == 0.0
    assert row('hp ink', 'Ink', brand='HP')['brand_in_query'] == 0.0         # brands shorter than 3 chars are skipped
    assert row('shampoo without sulfate', 'Shampoo with sulfate')['neg_violation'] == 1.0
    assert row('shampoo without sulfate', 'Sulfate-free shampoo')['neg_violation'] == 0.0
    assert row('shampoo without sulfate', 'Mild shampoo')['neg_violation'] == 0.0
    assert row('mild shampoo', 'Mild shampoo')['neg_violation'] == 0.5


def test_missing_fields():
    r = row('usb cable', None, None, None, None)
    assert r['overlap_title'] == 0.0 and r['title_words'] == 0.0
    df = match_features(pd.Series(['a b', 'c']), pd.Series(['a', None]), pd.Series([None, 'x']),
                        pd.Series([None, None]), pd.Series([np.nan, 'c']))
    assert list(df.columns) == MATCH_COLS and df.notna().all().all() and df['overlap_main'].tolist() == [0.5, 1.0]


def test_within_query():
    df = pd.DataFrame({'query_id': [1, 1, 1, 2, 3, 3], 'x': [3.0, 1.0, 2.0, 5.0, 4.0, 4.0]})
    w = within_query(df, 'x')
    assert w['x_rank'].tolist() == [0.0, 1.0, 0.5, 0.0, 0.5, 0.5]
    assert np.allclose(w['x_gap'], [0, 2, 1, 0, 0, 0])
    assert np.allclose(w['x_z'][:3], np.array([1, -1, 0]) / np.std([3, 1, 2]))
    assert w['x_z'][3:].tolist() == [0.0, 0.0, 0.0]                          # one item, all equal


def test_brand_list():
    q = pd.Series([1, 1, 1, 1, 2, 2])
    b = pd.Series(['Sony', 'sony ', 'LG', None, 'LG', 'Sony'])
    f = brand_list_features(q, b)
    assert np.allclose(f['brand_share'], [0.5, 0.5, 0.25, 0, 0.5, 0.5])
    assert f['top_brand'].tolist() == [1, 1, 0, 0, 0, 0]                      # query 2: no brand appears twice
    assert f['q_n_brands'].tolist() == [2, 2, 2, 2, 2, 2]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)

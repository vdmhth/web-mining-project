"""Lexical and list features. BM25 moved unchanged from notebook 02_bm25.

BM25 is used for baseline B1 (02_bm25) and as a feature of the LightGBM re-ranker (Direction 3).
The second part holds the other Direction 3 features (08_features): match rules, within-query and brand-list features.
"""
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

TOKEN = re.compile(r'[a-z0-9]+(?:\.[0-9]+)?')   # as in the EDA: lowercase, decimals such as 10.2 kept


def tokens(s):
    return TOKEN.findall(s.lower()) if isinstance(s, str) else []


class BM25:
    """Okapi BM25; document frequencies and average length come from the whole catalogue."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.df, total = Counter(), 0
        for d in docs:
            t = tokens(d)
            self.df.update(set(t))
            total += len(t)
        self.n = len(docs)
        self.avgdl = total / self.n

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n - n + 0.5) / (n + 0.5))

    def score(self, query, doc):
        # each query term counts once
        dt = tokens(doc)
        if not dt:
            return 0.0
        tf = Counter(dt)
        norm = self.k1 * (1 - self.b + self.b * len(dt) / self.avgdl)
        return sum(self.idf(t) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in set(tokens(query)) if t in tf)


# ---------------------------------------------------------------- Direction 3 (08_features)
# Query-product match rules, ported from eda/eda_audit.py (EDA single-feature table). Neutral value 0.5 when the
# query has nothing to check (no number, color or negation), so "no constraint" sits between "met" and "broken".

NUM = re.compile(r'\d+(?:\.\d+)?')
UNIT = re.compile(r'(\d+(?:\.\d+)?)\s*(gb|tb|mb|oz|ounces?|lbs?|pounds?|inch(?:es)?|in|"|ft|feet|foot|mm|cm|m|ml|l|'
                  r'liters?|litres?|w|watts?|v|volts?|mah|packs?|pk|pcs|count|ct|qt|gal|gallons?|xl|xxl|x)(?![a-z0-9])')
UNIT_ALIAS = {'ounce': 'oz', 'ounces': 'oz', 'lbs': 'lb', 'pound': 'lb', 'pounds': 'lb', 'inch': 'in', 'inches': 'in',
              '"': 'in', 'feet': 'ft', 'foot': 'ft', 'liter': 'l', 'liters': 'l', 'litre': 'l', 'litres': 'l',
              'watt': 'w', 'watts': 'w', 'volt': 'v', 'volts': 'v', 'pack': 'ct', 'packs': 'ct', 'pk': 'ct', 'pcs': 'ct',
              'count': 'ct', 'gallon': 'gal', 'gallons': 'gal'}
NEG = re.compile(r'\b(?:without|w/o|no|not|non)[\s-]+([a-z0-9]+)')
COLORS = {'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'gray', 'grey', 'silver',
          'gold', 'brown', 'orange', 'beige', 'clear', 'navy', 'teal'}

MATCH_COLS = ['overlap_title', 'overlap_main', 'num_match', 'unit_match', 'color_match', 'brand_in_query',
              'neg_violation', 'q_words', 'q_has_num', 'q_has_unit', 'q_has_color', 'q_has_neg',
              'title_words', 'bullet_words']


def _s(x):
    return x if isinstance(x, str) else ''


def units_of(s):
    """{(number, unit)} with unit aliases merged: '64 GB' and '64gb' -> ('64', 'gb'); '12 pack' and '12 count' -> ct."""
    return {(f'{float(n):g}', UNIT_ALIAS.get(u, u)) for n, u in UNIT.findall(s.lower())}


def colors_of(s):
    return {('gray' if c == 'grey' else c) for c in tokens(s) if c in COLORS}


def title_negates(title, term):
    """True when the (lower-case) title itself excludes the term, e.g. 'no nails', 'without hose', 'bpa-free'."""
    t = re.escape(term)
    return re.search(rf'\b(?:without|w/o|no|non|free of)[\s-]+(?:\w+[\s-]+)?{t}|{t}[\s-]*free\b', title) is not None


def match_row(query, title, brand, color, bullets):
    """Values of MATCH_COLS for one pair; missing product fields count as empty."""
    ql, tl, bl = _s(query).lower(), _s(title).lower(), _s(bullets).lower()
    qt, tt, bt = tokens(ql), tokens(tl), tokens(brand)
    qs, ts = set(qt), set(tt)
    bullet_t = tokens(bl)
    main_s = ts | set(bt) | set(tokens(color)) | set(bullet_t)

    qn = set(NUM.findall(ql))
    qu = units_of(ql)
    qc, pc = colors_of(ql), colors_of(tl + ' ' + _s(color))
    neg = set(NEG.findall(ql))
    hit = [x for x in neg if x in ts]
    bj = ' '.join(bt)
    return (
        len(qs & ts) / len(qs) if qs else 0.0,
        len(qs & main_s) / len(qs) if qs else 0.0,
        0.5 if not qn else float(qn <= set(NUM.findall(tl))),
        0.5 if not qu else len(qu & units_of(tl + ' ' + bl)) / len(qu),
        0.5 if not qc or not pc else float(bool(qc & pc)),
        float(len(bj) >= 3 and f' {bj} ' in f" {' '.join(qt)} "),
        0.5 if not neg else float(any(not title_negates(tl, x) for x in hit)),
        float(len(qt)), float(bool(qn)), float(bool(qu)), float(bool(qc)), float(bool(neg)),
        float(len(tt)), float(len(bullet_t)),
    )


def match_features(query, title, brand, color, bullets):
    """MATCH_COLS for aligned Series of query text and cleaned product fields; same index as `query`."""
    rows = [match_row(*r) for r in zip(query, title, brand, color, bullets)]
    return pd.DataFrame(rows, columns=MATCH_COLS, index=query.index)


def within_query(df, col, by='query_id'):
    """Position of each pair's `col` among its query's candidates.

    `<col>_rank`: 0 = highest in the query, 1 = lowest (ties averaged; 0 for a one-item list);
    `<col>_z`: z-score inside the query (0 when all equal); `<col>_gap`: query maximum minus the value.
    """
    x = df[col].astype('float64')
    g = x.groupby(df[by])
    n = g.transform('size')
    rank = g.rank(ascending=False, method='average')
    mean = g.transform('mean')
    std = np.sqrt(np.maximum((x * x).groupby(df[by]).transform('mean') - mean * mean, 0.0))
    return pd.DataFrame({
        f'{col}_rank': np.where(n > 1, (rank - 1) / np.maximum(n - 1, 1), 0.0),
        f'{col}_z': np.where(std > 1e-9, (x - mean) / np.where(std > 1e-9, std, 1.0), 0.0),
        f'{col}_gap': g.transform('max') - x,
    }, index=df.index)


def brand_list_features(query_id, brand):
    """Brand of a pair relative to its candidate list.

    `brand_share`: share of the query's candidates with the same brand (0 when the brand is missing);
    `top_brand`: 1 when the brand is the most frequent one in the list and appears more than once;
    `q_n_brands`: number of distinct known brands in the list.
    """
    b = brand.fillna('').astype(str).str.lower().str.strip()
    f = pd.DataFrame({'q': query_id.to_numpy(), 'b': b.to_numpy()}, index=query_id.index)
    n = f.groupby('q')['b'].transform('size')
    known = f[f['b'] != '']
    cnt = known.groupby(['q', 'b']).size().rename('cnt').reset_index()
    top = cnt.sort_values(['q', 'cnt', 'b'], ascending=[True, False, True]).drop_duplicates('q').set_index('q')
    c = f.merge(cnt, on=['q', 'b'], how='left')['cnt'].fillna(0).to_numpy()
    top_b, top_c = f['q'].map(top['b']), f['q'].map(top['cnt']).fillna(0)
    return pd.DataFrame({
        'brand_share': c / n.to_numpy(),
        'top_brand': ((f['b'] != '') & (f['b'] == top_b) & (top_c > 1)).astype(float).to_numpy(),
        'q_n_brands': f['q'].map(cnt.groupby('q').size()).fillna(0).astype(float).to_numpy(),
    }, index=query_id.index)

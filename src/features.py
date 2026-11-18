"""Lexical features. Moved unchanged from notebook 02_bm25.

BM25 is used for baseline B1 (02_bm25) and as a feature of the LightGBM re-ranker (Direction 3).
"""
import math
import re
from collections import Counter

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

"""Product-field cleaning and text presets. Moved unchanged from notebook 01_preprocess.

`products_clean.parquet` already holds the cleaned fields plus `text_title_bc` and `text_main`; use
`product_text(prod, 'all')` when a run needs the description too.
"""
import html
import re
import unicodedata

SEP = ' [SEP] '   # separator between product fields in the text_* columns
TAG = re.compile(r'<[a-zA-Z/!][^>]{0,200}>')   # '<' followed by a letter, so '< 5' or '<3' survive
WS = re.compile(r'\s+')

PRESETS = {
    'title': ['title'],
    'title_bc': ['title', 'brand', 'color'],
    'main': ['title', 'brand', 'color', 'bullets'],
    'all': ['title', 'brand', 'color', 'bullets', 'description'],
}
LABEL = {'brand': 'brand: ', 'color': 'color: ', 'bullets': '', 'description': ''}


def clean_text(s):
    """HTML entities and tags out, NFC, single spaces. Case, digits, units and punctuation are kept."""
    if not isinstance(s, str):
        return None
    s = html.unescape(TAG.sub(' ', html.unescape(s)))   # unescape first so '&lt;br&gt;' is removed too
    s = WS.sub(' ', unicodedata.normalize('NFC', s)).strip()
    return s or None


def clean_bullets(s):
    """Bullet points are newline-separated: clean each one and join them with ' ; '."""
    if not isinstance(s, str):
        return None
    items = [b for b in (clean_text(x) for x in s.split('\n')) if b]
    return ' ; '.join(items) or None


def product_text(df, fields):
    """Text of each product for a preset; missing fields are dropped and the title always comes first."""
    cols = PRESETS[fields]
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        v = df[c]
        out = out + (SEP + LABEL[c] + v.fillna('')).where(v.notna(), '')
    return out

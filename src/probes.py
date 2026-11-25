"""Negation probes (supplement N1-B, notebook 16; also used by the LLM reranker of notebook 19).

A query such as "ipad case 8th generation without front cover" excludes a term ("front cover"). The probe scores
the same candidate list under four versions of the query and checks that the model moves in the right direction:
- orig: the query as typed;
- drop: the negated phrase removed ("ipad case 8th generation");
- flip: the exclusion turned into an inclusion ("ipad case 8th generation with front cover");
- para: the same exclusion with another cue ("ipad case 8th generation no front cover").
`term_status` says whether a product text contains the term ('has'), excludes it itself ('negated', e.g.
"no front cover", "BPA-free") or does not mention it ('absent').

The parser is a heuristic: titles and idioms ("gods not dead", "i did not miss the boat") also parse, so the
query list of a probe is filtered by hand first and the excluded queries are passed as `exclude`.
"""
import re

PREFIX_CUES = ('without', 'w/o', 'no', 'not', 'non')
STOP = {'and', 'or', 'for', 'with', 'in', 'on', 'to', 'of', 'by', 'from', 'at', '&', 'plus', 'but', 'that'}
MAX_TERM = 3                                  # tokens kept after a prefix cue
ALT_CUE = {'without': 'no', 'no': 'without', 'w/o': 'without', 'not': 'without', 'non': 'without'}

_KEEP = re.compile(r"[^a-z0-9/'\- ]+")
_HYPHEN_CUE = re.compile(r'\b(non|no)-(?=[a-z0-9])')   # "non-gmo", "no-show": cue + one token
_NEG_BEFORE = re.compile(r"(?:\b(?:no|without|non|not|zero)|w/o|\bfree of)[\s\-]+(?:[a-z0-9']+[\s\-]+){0,2}$")
_FREE_AFTER = re.compile(r'[\s\-]*free\b')


def normalize(text):
    """Lower case, punctuation other than / ' - dropped, single spaces."""
    return ' '.join(_KEEP.sub(' ', str(text).lower()).split())


def negation_phrase(query, exclude=()):
    """First exclusion in the query as a dict (cue, term, form, tokens, start, end), or None.
    form 'prefix': tokens[start] is the cue and tokens[start + 1:end] the term ("without front cover");
    form 'suffix': tokens[start:end - 1] is the term and tokens[end - 1] is 'free' ("gluten free", "bpa-free").
    `exclude`: queries known not to be exclusions (hand-checked); they return None."""
    q = normalize(query)
    if q in {normalize(e) for e in exclude}:
        return None
    tokens, one_token_cue = [], set()
    for tok in q.split():   # "non-gmo" -> "non" "gmo"; such a cue only negates the next token
        m = _HYPHEN_CUE.match(tok)
        if m:
            one_token_cue.add(len(tokens))
            tokens += [m.group(1), tok[m.end():]]
        else:
            tokens.append(tok)
    for i, tok in enumerate(tokens):
        if tok in PREFIX_CUES:
            limit = 1 if i in one_token_cue else MAX_TERM
            j = i + 1
            while j < len(tokens) and j - i - 1 < limit and tokens[j] not in STOP:
                j += 1
            if j > i + 1:
                return {'cue': tok, 'term': ' '.join(tokens[i + 1:j]), 'form': 'prefix', 'tokens': tokens,
                        'start': i, 'end': j}
        if tok.endswith('-free') and len(tok) > len('-free'):
            return {'cue': 'free', 'term': tok[:-len('-free')], 'form': 'suffix', 'tokens': tokens,
                    'start': i, 'end': i + 1, 'joined': True}
        if tok == 'free' and i > 0 and tokens[i - 1] not in STOP and tokens[i - 1] not in PREFIX_CUES:
            return {'cue': 'free', 'term': tokens[i - 1], 'form': 'suffix', 'tokens': tokens,
                    'start': i - 1, 'end': i + 1}
    return None


def variants(query, exclude=()):
    """{orig, drop, flip, para, cue, term} for a query with an exclusion, else None (also None when nothing is
    left once the negated phrase is dropped, e.g. "without hooks")."""
    m = negation_phrase(query, exclude)
    if m is None:
        return None
    tok, s, e, term = m['tokens'], m['start'], m['end'], m['term']
    rest = tok[:s] + tok[e:]
    if not rest:
        return None
    if m['form'] == 'prefix':
        flip = tok[:s] + ['with'] + term.split() + tok[e:]
        para = tok[:s] + [ALT_CUE[m['cue']]] + term.split() + tok[e:]
    else:
        flip = rest + ['with', term]
        para = rest + ['without', term]
    return {'orig': ' '.join(tok), 'drop': ' '.join(rest), 'flip': ' '.join(flip), 'para': ' '.join(para),
            'cue': m['cue'], 'term': term}


def _term_pattern(term):
    words = normalize(term).split()
    last = words[-1]
    if len(last) > 4 and last.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):   # glasses, boxes, brushes
        stem = last[:-2]
    elif len(last) > 3 and last.endswith('s') and not last.endswith('ss'):        # hooks, nails, laces
        stem = last[:-1]
    else:
        stem = last
    words[-1] = re.escape(stem) + r'(?:s|es)?'
    body = r'[\s\-]+'.join([re.escape(w) for w in words[:-1]] + [words[-1]])
    return re.compile(r'(?<![a-z0-9])' + body + r'(?![a-z0-9])')


def term_status(text, term):
    """'has' if the text mentions the term without excluding it at least once, 'negated' if every mention is
    excluded by the text itself ("no front cover", "without hooks", "bpa-free", "free of nails"), 'absent' if the
    term (singular or plural) is not mentioned."""
    t = normalize(text)
    found = []
    for m in _term_pattern(term).finditer(t):
        negated = _NEG_BEFORE.search(t[max(0, m.start() - 40):m.start()]) is not None
        negated = negated or _FREE_AFTER.match(t, m.end()) is not None
        found.append(negated)
    if not found:
        return 'absent'
    return 'negated' if all(found) else 'has'

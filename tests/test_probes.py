"""Checks for the negation probes in src/probes.py on hand-written queries and product texts.

Run from the repo root: `python -m pytest tests` or, without pytest, `python tests/test_probes.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.probes import negation_phrase, normalize, term_status, variants   # noqa: E402


def cue_term(query, **kw):
    m = negation_phrase(query, **kw)
    return None if m is None else (m['cue'], m['term'])


def test_negation_phrase_prefix():
    assert cue_term('ipad case 8th generation without front cover') == ('without', 'front cover')
    assert cue_term("10' ratchet straps without hooks") == ('without', 'hooks')
    assert cue_term('printers without ink') == ('without', 'ink')
    assert cue_term('double ended markers not brush') == ('not', 'brush')
    assert cue_term('frame hangers without nails for drywall') == ('without', 'nails')   # stops at "for"
    assert cue_term('100w solar panel w/o controller') == ('w/o', 'controller')
    assert cue_term('Sleep Supplement WITHOUT Melatonin!') == ('without', 'melatonin')
    assert cue_term('dress shoes without laces size 10 black') == ('without', 'laces size 10')   # at most 3 tokens


def test_negation_phrase_hyphen_and_suffix():
    assert cue_term('non-gmo soybean oil') == ('non', 'gmo')        # hyphen cue negates one token only
    assert cue_term('no-show socks men') == ('no', 'show')
    assert cue_term('bpa-free water bottle') == ('free', 'bpa')
    assert cue_term('gluten free bread') == ('free', 'gluten')


def test_negation_phrase_none_and_exclude():
    assert negation_phrase('iphone 12 case') is None
    assert negation_phrase('notebook without') is None               # cue with nothing after it
    assert negation_phrase('free shipping') is None                  # nothing before "free"
    assert negation_phrase('nothing bundt cakes') is None            # "nothing" is not the cue "not"
    assert cue_term('gods not dead dvd') == ('not', 'dead dvd')      # parses: idioms are removed by hand ...
    assert negation_phrase('gods not dead dvd', exclude=['Gods Not Dead DVD']) is None   # ... via `exclude`


def test_variants():
    v = variants('ipad case 8th generation without front cover')
    assert v['orig'] == 'ipad case 8th generation without front cover'
    assert v['drop'] == 'ipad case 8th generation'
    assert v['flip'] == 'ipad case 8th generation with front cover'
    assert v['para'] == 'ipad case 8th generation no front cover'
    v = variants('frame hangers without nails for drywall')
    assert (v['drop'], v['flip']) == ('frame hangers for drywall', 'frame hangers with nails for drywall')
    assert variants('printers no ink')['para'] == 'printers without ink'
    v = variants('gluten free bread')
    assert (v['drop'], v['flip'], v['para']) == ('bread', 'bread with gluten', 'bread without gluten')
    v = variants('bpa-free water bottle')
    assert (v['drop'], v['flip']) == ('water bottle', 'water bottle with bpa')
    assert variants('without hooks') is None                         # nothing left after dropping the phrase
    assert variants('iphone 12 case') is None


def test_term_status():
    assert term_status('Folio Case with Front Cover, auto sleep/wake', 'front cover') == 'has'
    assert term_status('Clear back shell, no front cover', 'front cover') == 'negated'
    assert term_status('Transparent back case for iPad 10.2 inch', 'front cover') == 'absent'
    assert term_status('Front-cover folio for iPad', 'front cover') == 'has'              # hyphen between words
    assert term_status('Includes 2 hook sets', 'hooks') == 'has'                          # singular in the text
    assert term_status('Ratchet strap, 4 pack, without hook', 'hooks') == 'negated'
    assert term_status('BPA-free bottle, 32 oz', 'bpa') == 'negated'
    assert term_status('Adhesive strips, free of nails and screws', 'nails') == 'negated'
    assert term_status('No front cover version; front cover sold separately', 'front cover') == 'has'
    assert term_status('Glass bottle', 'glasses') == 'has'                                # -es plural
    assert term_status('Glasses case', 'glass') == 'has'
    assert term_status('Glassware set of 6', 'glass') == 'absent'                         # no partial words
    assert term_status('Brush pens, 24 colors', 'brushes') == 'has'


def test_normalize():
    assert normalize('  Hello,   W/O "Cover"!  ') == 'hello w/o cover'


if __name__ == '__main__':
    tests = [f for n, f in sorted(globals().items()) if n.startswith('test_')]
    for f in tests:
        f()
        print('ok  ', f.__name__)
    print(f'{len(tests)} tests passed')

"""Pure-Python port of the phase-1 query attribute extractors (`csre.data.text`), for the online path.

The phase-1 extractors are Polars expressions: ideal for 1.8M products, but on a single query the ~20 regex
expressions cost ~6.5 ms of per-expression overhead (the largest stage of a reranked request), and Polars cannot
run in forked workers. The same regular expressions run here with Python's `re` (identical syntax and leftmost-first
semantics for these patterns), so a query parses in well under a millisecond. `tests/test_attributes.py` checks
parity with the Polars implementation on a multilingual query set.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from ..data import text as T

_QUOTES = re.compile(r"[’`´]")
_THOUSANDS = re.compile(r"(\d),(\d{3})\b")
_DEC_COMMA = re.compile(r"(\d),(\d)")
_SPACES = re.compile(r"\s+")

_MEASURE = re.compile(T.MEASURE_RE)
_UNIT_GROUP = re.compile(T._UNIT_GROUP)
_NUM = re.compile(rf"({T.NUM})")
_IN_DIGIT = re.compile(r"\bin \d$")
_TRAIL_DIGIT = re.compile(r"\s\d$")
_DIMENSION = re.compile(T.DIMENSION_RE)
_PACK = re.compile(T.PACK_RE)
_FIRST_INT = re.compile(r"(\d+)")
_SIZE_NUM = re.compile(T.SIZE_NUM_RE)
_COMPAT = re.compile(T.COMPAT_RE)
_COMPAT_JP = re.compile(T.COMPAT_JP_RE)
_NEGATION = re.compile(T.NEGATION_RE)
_DIGIT = re.compile(r"\d")
_KW = {name: (re.compile(pat), canon) for name, pat, canon in [
    ("audience", T.AUDIENCE_RE, T.AUDIENCE_CANON), ("materials", T.MATERIAL_RE, T.MATERIAL_CANON),
    ("colors", T.COLOR_RE, T.COLOR_CANON), ("sizes", T.SIZE_RE, T.SIZE_CANON)]}
_MEASURE_CANON = {**T.UNIT_CANON, '"': "in"}


def for_matching(s: str | None) -> str:
    """`csre.data.text.for_matching` in Python."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _QUOTES.sub("'", s)
    s = _THOUSANDS.sub(r"\1\2", s)
    s = _DEC_COMMA.sub(r"\1.\2", s)
    return _SPACES.sub(" ", s).strip()


def _uniq(xs):
    return list(dict.fromkeys(x for x in xs if x is not None))


def _all(rx: re.Pattern, s: str) -> list[str]:
    return [m.group(0) for m in rx.finditer(s)]


def _canon(name: str, s: str) -> list[str]:
    rx, canon = _KW[name]
    return _uniq(canon.get(x, x) for x in _all(rx, s))


def _measures(s: str) -> list[str]:
    out = []
    for e in _all(_MEASURE, s):
        if _IN_DIGIT.search(e):            # "2 in 1"
            continue
        e = _TRAIL_DIGIT.sub("", e, count=1)
        n = _NUM.search(e)
        u = _UNIT_GROUP.search(e)
        if n is None or u is None:
            continue
        num = re.sub(r"\.0+$", "", n.group(1))
        out.append(f"{num} {_MEASURE_CANON.get(u.group(1), u.group(1))}")
    return _uniq(out)


def _dimensions(s: str) -> list[str]:
    return _uniq(re.sub(r"[×*]", "x", re.sub(r"\s+", "", e)) for e in _all(_DIMENSION, s))


def _pack(s: str) -> int | None:
    m = _PACK.search(s)
    if m is None:
        return None
    d = _FIRST_INT.search(m.group(0))
    return int(d.group(1)) if d else None


def _sizes(s: str) -> list[str]:
    labels = _canon("sizes", s)
    numeric = []
    for m in _SIZE_NUM.finditer(s):
        e = m.group(0)
        mm = _SIZE_NUM.search(e)
        v = (mm.group(1) or mm.group(2)) if mm else None
        if v is not None:
            numeric.append("size:" + v)
    return _uniq(labels + numeric)


def _compat(s: str) -> list[str]:
    raw = [_COMPAT.search(e).group(1) for e in _all(_COMPAT, s)] + \
          [_COMPAT_JP.search(e).group(1) for e in _all(_COMPAT_JP, s)]
    out = []
    for c in raw:
        c = " ".join(c.split(" ")[:4]).strip(" .-+")
        if len(c) >= 3 and re.search(r"[a-z]", c) and c.split(" ")[0] not in T.COMPAT_STOP:
            out.append(c)
    return _uniq(out)


@lru_cache(maxsize=100_000)
def query_attributes(query: str) -> dict:
    """Same fields as `csre.data.text.query_attribute_exprs` (without the q_ prefix) plus the normalised text."""
    s = for_matching(query)
    return {
        "norm": s, "measures": _measures(s), "dimensions": _dimensions(s), "pack_count": _pack(s),
        "sizes": _sizes(s), "audience": _canon("audience", s), "compat": _compat(s),
        "materials": _canon("materials", s), "colors": _canon("colors", s),
        "has_negation": _NEGATION.search(s) is not None, "has_digit": _DIGIT.search(s) is not None,
    }

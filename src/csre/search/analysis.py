"""Text analysis shared by BM25, the dense encoder and online query parsing.

One pure-Python implementation serves both paths, so offline indexing and online queries cannot drift:

1. `normalize` — the phase-1 matching normalisation (NFKC, lower-case, number formats) plus a
   digit/letter boundary split ("12oz" -> "12 oz", "rtx3080" -> "rtx 3080").
2. `units` — latin/digit words (minus a small en/es stop list) and CJK runs.
3. Units map to index terms *once per unique unit* (`unit_terms`, `unit_features`), and documents are
   mapped with a sparse product  doc x unit  @  unit x term.  Products repeat the same words heavily, so
   this is ~20x faster than expanding every occurrence, and bulk analysis is spread over processes.

* BM25 terms: plural-folded words ("batteries" -> "battery", "zapatos" -> "zapato") and overlapping
  character bigrams over CJK runs (the standard dictionary-free approach for Japanese).
* Encoder features: the word, fastText-style character n-grams of "<word>" sharing a total weight of 1
  (typo and inflection robustness), and character 1-3-grams of CJK runs.
"""
from __future__ import annotations

import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Sequence

import numpy as np
import re
import scipy.sparse as sp

# Han (+ext A, compat), hiragana, katakana (+ext, prolonged mark), 々 〆 — explicit ranges (std `re` is fast)
CJK_CHARS = "\u3005\u3006\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_RUN = re.compile(f"[{CJK_CHARS}]+")
_WORD = re.compile(r"\d+(?:\.\d+)?|[^\W_]+")   # CJK runs are removed before this runs
_QUOTES = re.compile(r"[’`´]")
_THOUSANDS = re.compile(r"(\d),(\d{3})\b")
_DEC_COMMA = re.compile(r"(\d),(\d)")
_SPACES = re.compile(r"\s+")
_DIGIT_ALPHA = re.compile(r"(\d)([a-zà-ɏ])")
_ALPHA_DIGIT = re.compile(r"([a-zà-ɏ])(\d)")
_IES = re.compile(r"^([^\W\d_]{2,}[^aeiou\d])ies$")
_ES = re.compile(r"^([^\W\d_]{2,}(?:ss|sh|ch|x|z))es$")
_S = re.compile(r"^([^\W\d_]{2,}[^su\d])s$")

CJK_MARK = "\x01"   # prefix marking a unit as a CJK run

STOPWORDS = frozenset({
    # en
    "a", "an", "and", "the", "of", "for", "with", "to", "in", "on", "by", "at", "or", "is", "are", "from", "as",
    "that", "this", "it", "be", "your", "you", "our",
    # es
    "de", "la", "el", "los", "las", "y", "con", "para", "en", "del", "un", "una", "por", "al", "o", "su", "sus",
    "que", "lo", "se",
})


# --------------------------------------------------------------------------------------
# single-string primitives (used online and, via workers, offline)
# --------------------------------------------------------------------------------------
def normalize(s: str | None) -> str:
    """Phase-1 `for_matching` semantics + digit/letter split."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _QUOTES.sub("'", s)
    s = _THOUSANDS.sub(r"\1\2", s)
    s = _DEC_COMMA.sub(r"\1.\2", s)
    s = _DIGIT_ALPHA.sub(r"\1 \2", s)
    s = _ALPHA_DIGIT.sub(r"\1 \2", s)
    return _SPACES.sub(" ", s).strip()


def cache_key(s: str) -> str:
    """Normalised query used as cache / feedback key: case, spacing, punctuation and stop-word variants
    collapse onto one key (word order is kept)."""
    return " ".join(u.lstrip(CJK_MARK) for u in units(s))


def query_key(s: str) -> str:
    """Order-insensitive query key for feedback aggregation and the result cache: the sorted bag of normalised
    words. Every ranker here is a bag-of-words model, so reordered queries ("shoes women size 8" / "women shoes
    size 8") get the same results and should share feedback; case, spacing and stop words are ignored too."""
    return " ".join(sorted(u.lstrip(CJK_MARK) for u in units(s)))


def units(s: str | None) -> list[str]:
    t = normalize(s)
    if not t:
        return []
    out = [w for w in _WORD.findall(_CJK_RUN.sub(" ", t)) if w not in STOPWORDS]
    out.extend(CJK_MARK + r for r in _CJK_RUN.findall(t))
    return out


def stem(w: str) -> str:
    if len(w) < 4 or not w.endswith("s"):
        return w
    m = _IES.match(w)
    if m:
        return m.group(1) + "y"
    m = _ES.match(w)
    if m:
        return m.group(1)
    m = _S.match(w)
    return m.group(1) if m else w


def _ngrams(s: str, lo: int, hi: int) -> list[str]:
    return [s[i:i + n] for n in range(lo, hi + 1) for i in range(len(s) - n + 1)]


def unit_terms(u: str, jp_ngram: int = 2, stem_plurals: bool = True) -> list[str]:
    """BM25 index terms for one unit."""
    if u.startswith(CJK_MARK):
        r = u[1:]
        return [r] if len(r) < jp_ngram else _ngrams(r, jp_ngram, jp_ngram)
    return [stem(u) if stem_plurals else u]


def unit_features(u: str, char_ngrams: Sequence[int] = (3, 5), jp_char_ngrams: Sequence[int] = (1, 3)
                  ) -> list[tuple[str, float]]:
    """Dense-encoder input features (name, weight) for one unit."""
    if u.startswith(CJK_MARK):
        return [("j:" + g, 1.0) for g in _ngrams(u[1:], jp_char_ngrams[0], jp_char_ngrams[1])]
    feats = [("w:" + stem(u), 1.0)]
    if len(u) >= 3 and not u.isdigit():
        grams = sorted(set(_ngrams(f"<{u}>", char_ngrams[0], char_ngrams[1])))
        wt = 1.0 / len(grams)
        feats.extend(("c:" + g, wt) for g in grams)
    return feats


def terms(s: str, jp_ngram: int = 2, stem_plurals: bool = True) -> list[str]:
    return [t for u in units(s) for t in unit_terms(u, jp_ngram, stem_plurals)]


# --------------------------------------------------------------------------------------
# bulk: texts -> sparse doc x unit counts (parallel), then unit -> term / feature maps
# --------------------------------------------------------------------------------------
def _chunk_units(texts: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    vocab: dict[str, int] = {}
    indptr = [0]
    idx: list[int] = []
    cnt: list[int] = []
    for t in texts:
        local: dict[int, int] = {}
        for u in units(t):
            j = vocab.setdefault(u, len(vocab))
            local[j] = local.get(j, 0) + 1
        idx.extend(local.keys())
        cnt.extend(local.values())
        indptr.append(len(idx))
    return (list(vocab), np.asarray(indptr, np.int64), np.asarray(idx, np.int32), np.asarray(cnt, np.float32))


def doc_units(texts: Iterable[str], n_jobs: int | None = None, chunk: int = 5_000
              ) -> tuple[sp.csr_matrix, list[str]]:
    """Analyse many texts in parallel. Returns (doc x unit count matrix, unit vocabulary)."""
    texts = [t or "" for t in texts]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2))
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)] or [[]]
    vocab: dict[str, int] = {}
    indptrs, indices, data = [], [], []
    offset = 0
    if n_jobs > 1 and len(chunks) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            results = ex.map(_chunk_units, chunks)
            for res in results:
                offset = _merge(res, vocab, indptrs, indices, data, offset)
    else:
        for c in chunks:
            offset = _merge(_chunk_units(c), vocab, indptrs, indices, data, offset)
    indptr = np.concatenate([np.zeros(1, np.int64)] + indptrs)
    m = sp.csr_matrix((np.concatenate(data) if data else np.zeros(0, np.float32),
                       np.concatenate(indices) if indices else np.zeros(0, np.int32), indptr),
                      shape=(len(texts), len(vocab)))
    return m, list(vocab)


def _merge(res, vocab, indptrs, indices, data, offset) -> int:
    local_vocab, indptr, idx, cnt = res
    gid = np.fromiter((vocab.setdefault(u, len(vocab)) for u in local_vocab), np.int32, len(local_vocab))
    indptrs.append(indptr[1:] + offset)
    indices.append(gid[idx] if len(idx) else idx)
    data.append(cnt)
    return offset + len(idx)


def unit_map(unit_vocab: list[str], fn, vocab: dict[str, int] | None = None, grow: bool = True
             ) -> tuple[sp.csr_matrix, dict[str, int]]:
    """unit x term matrix from a per-unit function returning terms or (term, weight) pairs."""
    vocab = {} if vocab is None else vocab
    rows, cols, vals = [], [], []
    for i, u in enumerate(unit_vocab):
        for item in fn(u):
            t, w = (item, 1.0) if isinstance(item, str) else item
            j = vocab.get(t)
            if j is None:
                if not grow:
                    continue
                j = vocab[t] = len(vocab)
            rows.append(i)
            cols.append(j)
            vals.append(w)
    m = sp.csr_matrix((np.asarray(vals, np.float32), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
                      shape=(len(unit_vocab), len(vocab)))
    m.sum_duplicates()
    return m, vocab

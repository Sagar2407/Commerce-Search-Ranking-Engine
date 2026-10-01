"""Query spelling correction against the catalog's own vocabulary (symmetric-delete / SymSpell style).

A single typo costs every ranking method ~0.23 nDCG@10 on the replay stream's typo variants: the misspelled
word matches nothing (BM25) or matches the wrong sub-words (dense). The corrector rewrites a query word only
when the catalog has (almost) never seen it and a much more frequent catalog word is one edit away:

* candidates come from a symmetric single-delete index over each market's BM25 vocabulary (frequent latin
  words of length >= 3), which finds insertions, deletions, substitutions and adjacent swaps;
* a word is left alone if it contains digits, is shorter than 4 characters, is a known brand, or appears in at
  least `max_df` products; the replacement must appear in `min_ratio` times more products;
* ties are broken by product frequency, then by an exact Damerau-Levenshtein check.

Corrections operate on plural-folded terms (the form BM25 indexes), so "headphnes" -> "headphone".
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import analysis as A


def _deletes(w: str) -> set[str]:
    return {w[:i] + w[i + 1:] for i in range(len(w))}


def damerau1(a: str, b: str) -> bool:
    """True when a and b are at most one insertion / deletion / substitution / adjacent swap apart."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        if len(diff) == 1:
            return True
        return len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


class SpellCorrector:
    def __init__(self, df: dict[str, int], index: dict[str, list[str]], max_df: int = 3, min_ratio: float = 20.0,
                 protected: set[str] | None = None):
        self.df, self.index = df, index
        self.max_df, self.min_ratio = max_df, min_ratio
        self.protected = protected or set()

    @classmethod
    def from_bm25(cls, bm25, min_df: int = 5, max_df: int = 3, min_ratio: float = 20.0,
                  protected: set[str] | None = None) -> "SpellCorrector":
        dfs = np.diff(bm25.W.indptr)
        df = {t: int(dfs[j]) for t, j in bm25.vocab.items()}
        index: dict[str, list[str]] = {}
        for t, n in df.items():
            if n >= min_df and len(t) >= 3 and t.isalpha() and all(ord(c) < 0x3000 for c in t):   # latin words only
                for d in _deletes(t) | {t}:
                    index.setdefault(d, []).append(t)
        return cls(df, index, max_df, min_ratio, protected)

    def suggest(self, term: str) -> str | None:
        if len(term) < 4 or not term.isalpha() or term in self.protected:
            return None
        n0 = self.df.get(term, 0)
        if n0 >= self.max_df:
            return None
        cands: set[str] = set()
        for d in _deletes(term) | {term}:
            cands.update(self.index.get(d, ()))
        best, best_n = None, 0
        for c in cands:
            if c == term:
                continue
            n = self.df.get(c, 0)
            if n > best_n and n >= self.min_ratio * max(n0, 1) and damerau1(term, c):
                best, best_n = c, n
        return best

    def correct(self, query: str, mode: str = "expand") -> tuple[str, list[tuple[str, str]]]:
        """Returns (rewritten query, [(original word, suggestion)]).

        mode="expand" keeps the original word and adds the suggestion (a word the catalog has never seen adds
        nothing to BM25, so this recovers typos without dropping rare real words such as small brands);
        mode="replace" substitutes it.
        """
        words = A.normalize(query).split(" ")
        out, changes = [], []
        for w in words:
            stemmed = A.stem(w)
            s = None
            if stemmed not in A.STOPWORDS and self.df.get(stemmed, 0) < self.max_df:
                # plural folding can garble a misspelling ("wireles" -> "wirele"), so try the raw word too
                cands = [c for c in {self.suggest(stemmed), self.suggest(w)} if c and c != stemmed]
                s = max(cands, key=lambda c: self.df.get(c, 0)) if cands else None
            if s is not None and s != stemmed:
                out.extend([w, s] if mode == "expand" else [s])
                changes.append((w, s))
            else:
                out.append(w)
        return (" ".join(out), changes) if changes else (query, [])

    def save(self, p: Path) -> None:
        with open(p, "wb") as f:
            pickle.dump({"df": self.df, "index": self.index, "max_df": self.max_df, "min_ratio": self.min_ratio,
                         "protected": self.protected}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, p: Path) -> "SpellCorrector":
        with open(p, "rb") as f:
            o = pickle.load(f)
        return cls(o["df"], o["index"], o["max_df"], o["min_ratio"], o["protected"])

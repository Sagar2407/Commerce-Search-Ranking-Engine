"""Query understanding: attributes, brands, negation and a query -> department model.

Attribute extraction reuses the phase-1 extractors (`csre.data.text.query_attribute_exprs`) on a
Polars frame, in bulk offline and on a one-row frame online — the same expressions either way, so the
features a ranker was trained on are the features it sees in production.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from ..data import text as T
from . import analysis as A

ATTR_TYPES = ["measures", "dimensions", "sizes", "audience", "compat", "materials", "colors"]
_NEG_TRIGGER = re.compile(r"\b(?:without|w/o|no|non|not|sin|except|excluding|free of)\b\s+((?:[^\W\d_]{2,}\s?){1,2})")
_NEG_FREE = re.compile(r"\b([^\W\d_]{3,})[- ]free\b")


@dataclass
class ParsedQuery:
    text: str
    locale: str
    norm: str
    key: str
    attrs: dict[str, list[str]] = field(default_factory=dict)
    pack_count: int | None = None
    brands: list[str] = field(default_factory=list)
    has_negation: bool = False
    negated_terms: list[str] = field(default_factory=list)
    n_units: int = 0
    has_digit: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "locale": self.locale, "key": self.key, "attrs": self.attrs,
                "pack_count": self.pack_count, "brands": self.brands, "has_negation": self.has_negation,
                "negated_terms": self.negated_terms}


def negated_terms(norm: str) -> list[str]:
    """Terms the shopper does *not* want ("without lactose", "sin azúcar", "bpa-free" is NOT negation)."""
    out = []
    for m in _NEG_TRIGGER.finditer(norm):
        out += [A.stem(w) for w in m.group(1).split() if w not in A.STOPWORDS]
    return list(dict.fromkeys(out))


class BrandMatcher:
    """Word-bounded brand dictionary match (latin locales) / substring match (jp), as in phase 1."""

    def __init__(self, patterns: dict[str, list[str]]):
        self.sets = {loc: set(p) for loc, p in patterns.items()}
        self.max_words = {loc: max((len(b.split()) for b in p), default=1) for loc, p in patterns.items()}
        self.jp = sorted(patterns.get("jp", []), key=len, reverse=True)

    def __call__(self, norm: str, locale: str) -> list[str]:
        if locale == "jp":
            return [b for b in self.jp if b in norm][:5]
        s = self.sets.get(locale)
        if not s:
            return []
        toks = re.sub(r"[^\w\s]", " ", norm).split()
        found = []
        for n in range(1, self.max_words.get(locale, 1) + 1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i + n])
                if g in s:
                    found.append(g)
        return list(dict.fromkeys(found))


class QueryParser:
    """Bulk + cached query parsing.

    Polars is not fork-safe: a forked worker that runs Polars expressions can deadlock on locks held by the
    parent's thread pool. Parallel jobs therefore `prime()` the cache with every query in the parent before
    forking, and workers only read it.
    """

    def __init__(self, brand_patterns: dict[str, list[str]], max_cache: int = 200_000):
        self.brands = BrandMatcher(brand_patterns)
        self.cache: dict[tuple[str, str], ParsedQuery] = {}
        self.max_cache = max_cache

    def parse(self, text: str, locale: str) -> ParsedQuery:
        hit = self.cache.get((text, locale))
        if hit is not None:
            return hit
        p = self.parse_many([text], [locale])[0]
        if len(self.cache) >= self.max_cache:
            self.cache.pop(next(iter(self.cache)))
        self.cache[(text, locale)] = p
        return p

    def prime(self, texts: list[str], locales: list[str], batch: int = 50_000) -> None:
        pairs = list(dict.fromkeys((t, l) for t, l in zip(texts, locales) if (t, l) not in self.cache))
        self.max_cache = max(self.max_cache, len(self.cache) + len(pairs))
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            for key, p in zip(chunk, self.parse_many([t for t, _ in chunk], [l for _, l in chunk])):
                self.cache[key] = p

    def parse_many(self, texts: list[str], locales: list[str]) -> list[ParsedQuery]:
        df = pl.DataFrame({"q": texts, "locale": locales}).with_columns(
            T.for_matching(pl.col("q")).alias("qn")
        ).with_columns(*T.query_attribute_exprs(pl.col("qn")))
        out = []
        for r in df.iter_rows(named=True):
            norm = r["qn"]
            out.append(ParsedQuery(
                text=r["q"], locale=r["locale"], norm=norm, key=A.query_key(r["q"]),
                attrs={t: list(r[f"q_{t}"] or []) for t in ATTR_TYPES},
                pack_count=r["q_pack_count"],
                brands=self.brands(norm, r["locale"]),
                has_negation=bool(r["q_has_negation"]),
                negated_terms=negated_terms(norm) if r["q_has_negation"] else [],
                n_units=len(A.units(r["q"])),
                has_digit=bool(r["q_has_digit"]),
            ))
        return out


class QueryCategoryModel:
    """P(department | query), trained on train-split queries labelled with their Exact products' departments.

    Used as a ranking feature (does this product's department fit the query?) and in the storefront to
    detect vague queries (high entropy -> show department chips).
    """

    def __init__(self, vocab: dict[str, int], coef: np.ndarray, intercept: np.ndarray, classes: list[str]):
        self.vocab, self.coef, self.intercept, self.classes = vocab, coef, intercept, [str(c) for c in classes]
        self.class_index = {c: i for i, c in enumerate(classes)}

    @staticmethod
    def _feats(text: str) -> list[str]:
        us = A.units(text)
        feats = [A.unit_terms(u)[0] if not u.startswith(A.CJK_MARK) else u for u in us]
        feats += [a + "_" + b for a, b in zip(feats, feats[1:])]
        for u in us:
            if u.startswith(A.CJK_MARK):
                feats += A._ngrams(u[1:], 1, 2)
            elif len(u) >= 4:
                feats += A._ngrams(f"<{u}>", 3, 3)
        return feats

    @classmethod
    def fit(cls, texts: list[str], labels: list[str], weights: np.ndarray | None = None, min_df: int = 2,
            seed: int = 0) -> "QueryCategoryModel":
        import scipy.sparse as sp  # noqa: PLC0415
        from sklearn.linear_model import SGDClassifier  # noqa: PLC0415
        counts: dict[str, int] = {}
        rows = [cls._feats(t) for t in texts]
        for r in rows:
            for f in set(r):
                counts[f] = counts.get(f, 0) + 1
        vocab = {f: i for i, f in enumerate(sorted(f for f, c in counts.items() if c >= min_df))}
        X = cls._matrix(rows, vocab)
        clf = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=20, tol=1e-4, random_state=seed)
        clf.fit(X, np.asarray(labels), sample_weight=weights)
        return cls(vocab, clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32), [str(c) for c in clf.classes_])

    @staticmethod
    def _matrix(rows: list[list[str]], vocab: dict[str, int]):
        import scipy.sparse as sp  # noqa: PLC0415
        indptr, idx = [0], []
        for r in rows:
            ids = sorted({vocab[f] for f in r if f in vocab})
            idx += ids
            indptr.append(len(idx))
        data = np.ones(len(idx), np.float32)
        X = sp.csr_matrix((data, np.asarray(idx, np.int64), np.asarray(indptr)), shape=(len(rows), len(vocab)))
        n = np.sqrt(np.asarray(X.sum(axis=1)).ravel())
        n[n == 0] = 1
        return sp.diags(1 / n) @ X

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._matrix([self._feats(t) for t in texts], self.vocab)
        z = np.asarray(X @ self.coef.T) + self.intercept
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vocab": self.vocab, "coef": self.coef, "intercept": self.intercept, "classes": self.classes},
                    d / "query_category.joblib", compress=3)

    @classmethod
    def load(cls, d: Path) -> "QueryCategoryModel":
        o = joblib.load(d / "query_category.joblib")
        return cls(o["vocab"], o["coef"], o["intercept"], o["classes"])


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=-1)


def load_brand_patterns(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text()) if path.exists() else {}

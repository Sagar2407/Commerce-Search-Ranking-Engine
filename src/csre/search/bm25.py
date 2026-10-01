"""BM25 with precomputed impact weights.

The index stores, per term, the BM25 contribution of every document that contains it
(``idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``) in a CSC matrix, so a query is a
sum of a few posting lists: no per-query scoring arithmetic beyond one multiply-add per posting.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import analysis as A


class BM25Index:
    def __init__(self, W: sp.csc_matrix, vocab: dict[str, int], idf: np.ndarray, k1: float, b: float,
                 jp_ngram: int = 2, stem_plurals: bool = True):
        self.W = W
        self.vocab = vocab
        self.idf = idf
        self.k1, self.b = k1, b
        self.jp_ngram, self.stem_plurals = jp_ngram, stem_plurals
        self.n_docs = W.shape[0]
        self._indptr, self._indices, self._data = W.indptr, W.indices, W.data

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, texts: list[str], k1: float = 1.2, b: float = 0.75, jp_ngram: int = 2,
              stem_plurals: bool = True, n_jobs: int | None = None,
              unit_matrix: tuple[sp.csr_matrix, list[str]] | None = None) -> "BM25Index":
        du, uv = unit_matrix or A.doc_units(texts, n_jobs=n_jobs)
        um, vocab = A.unit_map(uv, lambda u: A.unit_terms(u, jp_ngram, stem_plurals))
        tf = (du @ um).tocsr()                      # doc x term counts
        tf.sum_duplicates()
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel().astype(np.float32)
        avgdl = float(dl.mean()) if n else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        norm = k1 * (1.0 - b + b * dl[rows] / avgdl)
        w = idf[tf.indices] * tf.data * (k1 + 1.0) / (tf.data + norm)
        W = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape).tocsc()
        W.sort_indices()
        return cls(W, vocab, idf, k1, b, jp_ngram, stem_plurals)

    # ------------------------------------------------------------------ query
    def query_terms(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """(term ids, query term frequency) for in-vocabulary terms."""
        counts: dict[int, int] = {}
        for t in A.terms(query, self.jp_ngram, self.stem_plurals):
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        return np.fromiter(counts.keys(), np.int64, len(counts)), np.fromiter(counts.values(), np.float32, len(counts))

    def max_score(self, query: str) -> float:
        """Candidate-independent upper bound used to normalise scores across queries."""
        tids, qtf = self.query_terms(query)
        return float((self.idf[tids] * (self.k1 + 1.0) * qtf).sum()) if len(tids) else 0.0

    def scores(self, query: str | tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Dense score vector over all docs + ids of the docs that matched at least one term."""
        tids, qtf = self.query_terms(query) if isinstance(query, str) else query
        s = np.zeros(self.n_docs, np.float32)
        touched = []
        for t, q in zip(tids, qtf):
            lo, hi = self._indptr[t], self._indptr[t + 1]
            rows = self._indices[lo:hi]
            s[rows] += self._data[lo:hi] * q            # rows are unique within a posting list
            touched.append(rows)
        hit = np.unique(np.concatenate(touched)) if touched else np.zeros(0, np.int32)
        return s, hit

    def search(self, query: str, k: int = 100, allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        s, hit = self.scores(query)
        if allowed is not None:
            hit = hit[allowed[hit]]
        if len(hit) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        hs = s[hit]
        if len(hit) > k:
            top = np.argpartition(-hs, k - 1)[:k]
            hit, hs = hit[top], hs[top]
        order = np.lexsort((hit, -hs))               # score desc, doc index asc (deterministic ties)
        return hit[order].astype(np.int64), hs[order]

    def score_docs(self, query: str, docs: np.ndarray) -> np.ndarray:
        """BM25 of `query` for the given doc indices (feature / rerank path)."""
        tids, qtf = self.query_terms(query)
        if len(tids) == 0 or len(docs) == 0:
            return np.zeros(len(docs), np.float32)
        sub = self.W[:, tids]                         # csc: docs x qterms
        return np.asarray((sub[docs] @ qtf)).ravel().astype(np.float32)

    def term_doc_weights(self, tids: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """(len(docs) x len(tids)) BM25 weights, 0 where a doc lacks the term (sorted-postings lookup)."""
        out = np.zeros((len(docs), len(tids)), np.float32)
        for j, t in enumerate(tids):
            lo, hi = self._indptr[t], self._indptr[t + 1]
            if hi == lo:
                continue
            post = self._indices[lo:hi]
            pos = np.minimum(np.searchsorted(post, docs), hi - lo - 1)
            hit = post[pos] == docs
            out[hit, j] = self._data[lo + pos[hit]]
        return out

    def term_ids(self, words: list[str]) -> np.ndarray:
        return np.asarray([self.vocab[w] for w in words if w in self.vocab], np.int64)

    def matched_terms(self, query: str, doc: int) -> list[str]:
        """Query terms present in a document (for explanations)."""
        inv = {}
        out = []
        for t in dict.fromkeys(A.terms(query, self.jp_ngram, self.stem_plurals)):
            j = self.vocab.get(t)
            if j is None:
                continue
            lo, hi = self._indptr[j], self._indptr[j + 1]
            pos = np.searchsorted(self._indices[lo:hi], doc)
            if pos < hi - lo and self._indices[lo + pos] == doc:
                out.append(t)
        del inv
        return out

    # ------------------------------------------------------------------ io
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / "weights.npz", self.W, compressed=False)
        np.save(d / "idf.npy", self.idf)
        with open(d / "vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        (d / "meta.json").write_text(json.dumps({
            "k1": self.k1, "b": self.b, "jp_ngram": self.jp_ngram, "stem_plurals": self.stem_plurals,
            "n_docs": int(self.n_docs), "n_terms": len(self.vocab), "nnz": int(self.W.nnz)}))

    @classmethod
    def load(cls, d: Path) -> "BM25Index":
        meta = json.loads((d / "meta.json").read_text())
        W = sp.load_npz(d / "weights.npz").tocsc()
        with open(d / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return cls(W, vocab, np.load(d / "idf.npy"), meta["k1"], meta["b"], meta["jp_ngram"], meta["stem_plurals"])

    def nbytes(self) -> int:
        return int(self.W.data.nbytes + self.W.indices.nbytes + self.W.indptr.nbytes + self.idf.nbytes)

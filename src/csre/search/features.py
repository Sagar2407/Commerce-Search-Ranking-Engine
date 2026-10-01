"""Ranking features for (query, product) pairs — one implementation for training and serving.

All lookups are vectorised (sorted BM25 postings, sparse attribute-value matrices, numpy arrays), so the
same code computes ~2M training rows offline and ~200 candidate rows per request online.

Feature groups
--------------
query       length, digits, brand / attribute counts, department entropy (vagueness), BM25 upper bound
lexical     BM25 (retrieval text, title), normalised by the query's upper bound; query-term coverage
semantic    dense cosine similarity (in-domain two-tower encoder)
attributes  for measures / dimensions / sizes / audience / compatibility / materials / colours / pack count /
            brand: does the query state it, does the product match it, does the product *contradict* it
negation    product title contains a term the query excludes ("without lactose")
department  P(product department | query) from the query department model
product     description richness / sparsity, title length, department confidence
commerce    price, stars, rating count, stock, popularity (real ESCI-S where available, simulated otherwise)
feedback    (feedback-aware ranker only) IPS-debiased CTR, add-to-cart, purchase and thumbs rates for this
            (query, product), smoothed; query traffic volume
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

from .query import ATTR_TYPES, ParsedQuery, entropy_bits

LOCALE_CODE = {"us": 0, "es": 1, "jp": 2}

QUERY_FEATURES = ["q_n_units", "q_has_digit", "q_n_brands", "q_n_attrs", "q_has_negation", "q_cat_entropy",
                  "q_cat_top", "q_bm25_max", "locale_code"]
LEXICAL_FEATURES = ["bm25_text", "bm25_text_norm", "bm25_title", "bm25_title_norm", "cov_text", "cov_title",
                    "all_terms_title"]
SEMANTIC_FEATURES = ["dense_cos"]
ATTR_FEATURES = [f"{a}_{k}" for a in [*ATTR_TYPES, "pack", "brand"] for k in ("q", "match", "conflict")] + \
                ["neg_violation"]
DOC_FEATURES = ["is_sparse", "log_richness", "title_len", "title_only", "cat_known", "cat_conf", "p_cat"]
COMMERCE_FEATURES = ["log_price", "stars", "log_ratings", "in_stock", "popularity_z"]
FEEDBACK_FEATURES = ["fb_log_impr", "fb_ctr_ips", "fb_cart_rate", "fb_buy_rate", "fb_thumbs", "fb_q_log_impr"]

CONTENT_FEATURES = (QUERY_FEATURES + LEXICAL_FEATURES + SEMANTIC_FEATURES + ATTR_FEATURES + DOC_FEATURES
                    + COMMERCE_FEATURES)
ALL_FEATURES = CONTENT_FEATURES + FEEDBACK_FEATURES

FEATURE_GROUPS = {
    "query": QUERY_FEATURES, "lexical": LEXICAL_FEATURES, "semantic": SEMANTIC_FEATURES,
    "attributes": ATTR_FEATURES, "product": DOC_FEATURES, "commerce": COMMERCE_FEATURES,
    "feedback": FEEDBACK_FEATURES,
}


# ======================================================================================
# per-locale product table
# ======================================================================================
class DocTable:
    """Numeric / attribute arrays for one locale's products, aligned with the index row order."""

    def __init__(self, cat: pl.DataFrame, commerce: pl.DataFrame | None, departments: list[str]):
        n = cat.height
        self.n = n
        self.attr_vocab: dict[str, dict[str, int]] = {}
        self.attr: dict[str, sp.csr_matrix] = {}
        for a in ATTR_TYPES:
            col = f"attr_{a}"
            ex = cat.select(pl.int_range(0, n, dtype=pl.Int64).alias("row"), pl.col(col)).explode(col, empty_as_null=True).drop_nulls()
            vals = ex[col].unique().sort().to_list()
            voc = {v: i for i, v in enumerate(vals)}
            self.attr_vocab[a] = voc
            ids = ex[col].replace_strict(voc, return_dtype=pl.Int64).to_numpy() if ex.height else np.zeros(0, np.int64)
            m = sp.csr_matrix((np.ones(len(ids), np.float32), (ex["row"].to_numpy(), ids)), shape=(n, max(len(voc), 1)))
            m.sum_duplicates()
            m.data[:] = 1.0
            self.attr[a] = m
        self.has_attr = {a: np.diff(m.indptr) > 0 for a, m in self.attr.items()}
        self.pack = cat["attr_pack_count"].fill_null(-1).to_numpy().astype(np.int64)
        brands = cat["brand_norm"].fill_null("").to_list()
        self.brand_vocab = {b: i for i, b in enumerate(sorted(set(brands) - {""}))}
        self.brand = np.asarray([self.brand_vocab.get(b, -1) for b in brands], np.int64)
        if not departments:   # no query-department model: index the catalog's own departments (filters still work)
            departments = sorted(c for c in cat["category"].drop_nulls().unique().to_list() if c != "Unknown")
        self.dept_index = {d: i for i, d in enumerate(departments)}
        self.category = np.asarray([self.dept_index.get(c, -1) for c in cat["category"].fill_null("Unknown").to_list()],
                                   np.int64)
        self.cat_conf = cat["category_conf"].fill_null(0).to_numpy().astype(np.float32)
        self.is_sparse = cat["is_sparse"].fill_null(False).to_numpy().astype(np.float32)
        self.title_only = cat["is_title_only"].fill_null(False).to_numpy().astype(np.float32)
        self.log_richness = np.log1p(cat["richness_chars"].fill_null(0).to_numpy().astype(np.float32))
        self.title_len = cat["n_title_chars"].fill_null(0).to_numpy().astype(np.float32)
        if commerce is not None:
            c = cat.select("doc_id").join(commerce, on="doc_id", how="left")
            self.log_price = np.log1p(c["price"].fill_null(np.nan).to_numpy().astype(np.float32))
            self.stars = c["stars"].fill_null(np.nan).to_numpy().astype(np.float32)
            self.log_ratings = np.log1p(c["n_ratings"].fill_null(0).to_numpy().astype(np.float32))
            self.in_stock = c["in_stock"].fill_null(True).to_numpy().astype(np.float32)
            self.popularity_z = c["popularity_z"].fill_null(0).to_numpy().astype(np.float32)
        else:
            nan = np.full(n, np.nan, np.float32)
            self.log_price = self.stars = self.log_ratings = self.in_stock = self.popularity_z = nan

    def query_attr_matrix(self, a: str, values: list[list[str]]) -> sp.csr_matrix:
        voc = self.attr_vocab[a]
        rows, cols = [], []
        for i, vs in enumerate(values):
            for v in vs:
                j = voc.get(v)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(values), self.attr[a].shape[1]))


# ======================================================================================
# feedback store
# ======================================================================================
class FeedbackStore:
    """(locale, query key, product row) -> aggregated feedback, with live increments from the storefront."""

    COLS = ["impressions", "clicks", "ips_clicks", "ips_impressions", "carts", "purchases", "thumbs_up", "thumbs_down"]

    def __init__(self):
        self.base: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.q_impr: dict[tuple[str, str], float] = {}
        self.live: dict[tuple[str, str, int], np.ndarray] = {}
        self.ctr_prior = 0.05

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> "FeedbackStore":
        """df: locale, key, row (product row index), and COLS."""
        st = cls()
        df = df.sort(["locale", "key", "row"])
        tot = df["impressions"].sum()
        st.ctr_prior = float(df["clicks"].sum() / tot) if tot else 0.05
        arr = df.select(cls.COLS).to_numpy().astype(np.float32)
        rows = df["row"].to_numpy().astype(np.int64)
        keys = list(zip(df["locale"].to_list(), df["key"].to_list()))
        starts = np.flatnonzero(np.r_[True, [keys[i] != keys[i - 1] for i in range(1, len(keys))]]) if keys else []
        bounds = list(starts) + [len(keys)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            k = keys[s]
            st.base[k] = (rows[s:e], arr[s:e])
            st.q_impr[k] = float(arr[s:e, 0].sum())
        return st

    def add(self, locale: str, key: str, row: int, event: str, propensity: float = 1.0) -> None:
        v = self.live.setdefault((locale, key, row), np.zeros(len(self.COLS), np.float32))
        w = 1.0 / max(propensity, 0.05)
        if event == "impression":
            v[0] += 1
            v[3] += w
            self.q_impr[(locale, key)] = self.q_impr.get((locale, key), 0.0) + 1
        elif event == "click":
            v[1] += 1
            v[2] += w
        elif event == "cart":
            v[4] += 1
        elif event == "purchase":
            v[5] += 1
        elif event == "thumbs_up":
            v[6] += 1
        elif event == "thumbs_down":
            v[7] += 1

    def lookup(self, locale: str, key: str, docs: np.ndarray) -> tuple[np.ndarray, float]:
        out = np.zeros((len(docs), len(self.COLS)), np.float32)
        b = self.base.get((locale, key))
        if b is not None:
            rows, arr = b
            pos = np.minimum(np.searchsorted(rows, docs), len(rows) - 1)
            hit = rows[pos] == docs
            out[hit] = arr[pos[hit]]
        if self.live:
            for i, d in enumerate(docs):
                v = self.live.get((locale, key, int(d)))
                if v is not None:
                    out[i] += v
        return out, self.q_impr.get((locale, key), 0.0)

    def features(self, raw: np.ndarray, q_impr: float) -> np.ndarray:
        imp, clk, ipc, ipi, cart, buy, up, down = raw.T
        a = 20.0
        return np.stack([
            np.log1p(imp),
            (ipc + a * self.ctr_prior) / (ipi + a),
            (cart + 0.5) / (clk + 10.0),
            (buy + 0.2) / (clk + 10.0),
            (up - down) / (up + down + 2.0),
            np.full(len(imp), np.log1p(q_impr), np.float32),
        ], axis=1).astype(np.float32)


# ======================================================================================
# builder
# ======================================================================================
@dataclass
class QueryContext:
    parsed: ParsedQuery
    qvec: np.ndarray | None          # dense query embedding
    cat_proba: np.ndarray | None     # P(department | query)


class FeatureBuilder:
    def __init__(self, bm25_text, bm25_title, vectors, docs: DocTable, feedback: FeedbackStore | None = None):
        self.bt, self.bti, self.vec, self.docs, self.fb = bm25_text, bm25_title, vectors, docs, feedback

    def build(self, ctx: QueryContext, docs: np.ndarray, with_feedback: bool = False) -> np.ndarray:
        """Features for one query and candidate product rows -> (len(docs) x n_features)."""
        q = ctx.parsed
        D = self.docs
        n = len(docs)
        f: dict[str, np.ndarray | float] = {}

        # ---- query
        f["q_n_units"] = q.n_units
        f["q_has_digit"] = float(q.has_digit)
        f["q_n_brands"] = len(q.brands)
        f["q_n_attrs"] = sum(len(v) > 0 for v in q.attrs.values()) + (q.pack_count is not None)
        f["q_has_negation"] = float(q.has_negation)
        if ctx.cat_proba is not None:
            f["q_cat_entropy"] = float(entropy_bits(ctx.cat_proba))
            f["q_cat_top"] = float(ctx.cat_proba.max())
        else:
            f["q_cat_entropy"] = f["q_cat_top"] = np.nan
        f["locale_code"] = LOCALE_CODE.get(q.locale, -1)

        # ---- lexical
        tids, qtf = self.bt.query_terms(q.text)
        Wt = self.bt.term_doc_weights(tids, docs)
        qmax = float((self.bt.idf[tids] * (self.bt.k1 + 1.0) * qtf).sum()) if len(tids) else 0.0
        f["q_bm25_max"] = qmax
        f["bm25_text"] = Wt @ qtf if len(tids) else np.zeros(n, np.float32)
        f["bm25_text_norm"] = f["bm25_text"] / qmax if qmax > 0 else np.zeros(n, np.float32)
        f["cov_text"] = (Wt > 0).mean(axis=1) if len(tids) else np.zeros(n, np.float32)
        tids2, qtf2 = self.bti.query_terms(q.text)
        Wti = self.bti.term_doc_weights(tids2, docs)
        qmax2 = float((self.bti.idf[tids2] * (self.bti.k1 + 1.0) * qtf2).sum()) if len(tids2) else 0.0
        f["bm25_title"] = Wti @ qtf2 if len(tids2) else np.zeros(n, np.float32)
        f["bm25_title_norm"] = f["bm25_title"] / qmax2 if qmax2 > 0 else np.zeros(n, np.float32)
        f["cov_title"] = (Wti > 0).mean(axis=1) if len(tids2) else np.zeros(n, np.float32)
        f["all_terms_title"] = (f["cov_title"] >= 1.0).astype(np.float32) if len(tids2) else np.zeros(n, np.float32)

        # ---- semantic
        f["dense_cos"] = self.vec.score_docs(ctx.qvec, docs) if (self.vec is not None and ctx.qvec is not None) \
            else np.full(n, np.nan, np.float32)

        # ---- attributes
        for a in ATTR_TYPES:
            vals = q.attrs.get(a) or []
            has_q = len(vals) > 0
            f[f"{a}_q"] = float(has_q)
            if has_q:
                Qm = D.query_attr_matrix(a, [vals])
                hit = np.asarray((D.attr[a][docs] @ Qm.T).todense()).ravel() > 0
                dh = D.has_attr[a][docs]
                f[f"{a}_match"] = hit.astype(np.float32)
                f[f"{a}_conflict"] = (dh & ~hit).astype(np.float32)
            else:
                f[f"{a}_match"] = f[f"{a}_conflict"] = np.zeros(n, np.float32)
        pk = D.pack[docs]
        f["pack_q"] = float(q.pack_count is not None)
        if q.pack_count is not None:
            f["pack_match"] = (pk == q.pack_count).astype(np.float32)
            f["pack_conflict"] = ((pk > 0) & (pk != q.pack_count)).astype(np.float32)
        else:
            f["pack_match"] = f["pack_conflict"] = np.zeros(n, np.float32)
        f["brand_q"] = float(len(q.brands) > 0)
        if q.brands:
            qb = np.asarray([D.brand_vocab.get(b, -2) for b in q.brands], np.int64)
            db = D.brand[docs]
            hit = np.isin(db, qb)
            f["brand_match"] = hit.astype(np.float32)
            f["brand_conflict"] = ((db >= 0) & ~hit).astype(np.float32)
        else:
            f["brand_match"] = f["brand_conflict"] = np.zeros(n, np.float32)
        neg = self.bti.term_ids(q.negated_terms) if q.negated_terms else np.zeros(0, np.int64)
        f["neg_violation"] = (self.bti.term_doc_weights(neg, docs) > 0).any(axis=1).astype(np.float32) if len(neg) \
            else np.zeros(n, np.float32)

        # ---- product
        f["is_sparse"] = D.is_sparse[docs]
        f["log_richness"] = D.log_richness[docs]
        f["title_len"] = D.title_len[docs]
        f["title_only"] = D.title_only[docs]
        dc = D.category[docs]
        f["cat_known"] = (dc >= 0).astype(np.float32)
        f["cat_conf"] = D.cat_conf[docs]
        if ctx.cat_proba is not None:
            p = np.full(n, np.nan, np.float32)
            ok = (dc >= 0) & (dc < len(ctx.cat_proba))
            p[ok] = ctx.cat_proba[dc[ok]]
            f["p_cat"] = p
        else:
            f["p_cat"] = np.full(n, np.nan, np.float32)

        # ---- commerce
        f["log_price"] = D.log_price[docs]
        f["stars"] = D.stars[docs]
        f["log_ratings"] = D.log_ratings[docs]
        f["in_stock"] = D.in_stock[docs]
        f["popularity_z"] = D.popularity_z[docs]

        names = ALL_FEATURES if with_feedback else CONTENT_FEATURES
        X = np.empty((n, len(names)), np.float32)
        for j, name in enumerate(CONTENT_FEATURES):
            X[:, j] = f[name]
        if with_feedback:
            if self.fb is not None:
                raw, qi = self.fb.lookup(q.locale, q.key, docs)
                X[:, len(CONTENT_FEATURES):] = self.fb.features(raw, qi)
            else:
                X[:, len(CONTENT_FEATURES):] = np.nan
        return X

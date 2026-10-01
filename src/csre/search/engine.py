"""The search engine: per-locale indexes + query understanding + six interchangeable ranking methods.

    bm25        keyword retrieval (BM25 over title + brand + colour + bullets + description)
    dense       semantic retrieval (in-domain two-tower encoder, exact or HNSW search)
    hybrid_rrf  reciprocal-rank fusion of the bm25 and dense candidate lists
    hybrid      convex fusion of z-normalised bm25 and dense scores (alpha tuned on dev)
    ltr         LambdaMART reranker over the fused candidates, content features only
    ltr_fb      LambdaMART reranker that also uses (simulated) shopper feedback features

Every response carries per-stage timings, the model versions used, and — when the latency budget is
exhausted or a component is unavailable — which cheaper method actually served it and why.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..utils import get_logger
from . import analysis as A
from .bm25 import BM25Index
from .corpus import Dataset
from .dense import VectorIndex, encoder_text, load_encoder
from .features import CONTENT_FEATURES, FeatureBuilder, FeedbackStore, QueryContext, DocTable
from .query import QueryCategoryModel, QueryParser
from .registry import ModelRegistry

log = get_logger("csre.engine")

METHODS = ["bm25", "dense", "hybrid_rrf", "hybrid", "ltr", "ltr_fb"]
METHOD_LABELS = {
    "bm25": "Keyword (BM25)", "dense": "Semantic (dense)", "hybrid_rrf": "Hybrid (RRF)",
    "hybrid": "Hybrid (score fusion)", "ltr": "Reranker (LambdaMART)", "ltr_fb": "Reranker + feedback",
}
# cheaper method each one degrades to when its own stages are unavailable or over budget
FALLBACK = {"ltr_fb": "ltr", "ltr": "hybrid", "hybrid": "bm25", "hybrid_rrf": "bm25", "dense": "bm25", "bm25": None}

DISPLAY_COLS = ["doc_id", "title", "brand", "color", "category", "attr_measures", "attr_dimensions",
                "attr_pack_count", "attr_sizes", "attr_audience", "attr_compat", "attr_materials", "attr_colors",
                "is_sparse"]


# ======================================================================================
# per-locale index
# ======================================================================================
@dataclass
class LocaleIndex:
    locale: str
    doc_ids: np.ndarray
    bm25_text: BM25Index
    bm25_title: BM25Index
    vectors: VectorIndex | None
    docs: DocTable
    display: pl.DataFrame
    _pos: dict | None = None

    @property
    def n(self) -> int:
        return len(self.doc_ids)

    def rows(self, doc_ids: list[str]) -> np.ndarray:
        if self._pos is None:
            self._pos = {d: i for i, d in enumerate(self.doc_ids.tolist())}
        return np.asarray([self._pos.get(d, -1) for d in doc_ids], np.int64)


def build_locale_index(cfg: Config, ds: Dataset, locale: str, encoder=None) -> dict:
    """Build + save BM25 (text, title) and dense vectors for one locale. Returns build stats."""
    d = ds.index_dir() / locale
    d.mkdir(parents=True, exist_ok=True)
    B = cfg.get("search.bm25")
    stats: dict = {"locale": locale}
    t = time.time()
    cat = ds.catalog(locale, ["doc_id", "title", "doc_text", "n_title_chars"])
    stats["n_docs"] = cat.height
    pl.DataFrame({"doc_id": cat["doc_id"]}).write_parquet(d / "doc_ids.parquet")
    stats["load_s"] = round(time.time() - t, 1)

    t = time.time()
    du = A.doc_units(cat["doc_text"].to_list())
    bt = BM25Index.build([], B["k1"], B["b"], B["jp_ngram"], B["stem_plurals"], unit_matrix=du)
    bt.save(d / "bm25_text")
    stats["bm25_text"] = {"seconds": round(time.time() - t, 1), "terms": len(bt.vocab), "nnz": int(bt.W.nnz),
                          "bytes": bt.nbytes()}
    del du, bt
    t = time.time()
    bti = BM25Index.build(cat["title"].to_list(), B["k1"], B["b"], B["jp_ngram"], B["stem_plurals"])
    bti.save(d / "bm25_title")
    stats["bm25_title"] = {"seconds": round(time.time() - t, 1), "terms": len(bti.vocab), "nnz": int(bti.W.nnz),
                           "bytes": bti.nbytes()}
    del bti

    if encoder is not None:
        t = time.time()
        emb = encoder.encode(encoder_text(cat, int(cfg.get("search.dense.doc_bullet_chars", 200))))
        te = time.time() - t
        an = cfg.get("search.dense.ann")
        vi = VectorIndex.build(emb, an["kind"], an["hnsw_m"], an["ef_construction"], an["ef_search"],
                               an["min_docs_for_ann"])
        vi.save(d / "dense")
        stats["dense"] = {"encode_seconds": round(te, 1), "index_seconds": round(time.time() - t - te, 1),
                          "dim": int(emb.shape[1]), "ann": vi.ann is not None, "bytes": vi.nbytes()}
    (d / "build.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_locale_index(cfg: Config, ds: Dataset, locale: str, departments: list[str], load_dense: bool = True
                      ) -> LocaleIndex:
    d = ds.index_dir() / locale
    ids = pl.read_parquet(d / "doc_ids.parquet")["doc_id"]
    cat = ds.catalog(locale)
    if not cat["doc_id"].equals(ids):
        raise RuntimeError(f"index for {locale} is stale (catalog changed): rebuild with `csre index`")
    vec = None
    if load_dense and (d / "dense" / "emb.npy").exists():
        vec = VectorIndex.load(d / "dense", ef_search=int(cfg.get("search.dense.ann.ef_search", 128)))
    commerce = ds.commerce()
    docs = DocTable(cat, commerce, departments)
    disp = cat.select([c for c in DISPLAY_COLS if c in cat.columns])
    if commerce is not None:
        disp = disp.join(commerce.select("doc_id", "price", "currency", "stars", "n_ratings", "in_stock",
                                         "commerce_source"), on="doc_id", how="left")
    meta = ds.product_meta()
    if meta is not None:
        disp = disp.join(meta.select("doc_id", "image_url", "category_leaf"), on="doc_id", how="left")
    return LocaleIndex(locale, ids.to_numpy(), BM25Index.load(d / "bm25_text"), BM25Index.load(d / "bm25_title"),
                       vec, docs, disp)


# ======================================================================================
# responses
# ======================================================================================
@dataclass
class SearchResponse:
    query: str
    locale: str
    method: str
    served_by: str
    rows: np.ndarray
    scores: np.ndarray
    timings_ms: dict = field(default_factory=dict)
    fallback_reason: str | None = None
    versions: dict = field(default_factory=dict)
    candidates: int = 0
    context: QueryContext | None = None
    features: np.ndarray | None = None      # for ltr methods: features of the returned rows
    feature_names: list[str] | None = None
    extra: dict = field(default_factory=dict)
    rewritten: str | None = None             # query actually searched after spelling correction
    corrections: list = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 3)


def _z(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


# ======================================================================================
# engine
# ======================================================================================
class Engine:
    def __init__(self, cfg: Config, corpus: str | None = None, locales: list[str] | None = None,
                 load_dense: bool = True, load_models: bool = True, feedback: bool = True):
        self.cfg = cfg
        self.ds = Dataset(cfg, corpus)
        self.registry = ModelRegistry(cfg)
        self.versions: dict[str, str] = {}
        t = time.time()
        self.parser = QueryParser(self.ds.brand_patterns())
        self.encoder = None
        if load_dense and self.registry.path("dense_encoder"):
            self.encoder = load_encoder(self.registry.path("dense_encoder"))
            self.versions["dense_encoder"] = self.registry.resolve("dense_encoder")
        self.qcat = None
        if self.registry.path("query_category"):
            self.qcat = QueryCategoryModel.load(self.registry.path("query_category"))
            self.versions["query_category"] = self.registry.resolve("query_category")
        departments = self.qcat.classes if self.qcat else []
        self.locales = locales or self.ds.locales()
        self.idx: dict[str, LocaleIndex] = {}
        for loc in self.locales:
            if not (self.ds.index_dir() / loc / "doc_ids.parquet").exists():
                log.warning("no index for %s/%s — run `csre index`", self.ds.name, loc)
                continue
            self.idx[loc] = load_locale_index(cfg, self.ds, loc, departments, load_dense=self.encoder is not None)
        self._check_index_encoder()
        self.models: dict[str, object] = {}
        if load_models:
            import lightgbm as lgb  # noqa: PLC0415
            for name in ("ltr", "ltr_fb", "esci_class"):
                p = self.registry.path(name)
                if p and (p / "model.txt").exists():
                    self.models[name] = lgb.Booster(model_file=str(p / "model.txt"))
                    self.versions[name] = self.registry.resolve(name)
                    self.model_meta = getattr(self, "model_meta", {})
                    self.model_meta[name] = json.loads((p / "model.json").read_text())
        SP = cfg.get("search.spell", {}) or {}
        self.spell_mode = SP.get("mode", "expand") if SP.get("enabled", False) else None
        self.spellers = {}
        if self.spell_mode:
            from .spell import SpellCorrector  # noqa: PLC0415
            bp = self.ds.brand_patterns()
            for loc, ix in self.idx.items():
                self.spellers[loc] = SpellCorrector.from_bm25(
                    ix.bm25_text, int(SP.get("min_df", 5)), int(SP.get("max_df", 3)), float(SP.get("min_ratio", 20)),
                    protected={w for b in bp.get(loc, []) for w in b.split()})
        hyb = self.registry.meta("hybrid")
        self.alpha = float(hyb["params"]["alpha"]) if hyb else float(cfg.get("search.hybrid.alpha", 0.5))
        if hyb:
            self.versions["hybrid"] = self.registry.resolve("hybrid")
        self.feedback = self._load_feedback() if feedback else None
        self.builders = {loc: FeatureBuilder(ix.bm25_text, ix.bm25_title, ix.vectors, ix.docs, self.feedback)
                         for loc, ix in self.idx.items()}
        log.info("engine ready (%s, locales=%s, versions=%s) in %.1fs", self.ds.name, list(self.idx), self.versions,
                 time.time() - t)

    # ---------------------------------------------------------------- setup helpers
    def _check_index_encoder(self) -> None:
        mf = self.ds.index_dir() / "manifest.json"
        if not mf.exists() or self.encoder is None:
            return
        built_with = json.loads(mf.read_text()).get("dense_encoder")
        if built_with and built_with != self.versions.get("dense_encoder"):
            log.warning("dense index built with encoder %s but production is %s: dense disabled until `csre index`",
                        built_with, self.versions.get("dense_encoder"))
            self.encoder = None
            for ix in self.idx.values():
                ix.vectors = None

    def _load_feedback(self) -> FeedbackStore | None:
        qds = self.ds.query_doc_stats()
        if qds is None or not self.idx:
            return None
        q = self.ds.queries().select("query_id", "query", "locale")
        keys = q.with_columns(pl.col("query").map_elements(A.query_key, return_dtype=pl.Utf8).alias("key"))
        df = qds.join(keys.select("query_id", "locale", "key"), on="query_id")
        parts = []
        for loc, ix in self.idx.items():
            m = df.filter(pl.col("locale") == loc)
            rows = ix.rows(m["doc_id"].to_list())
            parts.append(m.with_columns(pl.Series("row", rows)).filter(pl.col("row") >= 0))
        if not parts:
            return None
        df = pl.concat(parts).group_by("locale", "key", "row").agg(
            [pl.col(c).sum() for c in FeedbackStore.COLS])
        return FeedbackStore.from_frame(df)

    def available_methods(self) -> list[str]:
        out = ["bm25"]
        if self.encoder is not None:
            out += ["dense", "hybrid_rrf", "hybrid"]
            if "ltr" in self.models:
                out.append("ltr")
            if "ltr_fb" in self.models and self.feedback is not None:
                out.append("ltr_fb")
        return out

    # ---------------------------------------------------------------- filters (facets)
    def filter_mask(self, locale: str, filters: dict) -> np.ndarray | None:
        """Boolean mask over a market's products for {department, in_stock, max_price}; cached per filter set."""
        key = (locale, tuple(sorted((k, v) for k, v in filters.items() if v not in (None, "", False))))
        if len(key[1]) == 0:
            return None
        cache = self.__dict__.setdefault("_mask_cache", {})
        if key in cache:
            return cache[key]
        D = self.idx[locale].docs
        m = np.ones(D.n, bool)
        f = dict(key[1])
        if "department" in f:
            m &= D.category == D.dept_index.get(f["department"], -9)
        if f.get("in_stock"):
            m &= D.in_stock > 0.5
        if "max_price" in f:
            m &= np.nan_to_num(np.expm1(D.log_price), nan=np.inf) <= float(f["max_price"])
        if len(cache) > 256:
            cache.clear()
        cache[key] = m
        return m

    # ---------------------------------------------------------------- query rewriting
    def rewrite(self, query: str, locale: str) -> tuple[str, list[tuple[str, str]]]:
        sp = self.spellers.get(locale)
        return sp.correct(query, self.spell_mode) if sp is not None else (query, [])

    def prime(self, texts: list[str], locales: list[str]) -> None:
        """Parse every (rewritten) query up front; required before forking workers (Polars is not fork-safe)."""
        rw = [self.rewrite(t, l)[0] for t, l in zip(texts, locales)]
        self.parser.prime(rw, list(locales))

    # ---------------------------------------------------------------- query context
    def context(self, query: str, locale: str, timings: dict | None = None, full: bool = True) -> QueryContext:
        """Query embedding, plus (full=True) parsed attributes and department distribution for the rerankers."""
        t = time.perf_counter()
        qvec = self.encoder.encode_query(query) if self.encoder is not None else None
        t1 = time.perf_counter()
        parsed = self.parser.parse(query, locale) if full else None
        cp = self.qcat.predict_proba([query])[0] if (full and self.qcat is not None) else None
        t2 = time.perf_counter()
        if timings is not None:
            timings["encode"] = (t1 - t) * 1e3
            if full:
                timings["parse"] = (t2 - t1) * 1e3
        return QueryContext(parsed, qvec, cp)

    def complete(self, ctx: QueryContext, query: str, locale: str, timings: dict | None = None) -> QueryContext:
        """Add query understanding to a light context (explanations, reranking)."""
        if ctx.parsed is None:
            t = time.perf_counter()
            ctx.parsed = self.parser.parse(query, locale)
            ctx.cat_proba = self.qcat.predict_proba([query])[0] if self.qcat is not None else None
            if timings is not None:
                timings["parse"] = (time.perf_counter() - t) * 1e3
        return ctx

    # ---------------------------------------------------------------- retrieval
    def search(self, query: str, locale: str, method: str = "ltr", k: int = 10, budget_ms: float | None = None,
               **kw) -> SearchResponse:
        resp = self._search(query, locale, method, k, budget_ms, **kw)
        rw, corr = self.rewrite(query, locale)
        resp.query = query
        if corr:
            resp.rewritten, resp.corrections = rw, corr
        return resp

    def _search(self, query: str, locale: str, method: str = "ltr", k: int = 10, budget_ms: float | None = None,
               n_candidates: int | None = None, rerank_depth: int | None = None, exact: bool = False,
               rescue: bool = False, filters: dict | None = None) -> SearchResponse:
        """Full-catalog retrieval with graceful degradation to cheaper methods.

        rescue: when keyword retrieval finds nothing, serve semantic results instead (production default path;
        off when a caller explicitly asks for one method, e.g. the demo's method switcher).
        """
        if locale not in self.idx:
            raise KeyError(f"locale {locale!r} is not indexed")
        ix = self.idx[locale]
        T: dict[str, float] = {}
        t0 = time.perf_counter()
        query, corrections = self.rewrite(query, locale)
        if corrections:
            T["spell"] = (time.perf_counter() - t0) * 1e3
        allowed = self.filter_mask(locale, filters) if filters else None
        H = self.cfg.get("search.hybrid")
        n_cand = n_candidates or int(H["candidates_per_retriever"])
        depth = rerank_depth or int(self.cfg.get("search.ltr.rerank_depth", 100))
        served, reason = method, None

        def over_budget() -> bool:
            return budget_ms is not None and (time.perf_counter() - t0) * 1e3 > budget_ms

        # downgrade up front when a component is missing
        while served not in self.available_methods():
            reason = reason or f"{served} unavailable"
            served = FALLBACK[served] or "bm25"

        ctx = None
        if served != "bm25":
            ctx = self.context(query, locale, T, full=served in ("ltr", "ltr_fb"))
            if ctx.qvec is None or not np.any(ctx.qvec):
                reason, served = "query has no known features for the dense encoder", "bm25"

        # ---- bm25 (not needed by the dense-only method)
        b_rows, b_sc = np.zeros(0, np.int64), np.zeros(0, np.float32)
        if served != "dense":
            t = time.perf_counter()
            b_rows, b_sc = ix.bm25_text.search(query, n_cand if served != "bm25" else max(k, 1), allowed=allowed)
            T["bm25"] = (time.perf_counter() - t) * 1e3
        if served == "bm25":
            if rescue and len(b_rows) == 0 and self.encoder is not None and ix.vectors is not None:
                # zero keyword matches (shopper vocabulary differs from the catalog): semantic rescue
                ctx = ctx or self.context(query, locale, T, full=False)
                if ctx.qvec is not None and np.any(ctx.qvec):
                    t = time.perf_counter()
                    d_rows, d_sc = ix.vectors.search(ctx.qvec, k, exact, allowed)
                    T["dense"] = (time.perf_counter() - t) * 1e3
                    return SearchResponse(query, locale, method, "dense", d_rows[:k], d_sc[:k], T,
                                          "no keyword matches: served by semantic retrieval", dict(self.versions),
                                          len(d_rows), ctx)
            return SearchResponse(query, locale, method, "bm25", b_rows[:k], b_sc[:k], T, reason, dict(self.versions),
                                  len(b_rows), ctx)

        if over_budget():
            return SearchResponse(query, locale, method, "bm25", b_rows[:k], b_sc[:k], T,
                                  "latency budget exhausted after keyword stage", dict(self.versions), len(b_rows), ctx)
        # ---- dense
        t = time.perf_counter()
        d_rows, d_sc = ix.vectors.search(ctx.qvec, n_cand if served != "dense" else k, exact, allowed)
        T["dense"] = (time.perf_counter() - t) * 1e3
        if served == "dense":
            return SearchResponse(query, locale, method, "dense", d_rows[:k], d_sc[:k], T, reason, dict(self.versions),
                                  len(d_rows), ctx)

        # ---- fusion
        t = time.perf_counter()
        if served == "hybrid_rrf":
            rows, sc = self._rrf(b_rows, d_rows, int(H["rrf_k"]))
            T["fuse"] = (time.perf_counter() - t) * 1e3
            return SearchResponse(query, locale, method, served, rows[:k], sc[:k], T, reason, dict(self.versions),
                                  len(rows), ctx)
        cand = np.unique(np.concatenate([b_rows, d_rows]))
        bs = ix.bm25_text.score_docs(query, cand)
        ds_ = ix.vectors.score_docs(ctx.qvec, cand)
        fused = self.alpha * _z(ds_) + (1 - self.alpha) * _z(bs)
        order = np.lexsort((cand, -fused))
        cand, fused = cand[order], fused[order]
        T["fuse"] = (time.perf_counter() - t) * 1e3
        if served == "hybrid" or over_budget():
            if served != "hybrid":
                reason = "latency budget exhausted before reranking"
            return SearchResponse(query, locale, method, "hybrid", cand[:k], fused[:k], T, reason, dict(self.versions),
                                  len(cand), ctx)

        # ---- rerank
        self.complete(ctx, query, locale, T)
        rr = cand[:depth]
        scores, X, names = self._ltr_scores(served, ctx, locale, rr, T)
        order = np.lexsort((rr, -scores))
        rows, sc, X = rr[order], scores[order], X[order]
        if k > len(rows) and len(cand) > len(rows):
            # beyond the rerank depth, results continue in fused order (below every reranked score)
            tail = cand[len(rr):len(rr) + (k - len(rows))]
            floor = (sc.min() if len(sc) else 0.0) - 1.0
            rows = np.concatenate([rows, tail])
            sc = np.concatenate([sc, floor - np.arange(1, len(tail) + 1, dtype=np.float32) * 1e-3])
        n_feat = min(k, len(X))
        resp = SearchResponse(query, locale, method, served, rows[:k], sc[:k], T, reason,
                              dict(self.versions), len(cand), ctx, X[:n_feat], names)
        return resp

    def _ltr_scores(self, name: str, ctx: QueryContext, locale: str, rows: np.ndarray, T: dict | None = None):
        t = time.perf_counter()
        with_fb = name == "ltr_fb"
        X = self.builders[locale].build(ctx, rows, with_feedback=with_fb)
        t1 = time.perf_counter()
        s = self.models[name].predict(X, num_threads=1)
        if T is not None:
            T["features"] = (t1 - t) * 1e3
            T["rerank"] = (time.perf_counter() - t1) * 1e3
        names = self.models[name].feature_name()
        return np.asarray(s, np.float32), X, names

    @staticmethod
    def _rrf(a: np.ndarray, b: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sc: dict[int, float] = {}
        for lst in (a, b):
            for r, d in enumerate(lst.tolist()):
                sc[d] = sc.get(d, 0.0) + 1.0 / (k + r + 1)
        rows = np.fromiter(sc.keys(), np.int64, len(sc))
        s = np.fromiter(sc.values(), np.float32, len(sc))
        o = np.lexsort((rows, -s))
        return rows[o], s[o]

    # ---------------------------------------------------------------- rerank a fixed candidate set (ESCI task 1)
    def rerank_scores(self, query: str, locale: str, rows: np.ndarray, methods: list[str],
                      ctx: QueryContext | None = None) -> dict[str, np.ndarray]:
        """Scores of every method for a *given* candidate set (the judged products of a query)."""
        ix = self.idx[locale]
        out: dict[str, np.ndarray] = {}
        query = self.rewrite(query, locale)[0]
        need_ctx = any(m != "bm25" for m in methods)
        if need_ctx and ctx is None:
            ctx = self.context(query, locale, full=any(m in ("ltr", "ltr_fb") for m in methods))
        bs = ix.bm25_text.score_docs(query, rows)
        out["bm25"] = bs
        if ctx is not None and ctx.qvec is not None and ix.vectors is not None:
            ds_ = ix.vectors.score_docs(ctx.qvec, rows)
            if "dense" in methods:
                out["dense"] = ds_
            if "hybrid" in methods:
                out["hybrid"] = self.alpha * _z(ds_) + (1 - self.alpha) * _z(bs)
            if "hybrid_rrf" in methods:
                rb = np.empty(len(rows))
                rb[np.lexsort((rows, -bs))] = np.arange(len(rows))
                rd = np.empty(len(rows))
                rd[np.lexsort((rows, -ds_))] = np.arange(len(rows))
                k = int(self.cfg.get("search.hybrid.rrf_k", 60))
                out["hybrid_rrf"] = (1.0 / (k + rb + 1) + 1.0 / (k + rd + 1)).astype(np.float32)
            for m in ("ltr", "ltr_fb"):
                if m in methods and m in self.models:
                    out[m] = self._ltr_scores(m, ctx, locale, rows)[0]
        return {m: out[m] for m in methods if m in out}

    # ---------------------------------------------------------------- result rendering helpers
    def result_rows(self, locale: str, rows: np.ndarray) -> list[dict]:
        ix = self.idx[locale]
        return ix.display[rows.tolist()].to_dicts() if len(rows) else []

    def esci_class_proba(self, X: np.ndarray, names: list[str]) -> np.ndarray | None:
        m = self.models.get("esci_class")
        if m is None or X is None:
            return None
        cols = [names.index(n) for n in m.feature_name()]
        return np.asarray(m.predict(X[:, cols], num_threads=1))

    def index_manifest(self) -> dict:
        p = self.ds.index_dir() / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}


def build_indexes(cfg: Config, corpus: str | None = None) -> dict:
    """`csre index`: BM25 + dense indexes for every locale of a corpus, with a manifest."""
    from ..utils import stage_timer  # noqa: PLC0415
    ds = Dataset(cfg, corpus)
    reg = ModelRegistry(cfg)
    enc_path = reg.path("dense_encoder")
    encoder = load_encoder(enc_path) if enc_path else None
    with stage_timer(cfg, f"index_{ds.name}") as info:
        per = {loc: build_locale_index(cfg, ds, loc, encoder) for loc in ds.locales()}
        manifest = {"corpus": ds.name, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "dense_encoder": reg.resolve("dense_encoder") if encoder else None, "locales": per,
                    "bm25": cfg.get("search.bm25"), "ann": cfg.get("search.dense.ann")}
        (ds.index_dir() / "manifest.json").write_text(json.dumps(manifest, indent=2))
        info.update(rows=sum(v["n_docs"] for v in per.values()), corpus=ds.name)
    return manifest

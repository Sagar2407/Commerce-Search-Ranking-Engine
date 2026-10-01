"""Serving logic shared by the HTTP API and the static snapshot builder.

* `ResultCache` — TTL + LRU, keyed by (locale, method, k, normalised query, model versions). Case / spacing /
  stop-word variants of a query share an entry; promoting a new model version naturally misses the cache.
* `SearchService.search` — engine call + cache + explanations + query understanding (attributes, brands,
  negation, department distribution and a "vague query" flag with department suggestions).
* `SearchService.compare` — the same query through several methods: rank movements, overlap, latency, and
  — when the query is a judged dataset query — nDCG@10 and the human label of every judged result.
* `SearchService.product` — product page data with substitutes / complements from the product graph.
* `SearchService.feedback` — logs an event (JSONL) and updates the feedback store the `ltr_fb` ranker reads.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..evaluation import metrics as M
from ..search import analysis as A
from ..search.engine import FALLBACK, METHOD_LABELS, METHODS, Engine
from ..search.query import entropy_bits
from .explain import explain_results


class ResultCache:
    def __init__(self, max_entries: int = 50_000, ttl_s: float = 900):
        self.max, self.ttl = max_entries, ttl_s
        self.data: OrderedDict = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            v = self.data.get(key)
            if v is None or time.time() - v[0] > self.ttl:
                if v is not None:
                    del self.data[key]
                self.misses += 1
                return None
            self.data.move_to_end(key)
            self.hits += 1
            return v[1]

    def put(self, key, value) -> None:
        with self.lock:
            self.data[key] = (time.time(), value)
            self.data.move_to_end(key)
            while len(self.data) > self.max:
                self.data.popitem(last=False)
                self.evictions += 1

    def invalidate(self, pred) -> int:
        with self.lock:
            ks = [k for k in self.data if pred(k)]
            for k in ks:
                del self.data[k]
            return len(ks)

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"entries": len(self.data), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else None, "evictions": self.evictions}


def _clean(v):
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


class SearchService:
    def __init__(self, cfg: Config, engine: Engine):
        self.cfg = cfg
        self.engine = engine
        S = cfg.get("serve")
        self.cache = ResultCache(int(S["cache"]["max_entries"]), float(S["cache"]["ttl_s"]))
        self.budget_ms = float(S["latency_budget_ms"])
        self.default_method = S["default_method"]
        self.top_n = int(S.get("explain_top_n", 3))
        self.fb_dir = cfg.path("feedback")
        self.fb_dir.mkdir(parents=True, exist_ok=True)
        self.fb_lock = threading.Lock()
        self.fb_count = 0
        ds = engine.ds
        q = ds.queries()
        self.queries = q.filter(pl.col("locale").is_in(list(engine.idx)))
        self._qkey = {(r["locale"], A.query_key(r["query"])): r["query_id"]
                      for r in self.queries.select("query_id", "query", "locale").iter_rows(named=True)}
        j = ds.judgments(["query_id", "doc_id", "esci_label", "gain"])
        self._judged = {qid: dict(zip(g["doc_id"].to_list(), zip(g["esci_label"].to_list(), g["gain"].to_list())))
                        for (qid,), g in j.join(self.queries.select("query_id"), on="query_id", how="semi")
                        .group_by(["query_id"])}
        self.edges = ds.edges()
        self.cat_comp = ds.category_complements()
        self.request_log: list[dict] = []

    # ---------------------------------------------------------------- helpers
    def judged_query(self, query: str, locale: str) -> int | None:
        return self._qkey.get((locale, A.query_key(query)))

    def understanding(self, ctx, locale: str) -> dict:
        p = ctx.parsed
        out = {"normalized": p.key, "attributes": {k: v for k, v in p.attrs.items() if v}, "pack_count": p.pack_count,
               "brands": p.brands, "negated_terms": p.negated_terms}
        if ctx.cat_proba is not None and self.engine.qcat is not None:
            cp = ctx.cat_proba
            order = np.argsort(-cp)[:4]
            ent = float(entropy_bits(cp))
            tau = float(self.cfg.get("search.query_category.min_confidence", 0.35))
            out["departments"] = [{"name": self.engine.qcat.classes[i], "p": round(float(cp[i]), 3)} for i in order]
            out["vague"] = bool(cp.max() < tau)
            out["department_entropy"] = round(ent, 3)
        return out

    # ---------------------------------------------------------------- search
    def search(self, query: str, locale: str, method: str | None = None, k: int = 10, explain: bool = True,
               use_cache: bool = True, budget_ms: float | None = None, rescue: bool | None = None) -> dict:
        method = method or self.default_method
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        key = (locale, method, k, A.query_key(query), explain, budget_ms,
               tuple(sorted(self.engine.versions.items())))
        t0 = time.perf_counter()
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                out = dict(hit)
                out["cache"] = {"hit": True, "lookup_ms": round((time.perf_counter() - t0) * 1e3, 3)}
                out["query"] = query
                return out
        rescue = (method == self.default_method) if rescue is None else rescue
        resp = self.engine.search(query, locale, method, k=k, budget_ms=budget_ms, rescue=rescue)
        products = self.engine.result_rows(locale, resp.rows)
        t1 = time.perf_counter()
        # cheaper paths skip query understanding; explanations, badges and the understanding panel need it
        q_eff = resp.rewritten or query
        resp.context = self.engine.complete(resp.context or self.engine.context(q_eff, locale, full=True),
                                            q_eff, locale)
        expl = explain_results(self.engine, resp, products, self.top_n) if explain and len(resp.rows) else None
        t_expl = (time.perf_counter() - t1) * 1e3
        qid = self.judged_query(query, locale)
        judged = self._judged.get(qid, {}) if qid is not None else {}
        results = []
        for i, (p, sc) in enumerate(zip(products, resp.scores.tolist())):
            r = {"rank": i + 1, "score": round(float(sc), 5), **{k_: _clean(v) for k_, v in p.items()}}
            if judged:
                lab = judged.get(p["doc_id"])
                r["judged_label"] = lab[0] if lab else None
            if expl:
                r["explanation"] = _clean(expl[i])
            results.append(r)
        ctx = resp.context
        out = {
            "query": query, "locale": locale, "method": method, "method_label": METHOD_LABELS[method],
            "served_by": resp.served_by, "fallback_reason": resp.fallback_reason, "k": k,
            "timings_ms": {k_: round(v, 3) for k_, v in resp.timings_ms.items()},
            "engine_ms": resp.total_ms, "explain_ms": round(t_expl, 3), "candidates": resp.candidates,
            "versions": resp.versions, "understanding": self.understanding(ctx, locale),
            "judged_query_id": qid, "results": results,
            "corrected_query": resp.rewritten, "corrections": [{"from": a, "to": b} for a, b in resp.corrections],
        }
        if judged:
            rel = {self.engine.idx[locale].rows([d])[0]: (g, lab) for d, (lab, g) in judged.items()}
            rel = {int(r): (float(g), lab) for r, (g, lab) in rel.items() if r >= 0}
            out["quality"] = _clean(M.retrieval_metrics(resp.rows, rel, k=min(k, 10), recall_k=k))
        if use_cache:
            self.cache.put(key, out)
        out = dict(out)
        out["cache"] = {"hit": False}
        self.request_log.append({"ts": time.time(), "method": method, "served_by": resp.served_by,
                                 "ms": resp.total_ms})
        self.request_log = self.request_log[-5000:]
        return out

    def compare(self, query: str, locale: str, methods: list[str] | None = None, k: int = 10) -> dict:
        methods = [m for m in (methods or METHODS) if m in self.engine.available_methods()]
        runs = {m: self.search(query, locale, m, k=k, explain=True, use_cache=False, rescue=False) for m in methods}
        base = methods[0] if methods else None
        pos = {m: {r["doc_id"]: r["rank"] for r in v["results"]} for m, v in runs.items()}
        for m, v in runs.items():
            for r in v["results"]:
                r["rank_in"] = {o: pos[o].get(r["doc_id"]) for o in methods if o != m}
        overlap = {f"{a}|{b}": round(len(set(pos[a]) & set(pos[b])) / max(1, k), 3)
                   for i, a in enumerate(methods) for b in methods[i + 1:]}
        return {"query": query, "locale": locale, "methods": methods, "baseline": base, "runs": runs,
                "overlap_at_k": overlap, "judged_query_id": next(iter(runs.values()))["judged_query_id"] if runs else None}

    # ---------------------------------------------------------------- product page
    def product(self, doc_id: str, locale: str, n: int = 8) -> dict:
        ix = self.engine.idx[locale]
        row = ix.rows([doc_id])[0]
        if row < 0:
            raise KeyError(doc_id)
        prod = _clean(self.engine.result_rows(locale, np.array([row]))[0])
        rel = {"substitute": [], "complement": []}
        if self.edges is not None:
            e = self.edges.filter(pl.col("src_doc") == doc_id).sort("support", descending=True)
            for rtype in rel:
                ids = e.filter(pl.col("relation") == rtype)["dst_doc"].head(n).to_list()
                rows = ix.rows(ids)
                rows = rows[rows >= 0]
                rel[rtype] = [_clean(p) for p in self.engine.result_rows(locale, rows)]
        comp_cats = []
        if self.cat_comp is not None and prod.get("category"):
            comp_cats = self.cat_comp.filter((pl.col("src_category") == prod["category"])
                                             & (pl.col("dst_category") != prod["category"])) \
                                     .sort("share", descending=True).head(3)["dst_category"].to_list()
        return {"product": prod, "substitutes": rel["substitute"], "complements": rel["complement"],
                "complement_departments": comp_cats,
                "note": "Substitute / complement links come from products judged for the same queries (ESCI)."}

    # ---------------------------------------------------------------- feedback
    def feedback(self, ev: dict) -> dict:
        locale, query, doc_id, event = ev["locale"], ev["query"], ev["doc_id"], ev["event"]
        if event not in ("impression", "click", "cart", "purchase", "thumbs_up", "thumbs_down"):
            raise ValueError(f"unknown event {event!r}")
        key = A.query_key(query)
        row = int(self.engine.idx[locale].rows([doc_id])[0])
        if row < 0:
            raise KeyError(doc_id)
        if self.engine.feedback is not None:
            # position-based propensity (1/rank) for IPS-weighted clicks, as in the simulator's click model
            prop = 1.0 / max(1, int(ev.get("position") or 1))
            self.engine.feedback.add(locale, key, row, event, propensity=prop)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "locale": locale, "query": query, "key": key,
               "doc_id": doc_id, "event": event, "position": ev.get("position"), "method": ev.get("method"),
               "versions": self.engine.versions}
        with self.fb_lock:
            with open(self.fb_dir / f"events-{time.strftime('%Y%m%d')}.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.fb_count += 1
        n = self.cache.invalidate(lambda k_: k_[0] == locale and k_[3] == key and k_[1] == "ltr_fb")
        return {"ok": True, "invalidated_cache_entries": n, "events_logged": self.fb_count}

    # ---------------------------------------------------------------- demo helpers
    def sample_queries(self, locale: str, per_slice: int = 4, seed: int = 0) -> dict:
        q = self.queries.filter(pl.col("locale") == locale)
        out = {}
        for col, label in [("slice_ambiguous", "Vague / ambiguous"), ("slice_spec", "States a spec"),
                           ("slice_negation", "Negation"), ("slice_brand", "Brand"),
                           ("slice_sparse_products", "Thin product descriptions"), ("slice_hard", "Hard (ESCI)")]:
            if col in q.columns:
                g = q.filter(pl.col(col).fill_null(False) & (pl.col("n_E") >= 2))
                if g.height:
                    out[label] = g.sample(n=min(per_slice, g.height), seed=seed)["query"].to_list()
        return out

    def stats(self) -> dict:
        lat = [r["ms"] for r in self.request_log]
        served = {}
        for r in self.request_log:
            served[r["served_by"]] = served.get(r["served_by"], 0) + 1
        return {"cache": self.cache.stats(), "requests": len(self.request_log),
                "latency_ms": M.latency_summary(np.array(lat)) if lat else {}, "served_by": served,
                "feedback_events": self.fb_count, "fallback_chain": FALLBACK}

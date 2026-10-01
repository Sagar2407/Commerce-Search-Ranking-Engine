"""`csre snapshot`: a self-contained storefront page with precomputed results (no server needed).

Runs a curated set of queries — judged test queries from every evaluation slice plus hand-written vague,
misspelled and cross-vocabulary queries — through every ranking approach, and embeds the responses,
product pages, evaluation report and model registry into the storefront HTML. Product images are not
embedded (the page shows department initials instead), so the file stays small enough to share.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import polars as pl

from ..config import Config
from ..search.engine import METHOD_LABELS, Engine
from ..utils import get_logger
from .app import page_html
from .service import SearchService

log = get_logger("csre.snapshot")

# Unjudged, hand-written queries: vague, misspelled, or phrased differently from catalog titles.
CURATED = {
    "us": {
        "Vague or misspelled (not in the dataset)": [
            "something to keep coffee hot", "gift for a 10 year old who likes science", "wireles earbuds",
            "shoes for standing all day", "bluetooth speeker waterproof", "stuff for a new puppy",
        ],
    },
    "es": {"Vague or misspelled (not in the dataset)": ["regalo para niño de 5 años", "auriculares inalambricos"]},
    "jp": {"Vague or misspelled (not in the dataset)": ["水筒 子供", "ワイヤレスイヤホン"]},
}


def _norm(q: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", q).lower().split())


def _slim(run: dict) -> dict:
    """Drop fields the page does not read, to keep the snapshot small."""
    for r in run.get("results", []):
        for k in ("attr_dimensions", "attr_compat", "is_sparse", "color"):
            r.pop(k, None)
        e = r.get("explanation") or {}
        if "ranker" in e:
            e["ranker"]["top"] = e["ranker"]["top"][:3]
    return run


def build_snapshot(cfg: Config, corpus: str | None = None, per_slice: int = 3, out: Path | None = None) -> Path:
    eng = Engine(cfg, corpus or "demo")
    svc = SearchService(cfg, eng)
    queries: dict[str, dict] = {}
    products: dict[str, dict] = {}
    examples: dict[str, dict] = {}
    default_query = None
    for loc in eng.idx:
        ex = svc.sample_queries(loc, per_slice=per_slice, seed=cfg.seed)
        ex.update(CURATED.get(loc, {}))
        examples[loc] = ex
        for group, qs in ex.items():
            for q in qs:
                c = svc.compare(q, loc, k=10)
                for m in c["runs"]:
                    _slim(c["runs"][m])
                queries[f"{loc}|{_norm(q)}"] = c
                best = c["runs"].get(svc.default_method) or next(iter(c["runs"].values()))
                for r in best["results"][:2]:
                    if r["doc_id"] not in products:
                        try:
                            products[r["doc_id"]] = svc.product(r["doc_id"], loc, n=6)
                        except KeyError:
                            pass
                default_query = default_query or q
        log.info("snapshot %s: %d queries", loc, sum(len(v) for v in ex.values()))
    rep = None
    for name in (f"eval_full_{cfg.get('eval.split')}.json", f"eval_{eng.ds.name}_{cfg.get('eval.split')}.json"):
        p = cfg.path("reports", name)
        if p.exists():
            rep = json.loads(p.read_text())
            break
    health = {"status": "snapshot", "corpus": eng.ds.name, "locales": list(eng.idx),
              "n_products": {l: ix.n for l, ix in eng.idx.items()}, "methods": eng.available_methods(),
              "method_labels": METHOD_LABELS,
              "default_method": svc.default_method, "versions": eng.versions, "latency_budget_ms": svc.budget_ms,
              "stats": {"cache": None, "feedback_events": 0}}
    extra = {}
    for name in ("bench", "simulated_ab"):
        p = cfg.path("reports", f"{name}.json")
        if p.exists():
            extra[name] = json.loads(p.read_text())
    snap = {"health": health, "examples": examples, "queries": queries, "products": products, "eval": rep,
            "models": eng.registry.listing(), "default_query": default_query, "reports": extra}
    js = json.dumps(snap, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    out = out or cfg.path("reports", "storefront_snapshot.html")
    out.write_text(page_html(js, wrap=False), encoding="utf-8")
    log.info("wrote %s (%.1f MB, %d queries, %d product pages)", out, out.stat().st_size / 1e6, len(queries),
             len(products))
    return out

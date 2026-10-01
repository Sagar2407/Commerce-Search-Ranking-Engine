"""HTTP API + storefront.  `csre serve --corpus demo`  ->  http://localhost:8000

GET  /                     storefront (search, compare approaches, evaluation, models)
GET  /api/health           corpus, locales, methods, model versions, cache + latency stats
GET  /api/search           q, locale, method, k, explain, budget_ms, department, in_stock, max_price, user_id
GET  /api/experiment       running A/B experiment and (with user_id) the shopper's arm
GET  /api/compare          q, locale, methods (comma separated), k
GET  /api/product/{doc_id} locale -> product + substitutes + complements
POST /api/feedback         {query, locale, doc_id, event, position, method}
GET  /api/suggest          q, locale -> autocomplete from popular queries
GET  /api/examples         locale -> example queries per evaluation slice
GET  /api/eval             latest evaluation report (JSON)
GET  /api/models           model registry
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..search.engine import METHOD_LABELS, Engine
from .service import SearchService

STATIC = Path(__file__).parent / "static"


def page_html(snapshot_json: str | None = None, wrap: bool = True) -> str:
    """The storefront page. `snapshot_json` embeds precomputed data so the page runs without a server."""
    body = (STATIC / "storefront.html").read_text(encoding="utf-8")
    if snapshot_json is not None:
        body = body.replace("<script>\n(() => {", "<script>window.__CSRE_SNAPSHOT__ = " + snapshot_json
                            + ";</script>\n<script>\n(() => {", 1)
    if not wrap:
        return body
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
            '<style>[hidden]{display:none!important}img{max-width:100%}</style></head><body>' + body + "</body></html>")


def create_app(cfg: Config, corpus: str | None = None, engine: Engine | None = None):
    from fastapi import Body, FastAPI, HTTPException, Query  # noqa: PLC0415
    from fastapi.responses import HTMLResponse, JSONResponse  # noqa: PLC0415

    eng = engine or Engine(cfg, corpus)
    svc = SearchService(cfg, eng)
    app = FastAPI(title="Commerce Search & Ranking Engine", version="0.2.0")
    app.state.service = svc

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page_html()

    @app.get("/api/health")
    def health():
        return {"status": "ok", "corpus": eng.ds.name, "locales": list(eng.idx),
                "n_products": {l: ix.n for l, ix in eng.idx.items()}, "methods": eng.available_methods(),
                "method_labels": METHOD_LABELS, "default_method": svc.default_method, "versions": eng.versions,
                "latency_budget_ms": svc.budget_ms, "stats": svc.stats(), "index": eng.index_manifest()}

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=1, max_length=300), locale: str = "us", method: str | None = None,
               k: int = Query(10, ge=1, le=100), explain: bool = True, budget_ms: float | None = None,
               department: str | None = None, in_stock: bool = False, max_price: float | None = None,
               user_id: str | None = None):
        try:
            return JSONResponse(svc.search(q, locale, method, k=k, explain=explain, budget_ms=budget_ms,
                                           filters={"department": department, "in_stock": in_stock,
                                                    "max_price": max_price}, user_id=user_id))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/compare")
    def compare(q: str = Query(..., min_length=1, max_length=300), locale: str = "us", methods: str | None = None,
                k: int = Query(10, ge=1, le=50)):
        try:
            return JSONResponse(svc.compare(q, locale, methods.split(",") if methods else None, k=k))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/product/{doc_id}")
    def product(doc_id: str, locale: str = "us"):
        try:
            return JSONResponse(svc.product(doc_id, locale))
        except KeyError as e:
            raise HTTPException(404, f"unknown product {e}") from e

    @app.post("/api/feedback")
    def feedback(ev: dict = Body(...)):
        try:
            return svc.feedback(ev)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/suggest")
    def suggest(q: str = "", locale: str = "us"):
        return svc.suggest(q, locale)

    @app.get("/api/examples")
    def examples(locale: str = "us"):
        return svc.sample_queries(locale)

    @app.get("/api/eval")
    def eval_report():
        for name in (f"eval_full_{cfg.get('eval.split')}.json", f"eval_{eng.ds.name}_{cfg.get('eval.split')}.json"):
            p = cfg.path("reports", name)
            if p.exists():
                return JSONResponse(json.loads(p.read_text()))
        raise HTTPException(404, "no evaluation report yet: run `csre eval`")

    @app.get("/api/report/{name}")
    def report(name: str):
        if name not in ("bench", "simulated_ab"):
            raise HTTPException(404, "unknown report")
        p = cfg.path("reports", f"{name}.json")
        if not p.exists():
            raise HTTPException(404, f"no {name} report yet")
        return JSONResponse(json.loads(p.read_text()))

    @app.get("/api/experiment")
    def experiment(user_id: str | None = None):
        e = svc.experiment
        if e is None:
            return {"enabled": False}
        out = {"enabled": True, "name": e.name, "arms": e.arms, "allocation": e.allocation}
        if user_id:
            out["assignment"] = e.assign(user_id)
        return out

    @app.get("/api/models")
    def models():
        return eng.registry.listing()

    return app


def serve(cfg: Config, corpus: str | None = None) -> None:
    import uvicorn  # noqa: PLC0415
    S = cfg.get("serve")
    uvicorn.run(create_app(cfg, corpus), host=S["host"], port=int(S["port"]), log_level="info")

"""`csre eval`: every ranking method under identical conditions, broken down by slice, with latency and cost.

Outputs
-------
data/runs/<corpus>/<split>/{rerank,retrieval,latency,robustness}.parquet   per-query results
data/reports/eval_<corpus>_<split>.json / .md                             aggregated report

Rerank and retrieval runs are parallelised with forked workers that share the loaded engine
(copy-on-write); latency is measured separately, single-threaded, so the cost model is not distorted by
the parallel evaluation.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config
from ..search.engine import METHOD_LABELS, METHODS, Engine
from ..utils import get_logger, stage_timer
from . import metrics as M

log = get_logger("csre.eval")

_ENGINE: Engine | None = None
_JUDGED: dict | None = None
_METHODS: list[str] = []
_K = (10, 100)

RERANK_METRICS = ["ndcg10", "ndcg5", "mrr_e", "p1_e"]
RETRIEVAL_METRICS = ["ndcg10_cond", "ndcg10_lb", "judged10", "recall100_e", "success10_e"]


def _limit_threads() -> None:
    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
        threadpool_limits(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import faiss  # noqa: PLC0415
        faiss.omp_set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _rerank_batch(batch: list[tuple]) -> list[dict]:
    out = []
    for qid, query, loc in batch:
        rows, gains, labels = _JUDGED[qid]
        if len(rows) == 0:
            continue
        scores = _ENGINE.rerank_scores(query, loc, rows, _METHODS)
        for m, s in scores.items():
            out.append({"query_id": qid, "method": m, **M.rerank_metrics(s, gains, labels, rows, _K[0])})
    return out


def _retrieval_batch(batch: list[tuple]) -> list[dict]:
    out = []
    for qid, query, loc in batch:
        rows, gains, labels = _JUDGED[qid]
        judged = {int(r): (float(g), str(l)) for r, g, l in zip(rows, gains, labels)}
        for m in _METHODS:
            resp = _ENGINE.search(query, loc, m, k=_K[1])
            out.append({"query_id": qid, "method": m, "served_by": resp.served_by,
                        **M.retrieval_metrics(resp.rows, judged, _K[0], _K[1])})
    return out


def _robust_batch(batch: list[tuple]) -> list[dict]:
    out = []
    for rid, qid, variant_text, orig_text, loc, variant in batch:
        rows, gains, labels = _JUDGED[qid]
        judged = {int(r): (float(g), str(l)) for r, g, l in zip(rows, gains, labels)}
        for m in _METHODS:
            for kind, text in (("variant", variant_text), ("original", orig_text)):
                resp = _ENGINE.search(text, loc, m, k=_K[1])
                out.append({"request_id": rid, "query_id": qid, "variant": variant, "method": m, "text": kind,
                            **M.retrieval_metrics(resp.rows, judged, _K[0], _K[1])})
    return out


def _parallel(fn, items: list, n_jobs: int, chunk: int = 50) -> list[dict]:
    batches = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    if n_jobs <= 1:
        _limit_threads()
        return [r for b in batches for r in fn(b)]
    ctx = mp.get_context("fork")
    out: list[dict] = []
    with ctx.Pool(n_jobs, initializer=_limit_threads) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, batches)):
            out += res
            if (i + 1) % max(1, len(batches) // 10) == 0:
                log.info("  %s: %d/%d batches", fn.__name__.strip("_"), i + 1, len(batches))
    return out


def _sample(q: pl.DataFrame, n_per_locale: int, seed: int) -> pl.DataFrame:
    if n_per_locale <= 0:
        return q
    return q.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("locale").alias("_r")) \
            .filter(pl.col("_r") < n_per_locale).drop("_r")


def load_judged(engine: Engine, split: str) -> tuple[pl.DataFrame, dict]:
    ds = engine.ds
    q = ds.queries().filter((pl.col("split") == split) & pl.col("locale").is_in(list(engine.idx)))
    j = ds.judgments(["query_id", "doc_id", "locale", "esci_label", "gain", "split"]).filter(pl.col("split") == split)
    judged = {}
    for (loc,), g in j.group_by(["locale"]):
        if loc not in engine.idx:
            continue
        g = g.with_columns(pl.Series("row", engine.idx[loc].rows(g["doc_id"].to_list()))).filter(pl.col("row") >= 0)
        for (qid,), gg in g.group_by(["query_id"]):
            judged[qid] = (gg["row"].to_numpy(), gg["gain"].to_numpy().astype(np.float64),
                           gg["esci_label"].to_numpy())
    q = q.filter(pl.col("query_id").is_in(list(judged)))
    return q, judged


# ======================================================================================
# aggregation
# ======================================================================================
SLICE_COLS = ["slice_ambiguous", "slice_underspecified", "slice_multi_intent", "slice_negation", "slice_spec",
              "slice_brand", "slice_sparse_products", "slice_hard"]


def _agg(df: pl.DataFrame, metrics: list[str], n_boot: int, seed: int) -> dict:
    out = {}
    for m, g in df.group_by("method", maintain_order=True):
        rec = {"n": g.height}
        for c in metrics:
            if c in g.columns:
                mean, lo, hi = M.mean_ci(g[c].cast(pl.Float64).fill_null(np.nan).to_numpy(), n_boot, seed)
                rec[c] = {"mean": mean, "lo": lo, "hi": hi}
        out[m[0]] = rec
    return out


def _deltas(df: pl.DataFrame, metric: str, base: str, n_boot: int, seed: int) -> dict:
    wide = df.select("query_id", "method", metric).pivot(on="method", index="query_id", values=metric)
    out = {}
    if base not in wide.columns:
        return out
    a = wide[base].cast(pl.Float64).fill_null(np.nan).to_numpy()
    for m in wide.columns:
        if m in ("query_id", base):
            continue
        out[m] = M.paired_delta(a, wide[m].cast(pl.Float64).fill_null(np.nan).to_numpy(), n_boot, seed)
    return out


def _by(df: pl.DataFrame, q: pl.DataFrame, metric: str) -> dict:
    d = df.join(q, on="query_id", how="left")
    res: dict = {}
    for col in ["locale", "traffic_bucket", "length_bucket", "source"]:
        if col in d.columns:
            t = d.group_by(col, "method").agg(pl.col(metric).mean().alias("mean"), pl.len().alias("n"))
            res[col] = {str(k): {r["method"]: {"mean": r["mean"], "n": r["n"]} for r in g.iter_rows(named=True)}
                        for (k,), g in t.group_by(col)}
    res["slice"] = {}
    for col in SLICE_COLS:
        if col in d.columns:
            g = d.filter(pl.col(col).fill_null(False))
            if g.height:
                t = g.group_by("method").agg(pl.col(metric).mean().alias("mean"), pl.len().alias("n"))
                res["slice"][col.removeprefix("slice_")] = {r["method"]: {"mean": r["mean"], "n": r["n"]}
                                                           for r in t.iter_rows(named=True)}
    return res


# ======================================================================================
# main
# ======================================================================================
def evaluate(cfg: Config, corpus: str | None = None, split: str | None = None, methods: list[str] | None = None,
             n_jobs: int | None = None, engine: Engine | None = None, parts: tuple[str, ...] = (
                 "rerank", "retrieval", "latency", "robustness")) -> dict:
    global _ENGINE, _JUDGED, _METHODS, _K
    E = cfg.get("eval")
    split = split or E["split"]
    seed = cfg.seed
    n_boot = int(E["bootstrap"])
    n_jobs = n_jobs or max(1, (mp.cpu_count() or 2))
    eng = engine or Engine(cfg, corpus)
    _ENGINE = eng
    _METHODS = [m for m in (methods or METHODS) if m in eng.available_methods()]
    _K = (int(E["k"]), int(E["recall_k"]))
    q, judged = load_judged(eng, split)
    _JUDGED = judged
    run_dir = cfg.path("runs", eng.ds.name, split + (f"_{cfg.get('eval.tag')}" if cfg.get("eval.tag") else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    qcols = ["query_id", "query", "locale", "traffic_bucket", "length_bucket", "source", *SLICE_COLS]
    qinfo = q.select([c for c in qcols if c in q.columns])
    report: dict = {"corpus": eng.ds.name, "split": split, "methods": _METHODS, "labels": METHOD_LABELS,
                    "versions": eng.versions, "n_queries": q.height, "k": _K[0], "recall_k": _K[1],
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "index": {loc: {kk: v for kk, v in b.items() if kk != "locale"}
                              for loc, b in eng.index_manifest().get("locales", {}).items()}}
    items = list(zip(q["query_id"].to_list(), q["query"].to_list(), q["locale"].to_list()))
    eng.prime(q["query"].to_list(), q["locale"].to_list())   # Polars is not fork-safe: parse up front

    with stage_timer(cfg, f"eval_{eng.ds.name}_{split}") as info:
        if "rerank" in parts:
            t = time.time()
            rr = pl.DataFrame(_parallel(_rerank_batch, items, n_jobs))
            rr.write_parquet(run_dir / "rerank.parquet")
            report["rerank"] = {"overall": _agg(rr, RERANK_METRICS, n_boot, seed),
                                "delta_vs_bm25": _deltas(rr, "ndcg10", "bm25", n_boot, seed),
                                "by": _by(rr, qinfo, "ndcg10"), "seconds": round(time.time() - t, 1)}
            log.info("rerank done: %s", {m: round(v["ndcg10"]["mean"], 4) for m, v in report["rerank"]["overall"].items()})

        if "retrieval" in parts:
            t = time.time()
            qs = _sample(q, int(E["retrieval_queries_per_locale"]), seed)
            ritems = list(zip(qs["query_id"].to_list(), qs["query"].to_list(), qs["locale"].to_list()))
            rt = pl.DataFrame(_parallel(_retrieval_batch, ritems, n_jobs, chunk=20))
            rt.write_parquet(run_dir / "retrieval.parquet")
            report["retrieval"] = {"n_queries": qs.height, "overall": _agg(rt, RETRIEVAL_METRICS, n_boot, seed),
                                   "delta_vs_bm25": _deltas(rt, "ndcg10_cond", "bm25", n_boot, seed),
                                   "recall_delta_vs_bm25": _deltas(rt, "recall100_e", "bm25", n_boot, seed),
                                   "by": _by(rt, qinfo, "ndcg10_cond"),
                                   "served_by": {m[0]: dict(g["served_by"].value_counts().iter_rows())
                                                 for m, g in rt.group_by("method")},
                                   "seconds": round(time.time() - t, 1)}
            log.info("retrieval done: %s", {m: round(v["ndcg10_cond"]["mean"], 4)
                                            for m, v in report["retrieval"]["overall"].items()})

        if "latency" in parts:
            report["latency"] = measure_latency(cfg, eng, q, seed, run_dir)

        if "robustness" in parts:
            rb = robustness(cfg, eng, q, seed, n_jobs)
            if rb is not None:
                rb.write_parquet(run_dir / "robustness.parquet")
                piv = rb.group_by("variant", "method", "text").agg(pl.col("ndcg10_cond").mean(),
                                                                    pl.col("recall100_e").mean(), pl.len())
                report["robustness"] = {
                    "n_requests": rb["request_id"].n_unique(),
                    "table": piv.sort("variant", "method", "text").to_dicts(),
                }
        info.update(rows=q.height, corpus=eng.ds.name, split=split)

    tag = cfg.get("eval.tag")
    out = cfg.path("reports", f"eval_{eng.ds.name}_{split}{'_' + tag if tag else ''}.json")
    if out.exists() and set(parts) != {"rerank", "retrieval", "latency", "robustness"}:
        # partial re-run (e.g. `--parts latency`): keep the other sections of the existing report
        old = json.loads(out.read_text())
        old.update({k: v for k, v in report.items() if k in parts or k not in old})
        old["versions"], old["created_at"] = report["versions"], report["created_at"]
        report = old
    out.write_text(json.dumps(report, indent=2, default=_json_default))
    from .report import write_markdown  # noqa: PLC0415
    write_markdown(cfg, report, out.with_suffix(".md"))
    log.info("wrote %s", out)
    return report


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


def measure_latency(cfg: Config, eng: Engine, q: pl.DataFrame, seed: int, run_dir: Path) -> dict:
    """Single-threaded latency per method on a fixed query sample (k = 10, warm process)."""
    _limit_threads()
    E = cfg.get("eval")
    C = cfg.get("cost")
    n = int(E["latency_queries"])
    qs = q.sample(n=min(n, q.height), seed=seed, shuffle=True)
    rows = []
    # warm-up
    for qq, loc in zip(qs["query"].head(20).to_list(), qs["locale"].head(20).to_list()):
        for m in _METHODS:
            eng.search(qq, loc, m, k=10)
    for qid, qq, loc in zip(qs["query_id"].to_list(), qs["query"].to_list(), qs["locale"].to_list()):
        for m in _METHODS:
            t = time.perf_counter()
            r = eng.search(qq, loc, m, k=10)
            wall = (time.perf_counter() - t) * 1e3
            rows.append({"query_id": qid, "method": m, "locale": loc, "wall_ms": wall,
                         **{f"stage_{k}": v for k, v in r.timings_ms.items()}})
    lat = pl.DataFrame(rows)
    lat.write_parquet(run_dir / "latency.parquet")
    out: dict = {"n_queries": qs.height, "threads": 1, "methods": {}}
    stages = [c for c in lat.columns if c.startswith("stage_")]
    for (m,), g in lat.group_by(["method"], maintain_order=True):
        s = M.latency_summary(g["wall_ms"].to_numpy())
        out["methods"][m] = {
            "wall_ms": s,
            "stages_mean_ms": {c.removeprefix("stage_"): float(g[c].mean()) for c in stages if g[c].null_count() < g.height},
            "cost": M.cost_per_million(s["mean"], int(C["vcpus"]), float(C["usd_per_hour"]),
                                       float(C["target_utilisation"])),
        }
    out["cost_assumptions"] = C
    return out


def robustness(cfg: Config, eng: Engine, q: pl.DataFrame, seed: int, n_jobs: int) -> pl.DataFrame | None:
    """Replay variants (typo / dropped / reordered token / modifier / case) scored against the parent's labels."""
    rep = eng.ds.replay()
    if rep is None:
        return None
    n = int(cfg.get("eval.robustness_requests", 5000))
    v = rep.filter(pl.col("is_novel") & pl.col("query_id").is_in(q["query_id"].implode())) \
           .unique("query", keep="first", maintain_order=True).sort("request_id") \
           .join(q.select("query_id", pl.col("query").alias("orig")), on="query_id", maintain_order="left")
    if v.height == 0:
        return None
    v = v.sample(n=min(n, v.height), seed=seed, shuffle=True)
    eng.prime(v["query"].to_list(), v["locale"].to_list())
    items = list(zip(v["request_id"].to_list(), v["query_id"].to_list(), v["query"].to_list(), v["orig"].to_list(),
                     v["locale"].to_list(), v["variant"].to_list()))
    return pl.DataFrame(_parallel(_robust_batch, items, n_jobs, chunk=20))

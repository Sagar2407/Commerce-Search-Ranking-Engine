"""`csre bench`: systems benchmarks — the quality / latency / cost trade-offs behind the design choices.

1. Rerank depth: how many fused candidates should the LambdaMART reranker see? (quality vs latency)
2. ANN: HNSW efSearch vs exact search — recall of the exact top-100 and latency.
3. Catalog scale: a fixed-size *shard* (real catalog + phase-1 distractors) is built and timed for real; larger
   catalogs are served as N shards queried in parallel (scatter-gather), so tier latency is projected from the
   measured shard latency plus the measured merge cost, and cost / memory scale with the shard count.
   Measured and projected numbers are labelled as such in the report.

All timings are single-threaded per request (as in `csre eval`).
"""
from __future__ import annotations

import gc
import json
import math
import time

import numpy as np
import polars as pl

from ..config import Config
from ..search.bm25 import BM25Index
from ..search.dense import VectorIndex, encoder_text
from ..search.engine import Engine
from ..utils import get_logger, stage_timer
from . import metrics as M
from .harness import _limit_threads, load_judged

log = get_logger("csre.bench")


def _queries(eng: Engine, split: str, n: int, seed: int) -> tuple[pl.DataFrame, dict]:
    q, judged = load_judged(eng, split)
    return q.sample(n=min(n, q.height), seed=seed, shuffle=True), judged


def rerank_depth(cfg: Config, eng: Engine, n: int = 600, depths=(0, 10, 20, 50, 100, 200)) -> list[dict]:
    q, judged = _queries(eng, cfg.get("eval.split"), n, cfg.seed)
    for text, loc in zip(q["query"].head(30).to_list(), q["locale"].head(30).to_list()):   # warm-up
        eng.search(text, loc, "ltr", k=100)
    out = []
    for d in depths:
        method = "hybrid" if d == 0 else "ltr"
        nd, lat = [], []
        for qid, text, loc in zip(q["query_id"].to_list(), q["query"].to_list(), q["locale"].to_list()):
            rows, gains, labels = judged[qid]
            jd = {int(r): (float(g), str(l)) for r, g, l in zip(rows, gains, labels)}
            eng.parser.cache.pop((eng.rewrite(text, loc)[0], loc), None)   # cold query: pays parsing
            t = time.perf_counter()
            # same basis as `csre eval` retrieval: top 100 returned, condensed nDCG@10 over them
            resp = eng.search(text, loc, method, k=100, rerank_depth=max(d, 10),
                              n_candidates=max(100, d // 2 + 1))
            lat.append((time.perf_counter() - t) * 1e3)
            m = M.retrieval_metrics(resp.rows, jd, 10, 100)
            nd.append(np.nan if m["ndcg10_cond"] is None else m["ndcg10_cond"])
        out.append({"depth": d, "method": method, "ndcg10_cond": float(np.nanmean(nd)), **M.latency_summary(np.array(lat))})
        log.info("rerank depth %d: nDCG %.4f p50 %.1f ms", d, out[-1]["ndcg10_cond"], out[-1]["p50"])
    return out


def ann_sweep(cfg: Config, eng: Engine, n: int = 500, efs=(16, 32, 64, 128, 256)) -> list[dict]:
    loc = max(eng.idx, key=lambda l: eng.idx[l].n)
    vi = eng.idx[loc].vectors
    if vi is None or vi.ann is None:
        return []
    q = eng.ds.queries().filter((pl.col("locale") == loc) & (pl.col("split") == cfg.get("eval.split")))
    q = q.sample(n=min(n, q.height), seed=cfg.seed)
    qv = np.stack([eng.encoder.encode_query(t) for t in q["query"].to_list()])
    exact, lat_exact = [], []
    for v in qv:
        t = time.perf_counter()
        exact.append(set(vi.search(v, 100, exact=True)[0].tolist()))
        lat_exact.append((time.perf_counter() - t) * 1e3)
    out = [{"ef_search": "exact", "recall_at_100": 1.0, **M.latency_summary(np.array(lat_exact))}]
    for ef in efs:
        vi.ann.hnsw.efSearch = ef
        rec, lat = [], []
        for v, ex in zip(qv, exact):
            t = time.perf_counter()
            got = vi.search(v, 100)[0]
            lat.append((time.perf_counter() - t) * 1e3)
            rec.append(len(ex & set(got.tolist())) / max(1, len(ex)))
        out.append({"ef_search": ef, "recall_at_100": float(np.mean(rec)), **M.latency_summary(np.array(lat))})
        log.info("HNSW ef=%d recall@100=%.4f p50=%.2f ms", ef, out[-1]["recall_at_100"], out[-1]["p50"])
    vi.ann.hnsw.efSearch = vi.ef_search
    return out


def _merge_cost(k: int, n_shards: int, reps: int = 2000) -> float:
    rng = np.random.default_rng(0)
    lists = [(rng.integers(0, 1 << 30, 100), rng.random(100).astype(np.float32)) for _ in range(n_shards)]
    t = time.perf_counter()
    for _ in range(reps):
        ids = np.concatenate([a for a, _ in lists])
        sc = np.concatenate([b for _, b in lists])
        top = np.argpartition(-sc, k - 1)[:k]
        ids[top[np.argsort(-sc[top])]]
    return (time.perf_counter() - t) * 1e3 / reps


class _Light:
    """Just what the scale benchmark needs (dataset, locales, encoder), without loading every index."""

    def __init__(self, cfg: Config, corpus: str):
        from ..search.corpus import Dataset  # noqa: PLC0415
        from ..search.dense import load_encoder  # noqa: PLC0415
        from ..search.registry import ModelRegistry  # noqa: PLC0415
        self.ds = Dataset(cfg, corpus)
        self.idx = {loc: None for loc in self.ds.locales()}
        p = ModelRegistry(cfg).path("dense_encoder")
        self.encoder = load_encoder(p) if p else None


def _shard_docs(ds, eng, n: int, scale_dir) -> pl.DataFrame:
    real = pl.concat([ds.catalog(l, ["doc_id", "title", "doc_text", "n_title_chars"]) for l in eng.idx])
    if n <= real.height:
        return real.sample(n=n, seed=0)
    syn = pl.scan_parquet(str(scale_dir / "**" / "*.parquet")).select("doc_id", "title", "doc_text") \
            .head(n - real.height).collect() \
            .with_columns(pl.col("title").str.len_chars().cast(pl.UInt32).alias("n_title_chars"))
    return pl.concat([real, syn], how="vertical_relaxed")


def catalog_scale(cfg: Config, eng, sizes=(500_000, 1_000_000, 1_500_000), n_queries: int = 300) -> dict:
    """Build BM25 + dense/HNSW shards of increasing size (real products + phase-1 distractors), time each one,
    and project larger catalogs as parallel shards of the largest measured size."""
    ds = eng.ds
    scale_dir = cfg.path("scale", "synthetic_catalog")
    if not scale_dir.exists():
        return {"skipped": "no phase-1 scale tiers (run `csre scale`)"}
    B = cfg.get("search.bm25")
    an = cfg.get("search.dense.ann")
    q = ds.queries().filter(pl.col("split") == cfg.get("eval.split")).sample(n=n_queries, seed=cfg.seed)
    texts_q = q["query"].to_list()
    qv = [eng.encoder.encode_query(x) for x in texts_q] if eng.encoder is not None else []
    _limit_threads()
    measured = []
    for n in sizes:
        docs = _shard_docs(ds, eng, n, scale_dir)
        t = time.time()
        bm = BM25Index.build(docs["doc_text"].to_list(), B["k1"], B["b"], B["jp_ngram"], B["stem_plurals"])
        build = time.time() - t
        lat = []
        for x in texts_q:
            t = time.perf_counter()
            bm.search(x, 100)
            lat.append((time.perf_counter() - t) * 1e3)
        rec = {"docs": docs.height, "bm25": {"build_s": round(build, 1), "bytes": bm.nbytes(), "terms": len(bm.vocab),
                                             "latency_ms": M.latency_summary(np.array(lat))}}
        del bm
        gc.collect()
        if eng.encoder is not None:
            t = time.time()
            emb = eng.encoder.encode(encoder_text(docs))
            enc_s = time.time() - t
            t = time.time()
            vi = VectorIndex.build(emb, "hnsw", an["hnsw_m"], an["ef_construction"], an["ef_search"], 0)
            ann_s = time.time() - t
            lat = []
            for v in qv:
                t = time.perf_counter()
                vi.search(v, 100)
                lat.append((time.perf_counter() - t) * 1e3)
            rec["dense"] = {"encode_s": round(enc_s, 1), "encode_docs_per_s": int(docs.height / max(enc_s, 1e-9)),
                            "hnsw_build_s": round(ann_s, 1), "bytes": vi.nbytes(),
                            "latency_ms": M.latency_summary(np.array(lat))}
            del vi, emb
        del docs
        gc.collect()
        measured.append(rec)
        log.info("scale shard %d docs: bm25 p50 %.1f ms, dense p50 %s ms", rec["docs"], rec["bm25"]["latency_ms"]["p50"],
                 rec.get("dense", {}).get("latency_ms", {}).get("p50"))
    # linear trend of BM25 latency in catalog size (postings grow with the catalog)
    xs = np.array([m["docs"] for m in measured], float)
    ys = np.array([m["bm25"]["latency_ms"]["p50"] for m in measured])
    slope, intercept = np.polyfit(xs / 1e6, ys, 1) if len(xs) > 1 else (0.0, float(ys[0]))
    big = measured[-1]
    C = cfg.get("cost")
    tiers = []
    for t_docs in sorted({int(x) for x in [1_814_924, *cfg.get("scale.tiers")]}):
        n_sh = max(1, math.ceil(t_docs / big["docs"]))
        merge = _merge_cost(100, n_sh) if n_sh > 1 else 0.0
        bm_p50 = (intercept + slope * t_docs / 1e6) if n_sh == 1 else big["bm25"]["latency_ms"]["p50"]
        d = big.get("dense", {})
        cpu_ms = (big["bm25"]["latency_ms"]["mean"] + d.get("latency_ms", {}).get("mean", 0.0)) * n_sh
        tiers.append({
            "catalog_docs": t_docs, "shards": n_sh, "measured": any(abs(m["docs"] - t_docs) < 1 for m in measured),
            "bm25_p50_ms": round(bm_p50 + merge, 3),
            "dense_p50_ms": None if not d else round(d["latency_ms"]["p50"] + merge, 3),
            "merge_ms": round(merge, 4),
            "memory_gb": round(n_sh * (big["bm25"]["bytes"] + d.get("bytes", 0)) / 1e9, 2),
            "retrieval_cpu_ms_per_query": round(cpu_ms, 2),
            "usd_per_million_retrieval": round(M.cost_per_million(cpu_ms, int(C["vcpus"]), float(C["usd_per_hour"]),
                                                                  float(C["target_utilisation"]))["usd_per_million"], 4),
        })
    return {"measured": measured, "shard_docs": big["docs"], "tiers": tiers,
            "bm25_p50_ms_per_million_docs": round(float(slope), 3),
            "note": f"Shards of 0.5M-{big['docs'] / 1e6:.1f}M products are built and timed (single-threaded). Catalogs "
                    f"larger than one shard are served as {big['docs'] / 1e6:.1f}M-product shards searched in parallel: "
                    "latency = slowest shard + measured top-100 merge; CPU cost and memory scale with the shard count."}


def run_bench(cfg: Config, corpus: str | None = None, parts: tuple[str, ...] = ("rerank_depth", "ann", "scale")
              ) -> dict:
    with stage_timer(cfg, "bench") as info:
        eng = Engine(cfg, corpus or "full") if set(parts) - {"scale"} else _Light(cfg, corpus or "full")
        _limit_threads()
        p = cfg.path("reports", "bench.json")
        res = json.loads(p.read_text()) if p.exists() else {}      # partial re-runs keep the other sections
        res.update({"corpus": eng.ds.name, "versions": getattr(eng, "versions", res.get("versions")), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "serving_depth": int(cfg.get("search.ltr.rerank_depth", 100))})
        if "rerank_depth" in parts:
            res["rerank_depth"] = rerank_depth(cfg, eng)
        if "ann" in parts:
            res["ann"] = ann_sweep(cfg, eng)
        p.write_text(json.dumps(res, indent=2, default=str))
        if "scale" in parts:
            light = _Light(cfg, eng.ds.name)
            del eng                       # free the full indexes before building a 2.5M-product shard
            gc.collect()
            res["scale"] = catalog_scale(cfg, light)
            p.write_text(json.dumps(res, indent=2, default=str))
        info.update(rows=1)
    log.info("wrote %s", p)
    return res

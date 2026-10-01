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
    out = []
    for d in depths:
        method = "hybrid" if d == 0 else "ltr"
        nd, lat = [], []
        for qid, text, loc in zip(q["query_id"].to_list(), q["query"].to_list(), q["locale"].to_list()):
            rows, gains, labels = judged[qid]
            jd = {int(r): (float(g), str(l)) for r, g, l in zip(rows, gains, labels)}
            t = time.perf_counter()
            resp = eng.search(text, loc, method, k=10, rerank_depth=max(d, 10),
                              n_candidates=max(100, d // 2 + 1))
            lat.append((time.perf_counter() - t) * 1e3)
            m = M.retrieval_metrics(resp.rows, jd, 10, 10)
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


def catalog_scale(cfg: Config, eng: Engine, shard_docs: int = 2_500_000, n_queries: int = 300) -> dict:
    """Build one BM25 + dense shard of `shard_docs` products (real + distractors) and time it."""
    ds = eng.ds
    scale_dir = cfg.path("scale", "synthetic_catalog")
    if not scale_dir.exists():
        return {"skipped": "no phase-1 scale tiers (run `csre scale`)"}
    real = pl.concat([ds.catalog(l, ["doc_id", "locale", "title", "doc_text", "n_title_chars"]) for l in eng.idx])
    need = max(0, shard_docs - real.height)
    syn = pl.scan_parquet(str(scale_dir / "**" / "*.parquet")).select("doc_id", "locale", "title", "doc_text") \
            .head(need).collect()
    docs = pl.concat([real.select("doc_id", "locale", "title", "doc_text"), syn])
    log.info("scale shard: %d real + %d synthetic = %d products", real.height, syn.height, docs.height)
    B = cfg.get("search.bm25")
    t = time.time()
    bm = BM25Index.build(docs["doc_text"].to_list(), B["k1"], B["b"], B["jp_ngram"], B["stem_plurals"])
    bm_build = time.time() - t
    q = ds.queries().filter(pl.col("split") == cfg.get("eval.split")).sample(n=n_queries, seed=cfg.seed)
    _limit_threads()
    lat = []
    for text in q["query"].to_list():
        t = time.perf_counter()
        bm.search(text, 100)
        lat.append((time.perf_counter() - t) * 1e3)
    shard = {"docs": docs.height, "real_docs": real.height, "synthetic_docs": syn.height,
             "bm25": {"build_s": round(bm_build, 1), "bytes": bm.nbytes(), "terms": len(bm.vocab),
                      "latency_ms": M.latency_summary(np.array(lat))}}
    del bm
    gc.collect()
    if eng.encoder is not None:
        t = time.time()
        texts = pl.concat([real.select("doc_text", "n_title_chars"),
                           syn.select("doc_text", pl.col("title").str.len_chars().cast(pl.UInt32).alias("n_title_chars"))])
        emb = eng.encoder.encode(encoder_text(texts))
        enc_s = time.time() - t
        an = cfg.get("search.dense.ann")
        t = time.time()
        vi = VectorIndex.build(emb, "hnsw", an["hnsw_m"], an["ef_construction"], an["ef_search"], 0)
        ann_s = time.time() - t
        qv = [eng.encoder.encode_query(x) for x in q["query"].to_list()]
        lat = []
        for v in qv:
            t = time.perf_counter()
            vi.search(v, 100)
            lat.append((time.perf_counter() - t) * 1e3)
        shard["dense"] = {"encode_s": round(enc_s, 1), "encode_docs_per_s": int(docs.height / max(enc_s, 1e-9)),
                          "hnsw_build_s": round(ann_s, 1), "bytes": vi.nbytes(), "latency_ms": M.latency_summary(np.array(lat))}
        del vi, emb
        gc.collect()
    tiers = []
    C = cfg.get("cost")
    rerank_ms = None
    for t_docs in [real.height, *cfg.get("scale.tiers")]:
        n_sh = max(1, math.ceil(t_docs / shard_docs))
        merge = _merge_cost(100, n_sh) if n_sh > 1 else 0.0
        bm_p50 = shard["bm25"]["latency_ms"]["p50"] * min(1.0, t_docs / shard_docs) if n_sh == 1 else shard["bm25"]["latency_ms"]["p50"]
        d_p50 = shard.get("dense", {}).get("latency_ms", {}).get("p50")
        cpu_ms = shard["bm25"]["latency_ms"]["mean"] * n_sh + (shard.get("dense", {}).get("latency_ms", {}).get("mean", 0) * n_sh)
        tiers.append({
            "catalog_docs": int(t_docs), "shards": n_sh, "measured": t_docs <= shard_docs,
            "bm25_p50_ms": round(bm_p50 + merge, 3), "dense_p50_ms": None if d_p50 is None else round(d_p50 + merge, 3),
            "merge_ms": round(merge, 4),
            "memory_gb": round(n_sh * (shard["bm25"]["bytes"] + shard.get("dense", {}).get("bytes", 0)) / 1e9, 2),
            "retrieval_cpu_ms_per_query": round(cpu_ms, 2),
            "usd_per_million_retrieval": round(M.cost_per_million(cpu_ms, int(C["vcpus"]), float(C["usd_per_hour"]),
                                                                  float(C["target_utilisation"]))["usd_per_million"], 4),
        })
    return {"shard": shard, "tiers": tiers, "shard_docs": shard_docs,
            "note": "Shard numbers are measured. Tier latency = slowest shard (shards searched in parallel) + measured "
                    "top-k merge; CPU cost and memory scale with the number of shards. A 1-shard tier smaller than the "
                    "shard scales BM25 latency down linearly (postings shrink with the catalog)."}


def run_bench(cfg: Config, corpus: str | None = None) -> dict:
    with stage_timer(cfg, "bench") as info:
        eng = Engine(cfg, corpus or "full")
        _limit_threads()
        res = {"corpus": eng.ds.name, "versions": eng.versions, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "serving_depth": int(cfg.get("search.ltr.rerank_depth", 100)),
               "rerank_depth": rerank_depth(cfg, eng), "ann": ann_sweep(cfg, eng)}
        p = cfg.path("reports", "bench.json")
        p.write_text(json.dumps(res, indent=2, default=str))
        res["scale"] = catalog_scale(cfg, eng, int(cfg.get("bench.shard_docs", 2_500_000)))
        p.write_text(json.dumps(res, indent=2, default=str))
        info.update(rows=1)
    log.info("wrote %s", p)
    return res

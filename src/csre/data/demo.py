"""Portable demo subset: a small, self-contained slice of every dataset for the storefront demo,
notebooks and CI. Same schemas as the full data, so code written against it runs unchanged at scale."""
from __future__ import annotations

import shutil

import polars as pl

from ..config import Config
from ..utils import dir_size_bytes, duck, get_logger, stage_timer

log = get_logger("csre.demo")


def build_demo(cfg: Config) -> dict:
    """Full demo subset (data/demo) + a download-sized portable one (data/demo_portable)."""
    D = cfg.get("demo")
    info = _build(cfg, D, cfg.path("demo"), drop_columns=[], replay_rows=200_000, stage="demo")
    P = {**D, **D.get("portable", {})}
    _build(cfg, P, cfg.path("demo").with_name("demo_portable"), drop_columns=P.get("drop_columns", []),
           replay_rows=int(P.get("replay_rows", 50_000)), stage="demo_portable",
           relations=P.get("relations"))
    return info


def _build(cfg: Config, D: dict, out, drop_columns: list[str], replay_rows: int, stage: str,
           relations: list[str] | None = None) -> dict:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    con = duck(cfg)
    P = lambda *a: str(cfg.path("processed", *a))  # noqa: E731
    S = lambda *a: str(cfg.path("synthetic", *a))  # noqa: E731
    cat = f"read_parquet('{P('catalog')}/**/*.parquet', hive_partitioning=true)"
    with stage_timer(cfg, stage) as info:
        # stratified: keep every slice represented by sampling within (slice-signature) groups
        con.execute(f"""
            CREATE TEMP TABLE dq AS
            SELECT * FROM '{P('queries.parquet')}'
            WHERE locale = '{D['locale']}' AND split = '{D['split']}'
            QUALIFY row_number() OVER (
                PARTITION BY slice_ambiguous, slice_negation, slice_spec, slice_brand, slice_sparse_products
                ORDER BY hash(query_id + {cfg.seed})) <= greatest(5, {int(D['n_queries'])} * count(*) OVER (
                PARTITION BY slice_ambiguous, slice_negation, slice_spec, slice_brand, slice_sparse_products)
                / count(*) OVER ())
        """)
        con.execute(f"CREATE TEMP TABLE dj AS SELECT j.* FROM '{P('judgments.parquet')}' j SEMI JOIN dq USING (query_id)")
        con.execute(f"""
            CREATE TEMP TABLE dd AS
            SELECT DISTINCT doc_id FROM dj
            UNION
            SELECT doc_id FROM (SELECT doc_id FROM {cat} WHERE locale = '{D['locale']}'
                                ORDER BY hash(doc_id || '{cfg.seed}') LIMIT {int(D['extra_distractors'])})
        """)
        jobs = {
            "queries.parquet": "SELECT * FROM dq",
            "judgments.parquet": "SELECT * FROM dj",
            "catalog.parquet": f"SELECT c.* {('EXCLUDE (' + ', '.join(drop_columns) + ')') if drop_columns else ''} "
                               f"FROM {cat} c SEMI JOIN dd USING (doc_id)",
            "product_commerce.parquet": f"SELECT c.* FROM '{S('product_commerce.parquet')}' c SEMI JOIN dd USING (doc_id)",
            "product_edges.parquet": f"""SELECT e.* FROM '{P('graph', 'product_edges.parquet')}' e
                                         WHERE e.src_doc IN (SELECT doc_id FROM dd) AND e.dst_doc IN (SELECT doc_id FROM dd)
                                         {"AND e.relation IN (" + ", ".join(f"'{r}'" for r in relations) + ")" if relations else ""}""",
            "related_queries.parquet": f"SELECT r.* FROM '{P('graph', 'related_queries.parquet')}' r SEMI JOIN dq USING (query_id)",
            "query_doc_stats.parquet": f"SELECT s.* FROM '{S('logs', 'query_doc_stats.parquet')}' s SEMI JOIN dq USING (query_id)",
            "query_popularity.parquet": f"SELECT p.* FROM '{S('query_popularity.parquet')}' p SEMI JOIN dq USING (query_id)",
            "replay_sample.parquet": f"SELECT * FROM '{S('replay', 'requests.parquet')}' WHERE locale = '{D['locale']}' LIMIT {replay_rows}",
            "category_complements.parquet": f"SELECT * FROM '{P('graph', 'category_complements.parquet')}'",
        }
        if cfg.path("processed", "product_meta.parquet").exists():   # real ESCI-S display metadata (images)
            jobs["product_meta.parquet"] = f"SELECT m.* FROM '{P('product_meta.parquet')}' m SEMI JOIN dd USING (doc_id)"
        counts = {}
        for name, sql in jobs.items():
            con.execute(f"COPY ({sql}) TO '{out / name}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            counts[name] = con.sql(f"SELECT count(*) FROM '{out / name}'").fetchone()[0]
        bp = cfg.path("processed", "brand_patterns.json")
        if bp.exists():  # query brand dictionary, so online query parsing works on the subset alone
            shutil.copy(bp, out / "brand_patterns.json")
        info.update(rows=sum(counts.values()), files=counts, bytes=dir_size_bytes(out))
        log.info("demo subset: %s", counts)
    return info

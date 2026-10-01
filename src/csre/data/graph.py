"""Stage 4 — product and query graphs derived from co-judgments.

ESCI labels say how a product relates to a *query*. Two products judged for the same query
therefore inherit a relation to each other:

* Exact x Exact       -> `co_exact`   (interchangeable for that intent; used for dedup / diversity)
* Exact -> Substitute -> `substitute` ("similar item" rails, out-of-stock fallback)
* Exact -> Complement -> `complement` ("goes well with" rails)

Support = number of distinct queries producing the edge. We also aggregate complement edges to a
department-level matrix (which departments complement which), and link queries that share Exact
products (`related_queries`, used for reformulation simulation and "related searches").
"""
from __future__ import annotations

from ..config import Config
from ..utils import dir_size_bytes, duck, get_logger, stage_timer

log = get_logger("csre.graph")


def build_graph(cfg: Config, max_exact_per_query: int = 25, max_queries_per_doc: int = 50) -> dict:
    con = duck(cfg)
    j = cfg.path("processed", "judgments.parquet")
    cat = cfg.path("processed", "catalog")
    out = cfg.path("processed", "graph")
    out.mkdir(parents=True, exist_ok=True)
    with stage_timer(cfg, "graph") as info:
        con.execute(f"""
            CREATE TEMP TABLE jj AS
            SELECT query_id, doc_id, locale, esci_label,
                   row_number() OVER (PARTITION BY query_id, esci_label ORDER BY doc_id) AS rk
            FROM '{j}'
        """)
        con.execute(f"""
            COPY (
              WITH e AS (SELECT query_id, doc_id FROM jj WHERE esci_label='E' AND rk <= {max_exact_per_query}),
              o AS (SELECT query_id, doc_id, esci_label FROM jj WHERE esci_label IN ('S','C')),
              pairs AS (
                SELECT a.doc_id AS src_doc, b.doc_id AS dst_doc, 'co_exact' AS relation, a.query_id
                FROM e a JOIN e b ON a.query_id = b.query_id AND a.doc_id < b.doc_id
                UNION ALL
                SELECT e.doc_id, o.doc_id, CASE o.esci_label WHEN 'S' THEN 'substitute' ELSE 'complement' END, e.query_id
                FROM e JOIN o USING (query_id)
              )
              SELECT src_doc, dst_doc, relation, count(DISTINCT query_id)::INTEGER AS support,
                     min(query_id) AS example_query_id
              FROM pairs GROUP BY ALL
            ) TO '{out / "product_edges.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        edges = con.sql(f"""SELECT relation, count(*) n, avg(support) avg_support
                            FROM '{out / "product_edges.parquet"}' GROUP BY 1""").fetchall()
        log.info("product edges: %s", edges)

        con.execute(f"""
            COPY (
              WITH c AS (SELECT doc_id, category FROM read_parquet('{cat}/**/*.parquet', hive_partitioning=true)),
              ed AS (
                SELECT cs.category AS src_category, cd.category AS dst_category, sum(e.support) AS weight
                FROM '{out / "product_edges.parquet"}' e
                JOIN c cs ON cs.doc_id = e.src_doc JOIN c cd ON cd.doc_id = e.dst_doc
                WHERE e.relation = 'complement' AND cs.category <> 'Unknown' AND cd.category <> 'Unknown'
                GROUP BY ALL)
              SELECT *, weight / sum(weight) OVER (PARTITION BY src_category) AS share
              FROM ed ORDER BY src_category, share DESC
            ) TO '{out / "category_complements.parquet"}' (FORMAT PARQUET)
        """)

        con.execute(f"""
            COPY (
              WITH e AS (
                SELECT query_id, doc_id FROM (
                  SELECT query_id, doc_id,
                         row_number() OVER (PARTITION BY doc_id ORDER BY hash(query_id)) AS r
                  FROM jj WHERE esci_label = 'E') WHERE r <= {max_queries_per_doc}),
              ne AS (SELECT query_id, count(*) AS n_e FROM jj WHERE esci_label='E' GROUP BY 1),
              p AS (
                SELECT a.query_id AS query_id, b.query_id AS related_query_id, count(*) AS n_shared_exact
                FROM e a JOIN e b ON a.doc_id = b.doc_id AND a.query_id <> b.query_id
                GROUP BY ALL)
              SELECT p.*, p.n_shared_exact / (n1.n_e + n2.n_e - p.n_shared_exact) AS jaccard
              FROM p JOIN ne n1 ON n1.query_id = p.query_id JOIN ne n2 ON n2.query_id = p.related_query_id
              QUALIFY row_number() OVER (PARTITION BY p.query_id ORDER BY jaccard DESC, related_query_id) <= 20
            ) TO '{out / "related_queries.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n_rel = con.sql(f"SELECT count(*), count(DISTINCT query_id) FROM '{out / 'related_queries.parquet'}'").fetchone()
        info.update(rows=sum(e[1] for e in edges), edges={e[0]: e[1] for e in edges},
                    related_query_pairs=n_rel[0], queries_with_related=n_rel[1], bytes=dir_size_bytes(out))
    return info

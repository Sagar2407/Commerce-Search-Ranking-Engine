"""Data-quality gate: schema, integrity, leakage and simulator sanity checks.

Every check returns (name, passed, detail). `csre validate` exits non-zero if any check fails, so
the pipeline can run in CI before any model is trained on the data.
"""
from __future__ import annotations

import json

from ..config import Config
from ..utils import duck, get_logger

log = get_logger("csre.validate")

ESCI_PRODUCTS = 1_814_924
ESCI_JUDGMENTS = 2_621_288   # rows in the released examples file (the ESCI README quotes 2,621,738)


def run_checks(cfg: Config) -> list[dict]:
    con = duck(cfg)
    P = lambda *a: str(cfg.path("processed", *a))  # noqa: E731
    S = lambda *a: str(cfg.path("synthetic", *a))  # noqa: E731
    cat = f"read_parquet('{P('catalog')}/**/*.parquet', hive_partitioning=true)"
    jud = f"'{P('judgments.parquet')}'"
    qry = f"'{P('queries.parquet')}'"
    res: list[dict] = []

    def check(name: str, sql: str, cond, fmt=lambda r: r):
        try:
            r = con.sql(sql).fetchone()
            ok = bool(cond(r))
            res.append({"check": name, "passed": ok, "detail": fmt(r)})
        except Exception as e:  # missing stage output etc.
            res.append({"check": name, "passed": False, "detail": f"error: {e}"[:300]})

    # catalog
    check("catalog.row_count_matches_esci", f"SELECT count(*) FROM {cat}", lambda r: r[0] == ESCI_PRODUCTS,
          lambda r: {"rows": r[0], "expected": ESCI_PRODUCTS})
    check("catalog.doc_id_unique", f"SELECT count(*) - count(DISTINCT doc_id) FROM {cat}", lambda r: r[0] == 0,
          lambda r: {"duplicates": r[0]})
    check("catalog.titles_present", f"SELECT sum((title = '')::INT) FROM {cat}", lambda r: r[0] < 100,
          lambda r: {"empty_titles": r[0]})
    check("catalog.category_coverage", f"SELECT avg((category <> 'Unknown')::INT) FROM {cat}",
          lambda r: r[0] > 0.4, lambda r: {"known_category_share": round(r[0], 4)})

    # judgments / splits
    raw_ex = cfg.path("raw", "esci", "shopping_queries_dataset_examples.parquet")
    check("judgments.row_count_after_leak_fix",
          f"""SELECT (SELECT count(*) FROM {jud}),
                     (SELECT count(*) FROM '{raw_ex}' WHERE split = 'train' AND query_id IN (
                        SELECT query_id FROM '{raw_ex}' GROUP BY 1 HAVING count(DISTINCT split) > 1))""",
          lambda r: r[0] == ESCI_JUDGMENTS - r[1],
          lambda r: {"rows": r[0], "esci_rows": ESCI_JUDGMENTS, "dropped_train_rows_of_cross_split_queries": r[1]})
    check("judgments.pair_unique", f"SELECT count(*) - count(DISTINCT (query_id, doc_id)) FROM {jud}",
          lambda r: r[0] == 0, lambda r: {"duplicates": r[0]})
    check("judgments.docs_in_catalog", f"SELECT count(*) FROM {jud} j ANTI JOIN {cat} c USING (doc_id)",
          lambda r: r[0] == 0, lambda r: {"orphans": r[0]})
    check("splits.no_query_in_two_splits",
          f"SELECT count(*) FROM (SELECT query_id FROM {jud} GROUP BY 1 HAVING count(DISTINCT split) > 1)",
          lambda r: r[0] == 0, lambda r: {"leaking_queries": r[0]})
    check("splits.test_is_esci_test",
          f"SELECT sum((split='test') <> (esci_split='test'))::INT FROM {jud}", lambda r: r[0] == 0,
          lambda r: {"mismatches": r[0]})
    check("splits.dev_fraction",
          f"""SELECT min(f), max(f) FROM (SELECT locale, avg((split='dev')::INT) f FROM {qry}
              WHERE split IN ('train','dev') GROUP BY 1)""",
          lambda r: 0.08 <= r[0] and r[1] <= 0.12, lambda r: {"min_locale_dev_share": round(r[0], 4),
                                                               "max_locale_dev_share": round(r[1], 4)})
    check("queries.one_row_per_query",
          f"SELECT (SELECT count(*) FROM {qry}) - (SELECT count(DISTINCT query_id) FROM {jud})",
          lambda r: r[0] == 0, lambda r: {"diff": r[0]})
    check("queries.slices_non_degenerate",
          f"""SELECT min(x), max(x) FROM (SELECT unnest([avg(slice_ambiguous::INT), avg(slice_negation::INT),
              avg(slice_spec::INT), avg(slice_brand::INT), avg(slice_sparse_products::INT),
              avg(slice_multi_intent::INT), avg(slice_underspecified::INT)]) AS x FROM {qry})""",
          lambda r: r[0] > 0.005 and r[1] < 0.8, lambda r: {"min_rate": round(r[0], 4), "max_rate": round(r[1], 4)})

    # graph
    g = P("graph", "product_edges.parquet")
    check("graph.no_self_edges", f"SELECT count(*) FROM '{g}' WHERE src_doc = dst_doc", lambda r: r[0] == 0,
          lambda r: {"self_edges": r[0]})
    check("graph.edges_in_catalog",
          f"SELECT count(*) FROM (SELECT src_doc AS doc_id FROM '{g}' UNION SELECT dst_doc FROM '{g}') e ANTI JOIN {cat} c USING (doc_id)",
          lambda r: r[0] == 0, lambda r: {"orphans": r[0]})

    # commerce
    com = S("product_commerce.parquet")
    check("commerce.one_row_per_doc", f"SELECT count(*), count(DISTINCT doc_id) FROM '{com}'",
          lambda r: r[0] == r[1] == ESCI_PRODUCTS, lambda r: {"rows": r[0], "distinct": r[1]})
    check("commerce.valid_ranges",
          f"SELECT sum((price <= 0)::INT), sum((stars < 1 OR stars > 5)::INT) FROM '{com}'",
          lambda r: r[0] == 0 and (r[1] or 0) == 0, lambda r: {"bad_price": r[0], "bad_stars": r[1]})

    # logs
    logs = cfg.path("synthetic", "logs")
    se = f"read_parquet('{logs}/searches/**/*.parquet', hive_partitioning=true)"
    im = f"read_parquet('{logs}/impressions/**/*.parquet', hive_partitioning=true)"
    check("logs.search_id_unique", f"SELECT count(*) - count(DISTINCT search_id) FROM {se}", lambda r: r[0] == 0,
          lambda r: {"duplicates": r[0]})
    check("logs.funnel_consistent",
          f"SELECT sum((carted AND NOT clicked)::INT) + sum((purchased AND NOT carted)::INT) FROM {im}",
          lambda r: r[0] == 0, lambda r: {"violations": r[0]})
    check("logs.position_range", f"SELECT min(position), max(position) FROM {im}",
          lambda r: r[0] == 1 and r[1] <= int(cfg.get("simulation.traffic.page_size")),
          lambda r: {"min": r[0], "max": r[1]})
    check("logs.ctr_monotone_in_label",
          f"""SELECT list(ctr ORDER BY esci_label) FROM (
                SELECT j.esci_label, avg(i.clicked::INT) ctr FROM {im} i
                JOIN {se} s USING (search_id) JOIN {jud} j ON j.query_id = s.query_id AND j.doc_id = i.doc_id
                WHERE s.search_id < 1000000000 GROUP BY 1)""",
          lambda r: r[0][1] > r[0][3] > r[0][0] > r[0][2],     # list is sorted C, E, I, S; need E > S > C > I
          lambda r: dict(zip(["C", "E", "I", "S"], [round(x, 4) for x in r[0]])))
    check("logs.explore_share",
          f"SELECT avg((policy LIKE 'explore%')::INT) FROM {se}",
          lambda r: abs(r[0] - float(cfg.get("simulation.traffic.logging_policy.explore_fraction"))) < 0.01,
          lambda r: {"explore_share": round(r[0], 4)})

    # replay + scale
    rp = S("replay", "requests.parquet")
    check("replay.row_count", f"SELECT count(*) FROM '{rp}'",
          lambda r: r[0] == int(cfg.get("simulation.replay.n_requests")), lambda r: {"rows": r[0]})
    sc = cfg.path("scale", "synthetic_catalog")
    if sc.exists():
        scat = f"read_parquet('{sc}/**/*.parquet', hive_partitioning=true)"
        check("scale.ids_unique_and_disjoint",
              f"SELECT count(*) - count(DISTINCT doc_id), sum((doc_id NOT LIKE 'syn-%')::INT) FROM {scat}",
              lambda r: r[0] == 0 and r[1] == 0, lambda r: {"duplicates": r[0], "non_synthetic_ids": r[1]})

    out = cfg.path("reports", "validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=str))
    n_fail = sum(not r["passed"] for r in res)
    log.info("validation: %d checks, %d failed", len(res), n_fail)
    for r in res:
        log.info("  [%s] %s %s", "PASS" if r["passed"] else "FAIL", r["check"], r["detail"])
    return res

"""Data card: one JSON with every number that describes the generated data, plus a Markdown render.

The JSON is the single source for the README tables and the published data-card page.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from ..config import Config
from ..utils import dir_size_bytes, duck, get_logger

log = get_logger("csre.datacard")


def _rows(con, sql: str) -> list[dict]:
    rel = con.sql(sql)
    cols = rel.columns
    return [dict(zip(cols, r)) for r in rel.fetchall()]


LEDGER = [
    # (key, relative path under data/, provenance, purpose)
    ("catalog", "processed/catalog", "real+derived", "Cleaned products, extracted attributes, sparsity flags, department, retrieval text"),
    ("judgments", "processed/judgments.parquet", "real", "ESCI Exact / Substitute / Complement / Irrelevant labels with gains and splits"),
    ("queries", "processed/queries.parquet", "real+derived", "Query text, parsed attributes, label profile, intent diversity, slice flags"),
    ("product_edges", "processed/graph/product_edges.parquet", "derived", "Substitute, complement and co-exact product relations"),
    ("related_queries", "processed/graph/related_queries.parquet", "derived", "Queries sharing Exact products"),
    ("category_complements", "processed/graph/category_complements.parquet", "derived", "Which departments complement which"),
    ("product_commerce", "synthetic/product_commerce.parquet", "simulated", "Price, rating, reviews, stock (real where ESCI-S has it)"),
    ("query_popularity", "synthetic/query_popularity.parquet", "simulated", "Zipf traffic share and head / torso / tail bucket"),
    ("searches", "synthetic/logs/searches", "simulated", "Sessions, users, time, logging policy, outcomes"),
    ("impressions", "synthetic/logs/impressions", "simulated", "Position, click, cart, purchase, dwell, feedback, propensity"),
    ("query_doc_stats", "synthetic/logs/query_doc_stats.parquet", "simulated", "Aggregated and IPS-weighted feedback per (query, product)"),
    ("replay", "synthetic/replay/requests.parquet", "simulated", "24h load-test stream incl. one-off query variants"),
    ("scale_catalog", "scale/synthetic_catalog", "simulated", "Distractor products for 2.5M-25M doc latency tiers"),
    ("demo", "demo", "subset", "Portable slice of every dataset for the storefront demo"),
]


def _parquet_rows(path: Path) -> int:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def _ledger(cfg: Config) -> list[dict]:
    out = []
    for key, rel, prov, purpose in LEDGER:
        p = cfg.root / "data" / rel
        if not p.exists():
            continue
        rows = None
        if key != "demo":
            rows = _parquet_rows(p)
        out.append({"dataset": key, "path": f"data/{rel}", "provenance": prov, "rows": rows,
                    "bytes": dir_size_bytes(p), "purpose": purpose})
    return out


def build_datacard(cfg: Config) -> dict:
    con = duck(cfg)
    P = lambda *a: str(cfg.path("processed", *a))  # noqa: E731
    S = lambda *a: str(cfg.path("synthetic", *a))  # noqa: E731
    cat = f"read_parquet('{P('catalog')}/**/*.parquet', hive_partitioning=true)"
    jud = f"'{P('judgments.parquet')}'"
    qry = f"'{P('queries.parquet')}'"
    logs = cfg.path("synthetic", "logs")
    se = f"read_parquet('{logs}/searches/**/*.parquet', hive_partitioning=true)"
    im = f"read_parquet('{logs}/impressions/**/*.parquet', hive_partitioning=true)"
    card: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "ledger": _ledger(cfg)}

    card["catalog"] = {
        "by_locale": _rows(con, f"""
            SELECT locale, count(*) AS products,
                   round(avg(is_sparse::INT), 4) AS sparse_share,
                   round(avg(is_title_only::INT), 4) AS title_only_share,
                   round(avg((brand_norm IS NOT NULL)::INT), 4) AS brand_share,
                   round(avg((len(attr_measures) > 0)::INT), 4) AS has_measure,
                   round(avg((attr_pack_count IS NOT NULL)::INT), 4) AS has_pack_count,
                   round(avg((len(attr_audience) > 0)::INT), 4) AS has_audience,
                   round(avg((len(attr_materials) > 0)::INT), 4) AS has_material,
                   round(avg((len(attr_colors) > 0)::INT), 4) AS has_color,
                   round(avg((category <> 'Unknown')::INT), 4) AS known_category_share,
                   round(median(n_title_chars)) AS median_title_chars,
                   round(median(richness_chars)) AS median_desc_bullet_chars
            FROM {cat} GROUP BY 1 ORDER BY products DESC"""),
        "categories": _rows(con, f"""SELECT category, count(*) AS products,
                                     round(count(*) / sum(count(*)) OVER (), 4) AS share
                                     FROM {cat} GROUP BY 1 ORDER BY 2 DESC"""),
    }
    mpath = cfg.path("reports", "category_model_metrics.json")
    if mpath.exists():
        m = json.loads(mpath.read_text())
        card["category_model"] = {k: m[k] for k in ("n_seed_labels", "n_weak_labels", "accuracy", "macro_f1",
                                                     "threshold", "accuracy_at_threshold", "coverage_at_threshold",
                                                     "by_locale", "coverage_accuracy_curve")}

    card["judgments"] = {
        "by_split_locale": _rows(con, f"""
            SELECT split, locale, count(DISTINCT query_id) AS queries, count(*) AS judgments,
                   round(count(*) / count(DISTINCT query_id), 1) AS avg_candidates,
                   round(avg((esci_label='E')::INT), 4) AS E, round(avg((esci_label='S')::INT), 4) AS S,
                   round(avg((esci_label='C')::INT), 4) AS C, round(avg((esci_label='I')::INT), 4) AS I
            FROM {jud} GROUP BY ALL ORDER BY split, locale"""),
        "totals": _rows(con, f"SELECT count(*) AS judgments, count(DISTINCT query_id) AS queries, count(DISTINCT doc_id) AS judged_docs FROM {jud}")[0],
    }
    slice_cols = ["slice_ambiguous", "slice_underspecified", "slice_multi_intent", "slice_negation",
                  "slice_spec", "slice_brand", "slice_sparse_products", "slice_hard", "slice_all_exact"]
    card["query_slices"] = {
        "by_locale": _rows(con, f"""SELECT locale, count(*) AS queries,
            {", ".join(f"round(avg({c}::INT), 4) AS {c.replace('slice_', '')}" for c in slice_cols)}
            FROM {qry} GROUP BY 1 ORDER BY 2 DESC"""),
        "test_counts": _rows(con, f"""SELECT locale, count(*) AS queries,
            {", ".join(f"sum({c}::INT) AS {c.replace('slice_', '')}" for c in slice_cols)}
            FROM {qry} WHERE split = 'test' GROUP BY 1 ORDER BY 2 DESC"""),
        "length_buckets": _rows(con, f"SELECT locale, length_bucket, count(*) AS queries FROM {qry} GROUP BY ALL ORDER BY ALL"),
        "sources": _rows(con, f"SELECT source, count(*) AS queries FROM {qry} GROUP BY 1 ORDER BY 2 DESC"),
        "examples": _rows(con, f"""
            SELECT slice, list(query ORDER BY query_id)[1:6] AS examples FROM (
              SELECT query, query_id, unnest(['ambiguous_multi_intent','underspecified','negation','spec','brand','sparse_products']) AS slice,
                     unnest([slice_multi_intent, slice_underspecified, slice_negation, slice_spec, slice_brand, slice_sparse_products]) AS f
              FROM {qry} WHERE locale = 'us' AND split = 'test' AND hash(query_id) % 97 = 0) WHERE f GROUP BY 1"""),
    }
    g = P("graph", "product_edges.parquet")
    card["graph"] = {
        "edges": _rows(con, f"SELECT relation, count(*) AS edges, round(avg(support), 3) AS avg_support, max(support) AS max_support FROM '{g}' GROUP BY 1 ORDER BY 2 DESC"),
        "related_queries": _rows(con, f"SELECT count(*) AS pairs, count(DISTINCT query_id) AS queries FROM '{P('graph', 'related_queries.parquet')}'")[0],
        "top_category_complements": _rows(con, f"""
            SELECT src_category, dst_category, round(share, 3) AS share FROM '{P('graph', 'category_complements.parquet')}'
            WHERE src_category <> dst_category QUALIFY row_number() OVER (PARTITION BY src_category ORDER BY share DESC) = 1
            ORDER BY src_category"""),
    }
    com = S("product_commerce.parquet")
    card["commerce"] = _rows(con, f"""
        SELECT locale, any_value(currency) AS currency, round(quantile_cont(price, 0.1), 2) AS price_p10,
               round(median(price), 2) AS price_median, round(quantile_cont(price, 0.9), 2) AS price_p90,
               round(avg(stars), 2) AS avg_stars, round(median(n_ratings)) AS median_ratings,
               round(avg(in_stock::INT), 4) AS in_stock_share, sum((commerce_source = 'esci_s')::INT) AS real_rows
        FROM '{com}' GROUP BY 1 ORDER BY 1""")

    card["traffic"] = {
        "totals": _rows(con, f"""SELECT count(*) AS searches, count(DISTINCT session_id) AS sessions,
                                 count(DISTINCT user_id) AS users, count(DISTINCT query_id) AS queries_with_traffic,
                                 min(day) AS first_day, max(day) AS last_day,
                                 round(avg(abandoned::INT), 4) AS abandonment_rate,
                                 round(avg((seq_in_session > 0)::INT), 4) AS reformulation_share,
                                 round(avg((n_clicks > 0)::INT), 4) AS search_ctr,
                                 round(avg((n_carts > 0)::INT), 4) AS search_add_to_cart_rate,
                                 round(avg((n_purchases > 0)::INT), 4) AS search_conversion_rate
                                 FROM {se}""")[0],
        "impressions": _rows(con, f"SELECT count(*) AS impressions, round(avg(clicked::INT), 4) AS ctr, sum((feedback IS NOT NULL)::INT) AS explicit_feedback FROM {im}")[0],
        "ctr_by_position": _rows(con, f"""SELECT position, round(avg(clicked::INT), 4) AS ctr FROM {im}
                                          WHERE search_id < 1000000000 GROUP BY 1 ORDER BY 1"""),
        "ctr_by_label": _rows(con, f"""SELECT j.esci_label, round(avg(i.clicked::INT), 4) AS ctr,
                                       round(avg(i.carted::INT), 5) AS cart_rate, round(avg(i.purchased::INT), 5) AS purchase_rate
                                       FROM {im} i JOIN {se} s USING (search_id)
                                       JOIN {jud} j ON j.query_id = s.query_id AND j.doc_id = i.doc_id
                                       WHERE s.search_id < 2000000000 GROUP BY 1 ORDER BY 1"""),
        "daily": _rows(con, f"SELECT day, count(*) AS searches FROM {se} GROUP BY 1 ORDER BY 1"),
        "by_hour_utc": _rows(con, f"SELECT hour(ts_utc) AS hour_utc, locale, count(*) AS searches FROM {se} WHERE search_id < 4000000000 GROUP BY ALL ORDER BY ALL"),
    }
    pop = S("query_popularity.parquet")
    card["traffic"]["popularity_buckets"] = _rows(con, f"""
        SELECT traffic_bucket, count(*) AS queries, round(sum(traffic_share), 4) AS traffic_share,
               sum((split='test')::INT) AS test_queries
        FROM '{pop}' GROUP BY 1 ORDER BY 3 DESC""")
    card["traffic"]["searches_per_query_quantiles"] = _rows(con, f"""
        SELECT quantile_cont(n, [0.1, 0.5, 0.9, 0.99]) AS q, max(n) AS max FROM (SELECT query_id, count(*) n FROM {se} GROUP BY 1)""")[0]

    rp = S("replay", "requests.parquet")
    card["replay"] = _rows(con, f"""SELECT count(*) AS requests, count(DISTINCT query_id) AS unique_queries,
                                    round(1 - count(DISTINCT query_id) / count(*), 4) AS ideal_cache_hit_rate,
                                    min(ts_utc) AS start, max(ts_utc) AS end FROM '{rp}'""")[0]
    card["replay"]["variants"] = _rows(con, f"""SELECT variant, count(*) AS requests FROM '{rp}' GROUP BY 1 ORDER BY 2 DESC""")
    card["replay"]["hit_rate_raw_vs_normalised"] = _rows(con, f"""
        SELECT round(1 - count(DISTINCT query) / count(*), 4) AS raw_text,
               round(1 - count(DISTINCT lower(regexp_replace(trim(query), '\\s+', ' ', 'g'))) / count(*), 4) AS normalised
        FROM '{rp}'""")[0]
    card["replay"]["variant_examples"] = _rows(con, f"""
        SELECT r.variant, q.query AS original, r.query AS typed FROM '{rp}' r JOIN {qry} q USING (query_id)
        WHERE r.is_novel AND r.locale = 'us' QUALIFY row_number() OVER (PARTITION BY r.variant ORDER BY r.request_id) = 1""")
    card["replay"]["requests_per_second_by_minute"] = _rows(con, f"""
        SELECT round(min(n)/60.0, 1) AS min_rps, round(median(n)/60.0, 1) AS median_rps, round(max(n)/60.0, 1) AS peak_rps
        FROM (SELECT date_trunc('minute', ts_utc) m, count(*) n FROM '{rp}' GROUP BY 1)""")[0]
    tiers = cfg.path("scale", "tiers.json")
    if tiers.exists():
        card["scale"] = json.loads(tiers.read_text())

    card["footprint_bytes"] = {
        "raw": dir_size_bytes(cfg.path("raw")),
        "processed": dir_size_bytes(cfg.path("processed")),
        "synthetic": dir_size_bytes(cfg.path("synthetic")),
        "scale": dir_size_bytes(cfg.path("scale")) if cfg.path("scale").exists() else 0,
    }
    man = cfg.path("reports", "manifest.json")
    if man.exists():
        card["stage_runs"] = json.loads(man.read_text())
    val = cfg.path("reports", "validation.json")
    if val.exists():
        v = json.loads(val.read_text())
        card["validation"] = {"checks": len(v), "failed": sum(not x["passed"] for x in v), "results": v}

    out = cfg.path("reports", "datacard.json")
    out.write_text(json.dumps(card, indent=2, default=str))
    _markdown(card, cfg.path("reports", "datacard.md"))
    render_html(cfg, card)
    log.info("data card written: %s", out)
    return card


def _table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    cols = cols or list(rows[0])
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def _markdown(card: dict, path: Path) -> None:
    gb = lambda b: f"{b / 1e9:.2f} GB"  # noqa: E731
    md = ["# Data card — Commerce Search & Ranking Engine\n"]
    md.append("## Catalog\n" + _table(card["catalog"]["by_locale"]))
    if "category_model" in card:
        cm = card["category_model"]
        md.append(f"\nCategory model: CV accuracy {cm['accuracy']:.3f}, macro-F1 {cm['macro_f1']:.3f}; "
                  f"at confidence >= {cm['threshold']}: accuracy {cm['accuracy_at_threshold']:.3f} on "
                  f"{cm['coverage_at_threshold']:.1%} of products (rest = Unknown). "
                  f"{cm['n_seed_labels']} real labels + {cm['n_weak_labels']} propagated weak labels.\n")
    md.append("\n## Judgments\n" + _table(card["judgments"]["by_split_locale"]))
    md.append("\n## Query slices (share of queries)\n" + _table(card["query_slices"]["by_locale"]))
    md.append("\n## Product graph\n" + _table(card["graph"]["edges"]))
    md.append("\n## Simulated commerce attributes\n" + _table(card["commerce"]))
    md.append("\n## Simulated traffic\n" + _table([card["traffic"]["totals"]]) + "\n" + _table([card["traffic"]["impressions"]]))
    md.append("\n" + _table(card["traffic"]["popularity_buckets"]))
    md.append("\n## Replay stream\n" + _table([{k: v for k, v in card["replay"].items() if not isinstance(v, dict)}]))
    if "scale" in card:
        md.append("\n## Scale tiers\n" + _table(card["scale"]["tiers"]))
    md.append("\n## Footprint\n" + _table([{k: gb(v) for k, v in card["footprint_bytes"].items()}]))
    if "validation" in card:
        md.append(f"\n## Validation\n{card['validation']['checks']} checks, {card['validation']['failed']} failed.\n")
    path.write_text("\n".join(md))


def render_html(cfg: Config, card: dict | None = None) -> Path:
    """Inject the data card JSON into the self-contained HTML template (no external data)."""
    card = card or json.loads(cfg.path("reports", "datacard.json").read_text())
    slim = {k: v for k, v in card.items() if k not in ("stage_runs",)}
    slim["stage_runs"] = {k: {kk: v.get(kk) for kk in ("rows", "seconds", "rows_per_sec", "bytes")}
                          for k, v in card.get("stage_runs", {}).items()}
    tpl = (Path(__file__).parent / "datacard_template.html").read_text()
    html = tpl.replace("/*__DATACARD__*/null", json.dumps(slim, default=str).replace("</", "<\\/"))
    out = cfg.path("reports", "datacard.html")
    out.write_text(html)
    return out

"""Stage 3 — relevance judgments, query table, splits and evaluation slices.

Outputs
-------
processed/judgments.parquet   one row per (query, product) ESCI judgment, with gain / grade / split
processed/queries.parquet     one row per query: text, locale, split, parsed attributes, label profile,
                              intent-diversity stats over its Exact products, and boolean slice flags
processed/qrels/<split>/<locale>.qrels   TREC-format qrels (grades 3/2/1/0) for standard tooling

Design notes
------------
* ESCI `test` is kept untouched as the held-out test set; `dev` is a 10% query-level carve-out of
  ESCI `train` (deterministic MD5 hash of query_id, so it never changes between runs).
* Relevance is only known inside each query's judged candidate set. Metrics computed on full-catalog
  retrieval must treat unjudged products as *unknown*, not irrelevant (see `judged_at_k` in eval).
"""
from __future__ import annotations

import math
import shutil
from collections import Counter
from itertools import combinations

import numpy as np
import polars as pl

from ..config import Config
from ..utils import duck, get_logger, stable_hash64, stage_timer, uniform_from_hash, write_parquet
from . import text as T

log = get_logger("csre.judgments")


# --------------------------------------------------------------------------------------
# judgments
# --------------------------------------------------------------------------------------
def _judgments(cfg: Config) -> pl.DataFrame:
    con = duck(cfg)
    ex = cfg.path("raw", "esci", "shopping_queries_dataset_examples.parquet")
    j = con.sql(f"""
        WITH q AS (  -- one query_id appears in both ESCI splits: pin it to test to avoid leakage
            SELECT query_id, CASE WHEN count(DISTINCT split) > 1 THEN 'test' ELSE any_value(split) END AS esci_split
            FROM '{ex}' GROUP BY 1)
        SELECT e.example_id, e.query_id, e.query, e.product_locale AS locale, e.product_id,
               e.product_locale || ':' || e.product_id AS doc_id, e.esci_label,
               q.esci_split, e.small_version = 1 AS is_hard
        FROM '{ex}' e JOIN q USING (query_id)
        WHERE NOT (e.split = 'train' AND q.esci_split = 'test')
    """).pl()
    gains, grades = cfg.get("relevance.gains"), cfg.get("relevance.grades")
    qids = j["query_id"].unique().sort()
    u = uniform_from_hash(stable_hash64((str(q) for q in qids.to_list()), salt=f"dev-split-{cfg.seed}"))
    dev_ids = set(qids.filter(pl.Series(u < float(cfg.get("splits.dev_fraction")))).to_list())
    return j.with_columns(
        pl.col("esci_label").replace_strict(gains, return_dtype=pl.Float32).alias("gain"),
        pl.col("esci_label").replace_strict(grades, return_dtype=pl.Int8).alias("grade"),
        pl.when(pl.col("esci_split") == "test").then(pl.lit("test"))
        .when(pl.col("query_id").is_in(list(dev_ids))).then(pl.lit("dev"))
        .otherwise(pl.lit("train")).alias("split"),
    )


# --------------------------------------------------------------------------------------
# brand vocabulary for query parsing
# --------------------------------------------------------------------------------------
def _brand_patterns(cfg: Config) -> dict[str, list[str]]:
    """Brands specific enough to be recognised in queries.

    A brand is kept if it names >= brand_min_freq products and is not a generic word, measured as
    brand-specificity = (#products carrying the brand) / (#titles mentioning the string) >= 0.05.
    """
    min_freq = int(cfg.get("slices.brand_min_freq", 15))
    brands = pl.read_parquet(cfg.path("processed", "brands.parquet")).filter(
        (pl.col("n_products") >= min_freq)
        & (pl.col("brand_norm").str.len_chars() >= pl.when(pl.col("locale") == "jp").then(2).otherwise(3))
        & ~pl.col("brand_norm").str.contains(r"^\d+$")
    )
    con = duck(cfg)
    out: dict[str, list[str]] = {}
    for loc in ["us", "es", "jp"]:
        b = brands.filter(pl.col("locale") == loc)
        pats = b["brand_norm"].to_list()
        pad = loc != "jp"
        titles = con.sql(f"""SELECT title FROM read_parquet('{cfg.path('processed', 'catalog')}/locale={loc}/*.parquet')""").pl()
        tm = titles.select(T.for_matching(pl.col("title")).alias("t"))
        if pad:
            tm = tm.with_columns((pl.lit(" ") + pl.col("t").str.replace_all(r"[^\w\s]", " ") + pl.lit(" ")).alias("t"))
            search = [f" {p} " for p in pats]
        else:
            search = pats
        hits = tm.select(pl.col("t").str.extract_many(search, overlapping=True).list.unique().alias("h")) \
                 .explode("h").drop_nulls().group_by("h").len()
        mention = dict(zip(hits["h"].to_list(), hits["len"].to_list()))
        keep = []
        for p, s, n in zip(pats, search, b["n_products"].to_list()):
            m = mention.get(s, 0)
            if m == 0 or n / m >= 0.05:
                keep.append(p)
        out[loc] = keep
        log.info("brand patterns %s: %d of %d kept", loc, len(keep), len(pats))
    return out


# --------------------------------------------------------------------------------------
# query table
# --------------------------------------------------------------------------------------
def _entropy_bits(counts: list[float]) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0)


def _token_set(title: str, locale: str) -> frozenset:
    t = title.lower()
    if locale == "jp":
        t = t.replace(" ", "")
        return frozenset(t[i:i + 2] for i in range(len(t) - 1))
    return frozenset(w for w in t.split() if len(w) > 1)


def _intent_stats(ej: pl.DataFrame, max_items: int = 15) -> pl.DataFrame:
    """Per query: category entropy / share and lexical dispersion over its Exact products."""
    rows = []
    for (qid,), g in ej.group_by(["query_id"], maintain_order=False):
        cats = [c for c in g["category"].to_list() if c and c != "Unknown"]
        cnt = Counter(cats)
        top = cnt.most_common(2)
        n_known = len(cats)
        titles = g["title"].to_list()[:max_items]
        loc = g["locale"][0]
        sets = [_token_set(t, loc) for t in titles]
        sims = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if (a | b)]
        rows.append({
            "query_id": qid,
            "e_n_known_cat": n_known,
            "e_n_categories": len(cnt),
            "e_top_category": top[0][0] if top else None,
            "e_top_category_share": top[0][1] / n_known if top else None,
            "e_second_category_share": top[1][1] / n_known if len(top) > 1 else 0.0,
            "e_category_entropy": _entropy_bits(list(cnt.values())),
            "e_lexical_dispersion": 1.0 - float(np.mean(sims)) if sims else None,
            "e_brand_diversity": g["brand_norm"].drop_nulls().n_unique() / max(len(g), 1),
            "e_frac_sparse": float(g["is_sparse"].mean()),
            "e_frac_title_only": float(g["is_title_only"].mean()),
        })
    return pl.DataFrame(rows)


def _query_table(cfg: Config, j: pl.DataFrame) -> pl.DataFrame:
    con = duck(cfg)
    src = cfg.path("raw", "esci", "shopping_queries_dataset_sources.csv")
    sources = con.sql(f"SELECT query_id, source FROM read_csv('{src}')").pl()

    q = (
        j.group_by("query_id")
        .agg(
            pl.col("query").first(), pl.col("locale").first(), pl.col("split").first(),
            pl.col("esci_split").first(), pl.col("is_hard").first(),
            pl.len().alias("n_judged"),
            *[(pl.col("esci_label") == lab).sum().alias(f"n_{lab}") for lab in "ESCI"],
        )
        .join(sources, on="query_id", how="left")
        .with_columns(
            pl.col("source").fill_null("other"),
            (pl.col("n_E") / pl.col("n_judged")).alias("frac_E"),
            T.for_matching(pl.col("query")).alias("query_norm"),
        )
        .with_columns(
            pl.col("query_norm").str.split(" ").list.len().alias("n_tokens"),
            pl.col("query_norm").str.replace_all(" ", "").str.len_chars().alias("n_chars"),
            *T.query_attribute_exprs(pl.col("query_norm")),
        )
    )

    # brand detection (Aho-Corasick via polars extract_many)
    pats = _brand_patterns(cfg)
    parts = []
    for loc, g in q.group_by("locale"):
        loc = loc[0]
        p = pats.get(loc, [])
        if not p:
            parts.append(g.with_columns(pl.lit([], dtype=pl.List(pl.Utf8)).alias("q_brands")))
            continue
        if loc == "jp":
            hay, needles = pl.col("query_norm"), p
        else:
            hay = pl.lit(" ") + pl.col("query_norm").str.replace_all(r"[^\w\s]", " ") + pl.lit(" ")
            needles = [f" {b} " for b in p]
        parts.append(g.with_columns(
            hay.str.extract_many(needles, overlapping=True).list.eval(pl.element().str.strip_chars())
            .list.unique(maintain_order=True).alias("q_brands")))
    q = pl.concat(parts, how="vertical_relaxed")

    # intent diversity over Exact products
    cat = cfg.path("processed", "catalog")
    con.register("ej_keys", j.filter(pl.col("esci_label") == "E").select("query_id", "doc_id", "locale").to_arrow())
    ej = con.sql(f"""
        SELECT k.query_id, k.locale, c.title, c.category, c.brand_norm, c.is_sparse, c.is_title_only
        FROM ej_keys k JOIN read_parquet('{cat}/**/*.parquet', hive_partitioning=true) c USING (doc_id)
    """).pl()
    stats = _intent_stats(ej)
    q = q.join(stats, on="query_id", how="left")

    s = cfg.get("slices")
    attr_cols = ["q_measures", "q_dimensions", "q_sizes", "q_colors", "q_audience", "q_materials", "q_compat"]
    has_attr = pl.any_horizontal([pl.col(c).list.len() > 0 for c in attr_cols]) | pl.col("q_pack_count").is_not_null()
    short = pl.when(pl.col("locale") == "jp").then(pl.col("n_chars") <= s["underspecified_max_chars_jp"]) \
              .otherwise(pl.col("n_tokens") <= s["underspecified_max_tokens"])
    q = q.with_columns(
        has_attr.alias("q_has_attribute"),
        (pl.col("q_brands").list.len() > 0).alias("q_has_brand"),
    ).with_columns(
        (short & ~pl.col("q_has_digit") & ~pl.col("q_has_attribute") & ~pl.col("q_has_brand")).alias("slice_underspecified"),
        ((pl.col("e_n_known_cat") >= 3) & (pl.col("e_category_entropy") >= s["multi_intent_min_entropy"])
         & (pl.col("e_second_category_share") >= s["multi_intent_min_share"])).fill_null(False).alias("slice_multi_intent"),
        ((pl.col("source") == "negations") | pl.col("q_has_negation")).alias("slice_negation"),
        # the query states a spec: a measure (12 oz), dimension (16x25x5), size (queen, size 8) or pack count
        (pl.col("q_measures").list.len() + pl.col("q_dimensions").list.len() + pl.col("q_sizes").list.len() > 0
         ).or_(pl.col("q_pack_count").is_not_null()).alias("slice_spec"),
        pl.col("q_has_brand").alias("slice_brand"),
        (pl.col("e_frac_sparse") >= s["sparse_query_min_share"]).fill_null(False).alias("slice_sparse_products"),
        pl.col("is_hard").alias("slice_hard"),
        (pl.col("n_E") == pl.col("n_judged")).alias("slice_all_exact"),
        pl.when(pl.col("locale") == "jp")
        .then(pl.when(pl.col("n_chars") <= 6).then(pl.lit("short")).when(pl.col("n_chars") <= 14).then(pl.lit("medium"))
              .otherwise(pl.lit("long")))
        .otherwise(pl.when(pl.col("n_tokens") <= 2).then(pl.lit("short")).when(pl.col("n_tokens") <= 5)
                   .then(pl.lit("medium")).otherwise(pl.lit("long")))
        .alias("length_bucket"),
    ).with_columns(
        (pl.col("slice_underspecified") | pl.col("slice_multi_intent")).alias("slice_ambiguous"),
    )
    return q.sort("query_id")


def _write_qrels(cfg: Config, j: pl.DataFrame) -> None:
    root = cfg.path("processed", "qrels")
    shutil.rmtree(root, ignore_errors=True)
    for (split, loc), g in j.group_by(["split", "locale"]):
        p = root / split / f"{loc}.qrels"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.sort("query_id", "doc_id").select(
            pl.col("query_id").cast(pl.Utf8), pl.lit("0"), pl.col("doc_id"), pl.col("grade").cast(pl.Utf8)
        ).write_csv(p, separator=" ", include_header=False)


def build_judgments(cfg: Config) -> dict:
    with stage_timer(cfg, "judgments") as info:
        j = _judgments(cfg)
        write_parquet(j.drop("query"), cfg.path("processed", "judgments.parquet"))
        q = _query_table(cfg, j)
        write_parquet(q, cfg.path("processed", "queries.parquet"))
        _write_qrels(cfg, j)
        info.update(rows=len(j), queries=len(q),
                    split_queries=dict(q.group_by("split").len().iter_rows()),
                    slice_rates={c: round(float(q[c].mean()), 4) for c in q.columns if c.startswith("slice_")})
    return info

"""Stage 2 — clean product catalog.

Streams the 1.8M-product ESCI catalog in fixed-size batches (constant memory, so the same
code handles a 10x larger catalog) and writes a locale-partitioned Parquet dataset with:

* cleaned title / description / bullets / brand / colour
* extracted attributes (measures, dimensions, pack count, sizes, audience, compatibility,
  materials, colours) used for ranking features and grounded result explanations
* text-richness stats and the "sparse description" flag used as an evaluation slice
* a department + confidence from the category model ("Unknown" when unsure)
* `doc_text`, a single retrieval field for BM25 / embedding models
"""
from __future__ import annotations

import shutil

import polars as pl
import pyarrow.parquet as pq

from ..config import Config
from ..utils import dir_size_bytes, duck, get_logger, stage_timer, write_parquet
from . import text as T
from .taxonomy import CategoryPredictor

log = get_logger("csre.catalog")

CATALOG_COLUMNS = [
    "doc_id", "product_id", "locale", "title", "brand", "brand_norm", "color", "description", "bullets",
    "attr_measures", "attr_dimensions", "attr_pack_count", "attr_sizes", "attr_audience", "attr_compat",
    "attr_materials", "attr_colors", "brand_in_title",
    "n_title_chars", "n_desc_chars", "n_bullets", "n_bullet_chars", "richness_chars", "is_title_only", "is_sparse",
    "category", "category_pred", "category_conf", "category_source", "doc_text",
]


def _stage_by_locale(cfg: Config) -> None:
    """Re-chunk the single-row-group source file into per-locale files with small row groups."""
    staging = cfg.path("duckdb_tmp", "products_by_locale")
    done = staging / "_SUCCESS"
    if done.exists():
        return
    shutil.rmtree(staging, ignore_errors=True)
    con = duck(cfg)
    src = cfg.path("raw", "esci", "shopping_queries_dataset_products.parquet")
    log.info("staging products by locale (streaming re-chunk) ...")
    con.execute(f"""
        COPY (SELECT * FROM '{src}') TO '{staging}'
        (FORMAT PARQUET, PARTITION_BY (product_locale), ROW_GROUP_SIZE {int(cfg.get('resources.batch_rows'))},
         COMPRESSION ZSTD)
    """)
    done.touch()


def transform_batch(df: pl.DataFrame, cfg: Config, predictor: CategoryPredictor | None) -> pl.DataFrame:
    sparse = cfg.get("catalog.sparse_chars")
    n_desc = int(cfg.get("catalog.doc_text_desc_chars", 600))
    n_bul = int(cfg.get("catalog.doc_text_bullets", 4))
    out = (
        df.rename({"product_locale": "locale"})
        .with_columns(
            T.clean_text(pl.col("product_title")).fill_null("").alias("title"),
            T.clean_text(pl.col("product_description")).alias("description"),
            T.clean_bullets(pl.col("product_bullet_point")).alias("bullets"),
            T.clean_text(pl.col("product_brand")).alias("brand"),
            T.clean_text(pl.col("product_color")).alias("color"),
        )
        .with_columns(
            T.normalize_brand(pl.col("brand")).alias("brand_norm"),
            pl.col("title").str.len_chars().alias("n_title_chars"),
            pl.col("description").fill_null("").str.len_chars().alias("n_desc_chars"),
            pl.col("bullets").list.len().alias("n_bullets"),
            pl.col("bullets").list.join(" ").str.len_chars().alias("n_bullet_chars"),
            T.for_matching(pl.col("title")).alias("_tm"),
            T.for_matching(pl.concat_str([pl.col("title"), pl.col("bullets").list.head(5).list.join(" ")],
                                         separator=" | ")).alias("_rm"),
            T.for_matching(pl.col("color")).alias("_cm"),
        )
        .with_columns(
            (pl.col("n_desc_chars") + pl.col("n_bullet_chars")).alias("richness_chars"),
            ((pl.col("n_desc_chars") == 0) & (pl.col("n_bullets") == 0)).alias("is_title_only"),
            pl.col("_tm").str.contains(pl.col("brand_norm"), literal=True).fill_null(False).alias("brand_in_title"),
            *T.attribute_exprs(pl.col("_tm"), pl.col("_rm")),
            pl.col("_cm").str.extract_all(T.COLOR_RE).list.eval(pl.element().replace(T.COLOR_CANON))
            .list.unique(maintain_order=True).alias("_colors_field"),
        )
        .with_columns(
            # colour field is authoritative when present; otherwise fall back to the title
            pl.when(pl.col("_colors_field").list.len() > 0).then(pl.col("_colors_field"))
            .otherwise(pl.col("attr_colors")).alias("attr_colors"),
            (pl.col("richness_chars") < pl.col("locale").replace_strict(sparse, return_dtype=pl.Int64))
            .alias("is_sparse"),
            (pl.col("locale") + ":" + pl.col("product_id")).alias("doc_id"),
            pl.concat_list([
                pl.col("title"),
                pl.col("brand").fill_null(""),
                pl.col("color").fill_null(""),
                pl.col("bullets").list.head(n_bul).list.join(" "),
                pl.col("description").fill_null("").str.slice(0, n_desc),
            ]).list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).list.join(" | ")
            .alias("doc_text"),
        )
    )
    if predictor is not None:
        out = predictor.predict(out)
    else:
        out = out.with_columns(pl.lit("Unknown").alias("category"), pl.lit(None, pl.Utf8).alias("category_pred"),
                               pl.lit(0.0, pl.Float32).alias("category_conf"), pl.lit("unknown").alias("category_source"))
    return out.select(CATALOG_COLUMNS)


def build_catalog(cfg: Config) -> dict:
    with stage_timer(cfg, "catalog") as info:
        _stage_by_locale(cfg)
        predictor = CategoryPredictor(cfg) if cfg.path("models", "category_model.joblib").exists() else None
        if predictor is None:
            log.warning("no category model found — categories will be 'Unknown'")
        out_dir = cfg.path("processed", "catalog")
        shutil.rmtree(out_dir, ignore_errors=True)
        batch_rows = int(cfg.get("resources.batch_rows"))
        total = 0
        for loc_dir in sorted(cfg.path("duckdb_tmp", "products_by_locale").glob("product_locale=*")):
            loc = loc_dir.name.split("=", 1)[1]
            part = 0
            for f in sorted(loc_dir.glob("*.parquet")):
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=batch_rows):
                    df = pl.from_arrow(batch).with_columns(pl.lit(loc).alias("product_locale"))
                    out = transform_batch(df, cfg, predictor)
                    write_parquet(out, out_dir / f"locale={loc}" / f"part-{part:05d}.parquet", row_group_size=50_000)
                    total += len(out)
                    part += 1
                    log.info("catalog %s part %d rows=%d (total %d)", loc, part, len(out), total)
        _build_side_tables(cfg)
        info.update(rows=total, bytes=dir_size_bytes(out_dir))
    return info


def _build_side_tables(cfg: Config) -> None:
    """Brand vocabulary (for query brand detection) and exact-title duplicate groups (for result dedup)."""
    con = duck(cfg)
    cat = cfg.path("processed", "catalog")
    brands = con.sql(f"""
        SELECT locale, brand_norm, count(*) AS n_products,
               mode(category) FILTER (WHERE category <> 'Unknown') AS top_category
        FROM read_parquet('{cat}/**/*.parquet', hive_partitioning=true)
        WHERE brand_norm IS NOT NULL
        GROUP BY ALL
    """).pl()
    write_parquet(brands, cfg.path("processed", "brands.parquet"))
    dups = con.sql(f"""
        SELECT doc_id, md5(locale || '|' || lower(title)) AS title_key,
               count(*) OVER (PARTITION BY locale, lower(title)) AS title_group_size
        FROM read_parquet('{cat}/**/*.parquet', hive_partitioning=true)
    """).pl().filter(pl.col("title_group_size") > 1)
    write_parquet(dups, cfg.path("processed", "catalog_title_groups.parquet"))
    log.info("brands=%d, docs in duplicate-title groups=%d", len(brands), len(dups))

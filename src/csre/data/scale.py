"""Scale-test catalog tiers (systems benchmarks only).

The real catalog is 1.8M products. To measure how index build time, memory, latency and cost grow
with catalog size, we add synthetic *distractor* products up to each tier (2.5M ... 25M+).

Each distractor is a perturbed copy of a real product (brand swapped for another brand of the same
locale, the leading number rescaled, colour swapped, a pack-size suffix added), so token statistics,
title lengths and locale mix stay realistic. `base_doc_id` is kept so later phases can derive an
embedding for a distractor from its base product's vector without re-encoding millions of texts.

Distractors are NEVER judged and must never enter relevance metrics: they exist to make the
retrieval problem bigger, not to measure quality.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import polars as pl

from ..config import Config
from ..utils import dir_size_bytes, duck, get_logger, stage_timer, write_parquet

log = get_logger("csre.scale")

_ASCII_COLORS = ["black", "white", "red", "blue", "green", "grey", "pink", "purple", "navy", "beige", "silver", "gold"]
_COLOR_RE = r"(?i)\b(?:black|white|red|blue|green|gray|grey|pink|purple|navy|beige|silver|gold|negro|blanco|rojo|azul|verde|gris|rosa)\b"
_SUFFIX = {
    "us": [", Pack of {n}", " ({n} Count)", " - {n} Pack", ", Set of {n}", " (Model {m})", " - Value Pack"],
    "es": [", Pack de {n}", " ({n} Unidades)", " - Juego de {n}", " (Modelo {m})", ", Paquete de {n}"],
    "jp": [" {n}個セット", " ({n}枚入り)", " {n}本入", " (モデル {m})", " お得用"],
}


def _load_base(cfg: Config) -> pl.DataFrame:
    con = duck(cfg)
    cat = cfg.path("processed", "catalog")
    return con.sql(f"""
        SELECT doc_id AS base_doc_id, locale, title, brand, color, category,
               left(array_to_string(bullets[1:2], ' '), 200) AS bullets_head
        FROM read_parquet('{cat}/**/*.parquet', hive_partitioning=true)
    """).pl()


def _synth_shard(rng: np.random.Generator, base: pl.DataFrame, brands: dict[str, np.ndarray],
                 n: int, shard: int, locale_mix: dict[str, float]) -> pl.DataFrame:
    locs = np.array(list(locale_mix))
    p = np.array([locale_mix[l] for l in locs])
    loc_draw = rng.choice(locs, size=n, p=p / p.sum())
    parts = []
    for loc in locs:
        k = int((loc_draw == loc).sum())
        if k == 0:
            continue
        b = base.filter(pl.col("locale") == loc)
        idx = rng.integers(0, len(b), size=k)
        new_brand = brands[loc][rng.integers(0, len(brands[loc]), size=k)]
        tmpl = np.array(_SUFFIX[loc])[rng.integers(0, len(_SUFFIX[loc]), size=k)]
        nums = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 50, 100], size=k)
        model = rng.integers(100, 9999, size=k)
        suffix = [t.format(n=a, m=f"{chr(65 + m % 26)}{m}") for t, a, m in zip(tmpl, nums, model)]
        scale = rng.choice([0.5, 0.75, 1.25, 1.5, 2.0, 3.0], size=k)
        color = np.array(_ASCII_COLORS)[rng.integers(0, len(_ASCII_COLORS), size=k)]
        parts.append(
            b[idx].with_columns(
                pl.Series("new_brand", new_brand), pl.Series("suffix", suffix),
                pl.Series("scale", scale), pl.Series("new_color", color),
            )
        )
    df = pl.concat(parts)
    first_num = df["title"].str.extract(r"(\d+(?:\.\d+)?)", 1).cast(pl.Float64, strict=False)
    new_num = (first_num * df["scale"]).round(1).cast(pl.Utf8).str.replace(r"\.0$", "")
    # swap the first occurrence of the brand (polars has no per-row replace pattern -> split on it)
    parts = pl.col("title").str.split(pl.col("brand").fill_null("\u0000"))
    df = df.with_columns(pl.Series("new_num", new_num)).with_columns(
        pl.when(parts.list.len() > 1)
        .then(parts.list.first() + pl.col("new_brand") + parts.list.slice(1).list.join(pl.col("brand")))
        .otherwise(pl.col("new_brand") + " " + pl.col("title")).alias("title"),
    ).with_columns(
        pl.when(pl.col("new_num").is_not_null())
        .then(pl.col("title").str.replace(r"\d+(?:\.\d+)?", pl.col("new_num")))
        .otherwise(pl.col("title")).alias("title"),
    ).with_columns(
        (pl.col("title").str.replace(_COLOR_RE, pl.col("new_color")) + pl.col("suffix")).alias("title"),
        pl.col("new_brand").alias("brand"),
    )
    ids = np.arange(n, dtype=np.int64) + np.int64(shard) * 10_000_000
    return df.with_columns(
        pl.Series("n", ids),
    ).with_columns(
        (pl.lit("syn-") + pl.col("locale") + pl.lit(":") + pl.col("n").cast(pl.Utf8).str.zfill(10)).alias("doc_id"),
        pl.concat_list([pl.col("title"), pl.col("brand").fill_null(""), pl.col("bullets_head").fill_null("")])
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).list.join(" | ").alias("doc_text"),
        pl.lit(True).alias("is_synthetic"),
    ).select("doc_id", "base_doc_id", "locale", "title", "brand", "category", "doc_text", "is_synthetic")


def build_scale(cfg: Config, max_tier: int | None = None) -> dict:
    S = cfg.get("scale")
    tiers = sorted(int(t) for t in S["tiers"])
    if max_tier:
        tiers = [t for t in tiers if t <= max_tier]
    shard_rows = int(S["shard_rows"])
    out = cfg.path("scale")
    shutil.rmtree(out, ignore_errors=True)
    with stage_timer(cfg, "scale") as info:
        con = duck(cfg)
        n_real = con.sql(f"SELECT count(*) FROM read_parquet('{cfg.path('processed', 'catalog')}/**/*.parquet')").fetchone()[0]
        base = _load_base(cfg)
        bt = pl.read_parquet(cfg.path("processed", "brands.parquet")).filter(pl.col("n_products") >= 5)
        brands = {l: bt.filter(pl.col("locale") == l)["brand_norm"].str.to_titlecase().to_numpy() for l in ["us", "es", "jp"]}
        n_syn_max = max(0, tiers[-1] - n_real)
        n_shards = int(np.ceil(n_syn_max / shard_rows))
        written = 0
        for s in range(n_shards):
            n = min(shard_rows, n_syn_max - written)
            rng = np.random.default_rng([cfg.seed, 31, s])
            df = _synth_shard(rng, base, brands, n, s, S["locale_mix"])
            write_parquet(df, out / "synthetic_catalog" / f"shard={s:04d}" / "part-0.parquet", row_group_size=250_000)
            written += len(df)
            log.info("scale shard %d/%d rows=%d (total synthetic %d)", s + 1, n_shards, len(df), written)
        manifest = {
            "real_catalog_rows": n_real,
            "shard_rows": shard_rows,
            "tiers": [{"total_docs": t, "real_docs": n_real, "synthetic_docs": max(0, t - n_real),
                       "synthetic_shards": int(np.ceil(max(0, t - n_real) / shard_rows))} for t in tiers],
            "note": "Tier T = full real catalog + the first (T - real) synthetic rows in shard order. "
                    "Synthetic docs are unjudged distractors for latency/throughput/cost scaling only.",
        }
        (out / "tiers.json").write_text(json.dumps(manifest, indent=2))
        info.update(rows=written, tiers=[t["total_docs"] for t in manifest["tiers"]], bytes=dir_size_bytes(out))
    return info

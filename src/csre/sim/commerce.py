"""Simulated commerce attributes: price, rating, review count, stock.

ESCI has no price / rating / stock. A storefront (and a click model) needs them, so we generate
them *calibrated to real data*: per (locale, department) distributions are fitted on the ESCI-S
sample of real product pages, shrunk towards the locale-level fit when a cell is small. For the
~4K products that are in the ESCI-S sample the real values are used directly.

Every value is a deterministic function of (seed, doc_id), so any shard of the catalog can be
generated independently and re-generated identically. All columns are flagged `commerce_source`
∈ {esci_s, simulated}; nothing simulated is ever used as a relevance label.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from ..config import Config
from ..utils import duck, get_logger, stable_hash64, stage_timer, uniform_from_hash, write_parquet

log = get_logger("csre.sim.commerce")

CURRENCY = {"us": "USD", "es": "EUR", "jp": "JPY"}
PRIOR_N = 20.0


def _fit(meta: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Per (locale, department) log-price and log-ratings Gaussians, shrunk to locale level."""
    m = meta.with_columns(
        pl.col("price").log().alias("lp"), pl.col("n_ratings").log1p().alias("lr"),
        pl.col("department").fill_null("Unknown"),
    )
    loc = m.group_by("locale").agg(
        pl.col("lp").mean().alias("lp_mu0"), pl.col("lp").std().alias("lp_sd0"),
        pl.col("lr").mean().alias("lr_mu0"), pl.col("lr").std().alias("lr_sd0"),
    )
    cell = m.group_by("locale", "department").agg(
        pl.col("lp").drop_nulls().len().alias("n_p"), pl.col("lp").mean().alias("lp_mu"),
        pl.col("lr").drop_nulls().len().alias("n_r"), pl.col("lr").mean().alias("lr_mu"),
    ).join(loc, on="locale").with_columns(
        ((pl.col("n_p") * pl.col("lp_mu").fill_null(0) + PRIOR_N * pl.col("lp_mu0")) / (pl.col("n_p") + PRIOR_N)).alias("lp_mu"),
        ((pl.col("n_r") * pl.col("lr_mu").fill_null(0) + PRIOR_N * pl.col("lr_mu0")) / (pl.col("n_r") + PRIOR_N)).alias("lr_mu"),
    )
    stars = {l: np.sort(m.filter((pl.col("locale") == l) & pl.col("stars").is_not_null())["stars"].to_numpy())
             for l in ["us", "es", "jp"]}
    return loc, cell, stars


def _round_price(p: np.ndarray, currency: np.ndarray) -> np.ndarray:
    jpy = currency == "JPY"
    out = np.where(jpy, np.maximum(np.round(p / 10) * 10, 100), np.maximum(np.floor(p) + 0.99, 0.99))
    return out.astype(np.float64)


def build_commerce(cfg: Config) -> dict:
    con = duck(cfg)
    with stage_timer(cfg, "sim_commerce") as info:
        meta = pl.read_parquet(cfg.path("processed", "esci_s_meta.parquet"))
        loc_fit, cell_fit, stars_emp = _fit(meta)
        cat = cfg.path("processed", "catalog")
        docs = con.sql(f"""
            WITH nj AS (SELECT doc_id, count(*) AS n_judgments FROM '{cfg.path('processed', 'judgments.parquet')}' GROUP BY 1)
            SELECT c.doc_id, c.product_id, c.locale, c.category, c.attr_pack_count, coalesce(nj.n_judgments, 0) AS n_judgments
            FROM read_parquet('{cat}/**/*.parquet', hive_partitioning=true) c LEFT JOIN nj USING (doc_id)
        """).pl()
        docs = docs.join(cell_fit.select("locale", pl.col("department").alias("category"), "lp_mu", "lr_mu"),
                         on=["locale", "category"], how="left") \
                   .join(loc_fit, on="locale", how="left") \
                   .with_columns(pl.col("lp_mu").fill_null(pl.col("lp_mu0")), pl.col("lr_mu").fill_null(pl.col("lr_mu0")))
        h = stable_hash64(docs["doc_id"].to_list(), salt=f"commerce-{cfg.seed}")
        z_price = norm.ppf(np.clip(uniform_from_hash(h, 1), 1e-6, 1 - 1e-6))
        z_rat = norm.ppf(np.clip(uniform_from_hash(h, 2), 1e-6, 1 - 1e-6))
        u_star = uniform_from_hash(h, 3)
        u_stock = uniform_from_hash(h, 4)

        pack = docs["attr_pack_count"].fill_null(1).clip(1, 500).to_numpy().astype(float)
        lp = docs["lp_mu"].to_numpy() + docs["lp_sd0"].to_numpy() * 0.85 * z_price + 0.35 * np.log(pack)
        currency = docs["locale"].replace_strict(CURRENCY).to_numpy()
        price = _round_price(np.exp(lp), currency)

        # review count: lognormal, lifted for products that surface in many queries (a real popularity proxy)
        nj = docs["n_judgments"].to_numpy().astype(float)
        lr = docs["lr_mu"].to_numpy() + docs["lr_sd0"].to_numpy() * 0.9 * z_rat + 0.6 * np.log(np.maximum(nj, 1.0))
        n_ratings = np.maximum(np.round(np.expm1(np.maximum(lr, 0))), 0).astype(np.int64)

        stars = np.empty(len(docs))
        locs = docs["locale"].to_numpy()
        for l, emp in stars_emp.items():
            msk = locs == l
            stars[msk] = np.quantile(emp, u_star[msk]) if len(emp) else 4.3
        # few-review products have more extreme ratings: pull towards 5.0 or 3.5 with weight ~ 1/sqrt(n)
        w = 1.0 / np.sqrt(1.0 + n_ratings)
        stars = np.round(np.clip((1 - w) * stars + w * np.where(u_star > 0.5, 5.0, 3.5), 1.0, 5.0), 1)
        stars = np.where(n_ratings == 0, np.nan, stars)

        out = docs.select("doc_id", "locale").with_columns(
            pl.Series("price", price), pl.Series("currency", currency),
            pl.Series("stars", stars).fill_nan(None).cast(pl.Float32),
            pl.Series("n_ratings", n_ratings),
            pl.Series("in_stock", u_stock > 0.06),
            pl.lit("simulated").alias("commerce_source"),
        )
        # overwrite with real values where the ESCI-S sample has them
        real = meta.select(
            (pl.col("locale") + ":" + pl.col("product_id")).alias("doc_id"),
            pl.col("price").alias("r_price"), pl.col("stars").alias("r_stars"), pl.col("n_ratings").alias("r_nr"),
        )
        out = out.join(real, on="doc_id", how="left").with_columns(
            pl.coalesce("r_price", "price").alias("price"),
            pl.coalesce(pl.col("r_stars").cast(pl.Float32), "stars").alias("stars"),
            pl.coalesce(pl.col("r_nr").cast(pl.Int64), "n_ratings").alias("n_ratings"),
            pl.when(pl.col("r_stars").is_not_null() | pl.col("r_price").is_not_null())
            .then(pl.lit("esci_s")).otherwise("commerce_source").alias("commerce_source"),
        ).drop("r_price", "r_stars", "r_nr")
        # popularity: standardised log review count within locale (used by the logging policy)
        out = out.with_columns(
            ((pl.col("n_ratings").log1p() - pl.col("n_ratings").log1p().mean().over("locale"))
             / pl.col("n_ratings").log1p().std().over("locale")).cast(pl.Float32).alias("popularity_z")
        )
        write_parquet(out, cfg.path("synthetic", "product_commerce.parquet"))
        info.update(rows=len(out),
                    real_rows=int((out["commerce_source"] == "esci_s").sum()),
                    median_price=dict(out.group_by("locale").agg(pl.col("price").median()).iter_rows()))
    return info

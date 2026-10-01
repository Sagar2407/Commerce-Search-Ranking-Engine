"""One interface over the three data layouts produced by phase 1.

* ``full``           data/processed + data/synthetic   (1.8M products, all splits)
* ``demo``           data/demo                         (flat files, 3K test queries, 151K products)
* ``demo_portable``  data/demo_portable                (flat files, 1.5K test queries, 43K products)

Everything downstream (indexes, models, evaluation, serving) asks this class for data, so the same code
runs on a laptop-sized slice and on the full catalog.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import polars as pl

from ..config import Config

LOCALES = ("us", "es", "jp")

# per-locale catalog columns the engine needs (doc_text keeps title + first bullets + 600 description chars)
CATALOG_COLS = [
    "doc_id", "locale", "title", "brand", "brand_norm", "color", "doc_text",
    "attr_measures", "attr_dimensions", "attr_pack_count", "attr_sizes", "attr_audience", "attr_compat",
    "attr_materials", "attr_colors", "brand_in_title", "n_title_chars", "richness_chars", "is_title_only",
    "is_sparse", "category", "category_conf",
]


class Dataset:
    def __init__(self, cfg: Config, corpus: str | None = None):
        self.cfg = cfg
        self.name = corpus or cfg.get("search.corpus", "full")
        if self.name not in ("full", "demo", "demo_portable"):
            raise ValueError(f"unknown corpus {self.name!r}")
        self.flat = self.name != "full"
        self.root = cfg.path("demo").with_name(self.name) if self.flat else cfg.root / "data"

    # ---------------------------------------------------------------- paths
    def _p(self, full_rel: str, flat_name: str) -> Path:
        return self.root / flat_name if self.flat else self.cfg.root / full_rel

    def _paths(self) -> dict[str, Path]:
        P = self.cfg.get("paths")
        return {
            "catalog": self._p(f"{P['processed']}/catalog", "catalog.parquet"),
            "queries": self._p(f"{P['processed']}/queries.parquet", "queries.parquet"),
            "judgments": self._p(f"{P['processed']}/judgments.parquet", "judgments.parquet"),
            "commerce": self._p(f"{P['synthetic']}/product_commerce.parquet", "product_commerce.parquet"),
            "query_doc_stats": self._p(f"{P['synthetic']}/logs/query_doc_stats.parquet", "query_doc_stats.parquet"),
            "query_popularity": self._p(f"{P['synthetic']}/query_popularity.parquet", "query_popularity.parquet"),
            "edges": self._p(f"{P['processed']}/graph/product_edges.parquet", "product_edges.parquet"),
            "related_queries": self._p(f"{P['processed']}/graph/related_queries.parquet", "related_queries.parquet"),
            "category_complements": self._p(f"{P['processed']}/graph/category_complements.parquet",
                                            "category_complements.parquet"),
            "replay": self._p(f"{P['synthetic']}/replay/requests.parquet", "replay_sample.parquet"),
            "brand_patterns": self._p(f"{P['processed']}/brand_patterns.json", "brand_patterns.json"),
            "product_meta": self._p(f"{P['processed']}/product_meta.parquet", "product_meta.parquet"),
        }

    @cached_property
    def paths(self) -> dict[str, Path]:
        return self._paths()

    def exists(self) -> bool:
        return self.paths["catalog"].exists() and self.paths["queries"].exists()

    def index_dir(self) -> Path:
        return self.cfg.path("indexes", self.name)

    # ---------------------------------------------------------------- tables
    def catalog(self, locale: str, columns: list[str] | None = None) -> pl.DataFrame:
        cols = columns or CATALOG_COLS
        p = self.paths["catalog"]
        if self.flat:
            lf = pl.scan_parquet(p).filter(pl.col("locale") == locale)
        else:
            lf = pl.scan_parquet(str(p / f"locale={locale}" / "*.parquet")).with_columns(pl.lit(locale).alias("locale"))
        have = set(lf.collect_schema().names())
        return lf.select([c for c in cols if c in have]).collect().sort("doc_id")

    def locales(self) -> list[str]:
        want = self.cfg.get("search.locales", list(LOCALES))
        if self.flat:
            have = set(pl.scan_parquet(self.paths["catalog"]).select("locale").unique().collect()["locale"])
            return [l for l in want if l in have]
        return [l for l in want if (self.paths["catalog"] / f"locale={l}").exists()]

    def queries(self) -> pl.DataFrame:
        q = pl.read_parquet(self.paths["queries"])
        pop = self.paths["query_popularity"]
        if pop.exists():
            q = q.join(pl.read_parquet(pop, columns=["query_id", "traffic_bucket", "traffic_share"]),
                       on="query_id", how="left")
        return q

    def judgments(self, columns: list[str] | None = None) -> pl.DataFrame:
        return pl.read_parquet(self.paths["judgments"], columns=columns)

    def _opt(self, key: str, columns: list[str] | None = None) -> pl.DataFrame | None:
        p = self.paths[key]
        return pl.read_parquet(p, columns=columns) if p.exists() else None

    def commerce(self) -> pl.DataFrame | None:
        return self._opt("commerce")

    def query_doc_stats(self) -> pl.DataFrame | None:
        return self._opt("query_doc_stats")

    def edges(self) -> pl.DataFrame | None:
        return self._opt("edges")

    def related_queries(self) -> pl.DataFrame | None:
        return self._opt("related_queries")

    def category_complements(self) -> pl.DataFrame | None:
        return self._opt("category_complements")

    def replay(self, n: int | None = None) -> pl.DataFrame | None:
        p = self.paths["replay"]
        if not p.exists():
            return None
        lf = pl.scan_parquet(p)
        return (lf.head(n) if n else lf).collect()

    def product_meta(self) -> pl.DataFrame | None:
        return self._opt("product_meta")

    def brand_patterns(self) -> dict[str, list[str]]:
        p = self.paths["brand_patterns"]
        return json.loads(p.read_text()) if p.exists() else {}

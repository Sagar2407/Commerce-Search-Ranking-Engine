"""ESCI-S: real product-page metadata for ESCI products (Apache-2.0, github.com/shuttie/esci-s).

The full dump (1.66M product pages, `esci.json.zst`) carries the real category path, price, star rating,
rating count, structured attributes and an image URL. This module streams it once into a tidy Parquet
table of raw strings (`processed/esci_s_raw.parquet`); `taxonomy.load_esci_s` turns that into typed
metadata. Reviews and long text are not kept.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import polars as pl

from ..config import Config
from ..utils import get_logger, stage_timer, write_parquet

log = get_logger("csre.esci_s")

BRAND_KEYS = ("Brand", "Marca", "ブランド", "Brand Name", "Nombre de la marca", "メーカー")


def full_dump_path(cfg: Config) -> Path:
    return cfg.path("raw", "esci_s", "esci.json.zst")


def parse_full_dump(cfg: Config, batch: int = 200_000) -> Path | None:
    """Stream esci.json.zst -> processed/esci_s_raw.parquet (one row per product page, raw strings)."""
    src = full_dump_path(cfg)
    out = cfg.path("processed", "esci_s_raw.parquet")
    if not src.exists():
        log.warning("full ESCI-S dump not found at %s — using the 4.4K sample only", src)
        return None
    if out.exists() and out.stat().st_mtime > src.stat().st_mtime:
        return out
    import zstandard  # noqa: PLC0415

    with stage_timer(cfg, "esci_s_parse") as info:
        parts, rows, n, n_err = [], [], 0, 0
        tmp = out.with_suffix(".parts")
        tmp.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as fh:
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
            for line in reader:
                r = json.loads(line)
                if r.get("type") != "product":
                    n_err += 1
                    continue
                attrs = r.get("attrs") if isinstance(r.get("attrs"), dict) else {}
                brand = next((attrs[k] for k in BRAND_KEYS if attrs.get(k)), None)
                cats = r.get("category") or []
                rows.append({
                    "product_id": r.get("asin"), "locale": r.get("locale"),
                    "category_path": [c for c in cats if isinstance(c, str)],
                    "stars_raw": r.get("stars") or None, "ratings_raw": r.get("ratings") or None,
                    "price_raw": r.get("price") or None, "template": r.get("template") or None,
                    "image_url": r.get("image") or None, "brand_attr": brand,
                    "attrs_json": json.dumps(attrs, ensure_ascii=False) if attrs else None,
                })
                if len(rows) >= batch:
                    p = tmp / f"part-{len(parts):04d}.parquet"
                    write_parquet(pl.DataFrame(rows), p)
                    parts.append(p)
                    n += len(rows)
                    rows = []
                    log.info("esci-s parsed %d product pages (%d non-product rows)", n, n_err)
        if rows:
            p = tmp / f"part-{len(parts):04d}.parquet"
            write_parquet(pl.DataFrame(rows), p)
            parts.append(p)
            n += len(rows)
        df = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed") \
               .unique(["locale", "product_id"], keep="first")
        write_parquet(df, out)
        for p in parts:
            p.unlink()
        tmp.rmdir()
        info.update(rows=len(df), non_product_rows=n_err)
    return out

"""Shared helpers: logging, timing, DuckDB, deterministic hashing, parquet IO, run manifest."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Config

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


log = get_logger("csre")


# --------------------------------------------------------------------------------------
# Timing + manifest (every stage records rows, bytes and throughput for the data card)
# --------------------------------------------------------------------------------------
@contextmanager
def stage_timer(cfg: Config, stage: str, **extra):
    t0 = time.time()
    info: dict = {"stage": stage, **extra}
    try:
        yield info
    finally:
        info["seconds"] = round(time.time() - t0, 2)
        if info.get("rows") and info["seconds"] > 0:
            info["rows_per_sec"] = int(info["rows"] / info["seconds"])
        record_manifest(cfg, stage, info)
        log.info("[%s] done in %.1fs %s", stage, info["seconds"],
                 {k: v for k, v in info.items() if k not in ("stage", "seconds")})


def record_manifest(cfg: Config, stage: str, info: dict) -> None:
    path = cfg.path("reports", "manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[stage] = {**info, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.write_text(json.dumps(data, indent=2, default=str))


def dir_size_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# --------------------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------------------
def duck(cfg: Config) -> duckdb.DuckDBPyConnection:
    tmp = cfg.path("duckdb_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{cfg.get('resources.duckdb_memory_limit', '4GB')}'")
    con.execute(f"SET threads={int(cfg.get('resources.threads', 2))}")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute("SET preserve_insertion_order=false")
    return con


# --------------------------------------------------------------------------------------
# Deterministic hashing (stable across machines / library versions)
# --------------------------------------------------------------------------------------
def stable_hash64(values: Iterable[str], salt: str = "") -> np.ndarray:
    """MD5-based 64-bit hash of strings; stable forever, unlike polars/duckdb hash()."""
    out = np.fromiter(
        (int.from_bytes(hashlib.md5(f"{salt}|{v}".encode()).digest()[:8], "little") for v in values),
        dtype=np.uint64,
    )
    return out


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser over uint64 — deterministic per-key randomness."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def uniform_from_hash(h: np.ndarray, stream: int = 0) -> np.ndarray:
    """Map uint64 keys to U[0,1) deterministically; `stream` gives independent draws per key."""
    with np.errstate(over="ignore"):
        z = splitmix64(h.astype(np.uint64) ^ np.uint64((stream * 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# --------------------------------------------------------------------------------------
# Parquet IO
# --------------------------------------------------------------------------------------
def write_parquet(df: pl.DataFrame | pa.Table, path: Path, row_group_size: int = 128_000) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df.to_arrow() if isinstance(df, pl.DataFrame) else df
    pq.write_table(table, path, compression="zstd", compression_level=6, row_group_size=row_group_size)
    return path


def read_parquet(path: Path | str, columns: list[str] | None = None) -> pl.DataFrame:
    path = Path(path)
    if path.is_dir():
        return pl.read_parquet(str(path / "**" / "*.parquet"), columns=columns, hive_partitioning=True)
    return pl.read_parquet(path, columns=columns)


def scan(path: Path | str) -> pl.LazyFrame:
    path = Path(path)
    if path.is_dir():
        return pl.scan_parquet(str(path / "**" / "*.parquet"), hive_partitioning=True)
    return pl.scan_parquet(path)

"""Stage 0 — acquire raw data (Amazon ESCI + ESCI-S product-page metadata) with integrity checks."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

from ..config import Config
from ..utils import get_logger, stage_timer

log = get_logger("csre.acquire")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dest)


def acquire(cfg: Config, force: bool = False) -> dict:
    raw = cfg.path("raw")
    out: dict = {}
    with stage_timer(cfg, "acquire") as info:
        for key, spec in cfg["esci"]["files"].items():
            dest = raw / "esci" / spec["name"]
            base = cfg["esci"]["base_url"] if spec.get("lfs") else cfg["esci"]["raw_url"]
            if force or not dest.exists():
                _download(f"{base}/{spec['name']}", dest)
            digest = sha256_file(dest)
            if spec.get("sha256") and digest != spec["sha256"]:
                raise ValueError(f"checksum mismatch for {dest.name}: {digest}")
            out[key] = {"path": str(dest.relative_to(cfg.root)), "bytes": dest.stat().st_size, "sha256": digest,
                        "verified": bool(spec.get("sha256"))}
        sample = raw / "esci_s" / "sample.json.gz"
        if force or not sample.exists():
            _download(cfg["esci_s"]["sample_url"], sample)
        out["esci_s_sample"] = {"path": str(sample.relative_to(cfg.root)), "bytes": sample.stat().st_size}
        full = raw / "esci_s" / "esci.json.zst"
        if cfg.get("esci_s.download_full", True) and (force or not full.exists()):
            try:
                _download(cfg["esci_s"]["full_url"], full)
            except Exception as e:  # noqa: BLE001 — optional enrichment: the pipeline runs on the sample alone
                log.warning("full ESCI-S dump unavailable (%s); continuing with the 4.4K sample", e)
        out["esci_s_full_present"] = full.exists()
        if full.exists():
            out["esci_s_full"] = {"path": str(full.relative_to(cfg.root)), "bytes": full.stat().st_size}
        info["files"] = out
        info["bytes"] = sum(v["bytes"] for v in out.values() if isinstance(v, dict) and "bytes" in v)
    return out

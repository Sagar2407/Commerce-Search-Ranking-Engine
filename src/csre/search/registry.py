"""A small file-based model registry: versioned model directories + aliases (production / candidate).

    data/models/registry/registry.json
    data/models/registry/<name>/<version>/   model files + model.json (params, metrics, data, lineage)

Every artefact the engine serves is loaded through an alias, so a new version can be trained, evaluated
and promoted (or rolled back) without touching serving code; each search response reports the versions
that produced it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from ..config import Config


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


class ModelRegistry:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.path("models", "registry")
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "registry.json"

    def _read(self) -> dict:
        return json.loads(self.index.read_text()) if self.index.exists() else {}

    def _write(self, data: dict) -> None:
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(self.index)

    def register(self, name: str, save: Callable[[Path], None], *, params: dict | None = None,
                 metrics: dict | None = None, data: dict | None = None, alias: str | None = "production",
                 parents: dict | None = None) -> str:
        reg = self._read()
        entry = reg.setdefault(name, {"versions": {}, "aliases": {}})
        version = f"v{len(entry['versions']) + 1}"
        d = self.root / name / version
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        save(d)
        meta = {"name": name, "version": version, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "git_commit": _git_commit(self.cfg.root), "params": params or {}, "metrics": metrics or {},
                "data": data or {}, "parents": parents or {}}
        (d / "model.json").write_text(json.dumps(meta, indent=2, default=str))
        entry["versions"][version] = {k: meta[k] for k in ("created_at", "git_commit", "metrics", "parents")}
        if alias:
            entry["aliases"][alias] = version
        self._write(reg)
        return version

    def resolve(self, name: str, ref: str = "production") -> str | None:
        entry = self._read().get(name)
        if not entry:
            return None
        return entry["aliases"].get(ref, ref if ref in entry["versions"] else None)

    def path(self, name: str, ref: str = "production") -> Path | None:
        v = self.resolve(name, ref)
        return self.root / name / v if v else None

    def meta(self, name: str, ref: str = "production") -> dict | None:
        p = self.path(name, ref)
        return json.loads((p / "model.json").read_text()) if p and (p / "model.json").exists() else None

    def promote(self, name: str, version: str, alias: str = "production") -> None:
        reg = self._read()
        if version not in reg.get(name, {}).get("versions", {}):
            raise KeyError(f"{name} has no version {version}")
        reg[name]["aliases"][alias] = version
        self._write(reg)

    def listing(self) -> dict:
        return self._read()

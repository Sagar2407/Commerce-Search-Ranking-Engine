"""Configuration loading and path resolution."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/data.yaml"
# Merged on top of the default config, in order (phase 2: retrieval, ranking, evaluation, serving).
EXTRA_CONFIGS = ["configs/search.yaml"]


def repo_root() -> Path:
    env = os.environ.get("CSRE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [Path.cwd(), *here.parents]:
        if (parent / "configs" / "data.yaml").exists():
            return parent
    return Path.cwd()


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value


@dataclass
class Config:
    raw: dict
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def path(self, name: str, *parts: str) -> Path:
        p = self.root / self.raw["paths"][name]
        p = p.joinpath(*parts) if parts else p
        return p

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    root = repo_root()
    cfg_path = Path(path) if path else root / DEFAULT_CONFIG
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    if path is None:
        for extra in EXTRA_CONFIGS:
            if (root / extra).exists():
                with open(root / extra) as f:
                    raw = _deep_merge(raw, yaml.safe_load(f) or {})
    raw = copy.deepcopy(raw)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        _set_dotted(raw, k, v)
    return Config(raw=raw, root=root)

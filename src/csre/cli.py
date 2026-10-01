"""`csre` command line: data stages (phase 1) and search stages (phase 2) from one config.

    csre all                       # phase-1 data pipeline, in dependency order
    csre traffic --set simulation.traffic.n_sessions=50000000 --set simulation.traffic.shards=200
    csre validate                  # data-quality gate (non-zero exit on failure)

    csre search-all                # train-dense, index, train-qcat, tune-hybrid, train-ltr, eval (full corpus)
    csre index --corpus demo       # build indexes for the storefront demo corpus with the production models
    csre eval --corpus full --split test
    csre serve --corpus demo       # storefront + API on :8000
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config

STAGES = ["acquire", "taxonomy", "catalog", "judgments", "graph", "commerce", "traffic", "replay", "scale",
          "demo", "validate", "datacard"]
SEARCH_STAGES = ["train-dense", "index", "train-qcat", "tune-hybrid", "train-ltr", "eval"]
EXTRA_SEARCH = ["train-esci-class"]
OTHER = ["serve", "bench", "snapshot", "models", "promote", "simulate-ab", *EXTRA_SEARCH]


def run_search_stage(stage: str, cfg, args) -> None:
    if stage == "train-dense":
        from .search.train import train_dense
        train_dense(cfg, args.corpus or "full")
    elif stage == "index":
        from .search.engine import build_indexes
        build_indexes(cfg, args.corpus)
    elif stage == "train-qcat":
        from .search.train import train_qcat
        train_qcat(cfg, args.corpus or "full")
    elif stage == "tune-hybrid":
        from .search.train import tune_hybrid
        tune_hybrid(cfg, corpus=args.corpus or "full")
    elif stage == "train-ltr":
        from .search.train import train_ltr
        train_ltr(cfg, corpus=args.corpus or "full")
    elif stage == "eval":
        from .evaluation.harness import evaluate
        parts = tuple(args.parts.split(",")) if args.parts else ("rerank", "retrieval", "latency", "robustness")
        evaluate(cfg, args.corpus, args.split, args.methods.split(",") if args.methods else None, parts=parts)
    elif stage == "serve":
        from .serve.app import serve
        serve(cfg, args.corpus)
    elif stage == "bench":
        from .evaluation.bench import run_bench
        run_bench(cfg, args.corpus, tuple(args.parts.split(",")) if args.parts else ("rerank_depth", "ann", "scale"))
    elif stage == "train-esci-class":
        from .search.train import train_esci_class
        train_esci_class(cfg, args.corpus or "full")
    elif stage == "simulate-ab":
        from .evaluation.online_sim import simulate_ab
        simulate_ab(cfg, args.corpus)
    elif stage == "snapshot":
        from .serve.snapshot import build_snapshot
        build_snapshot(cfg, args.corpus)
    elif stage == "models":
        import json
        from .search.registry import ModelRegistry
        print(json.dumps(ModelRegistry(cfg).listing(), indent=2))
    elif stage == "promote":
        from .search.registry import ModelRegistry
        name, version = args.target.split("=", 1)
        ModelRegistry(cfg).promote(name, version, args.alias)
    else:
        raise SystemExit(f"unknown stage {stage}")


def run_stage(stage: str, cfg) -> None:
    if stage == "acquire":
        from .data.acquire import acquire
        acquire(cfg)
    elif stage == "taxonomy":
        from .data.taxonomy import train_category_model
        train_category_model(cfg, propagate=True)
    elif stage == "catalog":
        from .data.catalog import build_catalog
        build_catalog(cfg)
    elif stage == "judgments":
        from .data.judgments import build_judgments
        build_judgments(cfg)
    elif stage == "graph":
        from .data.graph import build_graph
        build_graph(cfg)
    elif stage == "commerce":
        from .sim.commerce import build_commerce
        build_commerce(cfg)
    elif stage == "traffic":
        from .sim.traffic import build_traffic
        build_traffic(cfg)
    elif stage == "replay":
        from .sim.traffic import build_replay
        build_replay(cfg)
    elif stage == "scale":
        from .data.scale import build_scale
        build_scale(cfg)
    elif stage == "demo":
        from .data.demo import build_demo
        build_demo(cfg)
    elif stage == "validate":
        from .data.validate import run_checks
        res = run_checks(cfg)
        if any(not r["passed"] for r in res):
            sys.exit(1)
    elif stage == "datacard":
        from .data.datacard import build_datacard
        build_datacard(cfg)
    else:
        raise SystemExit(f"unknown stage {stage}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="csre")
    ap.add_argument("stage", choices=STAGES + ["all"] + SEARCH_STAGES + ["search-all"] + OTHER)
    ap.add_argument("--config", default=None)
    ap.add_argument("--corpus", default=None, help="full | demo | demo_portable (default: search.corpus)")
    ap.add_argument("--split", default=None, help="evaluation split (default: eval.split)")
    ap.add_argument("--methods", default=None, help="comma-separated ranking methods to evaluate")
    ap.add_argument("--parts", default=None, help="eval parts: rerank,retrieval,latency,robustness")
    ap.add_argument("--target", default=None, help="promote: name=version")
    ap.add_argument("--alias", default="production")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set simulation.traffic.n_sessions=1000000")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.stage in SEARCH_STAGES + OTHER or args.stage == "search-all":
        for st in (SEARCH_STAGES if args.stage == "search-all" else [args.stage]):
            run_search_stage(st, cfg, args)
        return
    for st in (STAGES if args.stage == "all" else [args.stage]):
        run_stage(st, cfg)


if __name__ == "__main__":
    main()

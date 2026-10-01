"""`csre` command line: run any data stage, or the whole pipeline, from one config.

    csre all                       # everything, in dependency order
    csre traffic --set simulation.traffic.n_sessions=50000000 --set simulation.traffic.shards=200
    csre validate                  # data-quality gate (non-zero exit on failure)
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config

STAGES = ["acquire", "taxonomy", "catalog", "judgments", "graph", "commerce", "traffic", "replay", "scale",
          "demo", "validate", "datacard"]


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
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set simulation.traffic.n_sessions=1000000")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    for st in (STAGES if args.stage == "all" else [args.stage]):
        run_stage(st, cfg)


if __name__ == "__main__":
    main()

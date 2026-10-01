"""`csre simulate-ab`: simulated online A/B test of the ranking approaches, and a pilot power analysis.

Offline nDCG says which ranking puts better-labelled products higher. It does not say how many more shoppers
would click, add to cart or buy. A real pilot answers that; this module (a) estimates what the phase-1 click
model predicts for each approach on the same traffic, and (b) sizes the pilot needed to detect such a change.

Every number here is SIMULATED. The click model turns ESCI labels into behaviour (position-based examination,
label-dependent attractiveness, cart and purchase funnel), so it can only re-express offline relevance in
behavioural units; it cannot discover effects the labels do not contain. Products not judged for a query have
unknown labels, so results are reported under several explicit assumptions for them.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import time

import numpy as np
import polars as pl

from ..config import Config
from ..search.engine import METHOD_LABELS, METHODS, Engine
from ..sim.click_model import LABEL_CODE, ClickModelParams, simulate_interactions
from ..utils import get_logger, stage_timer
from . import metrics as M
from .harness import _limit_threads, load_judged

log = get_logger("csre.online_sim")

# label assumed for products not judged for the query (sensitivity analysis)
UNJUDGED_ASSUMPTIONS = {"pessimistic (unjudged = Irrelevant)": "I", "neutral (unjudged = Complement)": "C",
                        "optimistic (unjudged = Substitute)": "S"}

_ENG: Engine | None = None
_METHODS: list[str] = []


def _rank_batch(batch):
    _limit_threads()
    out = []
    for qid, text, loc in batch:
        for m in _METHODS:
            r = _ENG.search(text, loc, m, k=10)
            out.append((qid, m, r.rows.tolist()))
    return out


def z_power_n(p0: float, rel_lift: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Searches per arm for a two-sided two-proportion z-test to detect p0 -> p0 * (1 + rel_lift)."""
    if p0 <= 0 or rel_lift == 0:
        return float("inf")
    from statistics import NormalDist  # noqa: PLC0415
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    p1 = p0 * (1 + rel_lift)
    pbar = (p0 + p1) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return num / (p1 - p0) ** 2


def simulate_ab(cfg: Config, corpus: str | None = None, n_queries: int = 3000, sessions_per_query: int = 100,
                n_jobs: int | None = None) -> dict:
    global _ENG, _METHODS
    n_jobs = n_jobs or mp.cpu_count()
    with stage_timer(cfg, "simulate_ab") as info:
        eng = Engine(cfg, corpus or "full")
        _ENG = eng
        _METHODS = [m for m in METHODS if m in eng.available_methods()]
        q, judged = load_judged(eng, cfg.get("eval.split"))
        # traffic-weighted sample: head queries appear as often as they would in production
        w = q["traffic_share"].fill_null(q["traffic_share"].min() or 1e-7).to_numpy() if "traffic_share" in q.columns \
            else np.ones(q.height)
        rng = np.random.default_rng(cfg.seed)
        pick = rng.choice(q.height, size=min(n_queries, q.height), replace=False, p=w / w.sum())
        qs = q[pick.tolist()]
        eng.prime(qs["query"].to_list(), qs["locale"].to_list())
        items = list(zip(qs["query_id"].to_list(), qs["query"].to_list(), qs["locale"].to_list()))
        batches = [items[i:i + 25] for i in range(0, len(items), 25)]
        ranked = []
        with mp.get_context("fork").Pool(n_jobs, initializer=_limit_threads) as pool:
            for r in pool.imap_unordered(_rank_batch, batches):
                ranked += r
        q_loc = dict(zip(qs["query_id"].to_list(), qs["locale"].to_list()))
        params = ClickModelParams.from_config(cfg.get("simulation.traffic.click_model"))
        results: dict = {}
        for name, unj in UNJUDGED_ASSUMPTIONS.items():
            per_q = {m: [] for m in _METHODS}
            for qid, m, rows in sorted(ranked, key=lambda r: (r[0], r[1])):
                # common random numbers: every method faces the same simulated shoppers for a query (same draws
                # per session and position), so differences come from the rankings alone (paired comparison)
                sim_rng = np.random.default_rng([cfg.seed, 7, int(qid)])
                D = eng.idx[q_loc[qid]].docs
                jr, _, jl = judged[qid]
                lab = dict(zip(jr.tolist(), jl.tolist()))
                if not rows:
                    per_q[m].append((qid, 0.0, 0.0, 0.0, 1.0))
                    continue
                codes = np.array([LABEL_CODE[lab.get(r, unj)] for r in rows], np.int64)
                n = len(rows)
                pos = np.tile(np.arange(1, n + 1), sessions_per_query)
                c = np.tile(codes, sessions_per_query)
                ra = np.asarray(rows, np.int64)
                lp = D.log_price[ra].astype(np.float64)
                pz1 = np.nan_to_num((lp - np.nanmean(lp)) / (np.nanstd(lp) or 1.0)) if np.isfinite(lp).any() \
                    else np.zeros(n)
                st = np.tile(D.stars[ra].astype(np.float64), sessions_per_query)   # NaN -> neutral in the model
                pz = np.tile(pz1, sessions_per_query)
                o = simulate_interactions(sim_rng, pos, c, st, pz, params)
                clk = o["clicked"].reshape(sessions_per_query, n)
                cart = o["carted"].reshape(sessions_per_query, n)
                buy = o["purchased"].reshape(sessions_per_query, n)
                per_q[m].append((qid, float(clk.any(1).mean()), float(cart.any(1).mean()), float(buy.any(1).mean()),
                                 float((~clk.any(1)).mean())))
            frames = {m: pl.DataFrame(v, schema=["query_id", "ctr", "cart", "buy", "abandon"], orient="row")
                      for m, v in per_q.items()}
            base = frames["bm25"]
            res = {}
            for m, f in frames.items():
                rec = {k: M.mean_ci(f[k].to_numpy(), 1000, cfg.seed)[0] for k in ("ctr", "cart", "buy", "abandon")}
                if m != "bm25":
                    j = base.join(f, on="query_id", suffix="_m")
                    for k in ("ctr", "cart", "buy"):
                        d = M.paired_delta(j[k].to_numpy(), j[f"{k}_m"].to_numpy(), 1000, cfg.seed)
                        rec[f"{k}_rel_lift"] = d["delta"] / rec_b if (rec_b := float(base[k].mean())) else None
                        rec[f"{k}_delta_ci"] = [d["lo"], d["hi"]]
                    p0 = float(base["buy"].mean())
                    lift = rec.get("buy_rel_lift") or 0.0
                    rec["pilot_searches_per_arm_for_purchase"] = z_power_n(p0, lift)
                res[m] = rec
            results[name] = res
            log.info("%s: %s", name, {m: round(v["buy"], 4) for m, v in res.items()})
        out = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "corpus": eng.ds.name, "versions": eng.versions,
               "queries": len(items), "sessions_per_query": sessions_per_query, "methods": _METHODS,
               "labels": METHOD_LABELS, "click_model": cfg.get("simulation.traffic.click_model"),
               "assumptions": list(UNJUDGED_ASSUMPTIONS), "results": results,
               "note": "Simulated behaviour from a click model driven by ESCI labels: not evidence of conversion impact."}
        p = cfg.path("reports", "simulated_ab.json")
        p.write_text(json.dumps(out, indent=2, default=str))
        info.update(rows=len(ranked))
    return out

"""`csre simulate-pilot`: rehearse the A/B pilot end to end on simulated shoppers.

Simulated shoppers search through the real serving path (`SearchService.search` with a user id, so the experiment
assigns the arm, enforces the latency budget, caches and logs exposures), react to what they are shown through
the phase-1 click model, and send their clicks / carts / purchases through `SearchService.feedback`. The pilot
analysis (`csre analyze-experiment`) then runs on the logged events exactly as it would on real traffic.

What this is for: testing the instrumentation, the assignment, the analysis and the sample-size assumptions
before real shoppers are involved. What it is not: evidence of a real effect — behaviour is generated from the
same ESCI labels the rankers were evaluated on.

Shoppers differ: each has a purchase propensity (log-normal), which makes outcomes correlated within a user (the
reason the analysis is user-level) and gives CUPED a pre-period signal to remove.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import Config
from ..sim.click_model import LABEL_CODE, ClickModelParams, simulate_interactions
from ..utils import get_logger, stage_timer

log = get_logger("csre.pilot_sim")


def simulate_pilot(cfg: Config, corpus: str | None = None, n_users: int = 12_000, mean_searches: float = 3.0,
                   unjudged_label: str = "C") -> dict:
    from ..search.engine import Engine  # noqa: PLC0415
    from ..serve.experiment import analyze  # noqa: PLC0415
    from ..serve.service import SearchService  # noqa: PLC0415
    from .harness import _limit_threads, load_judged  # noqa: PLC0415

    E = dict(cfg.get("serve.experiment") or {})
    name = E.get("name", "pilot") + "_simulated"
    cfg.raw["serve"]["experiment"] = {**E, "enabled": True, "name": name, "salt": name}
    _limit_threads()
    with stage_timer(cfg, "simulate_pilot") as info:
        eng = Engine(cfg, corpus or "full")
        svc = SearchService(cfg, eng)
        q, judged = load_judged(eng, cfg.get("eval.split"))
        w = q["traffic_share"].fill_null(q["traffic_share"].min() or 1e-7).to_numpy() if "traffic_share" in q.columns \
            else np.ones(q.height)
        w = w / w.sum()
        params = ClickModelParams.from_config(cfg.get("simulation.traffic.click_model"))
        rng = np.random.default_rng(cfg.seed + 11)
        texts, locs, qids = q["query"].to_list(), q["locale"].to_list(), q["query_id"].to_list()
        t0 = time.time()
        n_searches = 0
        for u in range(n_users):
            uid = f"sim-{u:06d}"
            z = float(np.exp(rng.normal(-0.2, 0.6)))           # shopper's purchase propensity
            # pre-period: the same shopper's success rate before the experiment (CUPED covariate)
            n_pre = max(1, rng.poisson(5))
            pre = float(rng.binomial(n_pre, min(0.95, 0.08 * z)) / n_pre)
            for _ in range(max(1, rng.poisson(mean_searches))):
                i = int(rng.choice(len(texts), p=w))
                out = svc.search(texts[i], locs[i], user_id=uid, explain=False, pre_success=pre)
                n_searches += 1
                ids = [r["doc_id"] for r in out["results"]]
                if not ids:
                    continue
                rows, _, labels = judged[qids[i]]
                jl = dict(zip(rows.tolist(), labels.tolist()))
                shown = eng.idx[locs[i]].rows(ids).tolist()
                codes = np.array([LABEL_CODE[jl.get(r, unjudged_label)] for r in shown], np.int64)
                stars = np.array([r.get("stars") if r.get("stars") is not None else np.nan for r in out["results"]], float)
                o = simulate_interactions(rng, np.arange(1, len(ids) + 1), codes, stars, np.zeros(len(ids)), params)
                cart = o["carted"] & (rng.random(len(ids)) < min(1.0, z))
                buy = cart & o["purchased"]
                ex = out["experiment"]
                base = {"query": texts[i], "locale": locs[i], "user_id": uid, "search_id": ex["search_id"],
                        "experiment": ex["name"], "arm": ex["arm"], "method": out["method"]}
                for pos in np.flatnonzero(o["clicked"]):
                    svc.feedback({**base, "doc_id": ids[pos], "event": "click", "position": int(pos) + 1})
                    if cart[pos]:
                        svc.feedback({**base, "doc_id": ids[pos], "event": "cart", "position": int(pos) + 1})
                    if buy[pos]:
                        svc.feedback({**base, "doc_id": ids[pos], "event": "purchase", "position": int(pos) + 1})
            if (u + 1) % 2000 == 0:
                log.info("pilot: %d users, %d searches (%.0fs), cache %s", u + 1, n_searches, time.time() - t0,
                         svc.cache.stats())
        report = analyze(cfg, name)
        report["simulated"] = {"users": n_users, "searches": n_searches, "cache": svc.cache.stats(),
                               "unjudged_label": unjudged_label, "seconds": round(time.time() - t0, 1)}
        info.update(rows=n_searches, users=n_users)
    p = cfg.path("reports", f"experiment_{name}.json")
    import json  # noqa: PLC0415
    p.write_text(json.dumps(report, indent=2, default=str))
    return report

"""Online A/B experiments: deterministic assignment, exposure logging, and the pilot analysis.

Assignment
----------
A shopper (user id, e.g. a first-party cookie) is hashed with the experiment's salt into [0, 1); the allocation
splits that interval into arms. Assignment is deterministic and stateless, so every server agrees and a shopper
always sees the same ranking for the life of the experiment (randomisation by user, not by search).

Logging
-------
Every search served under an experiment writes an `exposure` event (arm, method actually served, model versions,
latency, returned products). Clicks, carts, purchases and thumbs arrive through `/api/feedback` with the same user
id. Events go to `data/feedback/events-YYYYMMDD.jsonl`.

Analysis (`csre analyze-experiment`)
------------------------------------
* Unit of analysis = user (searches by the same person are correlated): per-user ratios, user-level bootstrap.
* Sample-ratio-mismatch check (chi-square on users per arm): a failed SRM invalidates the comparison.
* CUPED: optional pre-period covariate (the user's metric before the experiment) to reduce variance.
* Primary metric: search success = share of a user's searches followed by an add-to-cart or purchase of a
  returned product. Secondary: CTR, purchases per search, zero-result rate. Guardrails: p95 latency, fallback rate.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config


@dataclass
class Experiment:
    name: str
    arms: dict[str, str]              # arm -> ranking method
    allocation: dict[str, float]      # arm -> share of users
    salt: str
    enabled: bool = True

    @classmethod
    def from_config(cls, cfg: Config) -> "Experiment | None":
        E = cfg.get("serve.experiment") or {}
        if not E.get("enabled"):
            return None
        arms = dict(E["arms"])
        alloc = dict(E.get("allocation") or {a: 1.0 / len(arms) for a in arms})
        tot = sum(alloc.values())
        return cls(E["name"], arms, {a: v / tot for a, v in alloc.items()}, str(E.get("salt", E["name"])))

    def bucket(self, user_id: str) -> float:
        h = hashlib.md5(f"{self.salt}|{user_id}".encode()).digest()
        return int.from_bytes(h[:8], "little") / 2.0 ** 64

    def assign(self, user_id: str) -> str:
        u, acc = self.bucket(user_id), 0.0
        for arm in sorted(self.arms):
            acc += self.allocation[arm]
            if u < acc:
                return arm
        return sorted(self.arms)[-1]


# ======================================================================================
# analysis
# ======================================================================================
def load_events(paths: list[Path], experiment: str) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                e = json.loads(line)
                if e.get("experiment") == experiment:
                    out.append(e)
    return out


def _srm_pvalue(counts: dict[str, int], alloc: dict[str, float]) -> float:
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    chi2 = sum((counts.get(a, 0) - n * p) ** 2 / (n * p) for a, p in alloc.items() if p > 0)
    df = max(1, len(alloc) - 1)
    # survival function of chi-square with df degrees of freedom (regularised upper gamma), df small
    from scipy.stats import chi2 as _chi2  # noqa: PLC0415
    return float(_chi2.sf(chi2, df))


def _user_table(events: list[dict]) -> dict[str, dict]:
    """Per user: searches, successful searches, clicked searches, purchases, zero-result searches, latencies."""
    users: dict[str, dict] = {}
    searches: dict[str, dict] = {}
    for e in events:
        u = e["user_id"]
        rec = users.setdefault(u, {"arm": e["arm"], "searches": 0, "success": 0, "clicked": 0, "purchases": 0,
                                   "zero": 0, "latency": [], "fallback": 0, "pre_success": e.get("pre_success")})
        if e["event"] == "exposure":
            rec["searches"] += 1
            rec["zero"] += int(e.get("n_results", 1) == 0)
            rec["latency"].append(e.get("engine_ms", np.nan))
            rec["fallback"] += int(bool(e.get("fallback_reason")))
            searches[e["search_id"]] = {"user": u, "click": False, "success": False}
        else:
            s = searches.get(e.get("search_id"))
            if s is None:
                continue
            if e["event"] == "click":
                s["click"] = True
            if e["event"] in ("cart", "purchase"):
                s["success"] = True
            if e["event"] == "purchase":
                rec["purchases"] += 1
    for s in searches.values():
        users[s["user"]]["clicked"] += int(s["click"])
        users[s["user"]]["success"] += int(s["success"])
    return users


def _boot_diff(a_num, a_den, b_num, b_den, n_boot=2000, seed=0):
    """Ratio-of-sums difference (b - a) with a user-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    def ratio(num, den, idx):
        return num[idx].sum() / max(den[idx].sum(), 1e-12)
    ia, ib = np.arange(len(a_num)), np.arange(len(b_num))
    point = ratio(b_num, b_den, ib) - ratio(a_num, a_den, ia)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = ratio(b_num, b_den, rng.choice(ib, len(ib))) - ratio(a_num, a_den, rng.choice(ia, len(ia)))
    return point, float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _cuped(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if ok.sum() < 10 or np.var(x[ok]) == 0:
        return y
    theta = np.cov(y[ok], x[ok])[0, 1] / np.var(x[ok])
    out = y.copy()
    out[ok] = y[ok] - theta * (x[ok] - x[ok].mean())
    return out


def analyze(cfg: Config, experiment: str | None = None, control: str | None = None) -> dict:
    exp = Experiment.from_config(cfg)
    name = experiment or (exp.name if exp else None)
    if name is None:
        raise SystemExit("no experiment configured (serve.experiment) and none given")
    paths = sorted(cfg.path("feedback").glob("events-*.jsonl"))
    events = load_events(paths, name)
    users = _user_table(events)
    arms = sorted({u["arm"] for u in users.values()})
    alloc = exp.allocation if exp and exp.name == name else {a: 1 / len(arms) for a in arms}
    counts = {a: sum(1 for u in users.values() if u["arm"] == a) for a in arms}
    control = control or ("control" if "control" in arms else arms[0])
    per_arm = {}
    for a in arms:
        us = [u for u in users.values() if u["arm"] == a and u["searches"] > 0]
        s = np.array([u["searches"] for u in us], float)
        lat = np.concatenate([np.array(u["latency"], float) for u in us]) if us else np.array([])
        per_arm[a] = {
            "users": len(us), "searches": int(s.sum()),
            "search_success": float(sum(u["success"] for u in us) / max(s.sum(), 1)),
            "ctr": float(sum(u["clicked"] for u in us) / max(s.sum(), 1)),
            "purchases_per_search": float(sum(u["purchases"] for u in us) / max(s.sum(), 1)),
            "zero_result_rate": float(sum(u["zero"] for u in us) / max(s.sum(), 1)),
            "fallback_rate": float(sum(u["fallback"] for u in us) / max(s.sum(), 1)),
            "latency_p50_ms": float(np.nanpercentile(lat, 50)) if len(lat) else None,
            "latency_p95_ms": float(np.nanpercentile(lat, 95)) if len(lat) else None,
        }
    comparisons = {}
    ctl = [u for u in users.values() if u["arm"] == control and u["searches"] > 0]
    for a in arms:
        if a == control:
            continue
        trt = [u for u in users.values() if u["arm"] == a and u["searches"] > 0]
        rec = {}
        for metric, key in (("search_success", "success"), ("ctr", "clicked"), ("purchases_per_search", "purchases")):
            an = np.array([u[key] for u in ctl], float)
            ad = np.array([u["searches"] for u in ctl], float)
            bn = np.array([u[key] for u in trt], float)
            bd = np.array([u["searches"] for u in trt], float)
            if metric == "search_success":   # CUPED on the per-user success rate when a pre-period exists
                xa = np.array([np.nan if u["pre_success"] is None else u["pre_success"] for u in ctl], float)
                xb = np.array([np.nan if u["pre_success"] is None else u["pre_success"] for u in trt], float)
                if (~np.isnan(np.concatenate([xa, xb]))).sum() >= 20:
                    ya, yb = an / np.maximum(ad, 1), bn / np.maximum(bd, 1)
                    y = _cuped(np.concatenate([ya, yb]), np.concatenate([xa, xb]))
                    an, bn = y[:len(ya)] * ad, y[len(ya):] * bd
                    rec["cuped"] = True
            point, lo, hi = _boot_diff(an, ad, bn, bd, seed=cfg.seed)
            base = per_arm[control][metric]
            rec[metric] = {"delta": point, "lo": lo, "hi": hi, "relative": point / base if base else None,
                           "significant": not (lo <= 0 <= hi)}
        rec["guardrails"] = {
            "latency_p95_ms": {"control": per_arm[control]["latency_p95_ms"], "treatment": per_arm[a]["latency_p95_ms"],
                               "budget_ms": cfg.get("serve.latency_budget_ms"),
                               "ok": (per_arm[a]["latency_p95_ms"] or 0) <= float(cfg.get("serve.latency_budget_ms"))},
            "fallback_rate": {"control": per_arm[control]["fallback_rate"], "treatment": per_arm[a]["fallback_rate"]},
        }
        comparisons[a] = rec
    srm_p = _srm_pvalue(counts, alloc)
    report = {"experiment": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "events": len(events),
              "users_by_arm": counts, "allocation": alloc, "srm_pvalue": srm_p, "srm_ok": bool(srm_p > 0.001),
              "control": control, "arms": per_arm, "comparisons": comparisons,
              "methods": exp.arms if exp and exp.name == name else None}
    out = cfg.path("reports", f"experiment_{name}.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    out.with_suffix(".md").write_text(_markdown(report))
    return report


def _markdown(r: dict) -> str:
    L = [f"# Experiment `{r['experiment']}`", "",
         f"{sum(r['users_by_arm'].values()):,} users, {r['events']:,} events. Sample-ratio check p = {r['srm_pvalue']:.3f} "
         f"({'ok' if r['srm_ok'] else 'FAILED: do not trust the comparison'}).", "",
         "| Arm | method | users | searches | search success | CTR | purchases / search | zero results | p95 ms |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a, v in r["arms"].items():
        m = (r.get("methods") or {}).get(a, "")
        L.append(f"| {a} | {m} | {v['users']:,} | {v['searches']:,} | {v['search_success']:.4f} | {v['ctr']:.4f} | "
                 f"{v['purchases_per_search']:.4f} | {v['zero_result_rate']:.4f} | "
                 f"{v['latency_p95_ms'] if v['latency_p95_ms'] is None else round(v['latency_p95_ms'], 1)} |")
    for a, c in r["comparisons"].items():
        L += ["", f"## {a} vs {r['control']}", "", "| Metric | delta | 95% CI | relative | significant |", "|---|---|---|---|---|"]
        for k in ("search_success", "ctr", "purchases_per_search"):
            d = c[k]
            rel = "–" if d["relative"] is None else f"{d['relative']:+.1%}"
            L.append(f"| {k.replace('_', ' ')} | {d['delta']:+.4f} | [{d['lo']:+.4f}, {d['hi']:+.4f}] | {rel} | "
                     f"{'yes' if d['significant'] else 'no'} |")
        g = c["guardrails"]["latency_p95_ms"]
        L.append(f"\nGuardrail p95 latency: {g['treatment']:.1f} ms vs budget {g['budget_ms']} ms "
                 f"({'ok' if g['ok'] else 'BREACHED'}).")
    return "\n".join(L) + "\n"


def required_users(p0: float, rel_lift: float, searches_per_user: float, icc: float = 0.05,
                   alpha: float = 0.05, power: float = 0.8) -> float:
    """Users per arm for a per-search proportion, inflated by the design effect of clustering searches by user."""
    from statistics import NormalDist  # noqa: PLC0415
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    p1 = p0 * (1 + rel_lift)
    n_searches = z ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2
    deff = 1 + (searches_per_user - 1) * icc
    return math.ceil(n_searches * deff / searches_per_user)

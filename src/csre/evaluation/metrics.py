"""Ranking metrics with ESCI gains, honest handling of unjudged results, and bootstrap intervals.

Two evaluation modes, decided in phase 1 and used for every method:

* **rerank** (ESCI task 1): rank a query's *judged* candidate set. Every result has a label, so nDCG is exact.
* **retrieval**: rank the full catalog. Most retrieved products are unjudged *for this query* — unknown, not
  irrelevant. We report the condensed nDCG@k (unjudged results removed before cutting at k; Sakai 2007),
  a pessimistic bound (unjudged counted as 0), judged@k (how much of the top-k we can actually assess) and
  recall of the query's Exact products at 100 (valid because Exact labels are known positives).
"""
from __future__ import annotations

import numpy as np

_DISC = 1.0 / np.log2(np.arange(2, 1002))


def dcg(gains: np.ndarray, k: int) -> float:
    g = np.asarray(gains, dtype=np.float64)[:k]
    return float((g * _DISC[: len(g)]).sum())


def ndcg(ranked_gains: np.ndarray, all_gains: np.ndarray, k: int) -> float | None:
    ideal = dcg(np.sort(np.asarray(all_gains, dtype=np.float64))[::-1], k)
    return dcg(ranked_gains, k) / ideal if ideal > 0 else None


def rerank_metrics(scores: np.ndarray, gains: np.ndarray, labels: np.ndarray, tiebreak: np.ndarray,
                   k: int = 10) -> dict:
    order = np.lexsort((tiebreak, -np.asarray(scores, dtype=np.float64)))
    g = gains[order]
    lab = labels[order]
    first_e = np.flatnonzero(lab == "E")
    return {
        "ndcg10": ndcg(g, gains, k),
        "ndcg5": ndcg(g, gains, 5),
        "mrr_e": float(1.0 / (first_e[0] + 1)) if len(first_e) else None,
        "p1_e": float(lab[0] == "E") if len(lab) else None,
    }


def retrieval_metrics(ranked: np.ndarray, judged: dict[int, tuple[float, str]], k: int = 10,
                      recall_k: int = 100) -> dict:
    """ranked: product rows in rank order; judged: row -> (gain, ESCI label) for this query."""
    ranked = list(ranked)
    all_g = np.array([g for g, _ in judged.values()])
    n_e = sum(1 for _, lab in judged.values() if lab == "E")
    cond = [judged[r][0] for r in ranked if r in judged]
    lb = [judged[r][0] if r in judged else 0.0 for r in ranked[:k]]
    top_k = ranked[:k]
    top_r = set(ranked[:recall_k])
    e_rows = {r for r, (_, lab) in judged.items() if lab == "E"}
    return {
        "ndcg10_cond": ndcg(np.array(cond), all_g, k),
        "ndcg10_lb": ndcg(np.array(lb), all_g, k),
        "judged10": float(sum(r in judged for r in top_k) / k),
        "recall100_e": float(len(e_rows & top_r) / n_e) if n_e else None,
        "success10_e": float(any(r in e_rows for r in top_k)) if n_e else None,
        "n_returned": len(ranked),
    }


def mean_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x))) if len(x) * n_boot <= 5e7 else None
    if idx is None:  # large n: normal approximation is indistinguishable from the bootstrap
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        return float(x.mean()), float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean of (b - a) over queries scored by both, 95% bootstrap CI, and share of resamples with delta <= 0."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = b[ok] - a[ok]
    if len(d) == 0:
        return {"delta": None, "lo": None, "hi": None, "p_le_0": None, "n": 0}
    rng = np.random.default_rng(seed)
    if len(d) * n_boot <= 5e7:
        m = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
        lo, hi, p = np.quantile(m, 0.025), np.quantile(m, 0.975), float((m <= 0).mean())
    else:
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        from math import erf, sqrt  # noqa: PLC0415
        z = d.mean() / se if se > 0 else np.inf
        p = float(0.5 * (1 - erf(z / sqrt(2))))
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_le_0": p, "n": int(len(d))}


def latency_summary(ms: np.ndarray) -> dict:
    ms = np.asarray(ms, dtype=np.float64)
    if len(ms) == 0:
        return {}
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)), "max": float(ms.max()), "n": int(len(ms))}


def cost_per_million(mean_ms: float, vcpus: int, usd_per_hour: float, utilisation: float) -> dict:
    """Compute-only serving cost for single-threaded request workers (one per vCPU)."""
    qps_per_core = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    qps = qps_per_core * vcpus * utilisation
    hours = 1e6 / qps / 3600.0
    return {"qps_per_core": qps_per_core, "qps_per_instance": qps, "usd_per_million": hours * usd_per_hour}

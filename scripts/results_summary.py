"""Print the README's headline result tables from data/reports/*.json (so README numbers are reproducible)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "data" / "reports"
LABEL = {"bm25": "Keyword (BM25)", "dense": "Semantic (dense)", "hybrid_rrf": "Hybrid (RRF)",
         "hybrid": "Hybrid (score fusion)", "ltr": "Reranker (LambdaMART)", "ltr_fb": "Reranker + feedback ¹"}


def f(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def delta(d):
    if not d or d.get("delta") is None:
        return "–"
    return f"{d['delta']:+.4f}" + ("" if d["lo"] <= 0 <= d["hi"] else " *")


def main() -> None:
    e = json.loads((R / "eval_full_test.json").read_text())
    rr, rt, la = e["rerank"], e["retrieval"], e["latency"]["methods"]
    out = [f"Test split: {e['n_queries']:,} held-out queries (rerank), {rt['n_queries']:,} sampled for full-catalog "
           "retrieval; 1.8M products in three markets. `*` = 95% paired-bootstrap interval excludes zero.", "",
           "| Approach | Rerank nDCG@10 | Δ vs BM25 | Retrieval nDCG@10 (condensed) | Δ vs BM25 | Recall@100 (Exact) "
           "| p50 / p95 ms | $ per 1M queries |",
           "|---|---|---|---|---|---|---|---|"]
    for m in e["methods"]:
        lat = la.get(m, {})
        w = lat.get("wall_ms", {})
        out.append(f"| {LABEL[m]} | {f(rr['overall'][m]['ndcg10']['mean'])} | "
                   f"{'–' if m == 'bm25' else delta(rr['delta_vs_bm25'].get(m))} | "
                   f"{f(rt['overall'][m]['ndcg10_cond']['mean'])} | "
                   f"{'–' if m == 'bm25' else delta(rt['delta_vs_bm25'].get(m))} | "
                   f"{f(rt['overall'][m]['recall100_e']['mean'], 3)} | "
                   f"{f(w.get('p50'), 1)} / {f(w.get('p95'), 1)} | ${f(lat.get('cost', {}).get('usd_per_million'), 3)} |")
    out += ["", "¹ Uses clicks simulated from the same relevance labels: an upper bound on what feedback could add, "
            "not an estimate of it.", ""]
    sl = rr["by"]["slice"]
    out += ["| Rerank nDCG@10 by slice | queries | BM25 | Dense | Hybrid | Reranker | Reranker gain |",
            "|---|---|---|---|---|---|---|"]
    for k in ["ambiguous", "underspecified", "multi_intent", "negation", "spec", "brand", "sparse_products", "hard"]:
        v = sl.get(k)
        if not v:
            continue
        out.append(f"| {k.replace('_', ' ')} | {v['bm25']['n']:,} | {f(v['bm25']['mean'])} | {f(v['dense']['mean'])} | "
                   f"{f(v['hybrid']['mean'])} | {f(v['ltr']['mean'])} | {v['ltr']['mean'] - v['bm25']['mean']:+.4f} |")
    for loc, v in sorted(rr["by"]["locale"].items()):
        out.append(f"| market {loc} | {v['bm25']['n']:,} | {f(v['bm25']['mean'])} | {f(v['dense']['mean'])} | "
                   f"{f(v['hybrid']['mean'])} | {f(v['ltr']['mean'])} | {v['ltr']['mean'] - v['bm25']['mean']:+.4f} |")
    if "robustness" in e:
        t = {(x["variant"], x["method"], x["text"]): x["ndcg10_cond"] for x in e["robustness"]["table"]}
        out += ["", "| Query variant (replay stream) | BM25 Δ | Dense Δ | Reranker Δ |", "|---|---|---|---|"]
        for v in ["typo", "token_drop", "modifier", "reorder", "case_space"]:
            cells = []
            for m in ["bm25", "dense", "ltr"]:
                o, x = t.get((v, m, "original")), t.get((v, m, "variant"))
                cells.append("–" if o is None or x is None else f"{x - o:+.4f}")
            out.append(f"| {v.replace('_', ' ')} | " + " | ".join(cells) + " |")
    sim = R / "simulated_ab.json"
    if sim.exists():
        s = json.loads(sim.read_text())
        key = next(k for k in s["results"] if k.startswith("neutral"))
        out += ["", f"Simulated A/B ({s['queries']:,} traffic-weighted queries × {s['sessions_per_query']} sessions, "
                f"{key}):", "", "| Approach | purchases / search | lift vs BM25 | searches per arm to detect |",
                "|---|---|---|---|"]
        for m in s["methods"]:
            r = s["results"][key][m]
            n = r.get("pilot_searches_per_arm_for_purchase")
            out.append(f"| {LABEL[m]} | {r['buy'] * 100:.2f}% | "
                       f"{'–' if m == 'bm25' else format(r.get('buy_rel_lift') or 0, '+.1%')} | "
                       f"{'–' if not n or n == float('inf') else f'{n:,.0f}'} |")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()

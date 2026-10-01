"""Render an evaluation report (JSON from `harness.evaluate`) as Markdown."""
from __future__ import annotations

from pathlib import Path

from ..config import Config


def _f(x, nd=4):
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def _ci(rec: dict | None, nd=4) -> str:
    if not rec:
        return "–"
    return f"{_f(rec['mean'], nd)} [{_f(rec['lo'], nd)}, {_f(rec['hi'], nd)}]"


def _delta(d: dict | None) -> str:
    if not d or d.get("delta") is None:
        return "–"
    sig = "" if d["lo"] <= 0 <= d["hi"] else " *"
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{sig}"


def write_markdown(cfg: Config, r: dict, path: Path) -> None:
    L = r.get("labels", {})
    out = [f"# Evaluation — corpus `{r['corpus']}`, split `{r['split']}`", "",
           f"{r['n_queries']:,} judged queries · k = {r['k']} · model versions: "
           + ", ".join(f"`{k}={v}`" for k, v in r.get("versions", {}).items()), "",
           "`*` = 95% paired-bootstrap interval excludes 0.", ""]
    if "rerank" in r:
        rr = r["rerank"]
        out += ["## Ranking quality on judged candidates (ESCI task 1)", "",
                "Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.", "",
                "| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |",
                "|---|---|---|---|---|---|"]
        for m, v in rr["overall"].items():
            out.append(f"| {L.get(m, m)} | {_ci(v.get('ndcg10'))} | {_delta(rr['delta_vs_bm25'].get(m)) if m != 'bm25' else '–'} "
                       f"| {_f(v.get('ndcg5', {}).get('mean'))} | {_f(v.get('mrr_e', {}).get('mean'))} "
                       f"| {_f(v.get('p1_e', {}).get('mean'))} |")
        out.append("")
    if "retrieval" in r:
        rt = r["retrieval"]
        out += ["## Full-catalog retrieval", "",
                f"{rt['n_queries']:,} sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower "
                "bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.", "",
                "| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |",
                "|---|---|---|---|---|---|---|"]
        for m, v in rt["overall"].items():
            out.append(f"| {L.get(m, m)} | {_ci(v.get('ndcg10_cond'))} | {_delta(rt['delta_vs_bm25'].get(m)) if m != 'bm25' else '–'} "
                       f"| {_f(v.get('ndcg10_lb', {}).get('mean'))} | {_f(v.get('judged10', {}).get('mean'), 3)} "
                       f"| {_f(v.get('recall100_e', {}).get('mean'))} | {_f(v.get('success10_e', {}).get('mean'), 3)} |")
        out.append("")
    for mode, metric in (("rerank", "nDCG@10"), ("retrieval", "condensed nDCG@10")):
        if mode not in r:
            continue
        by = r[mode]["by"]
        methods = list(r[mode]["overall"])
        rows = []
        for group in ("locale", "traffic_bucket", "length_bucket"):
            for k, v in sorted(by.get(group, {}).items()):
                n = max((x["n"] for x in v.values()), default=0)
                rows.append((f"{group}={k}", n, v))
        for k, v in by.get("slice", {}).items():
            n = max((x["n"] for x in v.values()), default=0)
            rows.append((k, n, v))
        if rows:
            out += [f"### {metric} by slice ({mode})", "",
                    "| Slice | queries | " + " | ".join(L.get(m, m) for m in methods) + " |",
                    "|---|---|" + "---|" * len(methods)]
            for name, n, v in rows:
                best = max((v[m]["mean"] for m in methods if m in v and v[m]["mean"] is not None), default=None)
                cells = []
                for m in methods:
                    x = v.get(m, {}).get("mean")
                    s = _f(x)
                    cells.append(f"**{s}**" if x is not None and best is not None and abs(x - best) < 1e-12 else s)
                out.append(f"| {name} | {n:,} | " + " | ".join(cells) + " |")
            out.append("")
    if "latency" in r:
        la = r["latency"]
        C = la["cost_assumptions"]
        out += ["## Latency and serving cost", "",
                f"{la['n_queries']} queries, single-threaded, warm process, k = 10. Cost = compute only on "
                f"`{C['instance']}` ({C['vcpus']} vCPU, ${C['usd_per_hour']}/h) at {int(C['target_utilisation'] * 100)}% "
                "utilisation; an assumption to edit in `configs/search.yaml`.", "",
                "| Method | p50 ms | p95 ms | p99 ms | QPS / core | $ / 1M queries | stage means (ms) |",
                "|---|---|---|---|---|---|---|"]
        for m, v in la["methods"].items():
            w = v["wall_ms"]
            st = ", ".join(f"{k} {x:.1f}" for k, x in v["stages_mean_ms"].items())
            out.append(f"| {L.get(m, m)} | {w['p50']:.1f} | {w['p95']:.1f} | {w['p99']:.1f} | "
                       f"{v['cost']['qps_per_core']:.0f} | ${v['cost']['usd_per_million']:.3f} | {st} |")
        out.append("")
    if "robustness" in r:
        rb = r["robustness"]
        out += ["## Robustness to query variants (replay stream)", "",
                f"{rb['n_requests']:,} one-off variants of test queries, scored against the parent query's labels.", "",
                "| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |", "|---|---|---|---|---|"]
        tab: dict = {}
        for row in rb["table"]:
            tab.setdefault((row["variant"], row["method"]), {})[row["text"]] = row
        for (var, m), d in sorted(tab.items()):
            o, v = d.get("original", {}).get("ndcg10_cond"), d.get("variant", {}).get("ndcg10_cond")
            dl = (v - o) if (o is not None and v is not None) else None
            out.append(f"| {var} | {L.get(m, m)} | {_f(o)} | {_f(v)} | {_f(dl) if dl is None else f'{dl:+.4f}'} |")
        out.append("")
    path.write_text("\n".join(out))

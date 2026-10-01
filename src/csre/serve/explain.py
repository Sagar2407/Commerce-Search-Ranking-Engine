"""Result explanations grounded in product attributes and the ranker's own feature contributions.

For every result we can say, from data rather than prose:
* which query words the product title / text actually contains (BM25 postings),
* which stated attributes it matches or contradicts ("12 oz" ✓, "size 8" ✗ product says "size 10"),
* whether the brand matches, whether it contains an excluded term ("without lactose"),
* how semantically close it is (dense cosine) — and whether it was found *only* semantically,
* how well its department fits the query,
* for the LambdaMART rankers, which feature groups pushed it up or down (exact TreeSHAP contributions),
* what kind of result it probably is (Exact / Substitute / Complement / Irrelevant).
"""
from __future__ import annotations

import numpy as np

from ..search.features import ALL_FEATURES, FEATURE_GROUPS
from ..search.query import ATTR_TYPES

ATTR_LABEL = {"measures": "size / volume", "dimensions": "dimensions", "sizes": "size", "audience": "for",
              "compat": "compatible with", "materials": "material", "colors": "colour", "pack": "pack of",
              "brand": "brand"}

FEATURE_LABEL = {
    "bm25_text": "keyword match (all text)", "bm25_text_norm": "keyword match strength", "bm25_title": "keyword match (title)",
    "bm25_title_norm": "title match strength", "cov_text": "share of query words in text", "cov_title": "share of query words in title",
    "all_terms_title": "every query word in title", "dense_cos": "semantic similarity", "p_cat": "department fits query",
    "q_cat_entropy": "query vagueness", "q_cat_top": "query department confidence", "log_price": "price",
    "stars": "star rating", "log_ratings": "number of ratings", "in_stock": "in stock", "popularity_z": "popularity",
    "is_sparse": "thin description", "log_richness": "description richness", "title_len": "title length",
    "title_only": "title-only listing", "cat_known": "department known", "cat_conf": "department confidence",
    "neg_violation": "contains an excluded term", "fb_ctr_ips": "shopper click-through (debiased)",
    "fb_cart_rate": "add-to-cart rate", "fb_buy_rate": "purchase rate", "fb_thumbs": "thumbs up/down",
    "fb_log_impr": "times shown", "fb_q_log_impr": "query traffic",
}
for a in [*ATTR_TYPES, "pack", "brand"]:
    FEATURE_LABEL[f"{a}_match"] = f"{ATTR_LABEL[a]} matches"
    FEATURE_LABEL[f"{a}_conflict"] = f"{ATTR_LABEL[a]} contradicts query"
    FEATURE_LABEL[f"{a}_q"] = f"query states {ATTR_LABEL[a]}"

_GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
TYPE_LABEL = {"E": "Exact match", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}


def attribute_checks(parsed, prod: dict) -> list[dict]:
    out = []
    for a in ATTR_TYPES:
        qv = parsed.attrs.get(a) or []
        if not qv:
            continue
        pv = prod.get(f"attr_{a}") or []
        hit = sorted(set(qv) & set(pv))
        status = "match" if hit else ("conflict" if pv else "unknown")
        out.append({"type": a, "label": ATTR_LABEL[a], "query": qv, "product": pv, "status": status})
    if parsed.pack_count is not None:
        pk = prod.get("attr_pack_count")
        out.append({"type": "pack", "label": ATTR_LABEL["pack"], "query": [str(parsed.pack_count)],
                    "product": [str(pk)] if pk else [],
                    "status": "unknown" if not pk else ("match" if pk == parsed.pack_count else "conflict")})
    if parsed.brands:
        b = (prod.get("brand") or "").lower()
        status = "match" if b and b in parsed.brands else ("conflict" if b else "unknown")
        out.append({"type": "brand", "label": "brand", "query": parsed.brands, "product": [prod.get("brand")] if b else [],
                    "status": status})
    return out


def shap_summary(contrib: np.ndarray, names: list[str], x: np.ndarray, top_n: int = 3) -> dict:
    """contrib: per-feature contributions (+ bias last) for one row."""
    vals = contrib[:-1]
    groups: dict[str, float] = {}
    for n, v in zip(names, vals):
        g = _GROUP_OF.get(n, "other")
        groups[g] = groups.get(g, 0.0) + float(v)
    order = np.argsort(-np.abs(vals))[:top_n]
    top = [{"feature": names[i], "label": FEATURE_LABEL.get(names[i], names[i]), "contribution": round(float(vals[i]), 4),
            "value": None if np.isnan(x[i]) else round(float(x[i]), 4)} for i in order if abs(vals[i]) > 1e-6]
    return {"by_group": {k: round(v, 4) for k, v in sorted(groups.items(), key=lambda kv: -abs(kv[1]))},
            "top": top, "bias": round(float(contrib[-1]), 4)}


def explain_results(engine, resp, products: list[dict], top_n: int = 3) -> list[dict]:
    ix = engine.idx[resp.locale]
    parsed = resp.context.parsed if resp.context else engine.parser.parse(resp.query, resp.locale)
    names = resp.feature_names
    contrib = None
    if resp.features is not None and resp.served_by in engine.models:
        contrib = engine.models[resp.served_by].predict(resp.features, pred_contrib=True, num_threads=1)
    probs = engine.esci_class_proba(resp.features, names) if resp.features is not None else None
    if probs is None and resp.context is not None and "esci_class" in engine.models and len(resp.rows):
        X = engine.builders[resp.locale].build(resp.context, resp.rows, with_feedback=False)
        probs = engine.esci_class_proba(X, ALL_FEATURES[:X.shape[1]])
    qv = resp.context.qvec if resp.context else None
    cos = ix.vectors.score_docs(qv, resp.rows) if (qv is not None and ix.vectors is not None and len(resp.rows)) else None
    cp = resp.context.cat_proba if resp.context else None
    out = []
    for i, (row, prod) in enumerate(zip(resp.rows.tolist(), products)):
        title_terms = ix.bm25_title.matched_terms(resp.query, row)
        text_terms = ix.bm25_text.matched_terms(resp.query, row)
        e = {
            "matched_title_terms": title_terms,
            "matched_text_terms": text_terms,
            "attributes": attribute_checks(parsed, prod),
            "negation_violation": bool(parsed.negated_terms and set(parsed.negated_terms) & set(title_terms)),
            "semantic_similarity": None if cos is None else round(float(cos[i]), 4),
            "semantic_only": bool(not text_terms),
        }
        if cp is not None and engine.qcat is not None and prod.get("category") in engine.qcat.class_index:
            e["department_fit"] = round(float(cp[engine.qcat.class_index[prod["category"]]]), 4)
        if contrib is not None:
            e["ranker"] = shap_summary(contrib[i], names, resp.features[i], top_n)
        if probs is not None:
            p = probs[i]
            labs = ["I", "C", "S", "E"]
            k = int(np.argmax(p))
            e["result_type"] = {"label": labs[k], "name": TYPE_LABEL[labs[k]],
                                "proba": {TYPE_LABEL[l]: round(float(p[j]), 3) for j, l in enumerate(labs)}}
        out.append(e)
    return out

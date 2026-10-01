"""Training stages (all on ESCI *train* queries; dev for early stopping / tuning; test untouched).

    csre train-dense    two-tower subword-bag encoder              -> registry: dense_encoder
    csre index          BM25 + dense indexes for a corpus          -> data/indexes/<corpus>/
    csre train-qcat     query -> department model                  -> registry: query_category
    csre tune-hybrid    convex-fusion weight alpha on dev          -> registry: hybrid
    csre train-ltr      LambdaMART rerankers (content / +feedback) -> registry: ltr, ltr_fb
                        + E/S/C/I result-type classifier           -> registry: esci_class
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np
import polars as pl

from ..config import Config
from ..evaluation import metrics as M
from ..utils import get_logger, stage_timer
from .corpus import Dataset
from .dense import encoder_text, train_encoder
from .features import ALL_FEATURES, CONTENT_FEATURES, FEATURE_GROUPS
from .query import QueryCategoryModel
from .registry import ModelRegistry

log = get_logger("csre.train")

LABEL_CODE = {"I": 0, "C": 1, "S": 2, "E": 3}
CODE_LABEL = {v: k for k, v in LABEL_CODE.items()}


def _queries(ds: Dataset, split: str) -> pl.DataFrame:
    return ds.queries().filter(pl.col("split") == split)


# ======================================================================================
# dense encoder
# ======================================================================================
def train_dense(cfg: Config, corpus: str = "full") -> str:
    ds = Dataset(cfg, corpus)
    P = cfg.get("search.dense")
    with stage_timer(cfg, "train_dense") as info:
        q = _queries(ds, "train").select("query_id", "query", "locale").sort("query_id").with_row_index("qi")
        j = ds.judgments(["query_id", "doc_id", "locale", "esci_label", "split"]).filter(pl.col("split") == "train")
        neg_labels = list(P["train"].get("negative_labels", ["I", "C"]))
        j = j.filter(pl.col("esci_label").is_in(["E", *neg_labels]))
        docs = j.select("doc_id", "locale").unique().sort("doc_id")
        texts = []
        for loc in sorted(docs["locale"].unique().to_list()):
            cat = ds.catalog(loc, ["doc_id", "doc_text", "n_title_chars"]).join(
                docs.filter(pl.col("locale") == loc).select("doc_id"), on="doc_id", how="semi")
            texts.append(cat.with_columns(pl.Series("t", encoder_text(cat, int(P["doc_bullet_chars"]))))
                         .select("doc_id", "t", pl.lit(loc).alias("locale")))
        dt = pl.concat(texts).sort("doc_id").with_row_index("di")
        j = j.join(q.select("query_id", "qi"), on="query_id").join(dt.select("doc_id", "di"), on="doc_id")
        pos = j.filter(pl.col("esci_label") == "E")
        neg = j.filter(pl.col("esci_label") != "E").sort("qi")
        nq = q.height
        neg_ptr = np.zeros(nq + 1, np.int64)
        np.add.at(neg_ptr, neg["qi"].to_numpy().astype(np.int64) + 1, 1)
        neg_ptr = np.cumsum(neg_ptr)
        log.info("dense training: %d queries, %d products, %d positive pairs, %d hard negatives",
                 nq, dt.height, pos.height, neg.height)

        dev_fn = _dense_dev_fn(cfg, ds)
        enc, hist = train_encoder(
            q["query"].to_list(), q["locale"].to_numpy(), dt["t"].to_list(),
            pos["qi"].to_numpy().astype(np.int64), pos["di"].to_numpy().astype(np.int64),
            neg_ptr, neg["di"].to_numpy().astype(np.int64), dt["locale"].to_numpy(), P, cfg.seed, dev_fn)
        version = ModelRegistry(cfg).register(
            "dense_encoder", enc.save, params={k: v for k, v in P.items() if k != "ann"},
            metrics={"history": hist, **{k: v for k, v in hist[-1].items() if k.startswith("dev_")}},
            data={"train_queries": nq, "train_products": dt.height, "positive_pairs": pos.height,
                  "hard_negatives": neg.height})
        info.update(rows=pos.height, version=version, dev=hist[-1])
    return version


def _dense_dev_fn(cfg: Config, ds: Dataset, n: int = 3000):
    """Dev rerank nDCG@10 of the encoder alone (judged candidates of sampled dev queries)."""
    q = _queries(ds, "dev").sample(n=min(n, ds.queries().filter(pl.col("split") == "dev").height), seed=cfg.seed)
    j = ds.judgments(["query_id", "doc_id", "locale", "gain", "split"]).filter(pl.col("split") == "dev") \
          .join(q.select("query_id"), on="query_id", how="semi")
    docs = j.select("doc_id", "locale").unique().sort("doc_id")
    parts = []
    for loc in sorted(docs["locale"].unique().to_list()):
        cat = ds.catalog(loc, ["doc_id", "doc_text", "n_title_chars"]).join(
            docs.filter(pl.col("locale") == loc).select("doc_id"), on="doc_id", how="semi")
        parts.append(cat.with_columns(pl.Series("t", encoder_text(cat, int(cfg.get("search.dense.doc_bullet_chars")))))
                     .select("doc_id", "t"))
    dt = pl.concat(parts).sort("doc_id").with_row_index("di")
    j = j.join(dt.select("doc_id", "di"), on="doc_id")
    qtext = dict(zip(q["query_id"].to_list(), q["query"].to_list()))
    # one pass over the groups: Polars does not guarantee group order across separate group_by calls
    gq, groups = [], []
    for (qid,), g in j.group_by(["query_id"]):
        gq.append(qtext[qid])
        groups.append((g["di"].to_numpy(), g["gain"].to_numpy().astype(np.float64)))
    cache: dict = {}

    def fn(enc) -> dict:
        if "X" not in cache:
            cache["X"] = enc.featurize(dt["t"].to_list())
        D = enc.encode_features(cache["X"])
        nd = []
        for text, (di, gains) in zip(gq, groups):
            s = D[di] @ enc.encode_query(text)
            v = M.ndcg(gains[np.lexsort((di, -s))], gains, 10)
            if v is not None:
                nd.append(v)
        return {"dev_ndcg10": round(float(np.mean(nd)), 4)}

    return fn


# ======================================================================================
# query -> department
# ======================================================================================
def train_qcat(cfg: Config, corpus: str = "full") -> str:
    ds = Dataset(cfg, corpus)
    with stage_timer(cfg, "train_qcat") as info:
        q = ds.queries()
        lab = q.filter(pl.col("e_top_category").is_not_null() & (pl.col("e_top_category") != "Unknown"))
        tr, dv = lab.filter(pl.col("split") == "train"), lab.filter(pl.col("split") == "dev")
        m = QueryCategoryModel.fit(tr["query"].to_list(), tr["e_top_category"].to_list(),
                                   tr["e_top_category_share"].fill_null(1.0).to_numpy(), seed=cfg.seed)
        p = m.predict_proba(dv["query"].to_list())
        pred = np.asarray(m.classes)[p.argmax(1)]
        acc = float((pred == dv["e_top_category"].to_numpy()).mean())
        conf = p.max(1)
        tau = float(cfg.get("search.query_category.min_confidence", 0.35))
        metrics = {"dev_accuracy": round(acc, 4),
                   "dev_accuracy_at_tau": round(float((pred == dv["e_top_category"].to_numpy())[conf >= tau].mean()), 4),
                   "dev_coverage_at_tau": round(float((conf >= tau).mean()), 4), "tau": tau, "classes": m.classes}
        version = ModelRegistry(cfg).register("query_category", m.save, metrics=metrics,
                                              data={"train_queries": tr.height, "dev_queries": dv.height})
        log.info("query category model %s: %s", version, metrics)
        info.update(rows=tr.height, version=version, **{k: metrics[k] for k in ("dev_accuracy",)})
    return version


# ======================================================================================
# LTR feature extraction (parallel over queries, engine shared copy-on-write)
# ======================================================================================
_ENG = None
_JUD: dict = {}


def _feat_batch(batch):
    from ..evaluation.harness import _limit_threads  # noqa: PLC0415
    _limit_threads()
    out = []
    for qid, query, loc in batch:
        rows, grades = _JUD[qid]
        ctx = _ENG.context(query, loc)
        X = _ENG.builders[loc].build(ctx, rows, with_feedback=True)
        out.append((qid, X, grades, rows))
    return out


def extract_features(eng, q: pl.DataFrame, split: str, n_jobs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features (ALL_FEATURES) for the judged candidates of the given queries."""
    global _ENG, _JUD
    _ENG = eng
    j = eng.ds.judgments(["query_id", "doc_id", "locale", "grade", "split"]).filter(pl.col("split") == split) \
           .join(q.select("query_id"), on="query_id", how="semi")
    _JUD = {}
    for (loc,), g in j.group_by(["locale"]):
        if loc not in eng.idx:
            continue
        g = g.with_columns(pl.Series("row", eng.idx[loc].rows(g["doc_id"].to_list()))).filter(pl.col("row") >= 0)
        for (qid,), gg in g.sort("row").group_by(["query_id"]):
            _JUD[qid] = (gg["row"].to_numpy(), gg["grade"].to_numpy().astype(np.int32))
    items = [(a, b, c) for a, b, c in zip(q["query_id"].to_list(), q["query"].to_list(), q["locale"].to_list())
             if a in _JUD]
    eng.parser.prime([b for _, b, _ in items], [c for _, _, c in items])   # no Polars in forked workers
    batches = [items[i:i + 200] for i in range(0, len(items), 200)]
    res = []
    with mp.get_context("fork").Pool(n_jobs) as pool:
        for i, r in enumerate(pool.imap(_feat_batch, batches)):
            res += r
            if (i + 1) % max(1, len(batches) // 10) == 0:
                log.info("  features %s: %d/%d", split, i + 1, len(batches))
    qids = np.concatenate([np.full(len(r[2]), r[0], np.int64) for r in res])
    X = np.vstack([r[1] for r in res])
    y = np.concatenate([r[2] for r in res])
    rows = np.concatenate([r[3] for r in res])
    return X, y, qids, rows


def _group_sizes(qids: np.ndarray) -> np.ndarray:
    _, idx, cnt = np.unique(qids, return_index=True, return_counts=True)
    return cnt[np.argsort(idx)]


def _dev_ndcg(scores: np.ndarray, y: np.ndarray, qids: np.ndarray, gains: list[float]) -> float:
    g = np.asarray(gains)[y]
    out, start = [], 0
    for n in _group_sizes(qids):
        s, gg = scores[start:start + n], g[start:start + n]
        v = M.ndcg(gg[np.argsort(-s, kind="stable")], gg, 10)
        if v is not None:
            out.append(v)
        start += n
    return float(np.mean(out))


def train_ltr(cfg: Config, n_jobs: int | None = None, corpus: str = "full") -> dict:
    import lightgbm as lgb  # noqa: PLC0415
    from .engine import Engine  # noqa: PLC0415
    L = cfg.get("search.ltr")
    n_jobs = n_jobs or mp.cpu_count()
    with stage_timer(cfg, "train_ltr") as info:
        eng = Engine(cfg, corpus, load_models=False)
        ds = eng.ds
        # Stacking: the dense encoder and the query-department model were trained on `train` queries, so their
        # scores on those queries are optimistic (the encoder has partly memorised their Exact products). A
        # reranker fit there learns to over-trust them and degrades on unseen queries. The reranker is therefore
        # fit on held-out `dev` queries the base models never saw (fit_fraction), early-stopped on the rest of
        # dev, and evaluated on the untouched test split.
        fit_split = L.get("fit_split", "dev")
        src = _queries(ds, fit_split)
        if fit_split == "dev":
            u = (src["query_id"].hash(seed=cfg.seed) % 1000).to_numpy() / 1000.0
            fit_mask = u < float(L.get("fit_fraction", 0.8))
            tr, dv = src.filter(pl.Series(fit_mask)), src.filter(pl.Series(~fit_mask))
        else:
            tr, dv = src, _queries(ds, "dev")
        if int(L["max_train_queries"]) and tr.height > int(L["max_train_queries"]):
            tr = tr.sample(n=int(L["max_train_queries"]), seed=cfg.seed)
        t = time.time()
        Xtr, ytr, qtr, rtr = extract_features(eng, tr, fit_split, n_jobs)
        Xdv, ydv, qdv, rdv = extract_features(eng, dv, "dev", n_jobs)
        log.info("LTR features: fit %s, validation %s (%.0fs)", Xtr.shape, Xdv.shape, time.time() - t)
        feat_dir = cfg.path("runs", eng.ds.name)
        feat_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(feat_dir / "ltr_features.npz", Xtr=Xtr, ytr=ytr, qtr=qtr, rtr=rtr, Xdv=Xdv, ydv=ydv,
                            qdv=qdv, rdv=rdv, names=np.array(ALL_FEATURES))
        gains = L["label_gain"]
        params = {**{k: v for k, v in L["lightgbm"].items() if k not in ("num_boost_round", "early_stopping_rounds")},
                  "label_gain": gains, "metric": "ndcg", "eval_at": [10], "verbosity": -1, "seed": cfg.seed,
                  "num_threads": n_jobs}
        reg = ModelRegistry(cfg)
        versions = {}
        for name, feats in (("ltr", CONTENT_FEATURES), ("ltr_fb", ALL_FEATURES)):
            cols = [ALL_FEATURES.index(f) for f in feats]
            dtr = lgb.Dataset(Xtr[:, cols], ytr, group=_group_sizes(qtr), feature_name=feats, free_raw_data=False)
            ddv = lgb.Dataset(Xdv[:, cols], ydv, group=_group_sizes(qdv), feature_name=feats, reference=dtr)
            t = time.time()
            booster = lgb.train(params, dtr, num_boost_round=int(L["lightgbm"]["num_boost_round"]), valid_sets=[ddv],
                                callbacks=[lgb.early_stopping(int(L["lightgbm"]["early_stopping_rounds"]), verbose=False),
                                           lgb.log_evaluation(100)])
            dev = _dev_ndcg(booster.predict(Xdv[:, cols]), ydv, qdv, gains)
            imp = dict(zip(feats, booster.feature_importance("gain").tolist()))
            tot = sum(imp.values()) or 1.0
            group_imp = {g: round(sum(imp.get(f, 0) for f in fs) / tot, 4) for g, fs in FEATURE_GROUPS.items()}
            metrics = {"dev_ndcg10": round(dev, 4), "best_iteration": booster.best_iteration,
                       "train_seconds": round(time.time() - t, 1), "importance_by_group": group_imp,
                       "importance": {k: round(v / tot, 5) for k, v in sorted(imp.items(), key=lambda x: -x[1])}}
            versions[name] = reg.register(
                name, lambda d, b=booster: b.save_model(str(d / "model.txt"), num_iteration=b.best_iteration),
                params=params, metrics=metrics, data={"fit_split": fit_split, "fit_queries": int(len(np.unique(qtr))),
                                                       "fit_rows": int(len(ytr)), "validation_queries": int(len(np.unique(qdv)))},
                parents={k: v for k, v in eng.versions.items()})
            log.info("%s %s: dev nDCG@10 %.4f (iter %d) groups %s", name, versions[name], dev, booster.best_iteration,
                     group_imp)

        versions["esci_class"] = _fit_esci_class(cfg, Xtr, ytr, Xdv, ydv, n_jobs)
        info.update(rows=int(len(ytr)), versions=versions)
    return versions


def _fit_esci_class(cfg: Config, Xtr, ytr, Xdv, ydv, n_jobs: int) -> str:
    """E / S / C / I result-type classifier on content features (storefront badges).

    Square-root class weights: Exact dominates the labels (~2/3), and unweighted training almost never predicts
    Substitute or Irrelevant; full balancing costs too much accuracy. Complements stay hard to recognise from
    content alone (they need product-relationship knowledge), which the storefront reflects by showing a type
    only when the classifier is confident.
    """
    import lightgbm as lgb  # noqa: PLC0415
    from sklearn.metrics import accuracy_score, f1_score  # noqa: PLC0415
    cols = [ALL_FEATURES.index(f) for f in CONTENT_FEATURES]
    cnt = np.bincount(ytr, minlength=4).astype(np.float64)
    w = np.sqrt(len(ytr) / (4 * np.maximum(cnt, 1)))[ytr]
    mc = lgb.train({"objective": "multiclass", "num_class": 4, "learning_rate": 0.1, "num_leaves": 63,
                    "min_data_in_leaf": 100, "verbosity": -1, "seed": cfg.seed, "num_threads": n_jobs},
                   lgb.Dataset(Xtr[:, cols], ytr, weight=w, feature_name=CONTENT_FEATURES), num_boost_round=200)
    pred = mc.predict(Xdv[:, cols]).argmax(1)
    metrics = {"dev_accuracy": round(float(accuracy_score(ydv, pred)), 4),
               "dev_macro_f1": round(float(f1_score(ydv, pred, average="macro")), 4),
               "dev_f1_by_class": dict(zip("ICSE", np.round(f1_score(ydv, pred, average=None), 4).tolist())),
               "classes": [CODE_LABEL[i] for i in range(4)], "class_weighting": "sqrt-balanced"}
    v = ModelRegistry(cfg).register("esci_class", lambda d: mc.save_model(str(d / "model.txt")), metrics=metrics,
                                    data={"fit_rows": int(len(ytr))})
    log.info("esci_class %s: %s", v, metrics)
    return v


def train_esci_class(cfg: Config, corpus: str = "full", n_jobs: int | None = None) -> str:
    """Refit only the result-type classifier from the features saved by `train-ltr`."""
    d = np.load(cfg.path("runs", corpus, "ltr_features.npz"), allow_pickle=True)
    return _fit_esci_class(cfg, d["Xtr"], d["ytr"], d["Xdv"], d["ydv"], n_jobs or 2)


# ======================================================================================
# hybrid fusion weight
# ======================================================================================
def tune_hybrid(cfg: Config, n_queries: int = 4000, corpus: str = "full") -> str:
    from .engine import Engine, _z  # noqa: PLC0415
    with stage_timer(cfg, "tune_hybrid") as info:
        eng = Engine(cfg, corpus, load_models=False, feedback=False)
        dv = _queries(eng.ds, "dev").sample(n=min(n_queries, _queries(eng.ds, "dev").height), seed=cfg.seed)
        j = eng.ds.judgments(["query_id", "doc_id", "locale", "gain", "split"]).filter(pl.col("split") == "dev") \
               .join(dv.select("query_id"), on="query_id", how="semi")
        qtext = dict(zip(dv["query_id"].to_list(), dv["query"].to_list()))
        per_q = []
        for (qid, loc), g in j.group_by(["query_id", "locale"]):
            ix = eng.idx[loc]
            rows = ix.rows(g["doc_id"].to_list())
            ok = rows >= 0
            rows, gains = rows[ok], g["gain"].to_numpy()[ok].astype(np.float64)
            qv = eng.encoder.encode_query(qtext[qid])
            per_q.append((rows, gains, _z(ix.bm25_text.score_docs(qtext[qid], rows)), _z(ix.vectors.score_docs(qv, rows))))
        grid = [round(a, 2) for a in np.linspace(0, 1, 21)]
        curve = []
        for a in grid:
            nd = [M.ndcg(gains[np.lexsort((rows, -(a * zd + (1 - a) * zb)))], gains, 10) for rows, gains, zb, zd in per_q]
            curve.append({"alpha": a, "dev_ndcg10": round(float(np.mean([x for x in nd if x is not None])), 5)})
        best = max(curve, key=lambda r: r["dev_ndcg10"])
        version = ModelRegistry(cfg).register(
            "hybrid", lambda d: (d / "hybrid.json").write_text(json.dumps({"alpha": best["alpha"]})),
            params={"alpha": best["alpha"], "fusion": "z-score convex"}, metrics={"curve": curve, **best},
            data={"dev_queries": len(per_q)}, parents={"dense_encoder": eng.versions.get("dense_encoder")})
        log.info("hybrid alpha=%.2f dev nDCG@10=%.4f (bm25-only %.4f, dense-only %.4f)", best["alpha"],
                 best["dev_ndcg10"], curve[0]["dev_ndcg10"], curve[-1]["dev_ndcg10"])
        info.update(rows=len(per_q), alpha=best["alpha"], version=version)
    return version

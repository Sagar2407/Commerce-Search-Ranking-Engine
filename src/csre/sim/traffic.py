"""Simulated search traffic: query popularity, sessions, impressions, clicks, carts, purchases.

Why simulate: ESCI has relevance labels but no behaviour. The project needs behaviour for
(a) a feedback loop (learning from clicks / explicit feedback), (b) caching and load tests, and
(c) contrasting offline relevance with (simulated) conversion. The simulator is grounded in the
real ESCI labels and real-calibrated commerce attributes, and every assumption is a config value.

Outputs (all under data/synthetic/)
-----------------------------------
query_popularity.parquet            per query: Zipf traffic share, rank, head/torso/tail bucket
logs/searches/shard=NNN/*.parquet   one row per search (session, user, time, query, policy, outcomes)
logs/impressions/shard=NNN/*.parquet one row per shown result (position, doc, click/cart/purchase,
                                    dwell, explicit feedback, true examination propensity)
logs/query_doc_stats.parquet        aggregated feedback per (query, doc) incl. IPS-weighted clicks
replay/requests.parquet             a 24h request stream for cache / latency / load testing

Generation is sharded and each shard is seeded independently -> embarrassingly parallel: more
sessions = more shards, same memory per shard.
"""
from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from ..config import Config
from ..utils import dir_size_bytes, duck, get_logger, stable_hash64, stage_timer, uniform_from_hash, write_parquet
from .click_model import LABEL_CODE, ClickModelParams, simulate_interactions

log = get_logger("csre.sim.traffic")

DIURNAL = np.array([1.2, 0.7, 0.4, 0.3, 0.3, 0.4, 0.8, 1.5, 2.4, 3.2, 3.8, 4.2,
                    4.5, 4.3, 4.1, 4.0, 4.1, 4.4, 4.8, 5.3, 5.6, 5.2, 3.9, 2.3])
UTC_OFFSET_H = {"us": -6, "es": 2, "jp": 9}
LOCALES = ["us", "es", "jp"]


# --------------------------------------------------------------------------------------
# query popularity
# --------------------------------------------------------------------------------------
def query_popularity(cfg: Config, queries: pl.DataFrame, zipf_s: float) -> pl.DataFrame:
    """Zipf traffic over queries. Rank is driven by a 'headness' score: short, unspecific and
    behaviour-sourced queries tend to be head queries; the rest is deterministic noise."""
    u = uniform_from_hash(stable_hash64((str(q) for q in queries["query_id"].to_list()), salt=f"pop-{cfg.seed}"))
    from scipy.stats import norm
    noise = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
    length = np.where(queries["locale"].to_numpy() == "jp", queries["n_chars"].to_numpy() / 4.0,
                      queries["n_tokens"].to_numpy().astype(float))
    head = (-0.45 * length + 0.8 * queries["slice_underspecified"].to_numpy()
            + 0.4 * (queries["source"].to_numpy() == "behavioral") + 1.0 * noise)
    order = np.argsort(-head, kind="stable")
    rank = np.empty(len(head), dtype=np.int64)
    rank[order] = np.arange(1, len(head) + 1)
    w = 1.0 / rank.astype(float) ** zipf_s
    share = w / w.sum()
    cum = np.cumsum(np.sort(share)[::-1])
    head_cut = np.searchsorted(cum, 0.30) + 1
    torso_cut = np.searchsorted(cum, 0.70) + 1
    bucket = np.where(rank <= head_cut, "head", np.where(rank <= torso_cut, "torso", "tail"))
    return queries.select("query_id", "locale", "split").with_columns(
        pl.Series("pop_rank", rank), pl.Series("traffic_share", share), pl.Series("traffic_bucket", bucket),
    )


# --------------------------------------------------------------------------------------
# candidate arrays (CSR layout: one contiguous slice of candidates per query)
# --------------------------------------------------------------------------------------
class Candidates:
    def __init__(self, cfg: Config, query_ids: np.ndarray):
        con = duck(cfg)
        j = cfg.path("processed", "judgments.parquet")
        com = cfg.path("synthetic", "product_commerce.parquet")
        df = con.sql(f"""
            SELECT j.query_id, j.doc_id, j.esci_label, j.grade, c.stars, c.popularity_z,
                   (ln(c.price) - avg(ln(c.price)) OVER w) / nullif(stddev_pop(ln(c.price)) OVER w, 0) AS price_z
            FROM '{j}' j JOIN '{com}' c USING (doc_id)
            WINDOW w AS (PARTITION BY j.query_id)
        """).pl()
        qindex = {q: i for i, q in enumerate(query_ids.tolist())}
        df = df.filter(pl.col("query_id").is_in(query_ids.tolist())).with_columns(
            pl.col("query_id").replace_strict(qindex, return_dtype=pl.Int32).alias("qidx")
        ).sort("qidx", "doc_id")
        self.doc_vocab = df["doc_id"].unique().sort()
        dindex = pl.DataFrame({"doc_id": self.doc_vocab, "didx": np.arange(len(self.doc_vocab), dtype=np.int32)})
        df = df.join(dindex, on="doc_id", how="left").sort("qidx", "doc_id")
        self.doc = df["didx"].to_numpy()
        self.label = df["esci_label"].replace_strict(LABEL_CODE, return_dtype=pl.Int8).to_numpy()
        self.grade = df["grade"].to_numpy().astype(np.float32)
        self.stars = df["stars"].fill_null(np.nan).to_numpy().astype(np.float32)
        self.pop = df["popularity_z"].fill_null(0).to_numpy().astype(np.float32)
        self.price_z = df["price_z"].fill_null(0).fill_nan(0).to_numpy().astype(np.float32)
        counts = np.bincount(df["qidx"].to_numpy(), minlength=len(query_ids))
        self.len = counts.astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)


# --------------------------------------------------------------------------------------
# one round of searches
# --------------------------------------------------------------------------------------
def _run_searches(rng, qidx: np.ndarray, cand: Candidates, lp: dict, page_size: int, params: ClickModelParams):
    n_s = len(qidx)
    lens = cand.len[qidx]
    total = int(lens.sum())
    rep = np.repeat(np.arange(n_s), lens)
    seg_start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = cand.start[qidx][rep] + (np.arange(total) - seg_start[rep])

    score = (lp["grade_weight"] * cand.grade[flat] + lp["popularity_weight"] * cand.pop[flat]
             + lp["noise_sd"] * rng.standard_normal(total).astype(np.float32))
    order = np.lexsort((-score, rep))
    rep_o, flat_o = rep[order], flat[order]
    pos = np.arange(total) - seg_start[rep_o] + 1

    explore = rng.random(n_s) < lp["explore_fraction"]
    k = int(lp["explore_top_k"])
    m = explore[rep_o] & (pos <= k)
    if m.any():
        key = pos.astype(np.float64)
        key[m] = rng.random(int(m.sum())) * k + 0.5
        order2 = np.lexsort((key, rep_o))
        rep_o, flat_o = rep_o[order2], flat_o[order2]
        pos = np.arange(total) - seg_start[rep_o] + 1

    keep = pos <= page_size
    rep_o, flat_o, pos = rep_o[keep], flat_o[keep], pos[keep]
    out = simulate_interactions(rng, pos, cand.label[flat_o], cand.stars[flat_o], cand.price_z[flat_o], params)
    imp = {"rep": rep_o, "doc": cand.doc[flat_o], "position": pos.astype(np.int16), **out}

    n_shown = np.bincount(rep_o, minlength=n_s)
    n_clicks = np.bincount(rep_o, weights=out["clicked"], minlength=n_s).astype(np.int16)
    n_carts = np.bincount(rep_o, weights=out["carted"], minlength=n_s).astype(np.int16)
    n_buys = np.bincount(rep_o, weights=out["purchased"], minlength=n_s).astype(np.int16)
    first_click = np.full(n_s, 0, dtype=np.int16)
    ck = out["clicked"]
    if ck.any():
        # positions are sorted ascending within each search -> first occurrence is the min position
        r, first_idx = np.unique(rep_o[ck], return_index=True)
        first_click[r] = pos[ck][first_idx]
    srch = {"n_shown": n_shown.astype(np.int16), "n_clicks": n_clicks, "n_carts": n_carts,
            "n_purchases": n_buys, "first_click_pos": first_click, "explore": explore}
    return srch, imp


def _session_starts(rng, n: int, start: date, days: int, locale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates = [start + timedelta(days=d) for d in range(days)]
    wd = np.array([1.15 if d.weekday() >= 5 else 1.0 for d in dates]) * (1 + 0.002 * np.arange(days))
    day = rng.choice(days, size=n, p=wd / wd.sum())
    hour = rng.choice(24, size=n, p=DIURNAL / DIURNAL.sum())
    sec = rng.integers(0, 3600, size=n)
    off = np.vectorize(UTC_OFFSET_H.get)(locale) if n else np.zeros(0)
    base = np.datetime64(datetime(start.year, start.month, start.day), "s")
    ts = base + (day * 86400 + (hour - off) * 3600 + sec).astype("timedelta64[s]")
    return ts, hour.astype(np.int8)


def build_traffic(cfg: Config, shards: int | None = None, n_sessions: int | None = None) -> dict:
    T = cfg.get("simulation.traffic")
    params = ClickModelParams.from_config(T["click_model"])
    lp = T["logging_policy"]
    shards = int(shards or T["shards"])
    n_sessions = int(n_sessions or T["n_sessions"])
    page_size = int(T["page_size"])
    out_root = cfg.path("synthetic", "logs")
    shutil.rmtree(out_root, ignore_errors=True)

    with stage_timer(cfg, "sim_traffic", shards=shards, n_sessions=n_sessions) as info:
        queries = pl.read_parquet(cfg.path("processed", "queries.parquet"),
                                  columns=["query_id", "query", "locale", "split", "n_tokens", "n_chars", "source",
                                           "slice_underspecified"]).sort("query_id")
        pop = query_popularity(cfg, queries, float(T["zipf_s"]))
        write_parquet(pop, cfg.path("synthetic", "query_popularity.parquet"))

        qids = queries["query_id"].to_numpy()
        q_locale = queries["locale"].to_numpy()
        q_split = queries["split"].to_numpy()
        cand = Candidates(cfg, qids)
        cum = np.cumsum(pop["traffic_share"].to_numpy())
        cum /= cum[-1]

        rel = pl.read_parquet(cfg.path("processed", "graph", "related_queries.parquet"))
        qpos = {q: i for i, q in enumerate(qids.tolist())}
        rel = rel.filter(pl.col("related_query_id").is_in(qids.tolist())) \
                 .sort("query_id", "jaccard", descending=[False, True]).group_by("query_id").first()
        related = np.full(len(qids), -1, dtype=np.int64)
        related[[qpos[q] for q in rel["query_id"].to_list()]] = [qpos[q] for q in rel["related_query_id"].to_list()]

        n_users = int(T["n_users"])
        user_w = np.random.default_rng(cfg.seed).lognormal(0, 1.2, n_users)
        user_cum = np.cumsum(user_w) / user_w.sum()
        start = date.fromisoformat(T["start_date"])
        days = int(T["history_days"])
        per_shard = int(np.ceil(n_sessions / shards))
        tot_searches = tot_imps = 0

        for s in range(shards):
            rng = np.random.default_rng([cfg.seed, 7, s])
            n = min(per_shard, n_sessions - s * per_shard)
            if n <= 0:
                break
            sess_id = np.int64(s) * 1_000_000_000 + np.arange(n, dtype=np.int64)
            user = np.searchsorted(user_cum, rng.random(n)).astype(np.int32)
            qidx = np.minimum(np.searchsorted(cum, rng.random(n)), len(qids) - 1)
            ts, hour = _session_starts(rng, n, start, days, q_locale[qidx])
            seq = np.zeros(n, dtype=np.int8)
            prev = np.full(n, -1, dtype=np.int64)
            cur_sess, cur_user = sess_id, user
            s_frames, i_frames = [], []
            next_sid = np.int64(s) * 1_000_000_000
            for r in range(int(T["max_reformulations"]) + 1):
                m = len(qidx)
                sid = next_sid + np.arange(m, dtype=np.int64)
                next_sid += m
                srch, imp = _run_searches(rng, qidx, cand, lp, page_size, params)
                abandoned = srch["n_clicks"] == 0
                no_cart = (srch["n_clicks"] > 0) & (srch["n_carts"] == 0)
                reform = ((abandoned & (rng.random(m) < params.p_reformulate_if_abandon))
                          | (no_cart & (rng.random(m) < params.p_reformulate_if_no_cart)))
                if r == int(T["max_reformulations"]):
                    reform[:] = False
                s_frames.append(pl.DataFrame({
                    "search_id": sid, "session_id": cur_sess, "user_id": cur_user,
                    "ts_utc": ts.astype("datetime64[ms]"), "local_hour": hour,
                    "locale": q_locale[qidx], "query_id": qids[qidx], "query_split": q_split[qidx],
                    "seq_in_session": seq, "prev_search_id": prev,
                    "policy": np.where(srch["explore"], "explore_randtop10", lp["name"]),
                    **{k: v for k, v in srch.items() if k != "explore"},
                    "abandoned": abandoned, "reformulated_next": reform,
                }))
                i_frames.append(pl.DataFrame({
                    # query_id / explore are denormalised onto impressions so aggregations need no join
                    "search_id": sid[imp["rep"]], "query_id": qids[qidx][imp["rep"]],
                    "explore": srch["explore"][imp["rep"]], "position": imp["position"],
                    "didx": imp["doc"], "clicked": imp["clicked"], "carted": imp["carted"],
                    "purchased": imp["purchased"], "dwell_s": imp["dwell_s"], "feedback": imp["feedback"],
                    "exam_propensity": imp["exam_propensity"],
                }))
                if not reform.any():
                    break
                idx = np.nonzero(reform)[0]
                rq = related[qidx[idx]]
                use_rel = (rq >= 0) & (rng.random(len(idx)) < 0.8)
                qidx = np.where(use_rel, rq, qidx[idx])
                cur_sess, cur_user = cur_sess[idx], cur_user[idx]
                ts = ts[idx] + rng.integers(5, 90, size=len(idx)).astype("timedelta64[s]")
                hour = hour[idx]
                prev = sid[idx]
                seq = (seq[idx] + 1).astype(np.int8)
            sdf = pl.concat(s_frames).with_columns(pl.col("ts_utc").dt.date().alias("day"))
            idf = pl.concat(i_frames)
            idf = idf.with_columns(pl.Series("doc_id", cand.doc_vocab.gather(idf["didx"].to_numpy()))).drop("didx") \
                     .with_columns(pl.col("dwell_s").fill_nan(None),
                                   pl.when(pl.col("feedback") == 0).then(None).otherwise(pl.col("feedback"))
                                   .alias("feedback")) \
                     .select("search_id", "query_id", "explore", "position", "doc_id", "clicked", "carted",
                             "purchased", "dwell_s", "feedback", "exam_propensity")
            write_parquet(sdf, out_root / "searches" / f"shard={s:03d}" / "part-0.parquet", row_group_size=250_000)
            write_parquet(idf, out_root / "impressions" / f"shard={s:03d}" / "part-0.parquet", row_group_size=1_000_000)
            tot_searches += len(sdf)
            tot_imps += len(idf)
            log.info("shard %d/%d: searches=%d impressions=%d (total %d / %d)", s + 1, shards, len(sdf), len(idf),
                     tot_searches, tot_imps)

        _aggregate(cfg)
        info.update(rows=tot_imps, searches=tot_searches, impressions=tot_imps, bytes=dir_size_bytes(out_root))
    return info


def _aggregate(cfg: Config) -> None:
    con = duck(cfg)
    root = cfg.path("synthetic", "logs")
    con.execute(f"""
        COPY (
          SELECT query_id, doc_id,
                 count(*)::INTEGER AS impressions,
                 sum(clicked::INT)::INTEGER AS clicks,
                 sum(carted::INT)::INTEGER AS carts,
                 sum(purchased::INT)::INTEGER AS purchases,
                 sum(CASE WHEN clicked THEN 1.0 / exam_propensity ELSE 0 END) AS ips_clicks,
                 sum(1.0 / exam_propensity) AS ips_impressions,
                 sum(explore::INT)::INTEGER AS explore_impressions,
                 sum((explore AND clicked)::INT)::INTEGER AS explore_clicks,
                 avg(position) AS avg_position,
                 sum(CASE WHEN feedback = 1 THEN 1 ELSE 0 END)::INTEGER AS thumbs_up,
                 sum(CASE WHEN feedback = -1 THEN 1 ELSE 0 END)::INTEGER AS thumbs_down,
                 avg(dwell_s) AS avg_dwell_s
          FROM read_parquet('{root}/impressions/**/*.parquet', hive_partitioning=true)
          GROUP BY ALL
        ) TO '{root / "query_doc_stats.parquet"}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


# --------------------------------------------------------------------------------------
# replay stream for cache / latency / load tests
# --------------------------------------------------------------------------------------
_MODIFIERS = {
    "us": ["cheap", "best", "for men", "for women", "2 pack", "sale", "new", "large", "black", "for kids"],
    "es": ["barato", "oferta", "para mujer", "para hombre", "pack 2", "negro", "grande", "para niños"],
    "jp": ["安い", "おすすめ", "セット", "メンズ", "レディース", "人気", "大容量"],
}
VARIANT_KINDS = np.array(["case_space", "typo", "token_drop", "reorder", "modifier"])
VARIANT_P = np.array([0.15, 0.35, 0.20, 0.10, 0.20])


def _typo(tok: str, r: float, r2: float) -> str:
    i = int(r2 * (len(tok) - 1))
    if r < 0.4 and len(tok) >= 2:                   # swap adjacent
        return tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]
    if r < 0.75:                                    # delete
        return tok[:i] + tok[i + 1:]
    return tok[:i] + tok[i] + tok[i:]               # duplicate


def make_variant(q: str, locale: str, kind: str, u: np.ndarray) -> tuple[str, str]:
    """Deterministic (given u) one-off rewrite of a query, preserving its intent.
    Returns (text, kind actually applied): single-token queries cannot be dropped / reordered,
    so those fall back to a typo."""
    toks = q.split()
    if kind == "case_space":
        return (("  ".join(toks) if u[0] < 0.5 else q.upper()) if locale != "jp" else q.replace(" ", "")), kind
    if kind == "typo" or (kind in ("token_drop", "reorder") and len(toks) < 2):
        cand = [i for i, t in enumerate(toks) if len(t) >= (2 if locale == "jp" else 4)] or [0]
        i = cand[int(u[1] * len(cand)) % len(cand)]
        toks[i] = _typo(toks[i], u[2], u[3]) if len(toks[i]) >= 2 else toks[i]
        return " ".join(toks), "typo"
    if kind == "token_drop":
        del toks[int(u[1] * len(toks)) % len(toks)]
        return " ".join(toks), kind
    if kind == "reorder":
        i = int(u[1] * (len(toks) - 1))
        toks[i], toks[i + 1] = toks[i + 1], toks[i]
        return " ".join(toks), kind
    mods = _MODIFIERS[locale]
    m = mods[int(u[1] * len(mods)) % len(mods)]
    return (f"{q} {m}" if u[0] < 0.7 else f"{m} {q}"), kind


def build_replay(cfg: Config) -> dict:
    R = cfg.get("simulation.replay")
    with stage_timer(cfg, "sim_replay") as info:
        pop = pl.read_parquet(cfg.path("synthetic", "query_popularity.parquet")).sort("query_id")
        q = pl.read_parquet(cfg.path("processed", "queries.parquet"), columns=["query_id", "query"]).sort("query_id")
        rank = pop["pop_rank"].to_numpy().astype(float)
        w = 1.0 / rank ** float(R["zipf_s"])
        cum = np.cumsum(w) / w.sum()
        n = int(R["n_requests"])
        rng = np.random.default_rng([cfg.seed, 99])
        qi = np.minimum(np.searchsorted(cum, rng.random(n)), len(cum) - 1)
        loc = pop["locale"].to_numpy()[qi]
        d = date.fromisoformat(R["date"])
        hour = rng.choice(24, size=n, p=DIURNAL / DIURNAL.sum())
        off = np.select([loc == "us", loc == "es"], [UTC_OFFSET_H["us"], UTC_OFFSET_H["es"]], UTC_OFFSET_H["jp"])
        ms = ((hour - off) * 3600 * 1000 + rng.integers(0, 3600 * 1000, size=n)).astype(np.int64)
        base = np.datetime64(datetime(d.year, d.month, d.day), "ms")

        # one-off variants: typos, dropped / reordered tokens, modifiers, case & spacing
        novel = rng.random(n) < float(R.get("novel_fraction", 0.0))
        kinds = np.where(novel, VARIANT_KINDS[rng.choice(len(VARIANT_KINDS), size=n, p=VARIANT_P)], "original") \
            .astype(object)
        texts = q["query"].to_numpy()[qi].astype(object)
        uu = rng.random((int(novel.sum()), 4))
        for j, i in enumerate(np.nonzero(novel)[0]):
            texts[i], kinds[i] = make_variant(texts[i], loc[i], kinds[i], uu[j])

        df = pl.DataFrame({
            "ts_utc": base + ms.astype("timedelta64[ms]"),
            "locale": loc, "query_id": pop["query_id"].to_numpy()[qi],
            "query": pl.Series(texts.tolist(), dtype=pl.Utf8), "variant": pl.Series(kinds.tolist(), dtype=pl.Utf8),
            "is_novel": novel,
        }).sort("ts_utc").with_row_index("request_id").with_columns(pl.col("request_id").cast(pl.Int64))
        write_parquet(df, cfg.path("synthetic", "replay", "requests.parquet"), row_group_size=1_000_000)
        uniq_text = df["query"].n_unique()
        uniq_norm = df.select(pl.col("query").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
                              .n_unique()).item()
        info.update(rows=n, unique_query_ids=df["query_id"].n_unique(), unique_texts=uniq_text,
                    ideal_hit_rate_raw_text=round(1 - uniq_text / n, 4),
                    ideal_hit_rate_normalised=round(1 - uniq_norm / n, 4),
                    peak_rps=int(df.group_by(pl.col("ts_utc").dt.truncate("1m")).len()["len"].max() / 60))
    return info

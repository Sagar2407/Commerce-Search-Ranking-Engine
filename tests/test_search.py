import numpy as np
import polars as pl
import pytest

from csre.evaluation import metrics as M
from csre.search import analysis as A
from csre.search.bm25 import BM25Index
from csre.search.dense import SubwordEncoder, VectorIndex, train_encoder, typo
from csre.search.features import FeedbackStore
from csre.search.query import BrandMatcher, negated_terms


# ---------------------------------------------------------------- analysis
@pytest.mark.parametrize("text,expected", [
    ("Stainless Steel Water Bottles 12oz, 2-Pack", ["stainless", "steel", "water", "bottle", "12", "oz", "2", "pack"]),
    ("batteries for watches", ["battery", "watch"]),
    ("zapatos de mujer", ["zapato", "mujer"]),
    ("glasses", ["glass"]),
    ("rtx3080 1,000", ["rtx", "3080", "1000"]),
])
def test_terms(text, expected):
    assert A.terms(text) == expected


def test_cjk_bigrams_and_cache_key():
    assert A.terms("水筒500ml") == ["500", "ml", "水筒"]
    assert set(A.terms("ステンレス")) == {"ステ", "テン", "ンレ", "レス"}
    assert A.cache_key("  Water   BOTTLE ") == A.cache_key("water bottle")
    assert A.cache_key("the water bottle") == "water bottle"


def test_doc_units_parallel_matches_serial():
    texts = [f"red shoe size {i}" for i in range(30)] + ["水筒 ステンレス"] * 5
    m1, v1 = A.doc_units(texts, n_jobs=1, chunk=7)
    m2, v2 = A.doc_units(texts, n_jobs=2, chunk=7)
    d1 = {(i, v1[j]) for i, j in zip(*m1.nonzero())}
    d2 = {(i, v2[j]) for i, j in zip(*m2.nonzero())}
    assert d1 == d2


# ---------------------------------------------------------------- bm25
@pytest.fixture(scope="module")
def bm25():
    docs = ["red running shoes for women", "blue running shoes", "red wine glass", "stainless steel water bottle",
            "water bottle for kids", "phone case"]
    return BM25Index.build(docs, n_jobs=1), docs


def test_bm25_search_and_consistency(bm25):
    idx, _ = bm25
    rows, sc = idx.search("red shoes", k=3)
    assert rows[0] == 0
    np.testing.assert_allclose(idx.score_docs("red shoes", rows), sc, rtol=1e-5)
    tids, qtf = idx.query_terms("red shoes")
    W = idx.term_doc_weights(tids, rows)
    np.testing.assert_allclose(W @ qtf, sc, rtol=1e-5)
    assert idx.matched_terms("red water bottle", 3) == ["water", "bottle"]
    assert len(idx.search("zzzz")[0]) == 0


def test_bm25_roundtrip(bm25, tmp_path):
    idx, _ = bm25
    idx.save(tmp_path / "b")
    idx2 = BM25Index.load(tmp_path / "b")
    np.testing.assert_allclose(idx.scores("water bottle")[0], idx2.scores("water bottle")[0])


# ---------------------------------------------------------------- dense
def test_typo_is_one_edit():
    rng = np.random.default_rng(0)
    q = "wireless headphones"
    t = typo(q, rng)
    assert t != q and abs(len(t) - len(q)) <= 1


def test_encoder_learns_toy_task():
    # queries must retrieve their paired product among distractors
    words = ["shoe", "bottle", "lamp", "chair", "phone", "watch", "knife", "pillow"]
    d_texts = [f"{w} premium {c}" for w in words for c in ("red", "blue")]
    q_texts = [f"{w}s" for w in words]
    pos_q = np.repeat(np.arange(len(words)), 2)
    pos_d = np.arange(len(d_texts))
    neg_ptr = np.zeros(len(words) + 1, np.int64)
    params = {"dim": 16, "min_df": 1, "max_vocab": 10_000, "char_ngrams": [3, 4], "jp_char_ngrams": [1, 2],
              "train": {"epochs": 30, "batch_size": 4, "lr": 0.2, "temperature": 0.1, "typo_augment": 0.0,
                        "init_scale": 0.1}}
    enc, hist = train_encoder(q_texts, np.array(["us"] * 8), d_texts, pos_q, pos_d, neg_ptr, np.zeros(0, np.int64),
                              np.array(["us"] * 16), params, seed=0)
    assert hist[-1]["loss"] < hist[0]["loss"]
    D = enc.encode(d_texts, n_jobs=1)
    vi = VectorIndex.build(D, kind="exact")
    hits = [d_texts[vi.search(enc.encode_query(q), k=1)[0][0]].split()[0] == q[:-1] for q in q_texts]
    assert np.mean(hits) >= 0.75
    assert isinstance(enc, SubwordEncoder)


# ---------------------------------------------------------------- query understanding
def test_brand_and_negation():
    bm = BrandMatcher({"us": ["apple", "hamilton beach"], "jp": ["パナソニック"]})
    assert bm("hamilton beach blender", "us") == ["hamilton beach"]
    assert bm("pineapple slicer", "us") == []
    assert bm("パナソニック ドライヤー", "jp") == ["パナソニック"]
    assert negated_terms("lotion without fragrance") == ["fragrance"]
    assert negated_terms("leche sin lactosa") == ["lactosa"]


# ---------------------------------------------------------------- metrics
def test_ndcg_and_condensed():
    gains = np.array([1.0, 0.1, 0.0])
    assert M.ndcg(gains, gains, 10) == pytest.approx(1.0)
    assert M.ndcg(gains[::-1], gains, 10) < 1.0
    judged = {10: (1.0, "E"), 11: (0.1, "S"), 12: (0.0, "I")}
    r = M.retrieval_metrics(np.array([99, 10, 98, 11]), judged, k=10, recall_k=100)
    assert r["ndcg10_cond"] == pytest.approx(1.0)       # unjudged 99/98 are removed, not counted as misses
    assert r["ndcg10_lb"] < 1.0
    assert r["judged10"] == pytest.approx(0.2)
    assert r["recall100_e"] == 1.0


def test_paired_delta():
    a = np.full(200, 0.5)
    b = a + 0.1
    d = M.paired_delta(a, b, n_boot=200)
    assert d["delta"] == pytest.approx(0.1) and d["lo"] > 0


def test_cost_model():
    c = M.cost_per_million(10.0, vcpus=4, usd_per_hour=0.2, utilisation=0.5)
    assert c["qps_per_core"] == pytest.approx(100)
    assert c["usd_per_million"] == pytest.approx(1e6 / 200 / 3600 * 0.2)


# ---------------------------------------------------------------- feedback store
def test_feedback_store_lookup_and_live_events():
    df = pl.DataFrame({"locale": ["us", "us"], "key": ["shoe", "shoe"], "row": [5, 9],
                       **{c: [10.0, 2.0] for c in FeedbackStore.COLS}})
    st = FeedbackStore.from_frame(df)
    raw, qi = st.lookup("us", "shoe", np.array([9, 5, 7]))
    assert raw[0, 0] == 2.0 and raw[1, 0] == 10.0 and raw[2, 0] == 0.0 and qi == 12.0
    st.add("us", "shoe", 7, "impression")
    st.add("us", "shoe", 7, "click")
    raw, _ = st.lookup("us", "shoe", np.array([7]))
    assert raw[0, 0] == 1 and raw[0, 1] == 1
    assert st.features(raw, 1.0).shape == (1, 6)


# ---------------------------------------------------------------- spelling correction / query keys
def test_damerau1():
    from csre.search.spell import damerau1
    assert damerau1("headphone", "headphne") and damerau1("headphone", "haedphone")
    assert damerau1("bottle", "bottlle") and damerau1("bottle", "botle") and damerau1("bottle", "bittle")
    assert not damerau1("bottle", "battles") and not damerau1("shoe", "shirt")


def test_spell_corrector_expands_unseen_words_only():
    from csre.search.spell import SpellCorrector
    docs = ["wireless headphones black"] * 30 + ["roka goggles"] * 2 + ["rope ladder"] * 60
    idx = BM25Index.build(docs, n_jobs=1)
    sc = SpellCorrector.from_bm25(idx, min_df=5, max_df=3, min_ratio=10)
    q, ch = sc.correct("wireles headphnes")
    assert ch == [("wireles", "wireless"), ("headphnes", "headphone")]
    assert q == "wireles wireless headphnes headphone"           # expand keeps the original words
    assert sc.correct("wireles", mode="replace")[0] == "wireless"
    assert sc.correct("roka goggles")[1] == []                     # a real (rare) catalog word is left alone
    assert sc.correct("rtx 3080")[1] == []                         # numbers / short tokens untouched


def test_query_key_is_order_insensitive():
    assert A.query_key("Shoes for Women size 8") == A.query_key("women shoes  size 8")
    assert A.query_key("water bottle") != A.query_key("bottle opener")


def test_spell_corrector_rare_seller_misspellings():
    from csre.search.spell import SpellCorrector
    docs = ["wireless earbuds"] * 400 + ["wireles earbuds cheap"] * 4
    idx = BM25Index.build(docs, n_jobs=1)
    strict = SpellCorrector.from_bm25(idx, min_df=5, max_df=3, min_ratio=20)
    assert strict.correct("wireles earbuds")[1] == []                       # seen in 4 products: left alone
    lenient = SpellCorrector.from_bm25(idx, min_df=5, max_df=3, min_ratio=20, rare_df=50, strong_ratio=50)
    assert lenient.correct("wireles earbuds")[1] == [("wireles", "wireless")]  # but 100x rarer than its neighbour


def test_result_cache_tag_invalidation_and_eviction():
    from csre.serve.service import ResultCache
    c = ResultCache(max_entries=3, ttl_s=60)
    for i, m in enumerate(["bm25", "ltr_fb", "ltr"]):
        c.put(("us", m, 10, "water bottle", True, None, (), ()), i)
    assert c.invalidate("us", "water bottle", lambda k: k[1] == "ltr_fb") == 1
    c.put(("us", "bm25", 10, "shoe", True, None, (), ()), 9)
    c.put(("us", "bm25", 10, "sock", True, None, (), ()), 9)        # evicts the oldest entry
    assert c.invalidate("us", "water bottle") == 1 and len(c.data) == 2 and ("us", "water bottle") not in c.by_tag

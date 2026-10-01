"""End-to-end: tiny synthetic corpus -> `csre index` -> engine -> HTTP API (BM25 path, no trained models)."""
import shutil
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]

PRODUCTS = [
    ("us:P1", "Hydro Flask Stainless Steel Water Bottle 32 oz", "Hydro Flask", "Sports & Outdoors"),
    ("us:P2", "Plastic Water Bottle 16 oz 12 Pack", "Generic", "Grocery & Gourmet Food"),
    ("us:P3", "Women's Running Shoes Size 8 Breathable", "Asics", "Clothing, Shoes & Jewelry"),
    ("us:P4", "Men's Running Shoes Size 10", "Nike", "Clothing, Shoes & Jewelry"),
    ("us:P5", "Bottle Brush Cleaner for Water Bottles", "OXO", "Home & Kitchen"),
    ("us:P6", "Lactose Free Milk 1 Gallon", "Lactaid", "Grocery & Gourmet Food"),
]


def build_tiny_corpus(root):
    """Six products, two judged queries, one complement edge, in the demo_portable layout under `root`."""
    shutil.copytree(ROOT / "configs", root / "configs")
    d = root / "data" / "demo_portable"
    d.mkdir(parents=True)
    from csre.data import text as T
    cat = pl.DataFrame({
        "doc_id": [p[0] for p in PRODUCTS], "product_id": [p[0][3:] for p in PRODUCTS], "locale": "us",
        "title": [p[1] for p in PRODUCTS], "brand": [p[2] for p in PRODUCTS], "color": None,
        "category": [p[3] for p in PRODUCTS], "category_conf": 1.0,
    }).with_columns(
        pl.col("brand").str.to_lowercase().alias("brand_norm"),
        (pl.col("title") + " | " + pl.col("brand")).alias("doc_text"),
        T.for_matching(pl.col("title")).alias("_tm"),
        pl.col("title").str.len_chars().alias("n_title_chars"),
        pl.lit(0, pl.UInt32).alias("richness_chars"), pl.lit(True).alias("is_title_only"),
        pl.lit(True).alias("is_sparse"), pl.lit(True).alias("brand_in_title"),
    ).with_columns(*T.attribute_exprs(pl.col("_tm"), pl.col("_tm"))).drop("_tm")
    cat.write_parquet(d / "catalog.parquet")
    q = pl.DataFrame({"query_id": [1, 2], "query": ["water bottle", "womens running shoes size 8"], "locale": "us",
                      "split": "test", "n_E": [2, 1]}).with_columns(
        pl.lit(False).alias("slice_ambiguous"), pl.lit(False).alias("slice_spec"))
    q.write_parquet(d / "queries.parquet")
    pl.DataFrame({"query_id": [1, 1, 1, 2, 2], "doc_id": ["us:P1", "us:P2", "us:P5", "us:P3", "us:P4"],
                  "locale": "us", "esci_label": ["E", "E", "C", "E", "S"], "gain": [1.0, 1.0, 0.01, 1.0, 0.1],
                  "grade": [3, 3, 1, 3, 2], "split": "test"}).write_parquet(d / "judgments.parquet")
    pl.DataFrame({"src_doc": ["us:P1"], "dst_doc": ["us:P5"], "relation": ["complement"], "support": [3],
                  "example_query_id": [1]}).write_parquet(d / "product_edges.parquet")
    return root


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    root = tmp_path_factory.mktemp("csre")
    build_tiny_corpus(root)

    import os
    old = os.environ.get("CSRE_ROOT")
    os.environ["CSRE_ROOT"] = str(root)
    try:
        from fastapi.testclient import TestClient

        from csre.config import load_config
        from csre.search.engine import build_indexes
        from csre.serve.app import create_app
        cfg = load_config(overrides=["search.corpus=demo_portable", "serve.default_method=bm25",
                                     "search.spell.min_df=1", "search.spell.min_ratio=2"])   # tiny corpus
        build_indexes(cfg, "demo_portable")
        yield TestClient(create_app(cfg, "demo_portable"))
    finally:
        if old is None:
            os.environ.pop("CSRE_ROOT", None)
        else:
            os.environ["CSRE_ROOT"] = old


def test_health(client):
    h = client.get("/api/health").json()
    assert h["methods"] == ["bm25"] and h["n_products"] == {"us": 6}


def test_search_explains_attributes_and_caches(client):
    r = client.get("/api/search", params={"q": "womens running shoes size 8", "method": "bm25"}).json()
    top = r["results"][0]
    assert top["doc_id"] == "us:P3"
    checks = {a["type"]: a["status"] for a in top["explanation"]["attributes"]}
    assert checks["sizes"] == "match" and checks["audience"] == "match"
    men = next(x for x in r["results"] if x["doc_id"] == "us:P4")
    assert {a["type"]: a["status"] for a in men["explanation"]["attributes"]}["sizes"] == "conflict"
    assert r["judged_query_id"] == 2 and r["quality"]["judged10"] > 0
    again = client.get("/api/search", params={"q": "Womens  Running Shoes size 8", "method": "bm25"}).json()
    assert again["cache"]["hit"] is True


def test_unavailable_method_degrades_to_bm25(client):
    r = client.get("/api/search", params={"q": "water bottle", "method": "ltr"}).json()
    assert r["served_by"] == "bm25" and r["fallback_reason"]


def test_product_graph_and_feedback(client):
    p = client.get("/api/product/us:P1").json()
    assert [c["doc_id"] for c in p["complements"]] == ["us:P5"]
    fb = client.post("/api/feedback", json={"query": "water bottle", "locale": "us", "doc_id": "us:P1",
                                            "event": "click", "position": 1}).json()
    assert fb["ok"]
    assert client.post("/api/feedback", json={"query": "x", "locale": "us", "doc_id": "us:P1",
                                              "event": "bogus"}).status_code == 400


def test_storefront_page(client):
    html = client.get("/").text
    assert "<title>Commerce Search Storefront</title>" in html and "/api/search" in html


def test_department_filter_and_suggest(client):
    r = client.get("/api/search", params={"q": "water bottle", "method": "bm25",
                                          "department": "Home & Kitchen"}).json()
    assert [x["doc_id"] for x in r["results"]] == ["us:P5"] and r["filters"] == {"department": "Home & Kitchen"}
    assert client.get("/api/suggest", params={"q": "wat"}).json() == ["water bottle"]


def test_spelling_correction_expands_query(client):
    r = client.get("/api/search", params={"q": "watter bottle", "method": "bm25"}).json()
    assert r["corrections"] == [{"from": "watter", "to": "water"}]
    assert {"us:P1", "us:P2"} <= {x["doc_id"] for x in r["results"][:3]}   # both water bottles are found

"""The online (pure-Python) attribute extractors must match the phase-1 Polars extractors exactly."""
import polars as pl
import pytest

from csre.data import text as T
from csre.search.attributes import query_attributes

QUERIES = [
    "12 oz stainless steel tumbler 2 pack", "womens running shoes size 8 without laces", "16x25x5 air filter",
    "queen size sheets cotton white", "iphone 13 pro max case compatible with magsafe", "2 in 1 laptop 15.6 inch",
    "kids rain boots size 10 boys", "leather wallet for men brown", "pack of 12 aa batteries", "lotion without fragrance",
    "funda iphone 13 silicona", "zapatillas mujer talla 38 negro", "sin azúcar galletas 500g", "juego de 6 vasos de cristal",
    "ステンレス 水筒 500ml", "レディース 靴下 黒 5足セット", "iphone13 ケース 対応", "lサイズ メンズ tシャツ",
    "fl oz 8.5 shampoo", "1,000 piece puzzle", "2,5 kg pesas", "replacement for dyson v8 battery", "rose gold watch",
    "100% cotton t shirt xl", "bpa free water bottle 32oz", "set of 4 wooden hangers", "glass 10 count", "3xl hoodie",
]


@pytest.mark.parametrize("q", QUERIES)
def test_python_extractors_match_polars(q):
    r = pl.DataFrame({"q": [q]}).with_columns(T.for_matching(pl.col("q")).alias("qn")) \
          .with_columns(*T.query_attribute_exprs(pl.col("qn"))).row(0, named=True)
    p = query_attributes(q)
    assert p["norm"] == r["qn"]
    for f in ["measures", "dimensions", "pack_count", "sizes", "audience", "compat", "materials", "colors",
              "has_negation", "has_digit"]:
        expected = r[f"q_{f}"]
        if isinstance(expected, list):
            expected = [x for x in expected if x is not None]
        assert p[f] == expected, f

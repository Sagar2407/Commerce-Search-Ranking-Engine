import polars as pl

from csre.data import text as T


def _q(texts):
    df = pl.DataFrame({"q": texts}).with_columns(T.for_matching(pl.col("q")).alias("m"))
    return df.with_columns(T.query_attribute_exprs(pl.col("m"))).to_dicts()


def test_measures_and_units():
    r = _q(["12 oz coffee mug", "2 in 1 laptop 13.3 inch", "paquete de 12 botellas 500ml", "30cm幅 ラック"])
    assert r[0]["q_measures"] == ["12 oz"]
    assert r[1]["q_measures"] == ["13.3 in"]          # "2 in 1" is not a measure
    assert r[2]["q_measures"] == ["500 ml"] and r[2]["q_pack_count"] == 12
    assert r[3]["q_measures"] == ["30 cm"]


def test_dimensions_sizes_audience():
    r = _q(["16x25x5 filter", "women's running shoes size 8", "iphone xs case", "queen size sheets"])
    assert r[0]["q_dimensions"] == ["16x25x5"]
    assert r[1]["q_sizes"] == ["size:8"] and r[1]["q_audience"] == ["women"]
    assert r[2]["q_sizes"] == []                      # "xs" alone is a phone model, not a size
    assert "queen" in r[3]["q_sizes"]


def test_negation_colors_materials_compat():
    r = _q(["queen sheets without elastic", "2 lápices negro", "stainless steel water bottle",
            "compatible with samsung galaxy s21 ultra case"])
    assert r[0]["q_has_negation"]
    assert r[1]["q_colors"] == ["black"] and r[1]["q_measures"] == []   # "2 l" + accent is not litres
    assert r[2]["q_materials"] == ["stainless steel"]
    assert r[3]["q_compat"] == ["samsung galaxy s21 ultra"]


def test_clean_text_and_bullets():
    df = pl.DataFrame({"d": ["<p>Hello&nbsp;<b>World</b></p> &amp; more"], "b": ["✅ First point\n\n• Second"]})
    out = df.select(T.clean_text(pl.col("d")).alias("d"), T.clean_bullets(pl.col("b")).alias("b")).to_dicts()[0]
    assert out["d"] == "Hello World & more"
    assert out["b"] == ["First point", "Second"]


def test_generic_brands_are_null():
    df = pl.DataFrame({"b": ["Generic", "Nike", "  ", "Desconocido"]})
    assert df.select(T.normalize_brand(pl.col("b"))).to_series().to_list() == [None, "nike", None, None]

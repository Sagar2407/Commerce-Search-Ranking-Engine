"""Text normalisation and attribute extraction (vectorised with Polars / Rust regex).

The same extractors run on product titles and on queries, so that ranking features and
result explanations can say *which* attribute matched ("12 oz", "stainless steel", "women").
Everything here is multilingual-aware for the three ESCI locales (us / es / jp).
"""
from __future__ import annotations

import polars as pl

# --------------------------------------------------------------------------------------
# Regex helpers (Rust regex: no look-arounds / back-references)
# --------------------------------------------------------------------------------------
_META = set("\\.^$|?*+()[]{}-")


def _esc(s: str) -> str:
    return "".join("\\" + c if c in _META else c for c in s)


def _alt(words) -> str:
    return "|".join(_esc(w) for w in sorted(set(words), key=len, reverse=True))


GENERIC_BRANDS = {
    "generic", "unknown", "desconocido", "no brand", "unbranded", "sin marca", "genérico", "generico",
    "n/a", "na", "none", "ノーブランド品", "ノーブランド", "noname", "no name", "various", "varios",
}

# ---------------- measures ----------------
UNIT_CANON = {
    # length
    "inch": "in", "inches": "in", "in": "in", "pulgadas": "in", "pulgada": "in", "インチ": "in",
    "cm": "cm", "mm": "mm", "meter": "m", "meters": "m", "metro": "m", "metros": "m",
    "ft": "ft", "feet": "ft", "foot": "ft", "pies": "ft", "yd": "yd", "yards": "yd",
    # volume
    "fl oz": "fl oz", "fl. oz": "fl oz", "floz": "fl oz", "fluid ounce": "fl oz", "fluid ounces": "fl oz",
    "oz": "oz", "ounce": "oz", "ounces": "oz", "onzas": "oz",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "cc": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l", "litro": "l", "litros": "l", "リットル": "l",
    "gallon": "gal", "gallons": "gal", "gal": "gal", "quart": "qt", "quarts": "qt", "qt": "qt",
    # weight
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb", "libras": "lb",
    "kg": "kg", "kilo": "kg", "kilos": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g", "gramos": "g", "mg": "mg",
    # digital / electrical
    "gb": "gb", "tb": "tb", "mb": "mb", "mah": "mah",
    "w": "w", "watt": "w", "watts": "w", "v": "v", "volt": "v", "volts": "v",
}
_ASCII_UNITS = [u for u in UNIT_CANON if u.isascii()]
_CJK_UNITS = [u for u in UNIT_CANON if not u.isascii()]
NUM = r"\d+(?:\.\d+)?"
# A unit must end at a non-letter (latin incl. accents); CJK letters may follow ("30cm幅").
# A following " <digit>" is captured so that "2 in 1" can be dropped afterwards.
_END = r"(?:\s\d|[^a-z0-9À-ɏ]|$)"
MEASURE_RE = (
    rf"{NUM}\s*-?\s*(?:{_alt(_ASCII_UNITS)}){_END}"
    rf"|{NUM}\s*(?:{_alt(_CJK_UNITS)})"
    rf'|{NUM}\s*"'
)
_UNIT_ALT_ALL = _alt(list(UNIT_CANON) + ['"'])
_UNIT_GROUP = rf"(?:{NUM})\s*-?\s*({_UNIT_ALT_ALL})"

DIMENSION_RE = rf"{NUM}\s*(?:x|×|\*)\s*{NUM}(?:\s*(?:x|×|\*)\s*{NUM})?"

# ---------------- pack / count ----------------
_COUNT_WORDS = ["pack", "pk", "packs", "count", "ct", "pcs", "pc", "pieces", "piece", "pairs", "pair", "rolls",
                "sheets", "piezas", "pieza", "unidades", "uds", "pares", "paquetes", "bolsas"]
_COUNT_CJK = ["個入り", "本入り", "枚入り", "袋入り", "個セット", "本セット", "枚セット", "個", "枚", "本", "袋", "足",
              "組", "セット", "パック", "入り"]
PACK_RE = (
    rf"(?:pack|set|box|case|bag|lot|bundle) of (\d+)"
    rf"|(?:paquete|juego|lote|set|pack) de (\d+)"
    rf"|(\d+)\s*-?\s*(?:{_alt(_COUNT_WORDS)})\b"
    rf"|(\d+)\s*(?:{_alt(_COUNT_CJK)})"
)

# ---------------- audience ----------------
AUDIENCE_CANON = {
    "men's": "men", "mens": "men", "men": "men", "hombre": "men", "hombres": "men", "caballero": "men",
    "メンズ": "men", "男性": "men", "紳士": "men",
    "women's": "women", "womens": "women", "women": "women", "woman": "women", "ladies": "women",
    "mujer": "women", "mujeres": "women", "dama": "women", "レディース": "women", "女性": "women", "婦人": "women",
    "unisex": "unisex", "ユニセックス": "unisex", "男女兼用": "unisex",
    "boys": "boys", "boy's": "boys", "boy": "boys", "男の子": "boys",
    "girls": "girls", "girl's": "girls", "girl": "girls", "niña": "girls", "niñas": "girls", "女の子": "girls",
    "kids": "kids", "kid": "kids", "children": "kids", "child": "kids", "toddler": "kids", "toddlers": "kids",
    "youth": "kids", "niño": "kids", "niños": "kids", "infantil": "kids", "キッズ": "kids", "子供": "kids",
    "子ども": "kids", "こども": "kids", "ジュニア": "kids",
    "baby": "baby", "babies": "baby", "infant": "baby", "newborn": "baby", "bebé": "baby", "bebe": "baby",
    "bebés": "baby", "ベビー": "baby", "赤ちゃん": "baby", "新生児": "baby",
}

# ---------------- materials ----------------
MATERIAL_CANON = {
    "cotton": "cotton", "organic cotton": "cotton", "algodón": "cotton", "algodon": "cotton", "綿": "cotton",
    "コットン": "cotton", "polyester": "polyester", "poliéster": "polyester", "ポリエステル": "polyester",
    "nylon": "nylon", "nailon": "nylon", "ナイロン": "nylon", "spandex": "spandex",
    "leather": "leather", "genuine leather": "leather", "cuero": "leather", "piel": "leather", "本革": "leather",
    "革": "leather", "レザー": "leather", "faux leather": "faux leather", "pu leather": "faux leather",
    "合皮": "faux leather", "suede": "suede", "wool": "wool", "merino": "wool", "lana": "wool", "ウール": "wool",
    "silk": "silk", "seda": "silk", "シルク": "silk", "linen": "linen", "lino": "linen", "リネン": "linen",
    "麻": "linen", "denim": "denim", "fleece": "fleece",
    "stainless steel": "stainless steel", "acero inoxidable": "stainless steel", "ステンレス": "stainless steel",
    "steel": "steel", "acero": "steel", "aluminum": "aluminum", "aluminium": "aluminum", "aluminio": "aluminum",
    "アルミ": "aluminum", "cast iron": "cast iron", "iron": "iron", "hierro": "iron", "鉄": "iron",
    "copper": "copper", "cobre": "copper", "銅": "copper", "brass": "brass", "titanium": "titanium",
    "metal": "metal", "金属": "metal", "plastic": "plastic", "plástico": "plastic", "plastico": "plastic",
    "プラスチック": "plastic", "silicone": "silicone", "silicona": "silicone", "シリコン": "silicone",
    "シリコーン": "silicone", "rubber": "rubber", "goma": "rubber", "ゴム": "rubber",
    "wood": "wood", "wooden": "wood", "madera": "wood", "木製": "wood", "天然木": "wood",
    "bamboo": "bamboo", "bambú": "bamboo", "竹": "bamboo", "glass": "glass", "vidrio": "glass", "cristal": "glass",
    "ガラス": "glass", "ceramic": "ceramic", "cerámica": "ceramic", "セラミック": "ceramic", "陶器": "ceramic",
    "porcelain": "porcelain", "porcelana": "porcelain", "磁器": "porcelain", "stoneware": "stoneware",
    "marble": "marble", "canvas": "canvas", "paper": "paper", "papel": "paper", "紙": "paper",
    "memory foam": "memory foam", "foam": "foam", "espuma": "foam", "ウレタン": "foam", "latex": "latex",
    "acrylic": "acrylic", "acrílico": "acrylic", "アクリル": "acrylic", "resin": "resin", "resina": "resin",
    "樹脂": "resin", "vinyl": "vinyl", "pvc": "pvc", "carbon fiber": "carbon fiber", "sterling silver": "sterling silver",
    "14k gold": "gold", "18k gold": "gold", "10k gold": "gold",
}

# ---------------- colors ----------------
COLOR_CANON = {
    "black": "black", "negro": "black", "negra": "black", "ブラック": "black",
    "white": "white", "blanco": "white", "blanca": "white", "ホワイト": "white",
    "red": "red", "rojo": "red", "roja": "red", "レッド": "red",
    "navy blue": "navy", "navy": "navy", "azul marino": "navy", "ネイビー": "navy",
    "blue": "blue", "azul": "blue", "ブルー": "blue",
    "green": "green", "verde": "green", "グリーン": "green",
    "yellow": "yellow", "amarillo": "yellow", "amarilla": "yellow", "イエロー": "yellow",
    "pink": "pink", "rosa": "pink", "ピンク": "pink",
    "purple": "purple", "morado": "purple", "morada": "purple", "violeta": "purple", "lila": "purple",
    "パープル": "purple",
    "orange": "orange", "naranja": "orange", "オレンジ": "orange",
    "brown": "brown", "marrón": "brown", "marron": "brown", "ブラウン": "brown",
    "gray": "gray", "grey": "gray", "gris": "gray", "グレー": "gray", "グレイ": "gray",
    "silver": "silver", "plateado": "silver", "plata": "silver", "シルバー": "silver",
    "gold": "gold", "dorado": "gold", "ゴールド": "gold",
    "beige": "beige", "ベージュ": "beige", "clear": "clear", "transparente": "clear", "クリア": "clear",
    "透明": "clear", "multicolor": "multicolor", "multi-color": "multicolor", "multicolour": "multicolor",
    "マルチカラー": "multicolor", "rose gold": "rose gold",
}

# ---------------- size labels ----------------
SIZE_CANON = {
    "x-small": "xs", "small": "small", "medium": "medium", "large": "large",
    "x-large": "xl", "xl": "xl", "xx-large": "xxl", "xxl": "xxl", "2xl": "xxl", "xxx-large": "xxxl",
    "xxxl": "xxxl", "3xl": "xxxl", "4xl": "4xl", "5xl": "5xl",
    "pequeño": "small", "pequeña": "small", "mediano": "medium", "mediana": "medium", "grande": "large",
    "extra grande": "xl", "queen size": "queen", "queen": "queen", "king size": "king",
    "california king": "california king", "cal king": "california king", "twin xl": "twin xl",
    "twin size": "twin", "full size": "full",
}
# explicit "size 10" / "talla 42" / "Mサイズ" (bare "xs" is too ambiguous: "iPhone XS")
SIZE_NUM_RE = r"(?:size|talla)\s*:?\s*(\d{1,2}(?:\.5)?|xx?s|x{0,3}l|[sml])\b|(?:^|[^a-z])(xs|s|m|l|ll|xl|xxl|3l)\s*サイズ"

COMPAT_RE = (
    r"(?:compatible with|compatible con|compatible para|designed for|replacement for|for use with|fits)"
    r"\s+([a-z0-9][a-z0-9 .+\-]{1,40})"
)
COMPAT_STOP = ["most", "all", "up", "any", "over", "in", "perfectly", "well", "comfortably", "true", "snugly",
               "securely", "easily", "into", "on", "under", "a", "the", "your", "you", "standard", "almost",
               "many", "nearly", "both", "for", "with", "to", "and", "el", "la", "los", "las", "tu", "su"]
COMPAT_JP_RE = r"([a-z0-9][a-z0-9 .+\-]{2,30}?)\s*(?:対応|専用|用)"

NEGATION_RE = r"\b(?:without|w/o|not|non|no|sin|except|excluding|free of)\b|なし|無し|不要|以外"


def _kw_re(canon: dict) -> str:
    ascii_ = [w for w in canon if w.isascii()]
    other = [w for w in canon if not w.isascii()]
    # \b is unicode-aware: fine for latin (incl. accents); CJK needs no boundaries
    parts = [rf"\b(?:{_alt(ascii_)})\b"]
    latin_ext = [w for w in other if all(ord(c) < 0x3000 for c in w)]
    cjk = [w for w in other if w not in latin_ext]
    if latin_ext:
        parts.append(rf"\b(?:{_alt(latin_ext)})\b")
    if cjk:
        parts.append(rf"(?:{_alt(cjk)})")
    return "|".join(parts)


AUDIENCE_RE = _kw_re(AUDIENCE_CANON)
MATERIAL_RE = _kw_re(MATERIAL_CANON)
COLOR_RE = _kw_re(COLOR_CANON)
SIZE_RE = _kw_re(SIZE_CANON)


# --------------------------------------------------------------------------------------
# Normalisation expressions
# --------------------------------------------------------------------------------------
_ENTITIES = (["&amp;", "&nbsp;", "&quot;", "&#39;", "&apos;", "&lt;", "&gt;", "&#34;"],
             ["&", " ", '"', "'", "'", "<", ">", '"'])


def clean_text(expr: pl.Expr) -> pl.Expr:
    """HTML strip + entity unescape + NFKC + whitespace collapse (keeps case)."""
    return (
        expr.str.replace_all(r"(?i)<br\s*/?>|</p>|</li>|</div>", " ")
        .str.replace_all(r"<[^>]{0,200}>", " ")
        .str.replace_many(*_ENTITIES)
        .str.normalize("NFKC")
        .str.replace_all(r"[​­﻿]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def clean_bullets(expr: pl.Expr) -> pl.Expr:
    """Newline separated bullet string -> list[str] of cleaned, non-empty bullets."""
    return (
        expr.fill_null("")
        .str.split("\n")
        .list.eval(
            clean_text(pl.element())
            .str.replace(r"^[^\p{L}\p{N}]+", "")
            .filter(pl.element().str.len_chars() > 1)
        )
    )


def for_matching(expr: pl.Expr) -> pl.Expr:
    """Lower-cased, NFKC, number-normalised text used by all attribute extractors."""
    return (
        expr.fill_null("")
        .str.normalize("NFKC")
        .str.to_lowercase()
        .str.replace_all("[’`´]", "'")
        .str.replace_all(r"(\d),(\d{3})\b", "${1}${2}")
        .str.replace_all(r"(\d),(\d)", "${1}.${2}")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalize_brand(expr: pl.Expr) -> pl.Expr:
    b = expr.str.normalize("NFKC").str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()
    return pl.when(b.is_in(list(GENERIC_BRANDS)) | (b.str.len_chars() == 0)).then(None).otherwise(b)


def _canon_list(text: pl.Expr, pattern: str, mapping: dict) -> pl.Expr:
    return (
        text.str.extract_all(pattern)
        .list.eval(pl.element().replace(mapping))
        .list.unique(maintain_order=True)
    )


def _measures(text: pl.Expr) -> pl.Expr:
    return (
        text.str.extract_all(MEASURE_RE)
        .list.eval(
            pl.element().filter(~pl.element().str.contains(r"\bin \d$"))    # "2 in 1"
            .str.replace(r"\s\d$", "")
        )
        .list.eval(
            pl.concat_str([
                pl.element().str.extract(rf"({NUM})", 1).str.replace(r"\.0+$", ""),
                pl.element().str.extract(_UNIT_GROUP, 1).replace({**UNIT_CANON, '"': "in"}),
            ], separator=" ")
        )
        .list.unique(maintain_order=True)
    )


def _dimensions(text: pl.Expr) -> pl.Expr:
    return (
        text.str.extract_all(DIMENSION_RE)
        .list.eval(pl.element().str.replace_all(r"\s+", "").str.replace_all(r"[×*]", "x"))
        .list.unique(maintain_order=True)
    )


def _pack_count(text: pl.Expr) -> pl.Expr:
    first = text.str.extract_all(PACK_RE).list.first()
    return first.str.extract(r"(\d+)", 1).cast(pl.Int32, strict=False)


def _sizes(text: pl.Expr) -> pl.Expr:
    labels = _canon_list(text, SIZE_RE, SIZE_CANON)
    numeric = text.str.extract_all(SIZE_NUM_RE).list.eval(
        pl.lit("size:") + pl.coalesce(pl.element().str.extract(SIZE_NUM_RE, 1),
                                      pl.element().str.extract(SIZE_NUM_RE, 2)))
    return pl.concat_list([labels, numeric]).list.unique(maintain_order=True)


def _compat(text: pl.Expr) -> pl.Expr:
    en = text.str.extract_all(COMPAT_RE).list.eval(pl.element().str.extract(COMPAT_RE, 1))
    jp = text.str.extract_all(COMPAT_JP_RE).list.eval(pl.element().str.extract(COMPAT_JP_RE, 1))
    return (
        pl.concat_list([en, jp])
        .list.eval(
            pl.element().str.split(" ").list.head(4).list.join(" ").str.strip_chars(" .-+")
            .filter((pl.element().str.len_chars() >= 3) & pl.element().str.contains(r"[a-z]")
                    & ~pl.element().str.split(" ").list.first().is_in(COMPAT_STOP))
        )
        .list.unique(maintain_order=True)
    )


def attribute_exprs(title_m: pl.Expr, rich_m: pl.Expr, prefix: str = "attr_") -> list[pl.Expr]:
    """Attribute extractors.

    title_m: matching-normalised *title* (high precision fields: measures, sizes, audience, ...)
    rich_m:  matching-normalised title + colour field + bullets (keyword fields: materials, colours)
    """
    return [
        _measures(title_m).alias(f"{prefix}measures"),
        _dimensions(title_m).alias(f"{prefix}dimensions"),
        _pack_count(title_m).alias(f"{prefix}pack_count"),
        _sizes(title_m).alias(f"{prefix}sizes"),
        _canon_list(title_m, AUDIENCE_RE, AUDIENCE_CANON).alias(f"{prefix}audience"),
        _compat(title_m).alias(f"{prefix}compat"),
        _canon_list(rich_m, MATERIAL_RE, MATERIAL_CANON).alias(f"{prefix}materials"),
        _canon_list(rich_m, COLOR_RE, COLOR_CANON).alias(f"{prefix}colors"),
    ]


def query_attribute_exprs(q_m: pl.Expr, prefix: str = "q_") -> list[pl.Expr]:
    return [
        *attribute_exprs(q_m, q_m, prefix=prefix),
        q_m.str.contains(NEGATION_RE).alias(f"{prefix}has_negation"),
        q_m.str.contains(r"\d").alias(f"{prefix}has_digit"),
    ]

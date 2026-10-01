"""Unified department taxonomy + product category classifier.

ESCI products carry no category. The ESCI-S sample (4.4K real product pages across us/es/jp)
does, so we (1) map its locale-specific top-level categories to one cross-locale department
list, (2) optionally add weak labels by propagating seed departments to products that were judged
*Exact* for the same query as a seed, (3) train a char+word TF-IDF logistic-regression model (SGD),
(4) evaluate on held-out *seed* products only, and (5) predict a department + confidence for the
whole catalog. Low-confidence predictions become "Unknown" rather than a guess.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from ..config import Config
from ..utils import duck, get_logger, stage_timer, write_parquet
from . import text as T

log = get_logger("csre.taxonomy")

DEPARTMENT_MAP: dict[str, list[str]] = {
    "Clothing, Shoes & Jewelry": ["Clothing, Shoes & Jewelry", "Moda", "ファッション", "Shoe, Jewelry & Watch Accessories"],
    "Home & Kitchen": ["Home & Kitchen", "Hogar y cocina", "Kitchen & Dining", "Iluminación",
                       "Lights & Lighting Accessories", "Instalación de baño y cocina", "Appliances", "ホーム＆キッチン",
                       "Dining & Entertaining", "Aspiración, limpieza y cuidado de suelos", "大型家電",
                       "Heating, Cooling & Air Quality", "Productos para cocina y repostería",
                       "Large Appliances", "Grandes electrodomésticos", "Small Appliance Parts & Accessories"],
    "Beauty & Personal Care": ["Beauty & Personal Care", "Belleza", "Beauty", "ビューティー"],
    "Health & Household": ["Health & Household", "Salud y cuidado personal", "Health & Personal Care",
                           "ドラッグストア", "Suministros y equipamiento médico",
                           "Sillas de ruedas, sillas de ruedas eléctricas, scooters para discapacitados y accesorios",
                           "Medical Supplies & Equipment", "Mobility & Daily Living Aids",
                           "Cepillos de dientes y accesorios", "Ayudas para la movilidad y vida diaria",
                           "Sports Care, Accessories & Compression", "Monitores de diagnóstico y salud",
                           "Cuidado dental de bebés y niños", "Suministros médicos profesionales"],
    "Sports & Outdoors": ["Sports & Outdoors", "Deportes y aire libre", "スポーツ＆アウトドア", "Hunting & Fishing",
                          "Sports & Outdoor Recreation Accessories"],
    "Tools, Home Improvement & Garden": ["Tools & Home Improvement", "Bricolaje y herramientas", "DIY, Tools & Garden",
                                         "DIY・工具・ガーデン", "Power & Hand Tools", "Herramientas manuales y eléctricas",
                                         "Patio, Lawn & Garden", "Jardín", "Materiales", "Power Tool Parts & Accessories",
                                         "Outdoor Power Tools", "Grills & Outdoor Cooking", "Safety & Security",
                                         "Prevención y seguridad", "Accesorios para herramientas eléctricas",
                                         "Cortacéspedes y herramientas eléctricas de jardín", "Lighting Assemblies & Accessories",
                                         "Herramientas eléctricas y de mano", "Replacement Parts"],
    "Electronics & Computers": ["Electronics", "Electrónica", "Computers", "Informática", "パソコン・周辺機器",
                                "Cell Phones & Accessories", "Video Games", "Videojuegos", "家電＆カメラ", "Software", "ゲーム", "PCソフト",
                                "Consumer Electronics", "Amazon Devices & Accessories", "Dispositivos Amazon y Accesorios",
                                "Car & Vehicle Electronics"],
    "Toys, Games & Hobbies": ["Toys & Games", "Juguetes y juegos", "おもちゃ", "Hobbies", "ホビー",
                              "Remote & App Controlled Vehicles & Parts", "Remote & App Controlled Vehicle Parts"],
    "Books & Media": ["Books", "Kindle Store", "Libros", "Tienda Kindle", "Japanese Books", "本", "Kindleストア",
                      "Foreign Language Books", "Audible Books & Originals", "Movies & TV", "Películas y TV", "DVD",
                      "CDs & Vinyl", "Music", "ミュージック", "CDs y vinilos", "Magazine Subscriptions",
                      "Digital Music", "Audible Libros y Originales"],
    "Automotive": ["Automotive", "Coche y moto", "車＆バイク", "Motores y piezas del motor",
                   "Heavy Duty & Commercial Vehicle Equipment", "Piezas para coche", "Motorcycle & Powersports"],
    "Office Products": ["Office Products", "Oficina y papelería", "Material de oficina", "文房具・オフィス用品",
                        "Accesorios de escritorio y productos de oficina"],
    "Grocery & Gourmet Food": ["Grocery & Gourmet Food", "Alimentación y bebidas", "Food, Beverages & Alcohol",
                               "食品・飲料・お酒", "Lácteos, huevos y alternativas vegetales"],
    "Pet Supplies": ["Pet Supplies", "Productos para mascotas", "ペット用品"],
    "Baby Products": ["Baby Products", "Bebé", "Baby", "ベビー＆マタニティ"],
    "Arts, Crafts & Sewing": ["Arts, Crafts & Sewing", "Handmade Products", "Productos Handmade",
                              "Collectibles & Fine Art", "Costura y manualidades"],
    "Industrial & Scientific": ["Industrial & Scientific", "Industria, empresas y ciencia",
                                "Productos de laboratorio y ciencias", "産業・研究開発用品",
                                "Restaurant Appliances & Equipment", "Food Service Equipment & Supplies", "Lab & Scientific Products",
                                "Equipos y suministros agrícolas", "Artículos y equipo de servicio de comida",
                                "Equipos e instrumental de laboratorio", "Janitorial & Sanitation Supplies",
                                "Suministros de limpieza y sanitarios", "Productos de mantenimiento de instalaciones"],
    "Musical Instruments": ["Musical Instruments", "Instrumentos musicales", "Instrument Accessories", "楽器・音響機器"],
}
RAW_TO_DEPT = {raw: dept for dept, raws in DEPARTMENT_MAP.items() for raw in raws}
DEPARTMENTS = list(DEPARTMENT_MAP)
UNKNOWN = "Unknown"

_CURRENCY = {"us": "USD", "es": "EUR", "jp": "JPY"}


def _parse_number(s: str | None, locale: str) -> float | None:
    if not s:
        return None
    s = s.strip()
    m = re.search(r"[\d.,]+", s)
    if not m:
        return None
    num = m.group(0)
    if locale == "es":           # 1.234,56
        num = num.replace(".", "").replace(",", ".")
    else:                        # 1,234.56 / ¥1,234
        num = num.replace(",", "")
    try:
        return float(num)
    except ValueError:
        return None


def _parse_stars(s: str | None) -> float | None:
    """'4.4 out of 5 stars' / '4,6 de 5 estrellas' / '5つ星のうち4.3' -> 4.4 / 4.6 / 4.3."""
    if not s:
        return None
    nums = re.findall(r"\d+(?:[.,]\d+)?", s)
    if not nums:
        return None
    v = nums[-1] if "うち" in s else nums[0]
    try:
        x = float(v.replace(",", "."))
    except ValueError:
        return None
    return x if 0 < x <= 5 else None


def _parse_count(s: str | None) -> float | None:
    """'1,116 ratings' / '53.170 valoraciones' -> integer count (separators differ by locale)."""
    if not s:
        return None
    m = re.search(r"\d[\d.,]*", s)
    return float(re.sub(r"[.,]", "", m.group(0))) if m else None


def _meta_row(r: dict) -> dict:
    loc = r.get("locale")
    cats = [c for c in (r.get("category") or []) if c]
    top = cats[0] if cats else None
    return {
        "product_id": r["asin"], "locale": loc, "page_type": r.get("type"),
        "category_path": " > ".join(cats) if cats else None,
        "category_top_raw": top,
        "department": RAW_TO_DEPT.get(top, "Other" if top else None),
        "price": _parse_number(r.get("price"), loc),
        "currency": _CURRENCY.get(loc),
        "stars": _parse_stars(r.get("stars")),
        "n_ratings": _parse_count(r.get("ratings")),
        "template": r.get("template") or None,
    }


def load_esci_s_sample(cfg: Config) -> pl.DataFrame:
    """Parse the ESCI-S sample into a tidy frame of real category/price/rating metadata."""
    path = cfg.path("raw", "esci_s", "sample.json.gz")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "error":
                continue
            rows.append(_meta_row(r))
    return pl.DataFrame(rows)


def load_esci_s(cfg: Config) -> pl.DataFrame:
    """Real product-page metadata: the full ESCI-S dump when present (1.48M pages), else the 4.4K sample.

    Adds `image_url`, `category_leaf` and `brand_attr` when the full dump is used.
    """
    from .esci_s import parse_full_dump  # noqa: PLC0415
    sample = load_esci_s_sample(cfg).with_columns(pl.lit("sample").alias("esci_s_source"))
    raw_path = parse_full_dump(cfg)
    if raw_path is None:
        return sample
    raw = pl.read_parquet(raw_path)
    rows = [_meta_row({"asin": r["product_id"], "locale": r["locale"], "type": "product",
                       "category": r["category_path"], "price": r["price_raw"], "stars": r["stars_raw"],
                       "ratings": r["ratings_raw"], "template": r["template"]})
            for r in raw.select("product_id", "locale", "category_path", "price_raw", "stars_raw", "ratings_raw",
                                "template").iter_rows(named=True)]
    full = pl.DataFrame(rows, schema=sample.drop("esci_s_source").schema).with_columns(
        pl.lit("full").alias("esci_s_source"),
        raw["image_url"], raw["category_path"].list.last().alias("category_leaf"), raw["brand_attr"],
    )
    extra = sample.join(full.select("product_id", "locale"), on=["product_id", "locale"], how="anti")
    return pl.concat([full, extra], how="diagonal_relaxed")


def _model_text(df: pl.DataFrame) -> pl.Series:
    return df.select(
        T.for_matching(pl.concat_str([
            pl.col("title"), pl.col("brand").fill_null(""),
            pl.col("bullets").list.head(3).list.join(" ").str.slice(0, 400),
        ], separator=" | "))
    ).to_series()


def build_model(alpha: float = 1e-5) -> Pipeline:
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                                 max_features=300_000, dtype=np.float32)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w\w+\b", max_features=200_000, dtype=np.float32)),
    ])
    # SGD logistic regression (alpha picked by CV: 1e-6..2e-5): linear-time in #docs, so it can be retrained on millions of
    # weak labels (lbfgs LogisticRegression did not scale past ~50K docs here).
    clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=30, tol=1e-4, class_weight="balanced",
                        n_jobs=2, random_state=0)
    return Pipeline([("features", feats), ("clf", clf)])


def _seed_products(cfg: Config, meta: pl.DataFrame) -> pl.DataFrame:
    """Join ESCI-S labels to cleaned ESCI product text."""
    con = duck(cfg)
    keys = meta.filter(pl.col("department").is_not_null() & (pl.col("department") != "Other")) \
               .select("product_id", "locale", "department")
    con.register("keys", keys.to_arrow())
    prod = con.sql(f"""
        SELECT p.product_id, p.product_locale AS locale, p.product_title, p.product_brand, p.product_bullet_point,
               k.department
        FROM '{cfg.path('raw', 'esci', 'shopping_queries_dataset_products.parquet')}' p
        JOIN keys k ON p.product_id = k.product_id AND p.product_locale = k.locale
    """).pl()
    return _prep_text(prod)


def _prep_text(prod: pl.DataFrame) -> pl.DataFrame:
    return prod.with_columns(
        T.clean_text(pl.col("product_title")).alias("title"),
        pl.col("product_brand").alias("brand"),
        T.clean_bullets(pl.col("product_bullet_point")).alias("bullets"),
    ).drop("product_title", "product_brand", "product_bullet_point")


def _propagated_labels(cfg: Config, seeds: pl.DataFrame, max_per_query: int = 40) -> pl.DataFrame:
    """Weak labels: products judged Exact for a query on which a seed product is also Exact."""
    con = duck(cfg)
    con.register("seeds", seeds.select("product_id", "locale", "department").to_arrow())
    ex = cfg.path("raw", "esci", "shopping_queries_dataset_examples.parquet")
    weak = con.sql(f"""
        WITH e AS (SELECT query_id, product_id, product_locale AS locale FROM '{ex}' WHERE esci_label='E'),
        sq AS (SELECT e.query_id, s.department, s.product_id AS seed_id FROM e JOIN seeds s USING (product_id, locale)),
        -- a query is usable only if all its seeds agree on the department
        qd AS (SELECT query_id, any_value(department) AS department, any_value(seed_id) AS seed_id FROM sq
               GROUP BY 1 HAVING count(DISTINCT department)=1)
        SELECT e.product_id, e.locale, qd.department, qd.seed_id, qd.query_id
        FROM e JOIN qd USING (query_id)
        WHERE e.product_id NOT IN (SELECT product_id FROM seeds)
        QUALIFY row_number() OVER (PARTITION BY e.query_id ORDER BY e.product_id) <= {max_per_query}
    """).pl()
    # a product reached from seeds of different departments is ambiguous: drop it
    weak = weak.group_by("product_id", "locale").agg(
        pl.col("department").n_unique().alias("nd"), pl.col("department").first(), pl.col("seed_id").first()
    ).filter(pl.col("nd") == 1).drop("nd")
    con.register("weak", weak.select("product_id", "locale").to_arrow())
    txt = con.sql(f"""
        SELECT p.product_id, p.product_locale AS locale, p.product_title, p.product_brand, p.product_bullet_point
        FROM '{cfg.path('raw', 'esci', 'shopping_queries_dataset_products.parquet')}' p
        JOIN weak w ON p.product_id = w.product_id AND p.product_locale = w.locale
    """).pl()
    return _prep_text(txt.join(weak, on=["product_id", "locale"]))


def train_category_model(cfg: Config, propagate: bool = True) -> dict:
    with stage_timer(cfg, "taxonomy") as info:
        meta = load_esci_s(cfg)
        write_parquet(meta, cfg.path("processed", "esci_s_meta.parquet"))
        n_real = int(meta.filter(pl.col("department").is_not_null() & (pl.col("department") != "Other")).height)
        # With the full dump most products have a real department; the classifier only fills the gaps, so it is
        # trained on a capped stratified sample, and graph propagation (weak labels) is only used when real
        # labels are scarce (the 4.4K sample).
        max_seeds = int(cfg.get("taxonomy.max_seed_labels", 150_000))
        if n_real > max_seeds:
            meta_seed = meta.filter(pl.col("department").is_not_null() & (pl.col("department") != "Other")) \
                .sample(fraction=1.0, shuffle=True, seed=cfg.seed) \
                .with_columns(pl.col("product_id").cum_count().over("department").alias("_r"),
                              (pl.col("product_id").count().over("department").cast(pl.Float64) * max_seeds / n_real).ceil().alias("_cap")) \
                .filter(pl.col("_r") <= pl.col("_cap")).drop("_r", "_cap")
        else:
            meta_seed = meta
        seeds = _seed_products(cfg, meta_seed)
        propagate = propagate and n_real < int(cfg.get("taxonomy.propagate_below", 50_000))
        weak = _propagated_labels(cfg, seeds) if propagate else None
        log.info("seed labels: %d  propagated weak labels: %d", len(seeds), 0 if weak is None else len(weak))

        X_seed = _model_text(seeds).to_numpy()
        y_seed = seeds["department"].to_numpy()
        folds = int(cfg.get("taxonomy.cv_folds", 5))
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
        oof_pred = np.empty(len(seeds), dtype=object)
        oof_conf = np.zeros(len(seeds))
        seed_ids = seeds["product_id"].to_numpy()
        for k, (tr, te) in enumerate(skf.split(X_seed, y_seed)):
            X_tr, y_tr = X_seed[tr], y_seed[tr]
            if weak is not None and len(weak):
                held = set(seed_ids[te])
                w = weak.filter(~pl.col("seed_id").is_in(list(held)) & ~pl.col("product_id").is_in(list(held)))
                X_tr = np.concatenate([X_tr, _model_text(w).to_numpy()])
                y_tr = np.concatenate([y_tr, w["department"].to_numpy()])
            m = build_model().fit(X_tr, y_tr)
            proba = m.predict_proba(X_seed[te])
            oof_pred[te] = m.classes_[proba.argmax(1)]
            oof_conf[te] = proba.max(1)
            log.info("fold %d acc=%.3f", k, accuracy_score(y_seed[te], oof_pred[te]))

        tau = float(cfg.get("taxonomy.min_confidence", 0.5))
        covered = oof_conf >= tau
        by_locale = {}
        for loc in ["us", "es", "jp"]:
            msk = seeds["locale"].to_numpy() == loc
            by_locale[loc] = {
                "n": int(msk.sum()),
                "accuracy": round(float(accuracy_score(y_seed[msk], oof_pred[msk])), 4),
                "accuracy_at_tau": round(float(accuracy_score(y_seed[msk & covered], oof_pred[msk & covered])), 4),
                "coverage_at_tau": round(float(covered[msk].mean()), 4),
            }
        curve = []
        for t in [0.0, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]:
            c = oof_conf >= t
            curve.append({"threshold": t, "coverage": round(float(c.mean()), 4),
                          "accuracy": round(float(accuracy_score(y_seed[c], oof_pred[c])), 4) if c.any() else None})
        metrics = {
            "esci_s_pages": int(meta.height),
            "esci_s_real_departments": n_real,
            "n_seed_labels": int(len(seeds)),
            "n_weak_labels": 0 if weak is None else int(len(weak)),
            "propagation": propagate,
            "cv_folds": folds,
            "accuracy": round(float(accuracy_score(y_seed, oof_pred)), 4),
            "macro_f1": round(float(f1_score(y_seed, oof_pred, average="macro")), 4),
            "threshold": tau,
            "accuracy_at_threshold": round(float(accuracy_score(y_seed[covered], oof_pred[covered])), 4),
            "coverage_at_threshold": round(float(covered.mean()), 4),
            "by_locale": by_locale,
            "coverage_accuracy_curve": curve,
            "label_distribution": seeds["department"].value_counts().sort("count", descending=True).to_dicts(),
        }
        log.info("category model CV: %s", {k: metrics[k] for k in ("accuracy", "macro_f1", "accuracy_at_threshold",
                                                                     "coverage_at_threshold")})
        # final model on all labels
        X_all, y_all = X_seed, y_seed
        if weak is not None and len(weak):
            X_all = np.concatenate([X_all, _model_text(weak).to_numpy()])
            y_all = np.concatenate([y_all, weak["department"].to_numpy()])
        model = build_model().fit(X_all, y_all)
        model_path = cfg.path("models", "category_model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        cfg.path("reports").mkdir(parents=True, exist_ok=True)
        cfg.path("reports", "category_model_metrics.json").write_text(json.dumps(metrics, indent=2))
        if "image_url" in meta.columns:   # display metadata for the storefront (real, from ESCI-S)
            write_parquet(product_meta(meta), cfg.path("processed", "product_meta.parquet"))
        info.update(rows=len(y_all), accuracy=metrics["accuracy"], coverage=metrics["coverage_at_threshold"])
    return metrics


def product_meta(meta: pl.DataFrame) -> pl.DataFrame:
    """Storefront display metadata (real, from ESCI-S product pages)."""
    return meta.filter(pl.col("product_id").is_not_null()).select(
        (pl.col("locale") + ":" + pl.col("product_id")).alias("doc_id"), "locale", "category_path",
        "category_leaf", "image_url", "brand_attr")


class CategoryPredictor:
    def __init__(self, cfg: Config):
        self.model: Pipeline = joblib.load(cfg.path("models", "category_model.joblib"))
        self.tau = float(cfg.get("taxonomy.min_confidence", 0.5))
        self.seed_labels = self._seed_labels(cfg)

    @staticmethod
    def _seed_labels(cfg: Config) -> dict[tuple[str, str], str]:
        p = cfg.path("processed", "esci_s_meta.parquet")
        if not Path(p).exists():
            return {}
        m = pl.read_parquet(p).filter(pl.col("department").is_not_null() & (pl.col("department") != "Other"))
        return {(r["locale"], r["product_id"]): r["department"] for r in m.iter_rows(named=True)}

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """df needs product_id, locale, title, brand, bullets -> adds category, category_conf, category_source."""
        proba = self.model.predict_proba(_model_text(df).to_numpy())
        classes = self.model.classes_
        pred = classes[proba.argmax(1)]
        conf = proba.max(1).astype(np.float32)
        cat = np.where(conf >= self.tau, pred, UNKNOWN).astype(object)
        src = np.where(conf >= self.tau, "model", "unknown").astype(object)
        if self.seed_labels:
            keys = zip(df["locale"].to_list(), df["product_id"].to_list())
            for i, k in enumerate(keys):
                lab = self.seed_labels.get(k)
                if lab is not None:
                    cat[i], conf[i], src[i] = lab, 1.0, "esci_s"
        return df.with_columns(
            pl.Series("category", cat.tolist(), dtype=pl.Utf8),
            pl.Series("category_pred", pred.tolist(), dtype=pl.Utf8),
            pl.Series("category_conf", conf),
            pl.Series("category_source", src.tolist(), dtype=pl.Utf8),
        )

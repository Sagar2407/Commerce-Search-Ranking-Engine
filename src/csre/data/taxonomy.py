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
    "Clothing, Shoes & Jewelry": ["Clothing, Shoes & Jewelry", "Moda", "ファッション"],
    "Home & Kitchen": ["Home & Kitchen", "Hogar y cocina", "Kitchen & Dining", "Iluminación",
                       "Lights & Lighting Accessories", "Instalación de baño y cocina", "Appliances",
                       "Large Appliances", "Grandes electrodomésticos", "Small Appliance Parts & Accessories"],
    "Beauty & Personal Care": ["Beauty & Personal Care", "Belleza", "Beauty"],
    "Health & Household": ["Health & Household", "Salud y cuidado personal", "Health & Personal Care",
                           "ドラッグストア", "Suministros y equipamiento médico",
                           "Sillas de ruedas, sillas de ruedas eléctricas, scooters para discapacitados y accesorios"],
    "Sports & Outdoors": ["Sports & Outdoors", "Deportes y aire libre"],
    "Tools, Home Improvement & Garden": ["Tools & Home Improvement", "Bricolaje y herramientas", "DIY, Tools & Garden",
                                         "DIY・工具・ガーデン", "Power & Hand Tools", "Herramientas manuales y eléctricas",
                                         "Patio, Lawn & Garden", "Jardín"],
    "Electronics & Computers": ["Electronics", "Electrónica", "Computers", "Informática", "パソコン・周辺機器",
                                "Cell Phones & Accessories", "Video Games", "Videojuegos"],
    "Toys, Games & Hobbies": ["Toys & Games", "Juguetes y juegos", "おもちゃ", "Hobbies", "ホビー"],
    "Books & Media": ["Books", "Kindle Store", "Libros", "Tienda Kindle", "Japanese Books", "本", "Kindleストア",
                      "Foreign Language Books", "Audible Books & Originals", "Movies & TV", "Películas y TV", "DVD",
                      "CDs & Vinyl", "Music", "ミュージック"],
    "Automotive": ["Automotive", "Coche y moto", "車＆バイク", "Motores y piezas del motor",
                   "Heavy Duty & Commercial Vehicle Equipment"],
    "Office Products": ["Office Products", "Oficina y papelería"],
    "Grocery & Gourmet Food": ["Grocery & Gourmet Food", "Alimentación y bebidas", "Food, Beverages & Alcohol",
                               "食品・飲料・お酒"],
    "Pet Supplies": ["Pet Supplies", "Productos para mascotas", "ペット用品"],
    "Baby Products": ["Baby Products", "Bebé", "Baby"],
    "Arts, Crafts & Sewing": ["Arts, Crafts & Sewing", "Handmade Products", "Productos Handmade",
                              "Collectibles & Fine Art"],
    "Industrial & Scientific": ["Industrial & Scientific", "Industria, empresas y ciencia",
                                "Productos de laboratorio y ciencias", "産業・研究開発用品",
                                "Restaurant Appliances & Equipment"],
    "Musical Instruments": ["Musical Instruments", "Instrumentos musicales"],
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


def load_esci_s_sample(cfg: Config) -> pl.DataFrame:
    """Parse the ESCI-S sample into a tidy frame of real category/price/rating metadata."""
    path = cfg.path("raw", "esci_s", "sample.json.gz")
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "error":
                continue
            loc = r.get("locale")
            cats = r.get("category") or []
            top = cats[0] if cats else None
            stars = _parse_number((r.get("stars") or "").split(" ")[0], loc if loc == "es" else "us")
            rows.append({
                "product_id": r["asin"], "locale": loc, "page_type": r.get("type"),
                "category_path": " > ".join(cats) if cats else None,
                "category_top_raw": top,
                "department": RAW_TO_DEPT.get(top, "Other" if top else None),
                "price": _parse_number(r.get("price"), loc),
                "currency": _CURRENCY.get(loc),
                "stars": stars if stars is not None and 0 < stars <= 5 else None,
                "n_ratings": _parse_number((r.get("ratings") or "").split(" ")[0], loc if loc != "es" else "us"),
                "template": r.get("template") or None,
            })
    return pl.DataFrame(rows)


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
        meta = load_esci_s_sample(cfg)
        write_parquet(meta, cfg.path("processed", "esci_s_meta.parquet"))
        seeds = _seed_products(cfg, meta)
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
        info.update(rows=len(y_all), accuracy=metrics["accuracy"], coverage=metrics["coverage_at_threshold"])
    return metrics


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

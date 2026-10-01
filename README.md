# Commerce Search & Ranking Engine — data layer

**Business question:** how can a shopper find the right product when their query is vague,
incomplete, or uses different words from the catalog?

This repo is phase 1 of the project: a reproducible, scalable data layer on which keyword
retrieval, embedding retrieval, hybrid search and a learned reranker can be built, compared under
identical evaluation conditions, and demoed in a storefront. One command rebuilds everything:

```bash
pip install -e ".[dev]"
make data            # or: csre all
```

Every stage is deterministic (seeded, content-hashed), streams in fixed-size batches or shards,
and records rows / bytes / throughput to `data/reports/manifest.json`. `csre validate` is a
data-quality gate (integrity, leakage, simulator sanity) that exits non-zero on failure.

## What gets built

Provenance matters here, so every dataset is tagged as **real**, **derived** (deterministic
transformations of real data) or **simulated** (generated, calibrated to real data, never used as a
relevance label).

| Dataset | Provenance | Rows | What it is for |
|---|---|---|---|
| `processed/catalog/locale=*/` | real + derived | 1.81M products | cleaned text, extracted attributes, sparsity flags, department, retrieval `doc_text` |
| `processed/judgments.parquet` | real | 2.62M (query, product) labels | ESCI Exact / Substitute / Complement / Irrelevant with gains + grades |
| `processed/queries.parquet` | real + derived | 130,652 queries | splits, parsed query attributes, label profile, intent diversity, **slice flags** |
| `processed/qrels/<split>/<locale>.qrels` | real | — | TREC qrels for standard IR tooling |
| `processed/graph/product_edges.parquet` | derived | 17.3M edges | substitute / complement / co-exact product relations from co-judgments |
| `processed/graph/category_complements.parquet` | derived | — | which departments complement which |
| `processed/graph/related_queries.parquet` | derived | 781K pairs | queries sharing Exact products (reformulation, related searches) |
| `synthetic/product_commerce.parquet` | simulated (real where available) | 1.81M | price, rating, review count, stock, popularity |
| `synthetic/logs/searches/` | simulated | 37.7M searches | sessions, users, timestamps, logging policy, outcomes |
| `synthetic/logs/impressions/` | simulated | 634M impressions | position, click, cart, purchase, dwell, explicit feedback, propensity |
| `synthetic/logs/query_doc_stats.parquet` | simulated | per (query, doc) | aggregated + IPS-weighted feedback for learning from clicks |
| `synthetic/replay/requests.parquet` | simulated | 10M requests | 24h load-test stream with one-off query variants (cache / latency / robustness) |
| `scale/synthetic_catalog/` | simulated | 23.2M distractors | catalog tiers 2.5M → 25M docs for latency / memory / cost scaling |
| `demo/` | subset | 3K queries, 151K products | slice of everything for the storefront demo and CI |
| `demo_portable/` | subset | 1.5K queries, 43K products | download-sized copy (< 30 MB), same schemas minus long description text |

Exact figures for your run are in `data/reports/datacard.md` / `datacard.json`.

## Sources

* **Amazon Shopping Queries Dataset (ESCI)** — 130K difficult queries in English, Spanish and
  Japanese, up to 40 human-judged candidates each (Reddy et al., 2022, Apache-2.0). Downloaded from
  the official repo's Git LFS and verified against its SHA-256 pointers.
* **ESCI-S sample** — 4.4K real product pages (category path, price, stars, review count) for
  ESCI products (Apache-2.0). Used as ground truth for the department classifier and to calibrate
  simulated commerce attributes. The full 1.66M-product ESCI-S dump (S3) was not reachable from
  the build environment; wiring it in to replace the classifier and simulated prices is a planned
  extension.

## Evaluation protocol decisions (made here, once, so every model is compared the same way)

* **Splits by query.** ESCI `test` is the untouched held-out set (30,969 queries). `dev` is a
  deterministic 10% of ESCI `train` queries (9,989). One ESCI query appears in both official splits;
  it is pinned to `test` and its 3 train rows are dropped (2,621,285 of the 2,621,288 released rows remain).
* **Graded gains** from the ESCI paper: E = 1.0, S = 0.1, C = 0.01, I = 0 (nDCG), with grades 3/2/1/0
  for qrels.
* **Unknown ≠ irrelevant.** Relevance is known only inside each query's judged candidate set. Every
  product is judged for *some* query, so full-catalog retrieval will surface products that are
  simply unjudged for *this* query. Retrieval metrics must report judged@k and treat unjudged
  results as unknown (condensed lists), never as irrelevant.
* **Slices** (boolean flags on `queries.parquet`), so every metric can be broken down:

  | Slice | Definition |
  |---|---|
  | `underspecified` | ≤ 2 tokens (≤ 6 chars for Japanese) with no brand, number or attribute |
  | `multi_intent` | Exact products span ≥ 2 departments (entropy ≥ 0.9 bits, runner-up ≥ 20%) |
  | `ambiguous` | underspecified OR multi-intent |
  | `negation` | ESCI "negations" source or explicit negation ("without", "sin", "なし") |
  | `spec` | states a measure, dimension, size or pack count ("12 oz", "16x25x5", "queen") |
  | `brand` | mentions a catalog brand (dictionary match with a specificity filter) |
  | `sparse_products` | ≥ 50% of its Exact products have little description text |
  | `hard` | ESCI "small version" (easy queries removed by the dataset authors) |
  | locale, length bucket, source, traffic bucket (head / torso / tail) | |

## What is derived, and how

* **Attributes** (`attr_*` columns) — multilingual regex extractors (Rust regex via Polars) for
  measures with canonical units, dimensions, pack counts, sizes, audience, compatibility, materials
  and colours. The same extractors parse queries, so a result explanation can say *which* attribute
  matched ("12 oz", "stainless steel", "women").
* **Department** — ESCI has no categories. A TF-IDF + SGD logistic model is trained on the 4.1K real
  ESCI-S labels plus 58K weak labels propagated through the relevance graph (products judged Exact
  for the same query as a labelled product). Evaluated on held-out real labels only: 77% accuracy
  overall, **90% at confidence ≥ 0.5 on 70% of products**; the rest is `Unknown` rather than a guess.
* **Sparse descriptions** — description + bullets shorter than 150 chars (60 for Japanese).
* **Product graph** — two products judged for the same query inherit a relation: Exact × Exact =
  co-exact, Exact → Substitute, Exact → Complement, with support counts.

## What is simulated, and why

ESCI has labels but no behaviour, prices or traffic. The project needs them for a feedback loop,
caching and load tests, and for separating *offline relevance* from *conversion*. The simulator
(`csre.sim`) is grounded in real labels and real-calibrated attributes, and every assumption is a
config value you can stress-test:

* **Commerce attributes** — per (locale, department) price and review-count distributions fitted
  on ESCI-S and shrunk to locale level; real values used where ESCI-S has them.
* **Traffic** — Zipf query popularity (short / generic queries skew head), diurnal and weekly
  patterns per locale time zone, heavy-tailed user activity, reformulation after abandonment
  (to a related query when one exists).
* **Logging policy** — a deliberately imperfect production ranker (label + popularity + noise)
  with 5% RandTop-10 exploration traffic, so position bias can be estimated from the logs.
* **Click model** — position-based examination × label attractiveness (modulated by rating and
  relative price) → click → add-to-cart → purchase, plus rare, noisy thumbs up/down. The true
  examination propensity is logged so IPS estimators can be checked against ground truth.
* **Replay stream** — a 24h, 10M-request stream; 35% of requests are one-off variants of a known
  query (typos, dropped / reordered tokens, modifiers, case and spacing) so cache hit rates are
  realistic and the variants double as a robustness slice (they keep `query_id` of the parent).

Simulated conversion is **not** real conversion impact. The same click model is reused later to
contrast offline nDCG with simulated online outcomes, and the write-up must say so.

Logs cover queries from every split (as production logs would), tagged with `query_split`. Click
features are legitimate ranking inputs, but because the simulator generates clicks *from* the ESCI
labels, any offline gain from click features on test queries is simulator-dependent: report
content-only rankers and feedback-aware rankers separately, and break results down by traffic
bucket (feedback can only help queries that have traffic).

## Scale

* The catalog stage streams 100K-row batches (constant memory); traffic generation is sharded
  (250K sessions per shard, independently seeded) — more data means more shards, not more RAM.
* All randomness that must be stable per entity (dev split, prices, popularity) is keyed on an
  MD5 hash of the entity id, so any shard can be regenerated alone and identically.
* `scale/` adds perturbed distractor products (brand swapped, numbers rescaled, colour swapped,
  pack suffix) up to 25M docs, keeping `base_doc_id` so embeddings can be derived without
  re-encoding. They are never judged and never enter quality metrics.

```bash
# 100M sessions instead of 30M: same code, more shards
csre traffic --set simulation.traffic.n_sessions=100000000 --set simulation.traffic.shards=400
# a 50M-doc tier
csre scale --set "scale.tiers=[2500000,5000000,10000000,25000000,50000000]"
```

## Layout

```
configs/data.yaml        every parameter (paths, thresholds, simulator settings, tiers)
src/csre/data/           acquire, text (normalise + attributes), taxonomy, catalog, judgments,
                         graph, scale, demo, validate, datacard
src/csre/sim/            commerce attributes, click model, traffic + replay
tests/                   extractor, hashing and click-model tests
data/                    generated (git-ignored)
```

## Known limitations

* Department labels come from a classifier trained on 4.1K real labels; treat `category` as a
  facet and slicing aid, not ground truth.
* Brand matching is dictionary-based; brands named after common words ("shark", "pop") can match
  generic queries.
* Negation queries from ESCI's "negations" source include titles that merely contain "not"
  ("Gods Not Dead"); the slice keeps ESCI's own definition.
* All behavioural data is simulated. Conclusions that depend on it are labelled as such.

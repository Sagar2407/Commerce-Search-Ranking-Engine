# Commerce Search & Ranking Engine

**Business question:** how can a shopper find the right product when their query is vague, incomplete, or uses
different words from the catalog?

A searchable storefront on Amazon's ESCI benchmark (1.8M products, 130K real shopping queries in English, Spanish
and Japanese, 2.6M human relevance judgments) with keyword retrieval, an in-domain embedding retriever, hybrid
search and a learned reranker. Every approach is evaluated under the same conditions on held-out queries, broken
down by query slice, with latency and serving cost next to quality.

## Results

Test split: 30,969 held-out queries (rerank), 9,000 sampled for full-catalog retrieval; 1.8M products in three
markets. `*` = 95% paired-bootstrap interval excludes zero. Latency: one CPU thread, warm process, top 10.

| Approach | Rerank nDCG@10 | Δ vs BM25 | Retrieval nDCG@10 (condensed) | Δ vs BM25 | Recall@100 (Exact) | p50 / p95 ms | $ per 1M queries |
|---|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.8422 | – | 0.6559 | – | 0.590 | 5.8 / 16.8 | $0.135 |
| Semantic (dense) | 0.8354 | -0.0068 * | 0.5695 | -0.0865 * | 0.465 | 1.1 / 1.6 | $0.024 |
| Hybrid (RRF) | 0.8460 | +0.0039 * | 0.6810 | +0.0251 * | 0.612 | 6.3 / 18.0 | $0.150 |
| Hybrid (score fusion) | 0.8499 | +0.0078 * | 0.6843 | +0.0284 * | 0.618 | 8.2 / 23.9 | $0.197 |
| **Reranker (LambdaMART)** | **0.8631** | **+0.0209 \*** | **0.6951** | **+0.0391 \*** | 0.618 | 11.8 / 29.2 | $0.276 |
| Reranker + feedback ¹ | 0.9912 | +0.1490 * | 0.7635 | +0.1076 * | 0.618 | 14.3 / 32.8 | $0.330 |

¹ Uses clicks simulated from the same relevance labels: an upper bound on what feedback could add, not an estimate.

**Where each approach wins** (rerank nDCG@10; the reranker leads in every slice):

| Slice | queries | BM25 | Dense | Hybrid | Reranker | Reranker gain |
|---|---|---|---|---|---|---|
| ambiguous | 8,270 | 0.8583 | 0.8505 | 0.8656 | 0.8750 | +0.0167 |
| negation ("without …") | 2,143 | 0.6417 | 0.6472 | 0.6522 | 0.6816 | +0.0399 |
| states a spec ("12 oz") | 1,831 | 0.8151 | 0.8014 | 0.8213 | 0.8540 | +0.0389 |
| brand | 4,801 | 0.8534 | 0.8439 | 0.8607 | 0.8786 | +0.0251 |
| thin product descriptions | 4,837 | 0.8223 | 0.8190 | 0.8327 | 0.8426 | +0.0203 |
| hard (ESCI) | 14,496 | 0.7077 | 0.7050 | 0.7200 | 0.7387 | +0.0310 |
| market es / jp / us | 3,844 / 4,667 / 22,458 | 0.785 / 0.800 / 0.861 | 0.790 / 0.784 / 0.854 | 0.798 / 0.806 / 0.868 | 0.816 / 0.819 / 0.880 | +0.031 / +0.018 / +0.020 |

Full tables: [`data/reports/eval_full_test.md`](data/reports/eval_full_test.md).

### What the evaluation says

* **Keyword search is a strong baseline on this data.** ESCI queries were collected *because* they are hard, and
  candidate sets are lexically close to the query, so word overlap carries a lot of signal.
* **Semantic retrieval alone is worse than keyword search, but it finds different products.** Fused with BM25 it
  lifts recall of exact matches from 0.590 to 0.618 and retrieval nDCG by +0.028. In Spanish it beats BM25 on its
  own (0.790 vs 0.785), and on negation queries too.
* **The reranker is where most of the gain is.** It adds +0.021 (rerank) and +0.039 (retrieval) over BM25, largest
  on queries with specs (+0.039), negation (+0.040) and hard queries (+0.031): attribute match / conflict
  features let it demote a "size 10" for a "size 8" query that shares every word. It costs about 2× BM25's
  latency (p50 11.8 vs 5.8 ms) and ~$0.28 vs $0.14 per million queries in compute; at retail scale that is
  negligible next to a +0.04 nDCG change, and `csre bench` shows the quality per millisecond flattens beyond
  ~100 reranked products.
* **Typos are the biggest remaining failure.** One typo cost every method ~0.23 nDCG@10. Catalog-vocabulary
  spelling correction halves that (−0.113 with correction) and also helps real typos in the test queries.
* **Offline gains, translated into shopper terms (simulated).** Replaying 3,000 traffic-weighted test queries
  through the phase-1 click model, the reranker lifts purchases per search by +6% to +9% over BM25 across three
  assumptions about unjudged products. A real pilot would need roughly 38K–59K searches per arm to detect that
  ([docs/pilot.md](docs/pilot.md)). These are simulated numbers built from the same labels: they size the pilot,
  they do not replace it.
* **Feedback helps only queries seen before.** The feedback-aware ranker loses its advantage on query variants it
  has no history for, and its offline gain is simulator-dependent; it is reported separately, never as a
  conversion claim.

## Storefront demo

The storefront searches the catalog with any approach, shows *why* each result ranked where it did (matched
words, attribute matches and conflicts such as "size 8 ✓" or "12 oz ✕ product: 16 oz", semantic similarity,
department fit, the reranker's exact feature contributions, predicted Exact / Substitute / Complement type),
compares all approaches side by side with rank movements and per-query nDCG, and shows the evaluation, latency,
cost, scaling and simulated A/B results. Vague queries get department refinements; misspellings are corrected
with "showing results for"; feedback buttons update the feedback-aware ranker live.

```bash
make serve       # http://localhost:8000, 151K-product demo catalog with product images
make snapshot    # self-contained page with precomputed results (data/reports/storefront_snapshot.html)
```

## Quickstart

```bash
pip install -e ".[dev]"
make data        # phase 1: download ESCI + ESCI-S, build every dataset (~1 h on 4 vCPU / 16 GB)
make search      # phase 2: train encoder, build indexes, train rerankers, evaluate on the test split (~1 h)
make serve       # storefront + API
make bench       # rerank depth, ANN recall / latency, catalog-size scaling
csre simulate-ab # click-model A/B and pilot sample sizes
pytest -q        # 35 tests, no data needed
```

## How it works

| Layer | What | Docs |
|---|---|---|
| Data | ESCI + ESCI-S product pages (real departments, prices, ratings, images for 82% of products), attribute extraction, product graph, simulated traffic / feedback / replay, scale tiers to 25M products | [docs/data_layer.md](docs/data_layer.md) · [docs/phase1.md](docs/phase1.md) |
| Retrieval | Multilingual BM25 (Japanese via character bigrams), two-tower subword encoder trained in-domain with hard negatives, HNSW, RRF / score fusion, catalog-vocabulary spelling correction, filters | [docs/architecture.md](docs/architecture.md) |
| Ranking | LambdaMART over 57 content features (lexical, semantic, attribute match / conflict, negation, department fit, product, commerce) fit on held-out queries; feedback-aware variant; E/S/C/I result-type classifier | [docs/architecture.md](docs/architecture.md) |
| Evaluation | Rerank (judged sets) and full-catalog retrieval with unknown-relevance handling, slices, paired bootstrap CIs, latency, cost, robustness to query variants, systems benchmarks, simulated A/B | [docs/architecture.md](docs/architecture.md#evaluation-protocol) · [docs/pilot.md](docs/pilot.md) |
| Serving | FastAPI, model registry with versions and aliases, result cache keyed by normalised query + model versions, latency-budget degradation (reranker → hybrid → keyword), semantic rescue for zero results, feedback loop, explanations | `src/csre/serve/` |

| Approach | `method` | Idea |
|---|---|---|
| Keyword | `bm25` | BM25 over title, brand, colour, bullets and description |
| Semantic | `dense` | Two-tower encoder over words and character n-grams, trained on ESCI train queries |
| Hybrid (RRF) | `hybrid_rrf` | Reciprocal-rank fusion of the keyword and semantic lists |
| Hybrid (score fusion) | `hybrid` | Convex blend of z-normalised scores, weight tuned on dev |
| Reranker | `ltr` | LambdaMART on content features over the fused candidates |
| Reranker + feedback | `ltr_fb` | Adds debiased click, cart, purchase and thumbs features |

## Choices worth knowing about

* **No pretrained language models.** They were not downloadable from the build environment, so the semantic
  retriever is trained in-domain (numpy, CPU, Hogwild). A `sentence-transformers` backend plugs into the same
  interface (`pip install ".[pretrained]"`, `search.dense.backend`).
* **Stacking without leakage.** The encoder and the query→department model are trained on `train`; the reranker
  that uses their scores is fit on held-out `dev` queries (fitting it on `train` made it over-trust the encoder
  and stop after two trees), and everything is reported on the untouched `test` split.
* **Unjudged is unknown.** Full-catalog metrics are condensed (unjudged results dropped), with a pessimistic bound
  and judged@10 alongside.
* **Simulated is labelled simulated.** Prices and ratings are real (ESCI-S) where available; clicks, carts and
  purchases are always simulated, and conclusions that depend on them say so.

## Repository

```
configs/            data.yaml (phase 1), search.yaml (retrieval, ranking, evaluation, serving, cost model)
src/csre/data/      acquisition, ESCI-S, attributes, taxonomy, catalog, judgments, graph, scale, demo, validation
src/csre/sim/       commerce attributes, click model, traffic, replay
src/csre/search/    analysis, BM25, dense encoder + vector index, query understanding, spelling, features,
                    engine, training, registry
src/csre/evaluation/ metrics, harness, report, benchmarks, simulated A/B
src/csre/serve/     FastAPI app, service (cache, feedback, compare), explanations, storefront, snapshot
data/reports/       data card, evaluation reports, benchmarks, simulated A/B (generated data is not in git)
```

# Phase 1: data layer (done 2026-09-18)

**Business question:** how can a shopper find the right product when their query is vague, incomplete, or
uses different words from the catalog?

Data card (published artifact): https://claude.ai/artifact/NRwJVKpU8Y13TA2RpigCRh.
`make data` rebuilds everything (~40 min on 2 vCPU / 8 GB). The full generated data (~12 GB) is not stored
in git; it is reproducible from the code.

## What exists

* **Real:** Amazon ESCI (1,814,924 products; 2,621,285 judgments; 130,652 queries; us/es/jp), sha256-verified.
  ESCI-S 4.4K-product sample (real category / price / stars).
* **Derived:** cleaned catalog + multilingual attribute extraction (measures, dimensions, pack, size, audience,
  compat, material, colour); department classifier (4.1K real + 58K graph-propagated weak labels; 77% acc, 90% at
  conf >= 0.5 on 70% of products, rest "Unknown"); product graph 17.3M edges (co-exact / substitute / complement);
  781K related-query pairs; category complement matrix.
* **Simulated (clearly flagged):** price / rating / stock calibrated to ESCI-S; 30M sessions / 37.7M searches /
  634M impressions from a PBM click model + cart / purchase funnel, logging policy with 5% RandTop-10 exploration,
  true propensities logged, explicit thumbs feedback; 10M-request 24h replay with 35% one-off query variants
  (typo, drop, reorder, modifier, case) keeping the parent query id; scale tiers 2.5M / 5M / 10M / 25M docs
  (23.2M perturbed distractors with `base_doc_id`).
* **Validation gate:** 23 checks, all pass.

## Decisions (carried forward)

* Splits by query: ESCI test untouched (30,969 q); dev = MD5-hashed 10% of train (9,989 q). Query 79706 appears in
  both ESCI splits -> pinned to test, 3 train rows dropped. The released examples file has 2,621,288 rows (the ESCI
  README says 2,621,738).
* Gains E 1.0 / S 0.1 / C 0.01 / I 0; qrels grades 3/2/1/0.
* Unjudged = unknown, not irrelevant: full-catalog retrieval must report judged@k / condensed lists.
* Slices on `queries.parquet`: underspecified, multi_intent, ambiguous, negation, spec, brand, sparse_products,
  hard, all_exact, length bucket, source, traffic bucket (head / torso / tail).
* Logs cover all splits (tagged `query_split`). Click-feature gains on test are simulator-dependent: report
  content-only vs feedback-aware rankers separately and by traffic bucket.
* Distractors never enter quality metrics.

## Known limitations

* Department = classifier output (facet / slicing aid, not ground truth).
* Brand matching is dictionary-based (ambiguous brand names like "shark" match).
* ESCI "negations" source includes titles containing "not".

# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v4`, `ltr_fb=v3`, `esci_class=v4`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8422 [0.8397, 0.8445] | – | 0.8250 | 0.8822 | 0.8176 |
| Hybrid (score fusion) | 0.8499 [0.8476, 0.8520] | +0.0078 [+0.0070, +0.0086] * | 0.8349 | 0.8914 | 0.8309 |
| Reranker (LambdaMART) | 0.8631 [0.8608, 0.8652] | +0.0209 [+0.0196, +0.0221] * | 0.8508 | 0.9053 | 0.8513 |

## Full-catalog retrieval

3,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6515 [0.6402, 0.6622] | – | 0.3735 | 0.385 | 0.5948 | 0.748 |
| Hybrid (score fusion) | 0.6811 [0.6701, 0.6911] | +0.0296 [+0.0259, +0.0334] * | 0.3876 | 0.395 | 0.6217 | 0.767 |
| Reranker (LambdaMART) | 0.7014 [0.6903, 0.7118] | +0.0499 [+0.0442, +0.0556] * | 0.4067 | 0.396 | 0.6275 | 0.786 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 3,844 | 0.7845 | 0.7975 | **0.8155** |
| locale=jp | 4,667 | 0.8003 | 0.8060 | **0.8187** |
| locale=us | 22,458 | 0.8607 | 0.8680 | **0.8804** |
| traffic_bucket=head | 195 | 0.8615 | 0.8620 | **0.8696** |
| traffic_bucket=tail | 24,949 | 0.8390 | 0.8467 | **0.8605** |
| traffic_bucket=torso | 5,825 | 0.8550 | 0.8634 | **0.8739** |
| length_bucket=long | 3,022 | 0.7963 | 0.8060 | **0.8239** |
| length_bucket=medium | 19,554 | 0.8376 | 0.8453 | **0.8591** |
| length_bucket=short | 8,393 | 0.8692 | 0.8764 | **0.8865** |
| ambiguous | 8,270 | 0.8583 | 0.8656 | **0.8750** |
| underspecified | 6,279 | 0.8674 | 0.8748 | **0.8830** |
| multi_intent | 2,812 | 0.8422 | 0.8486 | **0.8595** |
| negation | 2,143 | 0.6417 | 0.6522 | **0.6816** |
| spec | 1,831 | 0.8151 | 0.8213 | **0.8540** |
| brand | 4,801 | 0.8534 | 0.8607 | **0.8786** |
| sparse_products | 4,837 | 0.8223 | 0.8327 | **0.8426** |
| hard | 14,496 | 0.7077 | 0.7200 | **0.7387** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 1,000 | 0.6460 | 0.6821 | **0.7105** |
| locale=jp | 1,000 | 0.6461 | 0.6688 | **0.6889** |
| locale=us | 1,000 | 0.6623 | 0.6924 | **0.7049** |
| traffic_bucket=head | 20 | 0.6388 | 0.6408 | **0.6726** |
| traffic_bucket=tail | 2,323 | 0.6551 | 0.6836 | **0.7046** |
| traffic_bucket=torso | 657 | 0.6390 | 0.6735 | **0.6912** |
| length_bucket=long | 261 | 0.6265 | 0.6471 | **0.6786** |
| length_bucket=medium | 1,906 | 0.6635 | 0.6934 | **0.7114** |
| length_bucket=short | 833 | 0.6318 | 0.6636 | **0.6857** |
| ambiguous | 806 | 0.6284 | 0.6572 | **0.6779** |
| underspecified | 624 | 0.6223 | 0.6525 | **0.6731** |
| multi_intent | 263 | 0.6471 | 0.6729 | **0.6874** |
| negation | 306 | 0.5716 | 0.5910 | **0.6184** |
| spec | 153 | 0.6518 | 0.6877 | **0.7389** |
| brand | 392 | 0.7257 | 0.7591 | **0.7831** |
| sparse_products | 610 | 0.6367 | 0.6619 | **0.6859** |
| hard | 1,699 | 0.5598 | 0.5854 | **0.6077** |

## Robustness to query variants (replay stream)

3,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6926 | 0.6915 | -0.0011 |
| case_space | Hybrid (score fusion) | 0.7184 | 0.7178 | -0.0006 |
| case_space | Reranker (LambdaMART) | 0.7241 | 0.7213 | -0.0028 |
| modifier | Keyword (BM25) | 0.6699 | 0.6523 | -0.0176 |
| modifier | Hybrid (score fusion) | 0.6983 | 0.6826 | -0.0157 |
| modifier | Reranker (LambdaMART) | 0.7128 | 0.6676 | -0.0452 |
| reorder | Keyword (BM25) | 0.6607 | 0.6607 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6867 | 0.6867 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.7007 | 0.6989 | -0.0018 |
| token_drop | Keyword (BM25) | 0.6694 | 0.5297 | -0.1397 |
| token_drop | Hybrid (score fusion) | 0.6912 | 0.5409 | -0.1503 |
| token_drop | Reranker (LambdaMART) | 0.6992 | 0.5427 | -0.1565 |
| typo | Keyword (BM25) | 0.6682 | 0.5502 | -0.1180 |
| typo | Hybrid (score fusion) | 0.6953 | 0.5792 | -0.1162 |
| typo | Reranker (LambdaMART) | 0.7098 | 0.5830 | -0.1268 |

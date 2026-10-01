# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v3`, `ltr_fb=v2`, `esci_class=v3`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8422 [0.8399, 0.8445] | – | 0.8250 | 0.8822 | 0.8176 |
| Hybrid (score fusion) | 0.8499 [0.8477, 0.8523] | +0.0078 [+0.0069, +0.0085] * | 0.8349 | 0.8914 | 0.8309 |
| Reranker (LambdaMART) | 0.8628 [0.8608, 0.8651] | +0.0207 [+0.0193, +0.0220] * | 0.8503 | 0.9054 | 0.8515 |

## Full-catalog retrieval

3,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6515 [0.6403, 0.6629] | – | 0.3735 | 0.385 | 0.5948 | 0.748 |
| Hybrid (score fusion) | 0.6811 [0.6699, 0.6917] | +0.0296 [+0.0257, +0.0336] * | 0.3876 | 0.395 | 0.6217 | 0.767 |
| Reranker (LambdaMART) | 0.6926 [0.6814, 0.7035] | +0.0412 [+0.0363, +0.0460] * | 0.4053 | 0.396 | 0.6217 | 0.786 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 3,844 | 0.7845 | 0.7975 | **0.8152** |
| locale=jp | 4,667 | 0.8003 | 0.8060 | **0.8182** |
| locale=us | 22,458 | 0.8607 | 0.8680 | **0.8802** |
| traffic_bucket=head | 195 | 0.8615 | 0.8620 | **0.8683** |
| traffic_bucket=tail | 24,949 | 0.8390 | 0.8467 | **0.8601** |
| traffic_bucket=torso | 5,825 | 0.8550 | 0.8634 | **0.8745** |
| length_bucket=long | 3,022 | 0.7963 | 0.8060 | **0.8223** |
| length_bucket=medium | 19,554 | 0.8376 | 0.8453 | **0.8589** |
| length_bucket=short | 8,393 | 0.8692 | 0.8764 | **0.8866** |
| ambiguous | 8,270 | 0.8583 | 0.8656 | **0.8750** |
| underspecified | 6,279 | 0.8674 | 0.8748 | **0.8832** |
| multi_intent | 2,812 | 0.8422 | 0.8486 | **0.8595** |
| negation | 2,143 | 0.6417 | 0.6522 | **0.6779** |
| spec | 1,831 | 0.8151 | 0.8213 | **0.8526** |
| brand | 4,801 | 0.8534 | 0.8607 | **0.8790** |
| sparse_products | 4,837 | 0.8223 | 0.8327 | **0.8428** |
| hard | 14,496 | 0.7077 | 0.7200 | **0.7386** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 1,000 | 0.6460 | 0.6821 | **0.6987** |
| locale=jp | 1,000 | 0.6461 | 0.6688 | **0.6776** |
| locale=us | 1,000 | 0.6623 | 0.6924 | **0.7016** |
| traffic_bucket=head | 20 | 0.6388 | 0.6408 | **0.6626** |
| traffic_bucket=tail | 2,323 | 0.6551 | 0.6836 | **0.6961** |
| traffic_bucket=torso | 657 | 0.6390 | 0.6735 | **0.6814** |
| length_bucket=long | 261 | 0.6265 | 0.6471 | **0.6641** |
| length_bucket=medium | 1,906 | 0.6635 | 0.6934 | **0.7059** |
| length_bucket=short | 833 | 0.6318 | 0.6636 | **0.6712** |
| ambiguous | 806 | 0.6284 | 0.6572 | **0.6642** |
| underspecified | 624 | 0.6223 | 0.6525 | **0.6591** |
| multi_intent | 263 | 0.6471 | 0.6729 | **0.6823** |
| negation | 306 | 0.5716 | 0.5910 | **0.6163** |
| spec | 153 | 0.6518 | 0.6877 | **0.7219** |
| brand | 392 | 0.7257 | 0.7591 | **0.7793** |
| sparse_products | 610 | 0.6367 | 0.6619 | **0.6666** |
| hard | 1,699 | 0.5598 | 0.5854 | **0.6014** |

## Robustness to query variants (replay stream)

3,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6798 | 0.6800 | +0.0002 |
| case_space | Hybrid (score fusion) | 0.7096 | 0.7092 | -0.0003 |
| case_space | Reranker (LambdaMART) | 0.7153 | 0.7154 | +0.0001 |
| modifier | Keyword (BM25) | 0.6541 | 0.6395 | -0.0146 |
| modifier | Hybrid (score fusion) | 0.6811 | 0.6699 | -0.0113 |
| modifier | Reranker (LambdaMART) | 0.6867 | 0.6734 | -0.0133 |
| reorder | Keyword (BM25) | 0.6067 | 0.6067 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6467 | 0.6467 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.6549 | 0.6545 | -0.0004 |
| token_drop | Keyword (BM25) | 0.6777 | 0.5611 | -0.1166 |
| token_drop | Hybrid (score fusion) | 0.7024 | 0.5820 | -0.1204 |
| token_drop | Reranker (LambdaMART) | 0.7116 | 0.5910 | -0.1206 |
| typo | Keyword (BM25) | 0.6559 | 0.5299 | -0.1260 |
| typo | Hybrid (score fusion) | 0.6842 | 0.5635 | -0.1207 |
| typo | Reranker (LambdaMART) | 0.6925 | 0.5690 | -0.1235 |

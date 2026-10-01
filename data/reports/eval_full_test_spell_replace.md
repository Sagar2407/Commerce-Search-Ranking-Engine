# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v3`, `ltr_fb=v2`, `esci_class=v3`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8420 [0.8397, 0.8444] | – | 0.8249 | 0.8819 | 0.8171 |
| Hybrid (score fusion) | 0.8497 [0.8476, 0.8521] | +0.0077 [+0.0069, +0.0085] * | 0.8346 | 0.8911 | 0.8306 |
| Reranker (LambdaMART) | 0.8627 [0.8607, 0.8649] | +0.0207 [+0.0195, +0.0220] * | 0.8502 | 0.9054 | 0.8514 |

## Full-catalog retrieval

3,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6514 [0.6393, 0.6630] | – | 0.3735 | 0.385 | 0.5948 | 0.747 |
| Hybrid (score fusion) | 0.6809 [0.6688, 0.6917] | +0.0296 [+0.0253, +0.0335] * | 0.3875 | 0.395 | 0.6216 | 0.767 |
| Reranker (LambdaMART) | 0.6924 [0.6803, 0.7030] | +0.0410 [+0.0356, +0.0458] * | 0.4054 | 0.396 | 0.6216 | 0.786 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 3,844 | 0.7843 | 0.7973 | **0.8151** |
| locale=jp | 4,667 | 0.8002 | 0.8059 | **0.8183** |
| locale=us | 22,458 | 0.8606 | 0.8678 | **0.8801** |
| traffic_bucket=head | 195 | 0.8615 | 0.8621 | **0.8678** |
| traffic_bucket=tail | 24,949 | 0.8389 | 0.8466 | **0.8600** |
| traffic_bucket=torso | 5,825 | 0.8546 | 0.8630 | **0.8742** |
| length_bucket=long | 3,022 | 0.7962 | 0.8060 | **0.8223** |
| length_bucket=medium | 19,554 | 0.8375 | 0.8452 | **0.8588** |
| length_bucket=short | 8,393 | 0.8689 | 0.8761 | **0.8864** |
| ambiguous | 8,270 | 0.8579 | 0.8652 | **0.8748** |
| underspecified | 6,279 | 0.8670 | 0.8744 | **0.8829** |
| multi_intent | 2,812 | 0.8422 | 0.8484 | **0.8595** |
| negation | 2,143 | 0.6417 | 0.6521 | **0.6780** |
| spec | 1,831 | 0.8151 | 0.8212 | **0.8526** |
| brand | 4,801 | 0.8534 | 0.8607 | **0.8790** |
| sparse_products | 4,837 | 0.8219 | 0.8325 | **0.8424** |
| hard | 14,496 | 0.7077 | 0.7197 | **0.7385** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 1,000 | 0.6457 | 0.6815 | **0.6980** |
| locale=jp | 1,000 | 0.6461 | 0.6688 | **0.6776** |
| locale=us | 1,000 | 0.6623 | 0.6925 | **0.7016** |
| traffic_bucket=head | 20 | 0.6388 | 0.6408 | **0.6626** |
| traffic_bucket=tail | 2,323 | 0.6551 | 0.6835 | **0.6959** |
| traffic_bucket=torso | 657 | 0.6387 | 0.6731 | **0.6809** |
| length_bucket=long | 261 | 0.6265 | 0.6471 | **0.6641** |
| length_bucket=medium | 1,906 | 0.6635 | 0.6933 | **0.7058** |
| length_bucket=short | 833 | 0.6315 | 0.6634 | **0.6707** |
| ambiguous | 806 | 0.6281 | 0.6569 | **0.6637** |
| underspecified | 624 | 0.6219 | 0.6522 | **0.6585** |
| multi_intent | 263 | 0.6471 | 0.6729 | **0.6823** |
| negation | 306 | 0.5716 | 0.5910 | **0.6161** |
| spec | 153 | 0.6518 | 0.6877 | **0.7218** |
| brand | 392 | 0.7257 | 0.7591 | **0.7793** |
| sparse_products | 610 | 0.6363 | 0.6611 | **0.6660** |
| hard | 1,699 | 0.5598 | 0.5853 | **0.6011** |

## Robustness to query variants (replay stream)

3,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6663 | 0.6665 | +0.0002 |
| case_space | Hybrid (score fusion) | 0.6987 | 0.6994 | +0.0007 |
| case_space | Reranker (LambdaMART) | 0.7162 | 0.7160 | -0.0002 |
| modifier | Keyword (BM25) | 0.6361 | 0.6199 | -0.0162 |
| modifier | Hybrid (score fusion) | 0.6693 | 0.6519 | -0.0174 |
| modifier | Reranker (LambdaMART) | 0.6754 | 0.6531 | -0.0223 |
| reorder | Keyword (BM25) | 0.6579 | 0.6579 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6897 | 0.6897 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.6944 | 0.6946 | +0.0002 |
| token_drop | Keyword (BM25) | 0.6869 | 0.5409 | -0.1461 |
| token_drop | Hybrid (score fusion) | 0.7188 | 0.5628 | -0.1560 |
| token_drop | Reranker (LambdaMART) | 0.7225 | 0.5664 | -0.1561 |
| typo | Keyword (BM25) | 0.6669 | 0.5590 | -0.1079 |
| typo | Hybrid (score fusion) | 0.6957 | 0.5875 | -0.1082 |
| typo | Reranker (LambdaMART) | 0.7026 | 0.5934 | -0.1091 |

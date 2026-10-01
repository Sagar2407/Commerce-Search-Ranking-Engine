# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v4`, `ltr_fb=v3`, `esci_class=v4`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8424 [0.8400, 0.8449] | – | 0.8254 | 0.8826 | 0.8180 |
| Hybrid (score fusion) | 0.8501 [0.8477, 0.8524] | +0.0076 [+0.0068, +0.0084] * | 0.8350 | 0.8916 | 0.8313 |
| Reranker (LambdaMART) | 0.8633 [0.8611, 0.8654] | +0.0208 [+0.0195, +0.0221] * | 0.8510 | 0.9056 | 0.8516 |

## Full-catalog retrieval

3,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6518 [0.6407, 0.6638] | – | 0.3737 | 0.385 | 0.5949 | 0.748 |
| Hybrid (score fusion) | 0.6813 [0.6706, 0.6926] | +0.0295 [+0.0255, +0.0338] * | 0.3878 | 0.395 | 0.6218 | 0.768 |
| Reranker (LambdaMART) | 0.7013 [0.6903, 0.7122] | +0.0495 [+0.0434, +0.0553] * | 0.4051 | 0.395 | 0.6274 | 0.785 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 3,844 | 0.7846 | 0.7976 | **0.8157** |
| locale=jp | 4,667 | 0.8003 | 0.8060 | **0.8187** |
| locale=us | 22,458 | 0.8611 | 0.8682 | **0.8807** |
| traffic_bucket=head | 195 | 0.8615 | 0.8627 | **0.8709** |
| traffic_bucket=tail | 24,949 | 0.8393 | 0.8469 | **0.8607** |
| traffic_bucket=torso | 5,825 | 0.8551 | 0.8634 | **0.8741** |
| length_bucket=long | 3,022 | 0.7965 | 0.8062 | **0.8238** |
| length_bucket=medium | 19,554 | 0.8379 | 0.8455 | **0.8592** |
| length_bucket=short | 8,393 | 0.8695 | 0.8766 | **0.8870** |
| ambiguous | 8,270 | 0.8586 | 0.8657 | **0.8754** |
| underspecified | 6,279 | 0.8677 | 0.8750 | **0.8836** |
| multi_intent | 2,812 | 0.8426 | 0.8489 | **0.8599** |
| negation | 2,143 | 0.6418 | 0.6522 | **0.6814** |
| spec | 1,831 | 0.8151 | 0.8213 | **0.8540** |
| brand | 4,801 | 0.8537 | 0.8609 | **0.8785** |
| sparse_products | 4,837 | 0.8224 | 0.8327 | **0.8429** |
| hard | 14,496 | 0.7081 | 0.7202 | **0.7391** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 1,000 | 0.6461 | 0.6823 | **0.7098** |
| locale=jp | 1,000 | 0.6461 | 0.6688 | **0.6889** |
| locale=us | 1,000 | 0.6631 | 0.6929 | **0.7051** |
| traffic_bucket=head | 20 | 0.6388 | 0.6408 | **0.6726** |
| traffic_bucket=tail | 2,323 | 0.6555 | 0.6839 | **0.7043** |
| traffic_bucket=torso | 657 | 0.6390 | 0.6735 | **0.6913** |
| length_bucket=long | 261 | 0.6267 | 0.6472 | **0.6782** |
| length_bucket=medium | 1,906 | 0.6638 | 0.6935 | **0.7113** |
| length_bucket=short | 833 | 0.6322 | 0.6641 | **0.6855** |
| ambiguous | 806 | 0.6288 | 0.6576 | **0.6776** |
| underspecified | 624 | 0.6229 | 0.6528 | **0.6728** |
| multi_intent | 263 | 0.6470 | 0.6734 | **0.6873** |
| negation | 306 | 0.5716 | 0.5909 | **0.6186** |
| spec | 153 | 0.6518 | 0.6877 | **0.7385** |
| brand | 392 | 0.7256 | 0.7594 | **0.7826** |
| sparse_products | 610 | 0.6377 | 0.6625 | **0.6858** |
| hard | 1,699 | 0.5600 | 0.5856 | **0.6076** |

## Robustness to query variants (replay stream)

3,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6929 | 0.6918 | -0.0011 |
| case_space | Hybrid (score fusion) | 0.7188 | 0.7182 | -0.0006 |
| case_space | Reranker (LambdaMART) | 0.7226 | 0.7198 | -0.0028 |
| modifier | Keyword (BM25) | 0.6704 | 0.6528 | -0.0176 |
| modifier | Hybrid (score fusion) | 0.6991 | 0.6835 | -0.0155 |
| modifier | Reranker (LambdaMART) | 0.7126 | 0.6688 | -0.0437 |
| reorder | Keyword (BM25) | 0.6615 | 0.6615 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6874 | 0.6874 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.7007 | 0.6989 | -0.0018 |
| token_drop | Keyword (BM25) | 0.6699 | 0.5305 | -0.1394 |
| token_drop | Hybrid (score fusion) | 0.6933 | 0.5418 | -0.1516 |
| token_drop | Reranker (LambdaMART) | 0.6998 | 0.5441 | -0.1557 |
| typo | Keyword (BM25) | 0.6693 | 0.5665 | -0.1028 |
| typo | Hybrid (score fusion) | 0.6961 | 0.5946 | -0.1016 |
| typo | Reranker (LambdaMART) | 0.7109 | 0.5988 | -0.1122 |

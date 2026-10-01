# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v3`, `ltr_fb=v2`, `esci_class=v3`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8414 [0.8390, 0.8438] | – | 0.8241 | 0.8814 | 0.8163 |
| Hybrid (score fusion) | 0.8491 [0.8469, 0.8514] | +0.0077 [+0.0070, +0.0085] * | 0.8339 | 0.8904 | 0.8295 |
| Reranker (LambdaMART) | 0.8621 [0.8600, 0.8643] | +0.0207 [+0.0195, +0.0220] * | 0.8495 | 0.9048 | 0.8507 |

## Full-catalog retrieval

3,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6491 [0.6365, 0.6600] | – | 0.3714 | 0.382 | 0.5922 | 0.745 |
| Hybrid (score fusion) | 0.6787 [0.6665, 0.6892] | +0.0296 [+0.0254, +0.0334] * | 0.3856 | 0.393 | 0.6193 | 0.766 |
| Reranker (LambdaMART) | 0.6902 [0.6779, 0.7006] | +0.0411 [+0.0363, +0.0457] * | 0.4042 | 0.395 | 0.6193 | 0.785 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 3,844 | 0.7839 | 0.7964 | **0.8143** |
| locale=jp | 4,667 | 0.8001 | 0.8059 | **0.8181** |
| locale=us | 22,458 | 0.8598 | 0.8672 | **0.8795** |
| traffic_bucket=head | 195 | 0.8593 | 0.8604 | **0.8672** |
| traffic_bucket=tail | 24,949 | 0.8384 | 0.8460 | **0.8595** |
| traffic_bucket=torso | 5,825 | 0.8538 | 0.8623 | **0.8734** |
| length_bucket=long | 3,022 | 0.7961 | 0.8058 | **0.8223** |
| length_bucket=medium | 19,554 | 0.8371 | 0.8446 | **0.8582** |
| length_bucket=short | 8,393 | 0.8678 | 0.8752 | **0.8856** |
| ambiguous | 8,270 | 0.8570 | 0.8646 | **0.8742** |
| underspecified | 6,279 | 0.8657 | 0.8734 | **0.8820** |
| multi_intent | 2,812 | 0.8417 | 0.8483 | **0.8594** |
| negation | 2,143 | 0.6418 | 0.6523 | **0.6780** |
| spec | 1,831 | 0.8145 | 0.8206 | **0.8520** |
| brand | 4,801 | 0.8529 | 0.8602 | **0.8786** |
| sparse_products | 4,837 | 0.8212 | 0.8319 | **0.8417** |
| hard | 14,496 | 0.7066 | 0.7188 | **0.7375** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Hybrid (score fusion) | Reranker (LambdaMART) |
|---|---|---|---|---|
| locale=es | 1,000 | 0.6423 | 0.6780 | **0.6941** |
| locale=jp | 1,000 | 0.6461 | 0.6688 | **0.6776** |
| locale=us | 1,000 | 0.6589 | 0.6894 | **0.6989** |
| traffic_bucket=head | 20 | 0.6388 | 0.6408 | **0.6626** |
| traffic_bucket=tail | 2,323 | 0.6521 | 0.6808 | **0.6932** |
| traffic_bucket=torso | 657 | 0.6388 | 0.6727 | **0.6807** |
| length_bucket=long | 261 | 0.6261 | 0.6464 | **0.6630** |
| length_bucket=medium | 1,906 | 0.6617 | 0.6918 | **0.7045** |
| length_bucket=short | 833 | 0.6275 | 0.6589 | **0.6662** |
| ambiguous | 806 | 0.6265 | 0.6549 | **0.6619** |
| underspecified | 624 | 0.6200 | 0.6496 | **0.6561** |
| multi_intent | 263 | 0.6460 | 0.6725 | **0.6818** |
| negation | 306 | 0.5704 | 0.5898 | **0.6149** |
| spec | 153 | 0.6503 | 0.6856 | **0.7209** |
| brand | 392 | 0.7257 | 0.7591 | **0.7793** |
| sparse_products | 610 | 0.6368 | 0.6622 | **0.6670** |
| hard | 1,699 | 0.5578 | 0.5829 | **0.5989** |

## Robustness to query variants (replay stream)

3,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6488 | 0.6521 | +0.0033 |
| case_space | Hybrid (score fusion) | 0.6793 | 0.6799 | +0.0005 |
| case_space | Reranker (LambdaMART) | 0.6841 | 0.6850 | +0.0009 |
| modifier | Keyword (BM25) | 0.6279 | 0.6057 | -0.0222 |
| modifier | Hybrid (score fusion) | 0.6561 | 0.6313 | -0.0248 |
| modifier | Reranker (LambdaMART) | 0.6617 | 0.6364 | -0.0253 |
| reorder | Keyword (BM25) | 0.6563 | 0.6563 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6780 | 0.6780 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.6889 | 0.6891 | +0.0003 |
| token_drop | Keyword (BM25) | 0.6671 | 0.5414 | -0.1258 |
| token_drop | Hybrid (score fusion) | 0.6890 | 0.5638 | -0.1252 |
| token_drop | Reranker (LambdaMART) | 0.6938 | 0.5730 | -0.1208 |
| typo | Keyword (BM25) | 0.6404 | 0.4155 | -0.2249 |
| typo | Hybrid (score fusion) | 0.6667 | 0.4436 | -0.2231 |
| typo | Reranker (LambdaMART) | 0.6739 | 0.4466 | -0.2273 |

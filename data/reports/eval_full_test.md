# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v4`, `ltr_fb=v3`, `esci_class=v4`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8422 [0.8399, 0.8445] | – | 0.8250 | 0.8822 | 0.8176 |
| Semantic (dense) | 0.8354 [0.8330, 0.8375] | -0.0068 [-0.0083, -0.0052] * | 0.8165 | 0.8747 | 0.8066 |
| Hybrid (RRF) | 0.8460 [0.8439, 0.8482] | +0.0039 [+0.0029, +0.0048] * | 0.8300 | 0.8885 | 0.8265 |
| Hybrid (score fusion) | 0.8499 [0.8476, 0.8521] | +0.0078 [+0.0069, +0.0086] * | 0.8349 | 0.8914 | 0.8309 |
| Reranker (LambdaMART) | 0.8631 [0.8611, 0.8652] | +0.0209 [+0.0197, +0.0222] * | 0.8508 | 0.9053 | 0.8513 |
| Reranker + feedback | 0.9912 [0.9907, 0.9917] | +0.1490 [+0.1469, +0.1513] * | 0.9909 | 0.9951 | 0.9920 |

## Full-catalog retrieval

9,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6559 [0.6489, 0.6625] | – | 0.3710 | 0.378 | 0.5901 | 0.744 |
| Semantic (dense) | 0.5695 [0.5620, 0.5768] | -0.0865 [-0.0929, -0.0800] * | 0.2473 | 0.248 | 0.4646 | 0.621 |
| Hybrid (RRF) | 0.6810 [0.6743, 0.6877] | +0.0251 [+0.0225, +0.0278] * | 0.3283 | 0.331 | 0.6119 | 0.736 |
| Hybrid (score fusion) | 0.6843 [0.6774, 0.6910] | +0.0284 [+0.0262, +0.0306] * | 0.3830 | 0.387 | 0.6184 | 0.767 |
| Reranker (LambdaMART) | 0.6951 [0.6882, 0.7016] | +0.0391 [+0.0364, +0.0418] * | 0.4036 | 0.392 | 0.6184 | 0.784 |
| Reranker + feedback | 0.7635 [0.7568, 0.7700] | +0.1076 [+0.1038, +0.1112] * | 0.7529 | 0.714 | 0.6184 | 0.917 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Semantic (dense) | Hybrid (RRF) | Hybrid (score fusion) | Reranker (LambdaMART) | Reranker + feedback |
|---|---|---|---|---|---|---|---|
| locale=es | 3,844 | 0.7845 | 0.7895 | 0.7957 | 0.7975 | 0.8155 | **0.9871** |
| locale=jp | 4,667 | 0.8003 | 0.7839 | 0.8000 | 0.8060 | 0.8187 | **0.9946** |
| locale=us | 22,458 | 0.8607 | 0.8539 | 0.8642 | 0.8680 | 0.8804 | **0.9912** |
| traffic_bucket=head | 195 | 0.8615 | 0.8431 | 0.8634 | 0.8620 | 0.8696 | **0.9973** |
| traffic_bucket=tail | 24,949 | 0.8390 | 0.8325 | 0.8428 | 0.8467 | 0.8605 | **0.9905** |
| traffic_bucket=torso | 5,825 | 0.8550 | 0.8474 | 0.8592 | 0.8634 | 0.8739 | **0.9941** |
| length_bucket=long | 3,022 | 0.7963 | 0.7944 | 0.8037 | 0.8060 | 0.8239 | **0.9853** |
| length_bucket=medium | 19,554 | 0.8376 | 0.8305 | 0.8412 | 0.8453 | 0.8591 | **0.9911** |
| length_bucket=short | 8,393 | 0.8692 | 0.8613 | 0.8726 | 0.8764 | 0.8865 | **0.9936** |
| ambiguous | 8,270 | 0.8583 | 0.8505 | 0.8615 | 0.8656 | 0.8750 | **0.9931** |
| underspecified | 6,279 | 0.8674 | 0.8590 | 0.8703 | 0.8748 | 0.8830 | **0.9933** |
| multi_intent | 2,812 | 0.8422 | 0.8339 | 0.8461 | 0.8486 | 0.8595 | **0.9925** |
| negation | 2,143 | 0.6417 | 0.6472 | 0.6529 | 0.6522 | 0.6816 | **0.9871** |
| spec | 1,831 | 0.8151 | 0.8014 | 0.8163 | 0.8213 | 0.8540 | **0.9889** |
| brand | 4,801 | 0.8534 | 0.8439 | 0.8566 | 0.8607 | 0.8786 | **0.9921** |
| sparse_products | 4,837 | 0.8223 | 0.8190 | 0.8282 | 0.8327 | 0.8426 | **0.9899** |
| hard | 14,496 | 0.7077 | 0.7050 | 0.7160 | 0.7200 | 0.7387 | **0.9842** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Semantic (dense) | Hybrid (RRF) | Hybrid (score fusion) | Reranker (LambdaMART) | Reranker + feedback |
|---|---|---|---|---|---|---|---|
| locale=es | 3,000 | 0.6501 | 0.6037 | 0.6868 | 0.6838 | 0.6966 | **0.7791** |
| locale=jp | 3,000 | 0.6582 | 0.5300 | 0.6707 | 0.6772 | 0.6878 | **0.7673** |
| locale=us | 3,000 | 0.6595 | 0.5747 | 0.6855 | 0.6919 | 0.7008 | **0.7440** |
| traffic_bucket=head | 63 | 0.5513 | 0.3389 | 0.5672 | 0.5739 | 0.5837 | **0.6561** |
| traffic_bucket=tail | 6,943 | 0.6605 | 0.5872 | 0.6849 | 0.6881 | 0.6996 | **0.7708** |
| traffic_bucket=torso | 1,994 | 0.6434 | 0.5151 | 0.6710 | 0.6745 | 0.6827 | **0.7413** |
| length_bucket=long | 769 | 0.6487 | 0.6042 | 0.6689 | 0.6691 | 0.6859 | **0.7819** |
| length_bucket=medium | 5,673 | 0.6679 | 0.6000 | 0.6914 | 0.6965 | 0.7074 | **0.7810** |
| length_bucket=short | 2,558 | 0.6316 | 0.4913 | 0.6616 | 0.6618 | 0.6704 | **0.7192** |
| ambiguous | 2,494 | 0.6273 | 0.4939 | 0.6543 | 0.6541 | 0.6623 | **0.7156** |
| underspecified | 1,959 | 0.6197 | 0.4771 | 0.6480 | 0.6479 | 0.6542 | **0.7011** |
| multi_intent | 784 | 0.6487 | 0.5210 | 0.6676 | 0.6710 | 0.6825 | **0.7555** |
| negation | 940 | 0.5729 | 0.5342 | 0.5911 | 0.5938 | 0.6139 | **0.7540** |
| spec | 441 | 0.6664 | 0.6357 | 0.6953 | 0.7009 | 0.7263 | **0.8014** |
| brand | 1,175 | 0.7491 | 0.6558 | 0.7714 | 0.7750 | 0.7943 | **0.8756** |
| sparse_products | 1,746 | 0.6510 | 0.5369 | 0.6760 | 0.6764 | 0.6820 | **0.7479** |
| hard | 5,101 | 0.5632 | 0.4818 | 0.5841 | 0.5885 | 0.6031 | **0.7140** |

## Latency and serving cost

500 queries, single-threaded, warm process, k = 10. Cost = compute only on `c7i.xlarge` (4 vCPU, $0.1785/h) at 60% utilisation; an assumption to edit in `configs/search.yaml`.

| Method | p50 ms | p95 ms | p99 ms | QPS / core | $ / 1M queries | stage means (ms) |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 5.8 | 16.8 | 21.5 | 153 | $0.135 | bm25 6.4 |
| Semantic (dense) | 1.1 | 1.6 | 1.8 | 876 | $0.024 | encode 0.2, dense 0.8 |
| Hybrid (RRF) | 6.3 | 18.0 | 22.9 | 138 | $0.150 | bm25 6.1, encode 0.2, dense 0.7, fuse 0.1 |
| Hybrid (score fusion) | 8.2 | 23.9 | 31.8 | 105 | $0.197 | bm25 6.1, encode 0.2, dense 0.6, fuse 2.4 |
| Reranker (LambdaMART) | 11.8 | 29.2 | 36.9 | 75 | $0.276 | bm25 6.0, encode 0.2, dense 0.7, fuse 2.3, parse 1.5, features 1.1, rerank 1.3 |
| Reranker + feedback | 14.3 | 32.8 | 41.2 | 63 | $0.330 | bm25 6.1, encode 0.2, dense 0.7, fuse 2.4, parse 1.5, features 1.3, rerank 3.4 |

## Robustness to query variants (replay stream)

5,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6694 | 0.6634 | -0.0060 |
| case_space | Semantic (dense) | 0.5653 | 0.5642 | -0.0010 |
| case_space | Hybrid (score fusion) | 0.6978 | 0.6918 | -0.0060 |
| case_space | Hybrid (RRF) | 0.6899 | 0.6829 | -0.0070 |
| case_space | Reranker (LambdaMART) | 0.7038 | 0.6967 | -0.0071 |
| case_space | Reranker + feedback | 0.7552 | 0.7419 | -0.0133 |
| modifier | Keyword (BM25) | 0.6714 | 0.6517 | -0.0197 |
| modifier | Semantic (dense) | 0.5463 | 0.5040 | -0.0423 |
| modifier | Hybrid (score fusion) | 0.6979 | 0.6808 | -0.0172 |
| modifier | Hybrid (RRF) | 0.6916 | 0.6736 | -0.0179 |
| modifier | Reranker (LambdaMART) | 0.7066 | 0.6862 | -0.0204 |
| modifier | Reranker + feedback | 0.7529 | 0.6813 | -0.0716 |
| reorder | Keyword (BM25) | 0.6650 | 0.6650 | +0.0000 |
| reorder | Semantic (dense) | 0.5784 | 0.5784 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6905 | 0.6905 | +0.0000 |
| reorder | Hybrid (RRF) | 0.6851 | 0.6851 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.7001 | 0.6994 | -0.0007 |
| reorder | Reranker + feedback | 0.7557 | 0.7555 | -0.0002 |
| token_drop | Keyword (BM25) | 0.6798 | 0.5381 | -0.1417 |
| token_drop | Semantic (dense) | 0.5828 | 0.4081 | -0.1747 |
| token_drop | Hybrid (score fusion) | 0.7029 | 0.5531 | -0.1498 |
| token_drop | Hybrid (RRF) | 0.6963 | 0.5454 | -0.1509 |
| token_drop | Reranker (LambdaMART) | 0.7079 | 0.5548 | -0.1531 |
| token_drop | Reranker + feedback | 0.7671 | 0.5554 | -0.2118 |
| typo | Keyword (BM25) | 0.6546 | 0.5418 | -0.1128 |
| typo | Semantic (dense) | 0.5363 | 0.4554 | -0.0809 |
| typo | Hybrid (score fusion) | 0.6810 | 0.5703 | -0.1107 |
| typo | Hybrid (RRF) | 0.6779 | 0.5631 | -0.1148 |
| typo | Reranker (LambdaMART) | 0.6898 | 0.5762 | -0.1136 |
| typo | Reranker + feedback | 0.7404 | 0.5722 | -0.1682 |

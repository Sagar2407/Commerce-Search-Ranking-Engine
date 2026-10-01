# Evaluation — corpus `full`, split `test`

30,969 judged queries · k = 10 · model versions: `dense_encoder=v2`, `query_category=v1`, `ltr=v4`, `ltr_fb=v3`, `esci_class=v4`, `hybrid=v1`

`*` = 95% paired-bootstrap interval excludes 0.

## Ranking quality on judged candidates (ESCI task 1)

Every product a query was judged on is ranked; all results are labelled, so nDCG is exact.

| Method | nDCG@10 | Δ vs BM25 | nDCG@5 | MRR (first Exact) | P@1 Exact |
|---|---|---|---|---|---|
| Keyword (BM25) | 0.8424 [0.8400, 0.8447] | – | 0.8254 | 0.8826 | 0.8180 |
| Semantic (dense) | 0.8355 [0.8332, 0.8377] | -0.0070 [-0.0084, -0.0054] * | 0.8168 | 0.8751 | 0.8074 |
| Hybrid (RRF) | 0.8462 [0.8440, 0.8484] | +0.0038 [+0.0029, +0.0047] * | 0.8303 | 0.8888 | 0.8271 |
| Hybrid (score fusion) | 0.8501 [0.8478, 0.8522] | +0.0076 [+0.0069, +0.0084] * | 0.8350 | 0.8916 | 0.8313 |
| Reranker (LambdaMART) | 0.8633 [0.8612, 0.8654] | +0.0208 [+0.0196, +0.0222] * | 0.8510 | 0.9056 | 0.8516 |
| Reranker + feedback | 0.9885 [0.9879, 0.9892] | +0.1461 [+0.1439, +0.1484] * | 0.9878 | 0.9932 | 0.9891 |

## Full-catalog retrieval

9,000 sampled queries. Unjudged results are *unknown*: condensed nDCG drops them, the lower bound counts them as irrelevant, judged@10 shows how much of the page could be assessed.

| Method | condensed nDCG@10 | Δ vs BM25 | nDCG@10 lower bound | judged@10 | Recall@100 (Exact) | Success@10 |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 0.6564 [0.6496, 0.6629] | – | 0.3713 | 0.378 | 0.5904 | 0.745 |
| Semantic (dense) | 0.5691 [0.5620, 0.5766] | -0.0873 [-0.0935, -0.0810] * | 0.2470 | 0.248 | 0.4642 | 0.620 |
| Hybrid (RRF) | 0.6814 [0.6749, 0.6873] | +0.0250 [+0.0221, +0.0278] * | 0.3277 | 0.331 | 0.6123 | 0.735 |
| Hybrid (score fusion) | 0.6847 [0.6782, 0.6910] | +0.0283 [+0.0261, +0.0304] * | 0.3833 | 0.387 | 0.6187 | 0.768 |
| Reranker (LambdaMART) | 0.7017 [0.6950, 0.7079] | +0.0453 [+0.0423, +0.0484] * | 0.4019 | 0.390 | 0.6233 | 0.783 |
| Reranker + feedback | 0.8037 [0.7975, 0.8094] | +0.1473 [+0.1432, +0.1513] * | 0.7896 | 0.757 | 0.6680 | 0.931 |

### nDCG@10 by slice (rerank)

| Slice | queries | Keyword (BM25) | Semantic (dense) | Hybrid (RRF) | Hybrid (score fusion) | Reranker (LambdaMART) | Reranker + feedback |
|---|---|---|---|---|---|---|---|
| locale=es | 3,844 | 0.7846 | 0.7897 | 0.7960 | 0.7976 | 0.8157 | **0.9836** |
| locale=jp | 4,667 | 0.8003 | 0.7838 | 0.8000 | 0.8060 | 0.8187 | **0.9946** |
| locale=us | 22,458 | 0.8611 | 0.8540 | 0.8644 | 0.8682 | 0.8807 | **0.9881** |
| traffic_bucket=head | 195 | 0.8615 | 0.8443 | 0.8640 | 0.8627 | 0.8709 | **0.9967** |
| traffic_bucket=tail | 24,949 | 0.8393 | 0.8326 | 0.8431 | 0.8469 | 0.8607 | **0.9878** |
| traffic_bucket=torso | 5,825 | 0.8551 | 0.8472 | 0.8591 | 0.8634 | 0.8741 | **0.9915** |
| length_bucket=long | 3,022 | 0.7965 | 0.7945 | 0.8039 | 0.8062 | 0.8238 | **0.9827** |
| length_bucket=medium | 19,554 | 0.8379 | 0.8307 | 0.8413 | 0.8455 | 0.8592 | **0.9884** |
| length_bucket=short | 8,393 | 0.8695 | 0.8613 | 0.8728 | 0.8766 | 0.8870 | **0.9909** |
| ambiguous | 8,270 | 0.8586 | 0.8505 | 0.8617 | 0.8657 | 0.8754 | **0.9906** |
| underspecified | 6,279 | 0.8677 | 0.8589 | 0.8704 | 0.8750 | 0.8836 | **0.9906** |
| multi_intent | 2,812 | 0.8426 | 0.8343 | 0.8467 | 0.8489 | 0.8599 | **0.9909** |
| negation | 2,143 | 0.6418 | 0.6472 | 0.6528 | 0.6522 | 0.6814 | **0.9854** |
| spec | 1,831 | 0.8151 | 0.8017 | 0.8164 | 0.8213 | 0.8540 | **0.9873** |
| brand | 4,801 | 0.8537 | 0.8441 | 0.8569 | 0.8609 | 0.8785 | **0.9901** |
| sparse_products | 4,837 | 0.8224 | 0.8189 | 0.8281 | 0.8327 | 0.8429 | **0.9855** |
| hard | 14,496 | 0.7081 | 0.7052 | 0.7163 | 0.7202 | 0.7391 | **0.9793** |

### condensed nDCG@10 by slice (retrieval)

| Slice | queries | Keyword (BM25) | Semantic (dense) | Hybrid (RRF) | Hybrid (score fusion) | Reranker (LambdaMART) | Reranker + feedback |
|---|---|---|---|---|---|---|---|
| locale=es | 3,000 | 0.6505 | 0.6028 | 0.6871 | 0.6843 | 0.7071 | **0.8204** |
| locale=jp | 3,000 | 0.6582 | 0.5300 | 0.6707 | 0.6772 | 0.6923 | **0.8033** |
| locale=us | 3,000 | 0.6605 | 0.5746 | 0.6863 | 0.6926 | 0.7058 | **0.7875** |
| traffic_bucket=head | 63 | 0.5513 | 0.3389 | 0.5672 | 0.5739 | 0.5785 | **0.6879** |
| traffic_bucket=tail | 6,943 | 0.6609 | 0.5866 | 0.6853 | 0.6884 | 0.7063 | **0.8116** |
| traffic_bucket=torso | 1,994 | 0.6441 | 0.5158 | 0.6715 | 0.6754 | 0.6899 | **0.7799** |
| length_bucket=long | 769 | 0.6487 | 0.6042 | 0.6693 | 0.6691 | 0.6901 | **0.8205** |
| length_bucket=medium | 5,673 | 0.6683 | 0.5995 | 0.6917 | 0.6969 | 0.7130 | **0.8209** |
| length_bucket=short | 2,558 | 0.6322 | 0.4912 | 0.6621 | 0.6625 | 0.6802 | **0.7606** |
| ambiguous | 2,494 | 0.6280 | 0.4938 | 0.6550 | 0.6550 | 0.6694 | **0.7545** |
| underspecified | 1,959 | 0.6205 | 0.4772 | 0.6488 | 0.6488 | 0.6623 | **0.7421** |
| multi_intent | 784 | 0.6491 | 0.5205 | 0.6680 | 0.6716 | 0.6833 | **0.7871** |
| negation | 940 | 0.5730 | 0.5337 | 0.5914 | 0.5938 | 0.6149 | **0.7987** |
| spec | 441 | 0.6672 | 0.6357 | 0.6963 | 0.7015 | 0.7425 | **0.8483** |
| brand | 1,175 | 0.7490 | 0.6551 | 0.7714 | 0.7751 | 0.7946 | **0.8975** |
| sparse_products | 1,746 | 0.6513 | 0.5357 | 0.6764 | 0.6767 | 0.6925 | **0.7745** |
| hard | 5,101 | 0.5636 | 0.4814 | 0.5845 | 0.5889 | 0.6078 | **0.7531** |

## Latency and serving cost

500 queries, single-threaded, warm process, k = 10. Cost = compute only on `c7i.xlarge` (4 vCPU, $0.1785/h) at 60% utilisation; an assumption to edit in `configs/search.yaml`.

| Method | p50 ms | p95 ms | p99 ms | QPS / core | $ / 1M queries | stage means (ms) |
|---|---|---|---|---|---|---|
| Keyword (BM25) | 6.1 | 18.2 | 24.5 | 142 | $0.146 | bm25 6.9, spell 0.0 |
| Semantic (dense) | 1.1 | 1.6 | 2.1 | 836 | $0.025 | encode 0.3, dense 0.8, spell 0.0 |
| Hybrid (RRF) | 6.5 | 18.4 | 26.3 | 129 | $0.160 | bm25 6.6, encode 0.2, dense 0.7, fuse 0.1, spell 0.0 |
| Hybrid (score fusion) | 8.6 | 23.2 | 34.0 | 100 | $0.207 | bm25 6.4, encode 0.2, dense 0.7, fuse 2.6, spell 0.0 |
| Reranker (LambdaMART) | 11.6 | 28.1 | 35.7 | 75 | $0.275 | bm25 6.2, encode 0.2, dense 0.7, fuse 2.4, parse 0.3, features 1.3, rerank 1.9, spell 0.0 |
| Reranker + feedback | 15.5 | 33.1 | 41.1 | 58 | $0.357 | bm25 6.5, encode 0.2, dense 0.7, fuse 2.5, parse 0.3, features 1.4, rerank 5.2, spell 0.0 |

## Robustness to query variants (replay stream)

5,000 one-off variants of test queries, scored against the parent query's labels.

| Variant | Method | original nDCG@10 | variant nDCG@10 | Δ |
|---|---|---|---|---|
| case_space | Keyword (BM25) | 0.6702 | 0.6642 | -0.0060 |
| case_space | Semantic (dense) | 0.5670 | 0.5660 | -0.0010 |
| case_space | Hybrid (score fusion) | 0.6982 | 0.6922 | -0.0060 |
| case_space | Hybrid (RRF) | 0.6908 | 0.6839 | -0.0070 |
| case_space | Reranker (LambdaMART) | 0.7095 | 0.7028 | -0.0067 |
| case_space | Reranker + feedback | 0.7931 | 0.7772 | -0.0159 |
| modifier | Keyword (BM25) | 0.6719 | 0.6520 | -0.0199 |
| modifier | Semantic (dense) | 0.5463 | 0.5044 | -0.0419 |
| modifier | Hybrid (score fusion) | 0.6985 | 0.6816 | -0.0169 |
| modifier | Hybrid (RRF) | 0.6923 | 0.6743 | -0.0179 |
| modifier | Reranker (LambdaMART) | 0.7089 | 0.6652 | -0.0437 |
| modifier | Reranker + feedback | 0.7903 | 0.6533 | -0.1370 |
| reorder | Keyword (BM25) | 0.6663 | 0.6663 | +0.0000 |
| reorder | Semantic (dense) | 0.5770 | 0.5770 | +0.0000 |
| reorder | Hybrid (score fusion) | 0.6917 | 0.6917 | +0.0000 |
| reorder | Hybrid (RRF) | 0.6876 | 0.6876 | +0.0000 |
| reorder | Reranker (LambdaMART) | 0.6971 | 0.6961 | -0.0010 |
| reorder | Reranker + feedback | 0.7946 | 0.7944 | -0.0002 |
| token_drop | Keyword (BM25) | 0.6801 | 0.5385 | -0.1415 |
| token_drop | Semantic (dense) | 0.5796 | 0.4094 | -0.1702 |
| token_drop | Hybrid (score fusion) | 0.7038 | 0.5535 | -0.1502 |
| token_drop | Hybrid (RRF) | 0.6983 | 0.5464 | -0.1519 |
| token_drop | Reranker (LambdaMART) | 0.7086 | 0.5531 | -0.1555 |
| token_drop | Reranker + feedback | 0.8082 | 0.5426 | -0.2657 |
| typo | Keyword (BM25) | 0.6558 | 0.5562 | -0.0996 |
| typo | Semantic (dense) | 0.5377 | 0.4707 | -0.0670 |
| typo | Hybrid (score fusion) | 0.6818 | 0.5838 | -0.0980 |
| typo | Hybrid (RRF) | 0.6794 | 0.5771 | -0.1023 |
| typo | Reranker (LambdaMART) | 0.6978 | 0.5893 | -0.1085 |
| typo | Reranker + feedback | 0.7815 | 0.5798 | -0.2017 |

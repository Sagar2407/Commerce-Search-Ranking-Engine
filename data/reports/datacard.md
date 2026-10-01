# Data card — Commerce Search & Ranking Engine

## Catalog
| locale | products | sparse_share | title_only_share | brand_share | has_measure | has_pack_count | has_audience | has_material | has_color | known_category_share | median_title_chars | median_desc_bullet_chars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| us | 1215854 | 0.1928 | 0.1054 | 0.9373 | 0.3217 | 0.1923 | 0.27 | 0.4808 | 0.6034 | 0.7571 | 96.0 | 737.0 |
| jp | 339059 | 0.3042 | 0.1783 | 0.8111 | 0.2803 | 0.156 | 0.1416 | 0.3246 | 0.4229 | 0.6117 | 53.0 | 170.0 |
| es | 260011 | 0.2141 | 0.1063 | 0.9506 | 0.3492 | 0.1194 | 0.23 | 0.453 | 0.5939 | 0.6104 | 113.0 | 802.0 |


Category model: CV accuracy 0.769, macro-F1 0.670; at confidence >= 0.5: accuracy 0.902 on 70.4% of products (rest = Unknown). 4109 real labels + 57973 propagated weak labels.


## Judgments
| split | locale | queries | judgments | avg_candidates | E | S | C | I |
|---|---|---|---|---|---|---|---|---|
| dev | es | 1175 | 27499 | 23.4 | 0.5762 | 0.2427 | 0.0573 | 0.1237 |
| dev | jp | 1327 | 32049 | 24.2 | 0.5959 | 0.2565 | 0.028 | 0.1196 |
| dev | us | 7487 | 138935 | 18.6 | 0.7008 | 0.1955 | 0.0198 | 0.0839 |
| test | es | 3844 | 93347 | 24.3 | 0.5319 | 0.273 | 0.0586 | 0.1364 |
| test | jp | 4667 | 118907 | 25.5 | 0.5581 | 0.276 | 0.0375 | 0.1284 |
| test | us | 22458 | 425762 | 19.0 | 0.6514 | 0.2233 | 0.0257 | 0.0997 |
| train | es | 10161 | 235564 | 23.2 | 0.5797 | 0.2415 | 0.0571 | 0.1216 |
| train | jp | 12133 | 295097 | 24.3 | 0.5865 | 0.2538 | 0.033 | 0.1267 |
| train | us | 67400 | 1254125 | 18.6 | 0.696 | 0.197 | 0.021 | 0.086 |


## Query slices (share of queries)
| locale | queries | ambiguous | underspecified | multi_intent | negation | spec | brand | sparse_products | hard | all_exact |
|---|---|---|---|---|---|---|---|---|---|---|
| us | 97345 | 0.2622 | 0.1971 | 0.0919 | 0.037 | 0.0642 | 0.1753 | 0.1287 | 0.3066 | 0.2729 |
| jp | 18127 | 0.309 | 0.2683 | 0.0606 | 0.139 | 0.0254 | 0.0434 | 0.2589 | 0.5741 | 0.1684 |
| es | 15180 | 0.2657 | 0.2231 | 0.0617 | 0.1062 | 0.0553 | 0.1775 | 0.1756 | 0.5302 | 0.1887 |


## Product graph
| relation | edges | avg_support | max_support |
|---|---|---|---|
| co_exact | 11082493 | 1.079 | 29 |
| substitute | 5458582 | 1.022 | 12 |
| complement | 758695 | 1.013 | 8 |


## Simulated commerce attributes
| locale | currency | price_p10 | price_median | price_p90 | avg_stars | median_ratings | in_stock_share | real_rows |
|---|---|---|---|---|---|---|---|---|
| es | EUR | 9.99 | 32.99 | 114.99 | 4.21 | 43.0 | 0.9396 | 576 |
| jp | JPY | 820.0 | 2500.0 | 7990.0 | 4.12 | 60.0 | 0.9404 | 710 |
| us | USD | 9.99 | 29.99 | 96.99 | 4.4 | 362.0 | 0.9399 | 2841 |


## Simulated traffic
| searches | sessions | users | queries_with_traffic | first_day | last_day | abandonment_rate | reformulation_share | search_ctr | search_add_to_cart_rate | search_conversion_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 37709580 | 30000000 | 3463744 | 130652 | 2026-05-31 | 2026-07-31 | 0.3224 | 0.2044 | 0.6776 | 0.2286 | 0.1424 |

| impressions | ctr | explicit_feedback |
|---|---|---|
| 634358747 | 0.0629 | 645093 |


| traffic_bucket | queries | traffic_share | test_queries |
|---|---|---|---|
| torso | 25435 | 0.3999 | 5825 |
| head | 778 | 0.3001 | 195 |
| tail | 104439 | 0.3 | 24949 |


## Replay stream
| requests | unique_queries | ideal_cache_hit_rate | start | end | variants | variant_examples |
|---|---|---|---|---|---|---|
| 10000000 | 130652 | 0.9869 | 2026-07-31 15:00:00.039000 | 2026-08-02 05:59:59.979000 | [{'variant': 'original', 'requests': 6499953}, {'variant': 'typo', 'requests': 1570692}, {'variant': 'modifier', 'requests': 699051}, {'variant': 'case_space', 'requests': 525395}, {'variant': 'token_drop', 'requests': 469599}, {'variant': 'reorder', 'requests': 235310}] | [{'variant': 'token_drop', 'original': 'long dress sequin', 'typed': 'long dress'}, {'variant': 'case_space', 'original': 'half gallon water bottle', 'typed': 'HALF GALLON WATER BOTTLE'}, {'variant': 'typo', 'original': 'girls underwear size 10', 'typed': 'girls underwear ssize 10'}, {'variant': 'modifier', 'original': 'brooks glycerin 15 womens', 'typed': 'large brooks glycerin 15 womens'}, {'variant': 'reorder', 'original': 'peel and stick wall', 'typed': 'peel and wall stick'}] |


## Scale tiers
| total_docs | real_docs | synthetic_docs | synthetic_shards |
|---|---|---|---|
| 2500000 | 1814924 | 685076 | 1 |
| 5000000 | 1814924 | 3185076 | 4 |
| 10000000 | 1814924 | 8185076 | 9 |
| 25000000 | 1814924 | 23185076 | 24 |


## Footprint
| raw | processed | synthetic | scale |
|---|---|---|---|
| 1.17 GB | 1.51 GB | 5.39 GB | 4.22 GB |


## Validation
23 checks, 0 failed.

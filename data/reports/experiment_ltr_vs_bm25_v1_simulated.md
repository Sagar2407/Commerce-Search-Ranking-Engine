# Experiment `ltr_vs_bm25_v1_simulated`

40,000 users, 197,488 events. Sample-ratio check p = 0.219 (ok).

| Arm | method | users | searches | search success | CTR | purchases / search | zero results | p95 ms |
|---|---|---|---|---|---|---|---|---|
| control | bm25 | 19,877 | 60,642 | 0.0787 | 0.3740 | 0.0474 | 0.0062 | 12.3 |
| treatment | ltr | 20,123 | 61,268 | 0.0833 | 0.3946 | 0.0502 | 0.0000 | 20.0 |

## treatment vs control

| Metric | delta | 95% CI | relative | significant |
|---|---|---|---|---|
| search success | +0.0045 | [+0.0013, +0.0075] | +5.7% | yes |
| ctr | +0.0206 | [+0.0152, +0.0261] | +5.5% | yes |
| purchases per search | +0.0028 | [+0.0003, +0.0054] | +5.9% | yes |

Guardrail p95 latency: 20.0 ms vs budget 150 ms (ok).

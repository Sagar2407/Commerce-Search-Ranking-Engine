# From offline relevance to a live pilot

Offline evaluation (`csre eval`) measures whether a ranking puts products that annotators judged relevant
higher on the page. It cannot measure whether shoppers find what they want faster, buy more, or come back.
This page describes how a pilot would measure that, and what the simulator does and does not tell us before one.

## What the offline numbers support

* Which approach orders *judged* products better, by query slice, with confidence intervals.
* How much retrieval quality can be assessed at all (judged@10): unjudged products are unknown.
* What each approach costs in latency and compute.

They do not support claims about conversion, revenue, or long-term engagement.

## What the simulated A/B adds (`csre simulate-ab`)

The phase-1 click model replays a traffic-weighted sample of test queries against each approach's top 10 and
simulates examination, clicks, add-to-cart and purchase. Because behaviour is generated *from the same ESCI
labels*, it only re-expresses offline relevance in behavioural units; it cannot reveal effects the labels do
not contain (presentation, price sensitivity beyond the model, novelty). Unjudged products have unknown labels,
so every result is reported under three explicit assumptions (unjudged = Irrelevant / Complement / Substitute).

Its useful output is the *order of magnitude* of a plausible effect, which sizes the pilot.

Results: `data/reports/simulated_ab.json` (rendered in the storefront's Evaluation tab).

## Pilot design

| Item | Choice |
|---|---|
| Unit of randomisation | Shopper (cookie / account), not search, so one person sees one ranking |
| Arms | Control: production ranking. Treatment: the candidate approach (e.g. `ltr`), same candidate generation |
| Primary metric | Search success: share of searches followed by an add-to-cart or purchase of a returned product within the session |
| Secondary metrics | CTR@10, purchase rate per search, time to first click, reformulation rate, zero-result rate |
| Guardrails | p95 latency (budget from `serve.latency_budget_ms`), fallback rate, revenue per search, refund / return rate |
| Slices | Same as offline: locale, vague / spec / negation / brand queries, head vs tail traffic |
| Duration | At least two full weeks (weekly seasonality), and long enough to reach the sample size below |
| Analysis | Two-proportion tests per metric, CUPED with pre-period behaviour, sequential monitoring with alpha spending |

## What the simulation predicts (current models)

Replaying 3,000 traffic-weighted test queries through the click model with 100 simulated shoppers each (the same
shoppers for every approach), the content-only reranker lifts purchases per search over BM25 by:

| Assumption for unjudged products | Lift | 95% CI | Searches per arm for the pilot |
|---|---|---|---|
| Irrelevant (pessimistic) | +8.3% | +5.8% to +10.7% | ~45,000 |
| Complement (neutral) | +6.2% | +4.3% to +8.1% | ~66,000 |
| Substitute (optimistic) | +5.2% | +3.7% to +6.9% | ~80,000 |

Dense retrieval alone *loses* 20–39% in the same simulation (it surfaces more unjudged, often off-target products),
which is why it is only used inside the hybrid and the reranker.

## Sample size

For a two-sided test at alpha = 0.05 with 80% power, the searches needed per arm depend on the baseline
purchase-per-search rate and the relative lift to detect (`online_sim.z_power_n`). The simulated A/B
provides both for each approach; the table is generated into `data/reports/simulated_ab.json`
under `pilot_searches_per_arm_for_purchase`.

## Instrumentation already in place

* `POST /api/feedback` logs impressions, clicks, carts, purchases and thumbs with position, method and model
  versions (`data/feedback/events-*.jsonl`).
* Every response carries the model versions and whether a fallback served it, so pilot traffic can be split
  by what the shopper actually saw.
* Position-based propensities are logged with clicks, so the feedback-aware ranker learns from debiased
  (IPS-weighted) clicks rather than from position.

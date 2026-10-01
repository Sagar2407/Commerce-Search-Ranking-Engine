"""User-behaviour simulator (position-based click model + cart/purchase funnel + explicit feedback).

Used twice:
1. to generate the historical interaction logs of a *logging policy* (stage `sim_traffic`), and
2. later, as an online-evaluation stand-in: re-rank the same traffic with a candidate ranker and
   simulate what users would do, to contrast offline nDCG with simulated conversion.

It is deliberately simple and fully specified by config, so conclusions drawn from it can be
stress-tested by changing its parameters. Simulated conversion is NOT real conversion impact.

Model
-----
P(examine | position k)     = (1 / k) ** eta
P(attractive | doc, query)  = base[label] * exp(rating_effect * (stars - 4.3)) * exp(-price_effect * price_z)
P(click)                    = P(examine) * P(attractive)
P(cart | click)             = p_cart[label];   P(purchase | cart) = p_buy[label]
Explicit feedback (thumbs up/down) on a small share of examined results, agreeing with the
graded label (E/S -> up, C/I -> down) with probability `explicit_feedback_accuracy`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LABELS = ["E", "S", "C", "I"]
LABEL_CODE = {l: i for i, l in enumerate(LABELS)}


@dataclass
class ClickModelParams:
    eta: float = 1.0
    attractiveness: dict = field(default_factory=lambda: {"E": 0.35, "S": 0.15, "C": 0.07, "I": 0.02})
    rating_effect: float = 0.30
    price_effect: float = 0.15
    p_cart_given_click: dict = field(default_factory=lambda: {"E": 0.25, "S": 0.15, "C": 0.18, "I": 0.02})
    p_buy_given_cart: dict = field(default_factory=lambda: {"E": 0.60, "S": 0.50, "C": 0.50, "I": 0.20})
    p_reformulate_if_abandon: float = 0.45
    p_reformulate_if_no_cart: float = 0.15
    explicit_feedback_rate: float = 0.005
    explicit_feedback_accuracy: float = 0.88

    @classmethod
    def from_config(cls, d: dict) -> "ClickModelParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def arr(self, name: str) -> np.ndarray:
        d = getattr(self, name)
        return np.array([d[l] for l in LABELS], dtype=np.float64)


def examination(position: np.ndarray, eta: float) -> np.ndarray:
    return (1.0 / np.maximum(position, 1)) ** eta


def simulate_interactions(rng: np.random.Generator, position: np.ndarray, label_code: np.ndarray,
                          stars: np.ndarray, price_z: np.ndarray, p: ClickModelParams) -> dict[str, np.ndarray]:
    """Vectorised over a flat array of impressions. Returns per-impression outcome arrays."""
    n = len(position)
    exam_p = examination(position, p.eta)
    s = np.where(np.isnan(stars), 4.3, stars)
    attract = p.arr("attractiveness")[label_code] * np.exp(p.rating_effect * (s - 4.3)) \
        * np.exp(-p.price_effect * np.clip(price_z, -3, 3))
    attract = np.clip(attract, 0.0, 0.95)
    examined = rng.random(n) < exam_p
    clicked = examined & (rng.random(n) < attract)
    carted = clicked & (rng.random(n) < p.arr("p_cart_given_click")[label_code])
    purchased = carted & (rng.random(n) < p.arr("p_buy_given_cart")[label_code])
    # dwell time (s): longer on relevant items; NaN when not clicked
    mu = np.array([3.6, 3.2, 3.0, 2.3])[label_code]
    dwell = np.where(clicked, np.exp(mu + 0.8 * rng.standard_normal(n)), np.nan).astype(np.float32)
    give_fb = examined & (rng.random(n) < p.explicit_feedback_rate)
    positive = label_code <= 1
    agree = rng.random(n) < p.explicit_feedback_accuracy
    fb = np.where(give_fb, np.where(positive == agree, 1, -1), 0).astype(np.int8)
    return {"examined": examined, "clicked": clicked, "carted": carted, "purchased": purchased,
            "dwell_s": dwell, "feedback": fb, "exam_propensity": exam_p.astype(np.float32)}

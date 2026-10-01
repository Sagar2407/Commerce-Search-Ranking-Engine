import numpy as np

from csre.config import load_config
from csre.sim.click_model import ClickModelParams, simulate_interactions
from csre.utils import stable_hash64, uniform_from_hash


def test_hash_is_deterministic_and_uniform():
    a = stable_hash64(["us:B000", "es:B000", "jp:X1"], salt="s")
    b = stable_hash64(["us:B000", "es:B000", "jp:X1"], salt="s")
    assert (a == b).all() and len(set(a.tolist())) == 3
    u = uniform_from_hash(stable_hash64((str(i) for i in range(20000))))
    assert 0.48 < u.mean() < 0.52 and u.min() >= 0 and u.max() < 1
    assert not np.allclose(uniform_from_hash(a, 1), uniform_from_hash(a, 2))


def test_click_model_orders_labels_and_positions():
    rng = np.random.default_rng(0)
    n = 400_000
    pos = rng.integers(1, 21, n)
    lab = rng.integers(0, 4, n).astype(np.int8)
    out = simulate_interactions(rng, pos, lab, np.full(n, 4.3), np.zeros(n), ClickModelParams())
    ctr = [out["clicked"][lab == k].mean() for k in range(4)]            # E, S, C, I
    assert ctr[0] > ctr[1] > ctr[2] > ctr[3]
    assert out["clicked"][pos == 1].mean() > 3 * out["clicked"][pos == 10].mean()
    assert not (out["carted"] & ~out["clicked"]).any()
    assert not (out["purchased"] & ~out["carted"]).any()


def test_config_overrides():
    cfg = load_config(overrides=["simulation.traffic.n_sessions=1234", "scale.tiers=[3000000]"])
    assert cfg.get("simulation.traffic.n_sessions") == 1234
    assert cfg.get("scale.tiers") == [3000000]


def test_query_variants_keep_intent_and_report_kind():
    from csre.sim.traffic import make_variant
    u = np.array([0.1, 0.5, 0.2, 0.3])
    assert make_variant("iphone case", "us", "reorder", u) == ("case iphone", "reorder")
    assert make_variant("divination", "us", "reorder", u)[1] == "typo"        # single token falls back
    text, kind = make_variant("zapatillas hombre", "es", "modifier", u)
    assert kind == "modifier" and "zapatillas hombre" in text

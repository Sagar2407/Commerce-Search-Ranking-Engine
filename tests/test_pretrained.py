"""A pretrained encoder registers as a *candidate*, indexes into its own directory and leaves production alone."""
import hashlib
import os
import sys
import types

import numpy as np

from tests.test_api import build_tiny_corpus


class _StubST:
    """Stands in for sentence_transformers.SentenceTransformer (no download): hashed bag-of-words vectors."""

    def __init__(self, name):
        self.name = name

    def get_sentence_embedding_dimension(self):
        return 32

    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        out = np.zeros((len(texts), 32), np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(":", " ").split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)


def test_pretrained_candidate_is_isolated(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_StubST))
    build_tiny_corpus(tmp_path)
    monkeypatch.setenv("CSRE_ROOT", str(tmp_path))
    from csre.config import load_config
    from csre.search.engine import Engine, build_indexes
    from csre.search.registry import ModelRegistry
    from csre.search.train import register_pretrained

    v = register_pretrained(load_config(), "intfloat/multilingual-e5-small")
    reg = ModelRegistry(load_config())
    assert reg.resolve("dense_encoder", "candidate") == v and reg.resolve("dense_encoder") is None

    cfg = load_config(overrides=["search.dense.encoder_alias=candidate", "search.index_tag=pretrained"])
    manifest = build_indexes(cfg, "demo_portable")
    assert manifest["dense_encoder"] == v
    assert (tmp_path / "data" / "indexes" / "demo_portable_pretrained" / "us" / "dense" / "emb.npy").exists()
    assert not (tmp_path / "data" / "indexes" / "demo_portable").exists()

    eng = Engine(cfg, "demo_portable", load_models=False, feedback=False)
    assert "dense" in eng.available_methods()
    r = eng.search("water bottle", "us", "dense", k=3)
    assert r.served_by == "dense" and len(r.rows) == 3

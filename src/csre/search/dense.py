"""Dense retrieval: an in-domain two-tower *subword-bag* encoder + exact / HNSW vector search.

Encoder
-------
text -> analysis units -> features (word, char 3-5-grams of "<word>", CJK char 1-3-grams)
     -> weighted sum of learned feature embeddings  E[f]  -> L2 normalise

Both towers share E (a siamese bag-of-subwords model, StarSpace / fastText-style). It is trained on
ESCI *train* queries only with an in-batch softmax (InfoNCE) loss, plus one hard negative per query taken
from products the annotators judged Irrelevant or Complement for that same query — exactly the
"looks lexically right, is the wrong product" cases BM25 struggles with. A share of training queries get a
random character edit so the encoder is robust to the typos seen in the replay stream.

Implemented in numpy / scipy with sparse Adagrad (only the rows touched by a batch are read and written),
so it trains on CPU in minutes and serves without a GPU or a deep-learning runtime.

A `sentence_transformers` backend with the same interface is available for environments that can
download pretrained models (e.g. multilingual-e5); it is not required anywhere.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import scipy.sparse as sp

from ..utils import get_logger
from . import analysis as A

log = get_logger("csre.dense")


def encoder_text(df: pl.DataFrame, extra_chars: int = 200) -> list[str]:
    """Product text for the encoder: title + brand + colour + the first bullet characters."""
    if "n_title_chars" in df.columns:
        return df.select(pl.col("doc_text").str.slice(0, pl.col("n_title_chars").cast(pl.Int64) + extra_chars)
                         )["doc_text"].to_list()
    return df["doc_text"].str.slice(0, 300).to_list()


def _tf(du: sp.csr_matrix) -> sp.csr_matrix:
    du = du.copy()
    du.data = (1.0 + np.log(du.data)).astype(np.float32)
    return du


def _normalize(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(U, axis=1, keepdims=True)
    n = np.maximum(n, 1e-6)
    return U / n, n


# ======================================================================================
# encoder
# ======================================================================================
class SubwordEncoder:
    kind = "hashed_bag"   # name kept for config compatibility: vocabulary-based subword bag

    def __init__(self, vocab: dict[str, int], E: np.ndarray, char_ngrams=(3, 5), jp_char_ngrams=(1, 3)):
        self.vocab = vocab
        self.E = E
        self.char_ngrams = tuple(char_ngrams)
        self.jp_char_ngrams = tuple(jp_char_ngrams)
        self.dim = E.shape[1]

    # ---------------------------------------------------------------- features
    def _ufn(self) -> Callable:
        cn, jn = self.char_ngrams, self.jp_char_ngrams
        return lambda u: A.unit_features(u, cn, jn)

    def unit_feature_matrix(self, unit_vocab: list[str]) -> sp.csr_matrix:
        m, _ = A.unit_map(unit_vocab, self._ufn(), vocab=self.vocab, grow=False)
        return m

    def featurize(self, texts: list[str], n_jobs: int | None = None) -> sp.csr_matrix:
        du, uv = A.doc_units(texts, n_jobs=n_jobs)
        return (_tf(du) @ self.unit_feature_matrix(uv)).tocsr()

    def featurize_one(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        counts: dict[str, int] = {}
        for u in A.units(text):
            counts[u] = counts.get(u, 0) + 1
        acc: dict[int, float] = {}
        for u, c in counts.items():
            tf = 1.0 + math.log(c)
            for f, w in A.unit_features(u, self.char_ngrams, self.jp_char_ngrams):
                j = self.vocab.get(f)
                if j is not None:
                    acc[j] = acc.get(j, 0.0) + tf * w
        return np.fromiter(acc.keys(), np.int64, len(acc)), np.fromiter(acc.values(), np.float32, len(acc))

    # ---------------------------------------------------------------- encode
    def encode_features(self, X: sp.csr_matrix, batch: int = 50_000) -> np.ndarray:
        out = np.empty((X.shape[0], self.dim), np.float32)
        for i in range(0, X.shape[0], batch):
            out[i:i + batch] = _normalize(np.asarray(X[i:i + batch] @ self.E))[0]
        return out

    def encode(self, texts: list[str], n_jobs: int | None = None, chunk: int = 200_000) -> np.ndarray:
        parts = [self.encode_features(self.featurize(texts[i:i + chunk], n_jobs)) for i in range(0, len(texts), chunk)]
        return np.vstack(parts) if parts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        ids, w = self.featurize_one(text)
        if len(ids) == 0:
            return np.zeros(self.dim, np.float32)
        u = w @ self.E[ids]
        return (u / max(float(np.linalg.norm(u)), 1e-6)).astype(np.float32)

    def known_feature_share(self, text: str) -> float:
        """Share of a text's features the encoder knows (an out-of-vocabulary signal for fallbacks)."""
        n_all = sum(len(A.unit_features(u, self.char_ngrams, self.jp_char_ngrams)) for u in A.units(text))
        return len(self.featurize_one(text)[0]) / n_all if n_all else 0.0

    # ---------------------------------------------------------------- io
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "E.npy", self.E)
        with open(d / "vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        (d / "encoder.json").write_text(json.dumps({
            "kind": self.kind, "dim": self.dim, "n_features": len(self.vocab),
            "char_ngrams": self.char_ngrams, "jp_char_ngrams": self.jp_char_ngrams}))

    @classmethod
    def load(cls, d: Path) -> "SubwordEncoder":
        meta = json.loads((d / "encoder.json").read_text())
        with open(d / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return cls(vocab, np.load(d / "E.npy"), meta["char_ngrams"], meta["jp_char_ngrams"])


class SentenceTransformerEncoder:
    """Optional pretrained backend (needs `sentence-transformers` and Hugging Face access)."""
    kind = "sentence_transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 (optional dependency)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._e5 = "e5" in model_name

    def encode(self, texts: list[str], n_jobs: int | None = None, chunk: int = 0) -> np.ndarray:
        texts = [("passage: " + t) if self._e5 else t for t in texts]
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        t = ("query: " + text) if self._e5 else text
        return self.model.encode([t], normalize_embeddings=True)[0].astype(np.float32)

    def known_feature_share(self, text: str) -> float:
        return 1.0

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "encoder.json").write_text(json.dumps({"kind": self.kind, "model_name": self.model_name, "dim": self.dim}))

    @classmethod
    def load(cls, d: Path) -> "SentenceTransformerEncoder":
        return cls(json.loads((d / "encoder.json").read_text())["model_name"])


def load_encoder(d: Path):
    kind = json.loads((d / "encoder.json").read_text())["kind"]
    return SentenceTransformerEncoder.load(d) if kind == "sentence_transformers" else SubwordEncoder.load(d)


# ======================================================================================
# training
# ======================================================================================
_KEYBOARD = "abcdefghijklmnopqrstuvwxyz"


def typo(q: str, rng: np.random.Generator) -> str:
    """One random character edit (delete / swap / substitute / insert) inside a latin word."""
    words = q.split()
    cand = [i for i, w in enumerate(words) if len(w) >= 4 and w.isascii() and w.isalpha()]
    if not cand:
        return q
    i = cand[rng.integers(len(cand))]
    w = words[i]
    p = int(rng.integers(1, len(w) - 1))
    op = rng.integers(4)
    if op == 0:
        w = w[:p] + w[p + 1:]
    elif op == 1:
        w = w[:p - 1] + w[p] + w[p - 1] + w[p + 1:]
    elif op == 2:
        w = w[:p] + _KEYBOARD[rng.integers(26)] + w[p + 1:]
    else:
        w = w[:p] + _KEYBOARD[rng.integers(26)] + w[p:]
    words[i] = w
    return " ".join(words)


def build_feature_vocab(unit_vocab: list[str], unit_freq: np.ndarray, char_ngrams, jp_char_ngrams,
                        min_df: int, max_vocab: int) -> dict[str, int]:
    """Keep the most frequent features (frequency = occurrences of units containing the feature)."""
    um, fv = A.unit_map(unit_vocab, lambda u: A.unit_features(u, char_ngrams, jp_char_ngrams))
    um.data[:] = 1.0
    freq = np.asarray(um.T @ unit_freq).ravel()
    keep = np.flatnonzero(freq >= min_df)
    if len(keep) > max_vocab:
        keep = keep[np.argsort(-freq[keep], kind="stable")[:max_vocab]]
    names = np.array(list(fv), dtype=object)[np.sort(keep)]
    return {f: i for i, f in enumerate(names)}


class _Trainer:
    def __init__(self, E: np.ndarray, lr: float, temperature: float, G: np.ndarray | None = None):
        self.E = E
        # row-wise Adagrad accumulator (memory-light); shared between Hogwild workers when given
        self.G = G if G is not None else np.full(E.shape[0], 1e-3, np.float32)
        self.lr = lr
        self.tau = temperature

    def step(self, Xq: sp.csr_matrix, Xd: sp.csr_matrix, mask: np.ndarray) -> float:
        cols = np.unique(np.concatenate([Xq.indices, Xd.indices]))
        Xq_c = sp.csr_matrix((Xq.data, np.searchsorted(cols, Xq.indices), Xq.indptr), shape=(Xq.shape[0], len(cols)))
        Xd_c = sp.csr_matrix((Xd.data, np.searchsorted(cols, Xd.indices), Xd.indptr), shape=(Xd.shape[0], len(cols)))
        Ec = self.E[cols]
        Uq = np.asarray(Xq_c @ Ec)
        Ud = np.asarray(Xd_c @ Ec)
        Q, nq = _normalize(Uq)
        D, nd = _normalize(Ud)
        S = (Q @ D.T) / self.tau
        S[mask] = -1e9
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)
        B = Q.shape[0]
        diag = np.arange(B)
        loss = float(-np.log(np.maximum(P[diag, diag], 1e-12)).mean())
        dS = P
        dS[diag, diag] -= 1.0
        dS /= (B * self.tau)
        dQ = dS @ D
        dD = dS.T @ Q
        dUq = (dQ - Q * (dQ * Q).sum(axis=1, keepdims=True)) / nq
        dUd = (dD - D * (dD * D).sum(axis=1, keepdims=True)) / nd
        g = np.asarray(Xq_c.T @ dUq) + np.asarray(Xd_c.T @ dUd)
        self.G[cols] += (g * g).mean(axis=1)
        self.E[cols] = Ec - (self.lr / np.sqrt(self.G[cols]))[:, None] * g
        return loss


def train_encoder(
    q_texts: list[str], q_locale: np.ndarray, d_texts: list[str],
    pos_q: np.ndarray, pos_d: np.ndarray, neg_ptr: np.ndarray, neg_d: np.ndarray, d_locale: np.ndarray,
    params: dict, seed: int, dev_fn: Callable[["SubwordEncoder"], dict] | None = None,
) -> tuple[SubwordEncoder, list[dict]]:
    """Train the subword-bag two-tower encoder.

    pos_q / pos_d: (query, product) positive pairs (Exact).  neg_ptr / neg_d: CSR lists of hard negatives per
    query (Irrelevant / Complement for that query).  Batches are drawn within one locale so in-batch
    negatives are plausible competitors.
    """
    rng = np.random.default_rng(seed)
    T = params["train"]
    cn, jn = tuple(params["char_ngrams"]), tuple(params["jp_char_ngrams"])
    t0 = time.time()
    aug_share = float(T.get("typo_augment", 0.0))
    q_aug = [typo(q, rng) if rng.random() < aug_share else q for q in q_texts]

    # analysis: queries (+ augmented copies) and products share one unit vocabulary
    du_all, uv = A.doc_units(list(q_texts) + q_aug + list(d_texts))
    unit_freq = np.asarray((du_all > 0).sum(axis=0)).ravel().astype(np.float64)
    vocab = build_feature_vocab(uv, unit_freq, cn, jn, int(params["min_df"]), int(params["max_vocab"]))
    log.info("encoder: %d units -> %d features (%.0fs)", len(uv), len(vocab), time.time() - t0)
    E = (rng.standard_normal((len(vocab), int(params["dim"]))) * float(T.get("init_scale", 0.1))).astype(np.float32)
    enc = SubwordEncoder(vocab, E, cn, jn)
    # features are materialised per batch (doc x unit @ unit x feature): ~50 units vs ~400 features per
    # product, so the full catalog fits in memory comfortably
    UF = enc.unit_feature_matrix(uv).tocsr()
    du_all = _tf(du_all).tocsr()
    nq = len(q_texts)
    Uq, Uqa, Ud = du_all[:nq], du_all[nq:2 * nq], du_all[2 * nq:]
    log.info("encoder: analysed %d queries, %d products, unit nnz=%d (%.0fs)", nq, Ud.shape[0], du_all.nnz,
             time.time() - t0)
    del du_all

    B = int(T["batch_size"])
    by_loc = {loc: np.flatnonzero(q_locale[pos_q] == loc) for loc in np.unique(q_locale)}
    docs_by_loc = {loc: np.flatnonzero(d_locale == loc) for loc in np.unique(d_locale)}
    n_workers = max(1, int(T.get("workers", 1)))

    # Hogwild: workers update one shared embedding table without locks (sparse rows rarely collide),
    # as in word2vec / fastText / StarSpace training. Workers are forked and only use numpy / scipy.
    from multiprocessing import shared_memory  # noqa: PLC0415
    shm_e = shared_memory.SharedMemory(create=True, size=E.nbytes)
    shm_g = shared_memory.SharedMemory(create=True, size=E.shape[0] * 4)
    try:
        E_sh = np.ndarray(E.shape, np.float32, buffer=shm_e.buf)
        E_sh[:] = E
        G_sh = np.ndarray((E.shape[0],), np.float32, buffer=shm_g.buf)
        G_sh[:] = 1e-3
        enc.E = E_sh
        del E

        def run_batches(batches, wseed, out_q=None):
            if out_q is not None:
                try:
                    from threadpoolctl import threadpool_limits  # noqa: PLC0415
                    threadpool_limits(1)
                except Exception:  # noqa: BLE001
                    pass
            r = np.random.default_rng(wseed)
            tr = _Trainer(E_sh, float(T["lr"]), float(T["temperature"]), G_sh)
            n_hard = max(1, int(T.get("hard_negatives", 1)))
            losses = []
            for loc, b in batches:
                qi, di = pos_q[b], pos_d[b]
                use_aug = r.random(len(qi)) < 0.5
                order = np.concatenate([np.flatnonzero(~use_aug), np.flatnonzero(use_aug)])
                Xb = (sp.vstack([Uq[qi[~use_aug]], Uqa[qi[use_aug]]]).tocsr() @ UF).tocsr()
                qi, di = qi[order], di[order]
                nn = neg_ptr[qi + 1] - neg_ptr[qi]
                pool = docs_by_loc[loc]
                negs = []
                for _ in range(n_hard):   # judged negatives of the same query (random product if none)
                    pick = neg_ptr[qi] + (r.random(len(qi)) * np.maximum(nn, 1)).astype(np.int64)
                    rand = pool[r.integers(len(pool), size=len(qi))]
                    negs.append(np.where(nn > 0, neg_d[np.minimum(pick, len(neg_d) - 1)], rand) if len(neg_d) else rand)
                docs = np.concatenate([di, *negs])
                owner = np.concatenate([qi, np.full(len(docs) - len(qi), -1)])
                mask = (docs[None, :] == di[:, None]) | (owner[None, :] == qi[:, None])
                mask[np.arange(len(qi)), np.arange(len(qi))] = False
                losses.append(tr.step(Xb, (Ud[docs] @ UF).tocsr(), mask))
            if out_q is not None:
                out_q.put(losses)
            return losses

        history = []
        for ep in range(int(T["epochs"])):
            batches = []
            for loc, idx in by_loc.items():
                idx = rng.permutation(idx)
                batches += [(loc, idx[i:i + B]) for i in range(0, len(idx) - B + 1, B)]
            batches = [batches[i] for i in rng.permutation(len(batches))]
            te = time.time()
            if n_workers == 1 or len(batches) < 4 * n_workers:
                losses = run_batches(batches, int(rng.integers(1 << 31)))
            else:
                import multiprocessing as mp  # noqa: PLC0415
                ctx = mp.get_context("fork")
                q_out = ctx.Queue()
                procs = [ctx.Process(target=run_batches, args=(batches[w::n_workers], int(rng.integers(1 << 31)), q_out))
                         for w in range(n_workers)]
                for pr in procs:
                    pr.start()
                losses = [x for _ in procs for x in q_out.get()]
                for pr in procs:
                    pr.join()
                    if pr.exitcode != 0:
                        raise RuntimeError(f"encoder worker failed with exit code {pr.exitcode}")
            rec = {"epoch": ep + 1, "loss": round(float(np.mean(losses)), 4), "steps": len(losses),
                   "seconds": round(time.time() - te, 1), "workers": n_workers}
            if dev_fn is not None:
                rec.update(dev_fn(enc))
            history.append(rec)
            log.info("encoder epoch %s", rec)
        enc.E = np.array(E_sh)            # copy out of shared memory
    finally:
        shm_e.close()
        shm_e.unlink()
        shm_g.close()
        shm_g.unlink()
    return enc, history


# ======================================================================================
# vector index
# ======================================================================================
class VectorIndex:
    """Exact inner-product search, or faiss HNSW for large corpora."""

    def __init__(self, emb: np.ndarray, ann=None, ef_search: int = 128):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.ann = ann
        self.ef_search = ef_search
        if ann is not None:
            ann.hnsw.efSearch = ef_search

    @classmethod
    def build(cls, emb: np.ndarray, kind: str = "hnsw", m: int = 32, ef_construction: int = 80,
              ef_search: int = 128, min_docs_for_ann: int = 200_000) -> "VectorIndex":
        ann = None
        if kind == "hnsw" and len(emb) >= min_docs_for_ann:
            import faiss  # noqa: PLC0415
            ann = faiss.IndexHNSWFlat(emb.shape[1], m, faiss.METRIC_INNER_PRODUCT)
            ann.hnsw.efConstruction = ef_construction
            t = time.time()
            ann.add(np.ascontiguousarray(emb, dtype=np.float32))
            log.info("HNSW built over %d vectors in %.0fs", len(emb), time.time() - t)
        return cls(emb, ann, ef_search)

    def search(self, q: np.ndarray, k: int = 100, exact: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if self.ann is not None and not exact:
            s, i = self.ann.search(q[None, :].astype(np.float32), k)
            ok = i[0] >= 0
            return i[0][ok].astype(np.int64), s[0][ok].astype(np.float32)
        s = self.emb @ q
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        order = np.lexsort((top, -s[top]))
        return top[order].astype(np.int64), s[top[order]]

    def score_docs(self, q: np.ndarray, docs: np.ndarray) -> np.ndarray:
        return self.emb[docs] @ q

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "emb.npy", self.emb)
        if self.ann is not None:
            import faiss  # noqa: PLC0415
            faiss.write_index(self.ann, str(d / "hnsw.faiss"))

    @classmethod
    def load(cls, d: Path, ef_search: int = 128, mmap: bool = False) -> "VectorIndex":
        emb = np.load(d / "emb.npy", mmap_mode="r" if mmap else None)
        ann = None
        if (d / "hnsw.faiss").exists():
            import faiss  # noqa: PLC0415
            ann = faiss.read_index(str(d / "hnsw.faiss"))
        return cls(np.asarray(emb), ann, ef_search)

    def nbytes(self) -> int:
        return int(self.emb.nbytes + (self.ann.ntotal * (self.emb.shape[1] * 4 + 2 * 32 * 4) if self.ann else 0))

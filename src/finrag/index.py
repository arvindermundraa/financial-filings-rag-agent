"""Build and query the search index.

Two retrievers over the same chunks:
  * BM25 - exact term matching, strong on line-item names like
    "purchases of property, plant and equipment". Implemented as a precomputed sparse
    matrix of BM25 term weights, so scoring a query is one sparse column sum.
  * Dense LSA vectors - TF-IDF over word 1-2 grams reduced with truncated SVD, compared by
    cosine similarity. Runs fully offline (no model download); it catches paraphrases that
    share related vocabulary, which BM25 misses.
They are combined with reciprocal rank fusion (RRF), which only needs ranks, so the two
score scales never have to be calibrated against each other.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config
from .chunking import Chunk
from .query import norm_company

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
METHODS = ("bm25", "dense", "hybrid")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


def rrf_fuse(rankings: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank(d)), rank from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class BM25:
    """Okapi BM25 with Lucene-style idf, k1=1.2, b=0.75."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.counter = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        tf = self.counter.fit_transform(texts).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.weights = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.vocab = self.counter.vocabulary_

    def scores(self, query: str) -> np.ndarray:
        cols = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not cols:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1), dtype=np.float32).ravel()


class SearchIndex:
    def __init__(self, chunks: list[Chunk], bm25: BM25, vectorizer: TfidfVectorizer,
                 svd: TruncatedSVD, dense: np.ndarray):
        self.chunks = chunks
        self.bm25 = bm25
        self.vectorizer = vectorizer
        self.svd = svd
        self.dense = dense
        # (vocab x dim) projection; sparse query @ this only touches the query's terms
        self._proj = np.ascontiguousarray(svd.components_.T, dtype=np.float32)
        self._doc_of = np.array([c.doc_name for c in chunks])
        self._company_of = np.array([c.company for c in chunks])

    # ---------- build / persist ----------
    @classmethod
    def build(cls, chunks: list[Chunk], lsa_dim: int = config.LSA_DIM) -> "SearchIndex":
        texts = [c.text for c in chunks]
        bm25 = BM25(texts)
        vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                     ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
        tfidf = vectorizer.fit_transform(texts)
        dim = min(lsa_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        svd = TruncatedSVD(n_components=dim, random_state=0)
        dense = _l2norm(svd.fit_transform(tfidf).astype(np.float32))
        return cls(chunks, bm25, vectorizer, svd, dense)

    def save(self, directory: Path = config.INDEX_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "index.pkl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory: Path = config.INDEX_DIR) -> "SearchIndex":
        with open(directory / "index.pkl", "rb") as f:
            return pickle.load(f)

    # ---------- search ----------
    def _mask(self, doc_name: str | None, company: str | None) -> np.ndarray | None:
        if doc_name:
            return self._doc_of == doc_name
        if company:
            return self._company_of == norm_company(company)
        return None

    def _rank(self, scores: np.ndarray, mask: np.ndarray | None, n: int) -> list[int]:
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [int(i) for i in top if np.isfinite(scores[i])]

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(query)

    def dense_scores(self, query: str) -> np.ndarray:
        x = self.vectorizer.transform([query])
        # gather only the query's term rows instead of a sparse x dense product over the vocab
        q = _l2norm((self._proj[x.indices].T @ x.data.astype(np.float32))[None, :])
        return self.dense @ q[0]

    def search(self, query: str, k: int = 5, method: str = "hybrid", doc_name: str | None = None,
               company: str | None = None, pool: int = 50) -> list[Hit]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        mask = self._mask(doc_name, company)
        if method == "bm25":
            scores = self.bm25_scores(query)
            ranked = [(i, float(scores[i])) for i in self._rank(scores, mask, k)]
        elif method == "dense":
            scores = self.dense_scores(query)
            ranked = [(i, float(scores[i])) for i in self._rank(scores, mask, k)]
        else:
            lists = [self._rank(self.bm25_scores(query), mask, pool),
                     self._rank(self.dense_scores(query), mask, pool)]
            ranked = rrf_fuse(lists)[:k]
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(ranked, start=1)]


def _l2norm(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms

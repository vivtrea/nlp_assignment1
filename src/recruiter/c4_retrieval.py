"""C4 · Hybrid retrieval: BM25 (sparse) + LSA embeddings (dense) fused with RRF.

Retrieval runs only over the C3-eligible pool (`restrict_to`), so it is one stage of a
pipeline, not the core of the system. The dense backend is LSA (TF-IDF -> TruncatedSVD)
because it runs offline; see reports/04_modeling.md for the trade-off.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recruiter import config

TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
STOP = set(
    "a an the and or of to in on for with at by from as is are was were be been i my we our you your "
    "it this that will can have has had not but if so do did their they he she his her me us".split()
)


def tokenize(text: str) -> list[str]:
    return [t.strip(".") for t in TOKEN_RE.findall(text.lower()) if t.strip(".") and t not in STOP]


def rrf(rankings: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r + 1)
    return scores


@dataclass
class HybridIndex:
    ids: list[str]
    bm25: BM25Okapi
    tfidf: TfidfVectorizer
    svd: TruncatedSVD | None
    doc_vecs: np.ndarray
    id_pos: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, ids: list[str], texts: list[str], dim: int = config.LSA_DIM) -> HybridIndex:
        bm25 = BM25Okapi([tokenize(t) for t in texts])
        tfidf = TfidfVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True
        )
        X = tfidf.fit_transform(texts)
        n_comp = min(dim, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=config.SEED) if n_comp >= 2 else None
        vecs = normalize(svd.fit_transform(X)) if svd else normalize(X).toarray()
        idx = cls(ids, bm25, tfidf, svd, vecs.astype(np.float32))
        idx.id_pos = {d: i for i, d in enumerate(ids)}
        return idx

    def embed(self, text: str) -> np.ndarray:
        x = self.tfidf.transform([text])
        v = self.svd.transform(x) if self.svd else x.toarray()
        return normalize(v)[0]

    def _subset(self, restrict_to: list[str] | None) -> np.ndarray:
        if restrict_to is None:
            return np.arange(len(self.ids))
        return np.array([self.id_pos[i] for i in restrict_to if i in self.id_pos], dtype=int)

    def bm25_scores(self, query: str, pos: np.ndarray) -> np.ndarray:
        q = tokenize(query)
        if len(pos) == len(self.ids):
            return self.bm25.get_scores(q)
        return np.array(self.bm25.get_batch_scores(q, pos.tolist()))

    def dense_scores(self, query: str, pos: np.ndarray) -> np.ndarray:
        return self.doc_vecs[pos] @ self.embed(query)

    def search(
        self,
        query: str,
        top_n: int = config.RETRIEVAL_TOP_N,
        restrict_to: list[str] | None = None,
        mode: str = "hybrid",
    ) -> list[tuple[str, float]]:
        pos = self._subset(restrict_to)
        if len(pos) == 0:
            return []
        if mode == "bm25":
            s = self.bm25_scores(query, pos)
            order = np.argsort(-s)[:top_n]
            return [(self.ids[pos[i]], float(s[i])) for i in order]
        if mode == "dense":
            s = self.dense_scores(query, pos)
            order = np.argsort(-s)[:top_n]
            return [(self.ids[pos[i]], float(s[i])) for i in order]
        depth = max(top_n * 4, 200)
        b = np.argsort(-self.bm25_scores(query, pos))[:depth].tolist()
        d = np.argsort(-self.dense_scores(query, pos))[:depth].tolist()
        fused = rrf([b, d])
        best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_n]
        return [(self.ids[pos[i]], s) for i, s in best]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> HybridIndex:
        with path.open("rb") as f:
            return pickle.load(f)


def candidate_text(title: str, cv: str) -> str:
    return f"{title}\n{cv}"

"""Dense, lexical (BM25) and hybrid (RRF) retrievers.

Every retriever exposes `search(query, *, filters, top_k) -> list[chunk]`, the same
interface as `agentic.Retriever`, so any of them can be passed to `run_agent`.
Returned chunks keep the Day 7 format: id, text, source_id, source, page, metadata,
plus `score` and `strategy`.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from baseline import tokenize  # noqa: E402

# Small English stopword list: the corpus and benchmark are English.
STOPWORDS = frozenset('''a an and are as at be by can do does for from has have how in into is it its
of on or that the their them these this those to was were what when where which who why will with
you your chapter section according'''.split())


class SearchFn(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]: ...


def _as_text(value: Any) -> str:
    return json.dumps(value) if isinstance(value, bool) else str(value)


def matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Same semantics as the SQL filter in agentic.Retriever: exact match, list = any of."""
    for key, value in (filters or {}).items():
        wanted = {_as_text(v) for v in (value if isinstance(value, (list, tuple, set)) else [value])}
        if key in ('source_id', 'document_id'):
            actual = chunk.get('source_id')
        elif key in ('source', 'page'):
            actual = chunk.get(key)
        else:
            actual = (chunk.get('metadata') or {}).get(key)
        if actual is None or _as_text(actual) not in wanted:
            return False
    return True


def _check(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query must be a non-empty string')
    if top_k < 1:
        raise ValueError('top_k must be >= 1')


class LexicalRetriever:
    """Okapi BM25 over chunk text, in memory; filters are applied before ranking."""

    name = 'bm25'

    def __init__(self, chunks: Sequence[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75,
                 stopwords: bool = True) -> None:
        if not chunks:
            raise ValueError('chunks must be non-empty')
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.stopwords = STOPWORDS if stopwords else frozenset()
        self.tfs = [Counter(self._terms(c['text'])) for c in self.chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / len(self.lengths) or 1.0
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _terms(self, text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in self.stopwords]

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        _check(query, top_k)
        terms = [t for t in set(self._terms(query)) if t in self.idf]
        results = []
        for chunk, tf, length in zip(self.chunks, self.tfs, self.lengths):
            if not matches(chunk, filters):
                continue
            score = 0.0
            for t in terms:
                f = tf.get(t, 0)
                if f:
                    score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': self.name})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


class DenseRetriever:
    """Dense retrieval on the Day 6 pgvector store (wraps agentic.Retriever)."""

    name = 'dense'

    def __init__(self, strategy: str = 'B', **kwargs: Any) -> None:
        from agentic import Retriever  # lazy: needs psycopg + DATABASE_URL + OPENROUTER_API_KEY
        self.store = Retriever(strategy, **kwargs)

    @property
    def cost_usd(self) -> float:
        return self.store.embedding_cost_usd

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        _check(query, top_k)
        return [{**r, 'strategy': self.name} for r in self.store.search(query, filters=filters, top_k=top_k)]


def rrf_fuse(ranked_lists: Sequence[Sequence[dict[str, Any]]], *, k: int = 60,
             weights: Sequence[float] | None = None, labels: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1."""
    weights = weights or [1.0] * len(ranked_lists)
    labels = labels or [str(i) for i in range(len(ranked_lists))]
    if not (len(weights) == len(labels) == len(ranked_lists)):
        raise ValueError('weights/labels must match ranked_lists')
    fused: dict[str, dict[str, Any]] = {}
    for results, weight, label in zip(ranked_lists, weights, labels):
        for rank, chunk in enumerate(results, start=1):
            entry = fused.setdefault(chunk['id'], {**chunk, 'score': 0.0, 'ranks': {}})
            entry['score'] += weight / (k + rank)
            entry['ranks'][label] = rank
    return sorted(fused.values(), key=lambda c: (-c['score'], c['id']))


class HybridRetriever:
    """Fuses dense and lexical candidate lists with RRF."""

    name = 'hybrid'

    def __init__(self, dense: SearchFn, lexical: SearchFn, *, pool: int = 20, k: int = 60,
                 weights: tuple[float, float] = (1.0, 1.0)) -> None:
        if pool < 1:
            raise ValueError('pool must be >= 1')
        self.dense, self.lexical, self.pool, self.k, self.weights = dense, lexical, pool, k, weights

    @property
    def cost_usd(self) -> float:
        return getattr(self.dense, 'cost_usd', 0.0)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        _check(query, top_k)
        pool = max(self.pool, top_k)
        lists = [self.dense.search(query, filters=filters, top_k=pool),
                 self.lexical.search(query, filters=filters, top_k=pool)]
        fused = rrf_fuse(lists, k=self.k, weights=self.weights, labels=['dense', 'lexical'])
        return [{**c, 'strategy': self.name} for c in fused[:top_k]]

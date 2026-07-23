"""Cross-encoder reranking stage (P1.1).

Takes the fused hybrid-search candidates and re-scores them with a
cross-encoder that sees the (query, document) pair together, then keeps
the best. Recall stays with hybrid retrieval (bge-m3 hit@10 ~0.95 on the
neutral gold set); this stage exists to lift the right document toward
rank #1 (hit@1 was ~0.49).

Target backend: Qwen3-Reranker-0.6B in GGUF via llama.cpp (no torch).
The backend is injected so the ranking logic is testable without the model.
A backend must expose:

    score(query: str, documents: list[str]) -> list[float]   # higher = more relevant
"""
from dataclasses import replace

from sparcos_rag.retriever import FusedHit


class Reranker:
    """Re-orders FusedHit candidates by cross-encoder relevance.

    backend: object with score(query, documents) -> list[float]. If None,
    the real GGUF backend is built lazily on first use (optional dependency).
    """

    def __init__(self, backend=None, model_path: str | None = None):
        self._backend = backend
        self.model_path = model_path

    def _ensure_backend(self):
        if self._backend is None:
            from sparcos_rag._qwen_reranker import Qwen3RerankerBackend

            self._backend = Qwen3RerankerBackend(self.model_path)
        return self._backend

    def rerank(
        self, query: str, hits: list[FusedHit], top_k: int | None = None
    ) -> list[FusedHit]:
        if not hits:
            return []
        scores = self._ensure_backend().score(query, [h.content for h in hits])
        if len(scores) != len(hits):
            raise ValueError(
                f"reranker backend returned {len(scores)} scores for {len(hits)} hits"
            )
        rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores)]
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored if top_k is None else rescored[:top_k]

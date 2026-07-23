import pytest

from sparcos_rag.retriever import FusedHit
from sparcos_rag.reranker import Reranker


class FakeBackend:
    """Scores each document by how many query terms it contains."""

    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        terms = set(query.lower().split())
        return [float(sum(t in d.lower() for t in terms)) for d in documents]


def _hits():
    # Deliberately worst-first: the relevant doc starts at the bottom.
    return [
        FusedHit("a.md", "H", "totally unrelated filler", 0.9),
        FusedHit("b.md", "H", "some other text", 0.5),
        FusedHit("c.md", "H", "giovanni beggiato profile", 0.1),
    ]


def test_rerank_lifts_relevant_to_top():
    r = Reranker(backend=FakeBackend())
    out = r.rerank("giovanni beggiato", _hits())
    assert out[0].source_path == "c.md"
    assert [h.source_path for h in out] == ["c.md", "a.md", "b.md"]


def test_rerank_respects_top_k():
    r = Reranker(backend=FakeBackend())
    out = r.rerank("giovanni beggiato", _hits(), top_k=1)
    assert len(out) == 1
    assert out[0].source_path == "c.md"


def test_rerank_empty_returns_empty():
    r = Reranker(backend=FakeBackend())
    assert r.rerank("q", []) == []


def test_rerank_rewrites_score_from_backend():
    r = Reranker(backend=FakeBackend())
    out = r.rerank("giovanni beggiato", _hits())
    # Score is the backend's, not the original RRF score.
    assert out[0].score == 2.0


def test_rerank_length_mismatch_raises():
    class BadBackend:
        def score(self, query, documents):
            return [1.0]  # wrong length

    with pytest.raises(ValueError):
        Reranker(backend=BadBackend()).rerank("q", _hits())


def test_missing_model_path_raises_clear_error():
    from sparcos_rag._qwen_reranker import Qwen3RerankerBackend

    with pytest.raises(ValueError):
        Qwen3RerankerBackend(model_path=None)

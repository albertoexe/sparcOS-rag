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


# --- Integration: real GGUF backend (skipped if model or llama-cpp absent) ---
import os
from pathlib import Path

_MODEL = os.environ.get(
    "RERANK_MODEL_PATH",
    str(Path.home() / ".cache/qmd/models/hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf"),
)


@pytest.mark.skipif(
    os.environ.get("RERANK_INTEGRATION") != "1" or not Path(_MODEL).exists(),
    reason="opt-in only (set RERANK_INTEGRATION=1); loading the GGUF can crash "
    "on native teardown inside pytest on Windows — validated via measure_rerank.py",
)
def test_real_backend_ranks_relevant_above_irrelevant():
    pytest.importorskip("llama_cpp")
    from sparcos_rag._qwen_reranker import Qwen3RerankerBackend

    backend = Qwen3RerankerBackend(_MODEL, n_ctx=2048)
    scores = backend.score(
        "Chi e Giovanni Beggiato?",
        [
            "Giovanni Beggiato e un cliente storico, referente area produzione.",
            "La ricetta della carbonara richiede uova, guanciale e pecorino.",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]

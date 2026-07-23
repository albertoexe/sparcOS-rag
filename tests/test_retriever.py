from sparcos_rag.store import SearchHit
from sparcos_rag.retriever import hybrid_search


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0]]


class FakeStore:
    def vector_search(self, vec, k):
        return [SearchHit(i, "a.md", "H", f"c{i}", 1.0, i + 1) for i in range(5)]

    def fulltext_search(self, q, k):
        return [SearchHit(99, "b.md", "H", "hit", 1.0, 1)]


def test_per_file_cap_and_topk():
    res = hybrid_search("q", FakeEmbedder(), FakeStore(), top_k=10, per_file_cap=3)
    a_count = sum(1 for r in res if r.source_path == "a.md")
    assert a_count <= 3
    assert any(r.source_path == "b.md" for r in res)


class ReverseReranker:
    """Reverses the fused order, so we can prove the stage ran."""

    def rerank(self, query, hits, top_k=None):
        out = list(reversed(hits))
        return out if top_k is None else out[:top_k]


def test_reranker_reorders_when_provided():
    baseline = hybrid_search("q", FakeEmbedder(), FakeStore(), top_k=10, per_file_cap=3)
    reranked = hybrid_search(
        "q", FakeEmbedder(), FakeStore(), top_k=10, per_file_cap=3, reranker=ReverseReranker()
    )
    # The reranker reversed the fused order, so the top content must differ.
    assert reranked[0].content != baseline[0].content


def test_no_reranker_leaves_pipeline_unchanged():
    res = hybrid_search("q", FakeEmbedder(), FakeStore(), reranker=None)
    assert res  # default path still works with reranker defaulting to None

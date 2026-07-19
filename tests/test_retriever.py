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

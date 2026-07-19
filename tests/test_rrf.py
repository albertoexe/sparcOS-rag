from sparcos_rag.store import SearchHit
from sparcos_rag.retriever import rrf_fuse


def _h(cid, path, rank):
    return SearchHit(cid, path, "H", f"c{cid}", 1.0, rank)


def test_consensus_beats_single_strong():
    vec = [_h(9, "x.md", 1), _h(1, "a.md", 2)]
    fts = [_h(2, "y.md", 1), _h(1, "a.md", 2)]
    fused = rrf_fuse([vec, fts])
    assert fused[0].source_path == "a.md"


def test_scores_use_k_const():
    fused = rrf_fuse([[_h(1, "a.md", 1)]], k_const=60)
    assert abs(fused[0].score - (1.0 / 61)) < 1e-9

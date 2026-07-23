from dataclasses import dataclass
from sparcos_rag.store import SearchHit


@dataclass(frozen=True)
class FusedHit:
    source_path: str
    heading: str
    content: str
    score: float


def rrf_fuse(result_lists, k_const: int = 60, weights=None) -> list[FusedHit]:
    if weights is None:
        weights = [1.0] * len(result_lists)
    acc: dict[int, dict] = {}
    for w, hits in zip(weights, result_lists):
        for h in hits:
            entry = acc.setdefault(h.chunk_id, {"hit": h, "score": 0.0})
            entry["score"] += w / (k_const + h.rank)
    fused = [
        FusedHit(e["hit"].source_path, e["hit"].heading, e["hit"].content, e["score"])
        for e in acc.values()
    ]
    return sorted(fused, key=lambda f: f.score, reverse=True)


def hybrid_search(
    query,
    embedder,
    store,
    top_k=10,
    per_file_cap=3,
    candidate_k=20,
    reranker=None,
    rerank_candidate_k=None,
):
    qvec = embedder.embed([query])[0]
    vec_hits = store.vector_search(qvec, candidate_k)
    fts_hits = store.fulltext_search(query, candidate_k)
    fused = rrf_fuse([vec_hits, fts_hits])
    if reranker is not None:
        pool = fused[:rerank_candidate_k] if rerank_candidate_k else fused
        fused = reranker.rerank(query, pool)
    capped: list[FusedHit] = []
    per_file: dict[str, int] = {}
    for f in fused:
        n = per_file.get(f.source_path, 0)
        if n >= per_file_cap:
            continue
        per_file[f.source_path] = n + 1
        capped.append(f)
        if len(capped) >= top_k:
            break
    return capped

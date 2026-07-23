def hit_at_k(results_per_question, k: int) -> float:
    if not results_per_question:
        return 0.0
    hits = sum(1 for retrieved, expected in results_per_question
               if expected in retrieved[:k])
    return hits / len(results_per_question)


def _reciprocal_rank(retrieved, expected) -> float:
    for i, path in enumerate(retrieved, start=1):
        if path == expected:
            return 1.0 / i
    return 0.0


def mrr(results_per_question) -> float:
    if not results_per_question:
        return 0.0
    total = sum(_reciprocal_rank(r, e) for r, e in results_per_question)
    return total / len(results_per_question)


def run_eval(questions, embedder, store, k: int, reranker=None, rerank_candidate_k: int = 20) -> dict:
    from sparcos_rag.retriever import hybrid_search
    data = []
    for item in questions:
        hits = hybrid_search(
            item["q"],
            embedder,
            store,
            top_k=k,
            reranker=reranker,
            rerank_candidate_k=rerank_candidate_k if reranker is not None else None,
        )
        data.append(([h.source_path for h in hits], item["expect"]))
    return {
        "hit_at_k": hit_at_k(data, k),
        "hit@1": hit_at_k(data, 1),
        "hit@5": hit_at_k(data, 5),
        "mrr": mrr(data),
        "n": len(questions),
    }

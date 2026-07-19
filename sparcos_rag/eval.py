def hit_at_k(results_per_question, k: int) -> float:
    if not results_per_question:
        return 0.0
    hits = sum(1 for retrieved, expected in results_per_question
               if expected in retrieved[:k])
    return hits / len(results_per_question)


def run_eval(questions, embedder, store, k: int) -> dict:
    from sparcos_rag.retriever import hybrid_search
    data = []
    for item in questions:
        hits = hybrid_search(item["q"], embedder, store, top_k=k)
        data.append(([h.source_path for h in hits], item["expect"]))
    return {"hit_at_k": hit_at_k(data, k), "n": len(questions)}

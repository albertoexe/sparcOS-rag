from sparcos_rag.eval import hit_at_k, mrr


def test_hit_at_k_counts_presence_in_topk():
    data = [
        (["a.md", "b.md", "c.md"], "b.md"),
        (["x.md", "y.md", "z.md"], "b.md"),
    ]
    assert hit_at_k(data, k=3) == 0.5
    assert hit_at_k(data, k=1) == 0.0


def test_mrr_uses_reciprocal_of_rank():
    data = [
        (["a.md", "b.md", "c.md"], "b.md"),  # rank 2 -> 1/2
        (["b.md", "y.md", "z.md"], "b.md"),  # rank 1 -> 1/1
        (["x.md", "y.md", "z.md"], "b.md"),  # absent -> 0
    ]
    assert mrr(data) == (0.5 + 1.0 + 0.0) / 3


def test_mrr_empty_is_zero():
    assert mrr([]) == 0.0

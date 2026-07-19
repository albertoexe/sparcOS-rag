from sparcos_rag.eval import hit_at_k


def test_hit_at_k_counts_presence_in_topk():
    data = [
        (["a.md", "b.md", "c.md"], "b.md"),
        (["x.md", "y.md", "z.md"], "b.md"),
    ]
    assert hit_at_k(data, k=3) == 0.5
    assert hit_at_k(data, k=1) == 0.0

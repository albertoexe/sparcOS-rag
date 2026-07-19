from sparcos_rag.embedder import Embedder


class FakeOllama:
    def __init__(self):
        self.calls = []

    def embeddings(self, model, prompt):
        self.calls.append((model, prompt))
        return {"embedding": [float(len(prompt))] * 3}


def test_embed_returns_vector_per_text():
    fake = FakeOllama()
    emb = Embedder(model="bge-m3", host="x", client=fake)
    out = emb.embed(["ab", "abcd"])
    assert out == [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    assert emb.model_name == "bge-m3"
    assert fake.calls[0][0] == "bge-m3"

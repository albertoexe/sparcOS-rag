import pytest
from pathlib import Path
from sparcos_rag.indexer import index_vault, ModelMismatch

ROOT = Path(__file__).parent / "fixtures" / "vault"


class FakeEmbedder:
    model_name = "bge-m3"

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_indexes_then_skips_unchanged(store):
    emb = FakeEmbedder()
    r1 = index_vault(ROOT, emb, store)
    assert r1["indexed"] > 0
    r2 = index_vault(ROOT, emb, store)
    assert r2["indexed"] == 0 and r2["skipped"] > 0


def test_model_mismatch_raises(store):
    index_vault(ROOT, FakeEmbedder(), store)

    class Other(FakeEmbedder):
        model_name = "nemoretriever"

    with pytest.raises(ModelMismatch):
        index_vault(ROOT, Other(), store)

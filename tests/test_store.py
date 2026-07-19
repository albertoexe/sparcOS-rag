from sparcos_rag.store import StoredChunk


def _c(path, content, vec, h="h1"):
    return StoredChunk(path, "H", 0, content, vec, "bge-m3", h)


def test_upsert_and_vector_search(store):
    store.upsert_chunks([
        _c("a.md", "gatto nero", [1.0, 0.0, 0.0]),
        _c("b.md", "cane bianco", [0.0, 1.0, 0.0]),
    ])
    hits = store.vector_search([1.0, 0.0, 0.0], k=2)
    assert hits[0].source_path == "a.md"
    assert hits[0].rank == 1


def test_fulltext_search_exact_token(store):
    store.upsert_chunks([_c("a.md", "errore checkpoint stallo", [1.0, 0, 0])])
    hits = store.fulltext_search("checkpoint", k=5)
    assert hits[0].source_path == "a.md"


def test_indexed_model_and_hashes(store):
    store.upsert_chunks([_c("a.md", "x", [1.0, 0, 0], h="hash-a")])
    assert store.indexed_model() == "bge-m3"
    assert store.existing_hashes() == {"a.md": "hash-a"}


def test_delete_source(store):
    store.upsert_chunks([_c("a.md", "x", [1.0, 0, 0])])
    store.delete_source("a.md")
    assert store.existing_hashes() == {}

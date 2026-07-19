from pathlib import Path
from sparcos_rag.loader import load_vault
from sparcos_rag.chunker import chunk_document
from sparcos_rag.store import StoredChunk


class ModelMismatch(RuntimeError):
    pass


def index_vault(root: Path, embedder, store, batch_size: int = 64) -> dict:
    indexed_model = store.indexed_model()
    if indexed_model and indexed_model != embedder.model_name:
        raise ModelMismatch(
            f"index built with {indexed_model!r}, embedder is {embedder.model_name!r}")

    existing = store.existing_hashes()
    seen: set[str] = set()
    stats = {"indexed": 0, "skipped": 0, "deleted": 0}

    for doc in load_vault(root):
        seen.add(doc.source_path)
        if existing.get(doc.source_path) == doc.content_hash:
            stats["skipped"] += 1
            continue
        store.delete_source(doc.source_path)
        chunks = chunk_document(doc)
        if not chunks:
            continue
        vectors = embedder.embed([c.content for c in chunks])
        store.upsert_chunks([
            StoredChunk(c.source_path, c.heading, c.chunk_index, c.content,
                        v, embedder.model_name, doc.content_hash)
            for c, v in zip(chunks, vectors)])
        stats["indexed"] += 1

    for gone in set(existing) - seen:
        store.delete_source(gone)
        stats["deleted"] += 1
    return stats

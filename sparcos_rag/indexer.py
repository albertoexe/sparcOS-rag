from dataclasses import dataclass
from pathlib import Path
from sparcos_rag.loader import load_vault
from sparcos_rag.chunker import chunk_document
from sparcos_rag.store import StoredChunk


class ModelMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class StatusReport:
    """Diff between the vault on disk and what the index holds."""
    indexed: list[str]  # present in both, hash matches
    stale: list[str]    # present in both, hash differs (note edited)
    new: list[str]      # on disk, not indexed yet
    removed: list[str]  # indexed, no longer on disk

    def is_stale(self) -> bool:
        return bool(self.stale or self.new or self.removed)


def status_vault(root: Path, store) -> StatusReport:
    """Compare current vault hashes against the index without touching it.

    Mirrors index_vault's view of the vault (loader skips empty-body notes),
    so a clean report guarantees the next index would be a no-op.
    """
    existing = store.existing_hashes()
    seen: set[str] = set()
    indexed: list[str] = []
    stale: list[str] = []
    new: list[str] = []

    for doc in load_vault(root):
        seen.add(doc.source_path)
        stored_hash = existing.get(doc.source_path)
        if stored_hash is None:
            new.append(doc.source_path)
        elif stored_hash == doc.content_hash:
            indexed.append(doc.source_path)
        else:
            stale.append(doc.source_path)

    removed = sorted(set(existing) - seen)
    return StatusReport(sorted(indexed), sorted(stale), sorted(new), removed)


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

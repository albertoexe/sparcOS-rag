from pathlib import Path

from sparcos_rag.indexer import index_vault, status_vault
from sparcos_rag.store import StoredChunk

ROOT = Path(__file__).parent / "fixtures" / "vault"


class FakeEmbedder:
    model_name = "bge-m3"

    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_empty_index_reports_all_new(store):
    report = status_vault(ROOT, store)
    # fixtures/vault has two non-empty notes (empty.md is skipped by the loader)
    assert len(report.new) == 2
    assert report.indexed == [] and report.stale == [] and report.removed == []
    assert report.is_stale() is True


def test_after_index_report_is_clean(store):
    index_vault(ROOT, FakeEmbedder(), store)
    report = status_vault(ROOT, store)
    assert len(report.indexed) == 2
    assert report.stale == [] and report.new == [] and report.removed == []
    assert report.is_stale() is False


def test_changed_note_is_reported_stale(store):
    index_vault(ROOT, FakeEmbedder(), store)
    # simulate an edit: the stored hash no longer matches the file on disk
    store.conn.execute(
        "UPDATE chunks SET content_hash='stale-hash' WHERE source_path=%s",
        ("note-a.md",),
    )
    report = status_vault(ROOT, store)
    assert "note-a.md" in report.stale
    assert report.is_stale() is True


def test_note_only_in_db_is_reported_removed(store):
    index_vault(ROOT, FakeEmbedder(), store)
    store.upsert_chunks([
        StoredChunk("ghost.md", "H", 0, "gone from disk",
                    [1.0, 0.0, 0.0], "bge-m3", "ghost-hash"),
    ])
    report = status_vault(ROOT, store)
    assert "ghost.md" in report.removed
    assert report.is_stale() is True


def test_last_indexed_at_is_none_when_empty(store):
    assert store.last_indexed_at() is None


def test_last_indexed_at_set_after_index(store):
    index_vault(ROOT, FakeEmbedder(), store)
    assert store.last_indexed_at() is not None

from pathlib import Path
from sparcos_rag.loader import load_vault

ROOT = Path(__file__).parent / "fixtures" / "vault"


def test_loads_note_with_body():
    docs = {d.source_path: d for d in load_vault(ROOT)}
    a = docs["note-a.md"]
    assert a.frontmatter["title"] == "Nota A"
    assert "contenuto reale" in a.body
    assert len(a.content_hash) == 64


def test_skips_body_less_notes():
    paths = {d.source_path for d in load_vault(ROOT)}
    assert "empty.md" not in paths


def test_handles_unicode_paths():
    paths = {d.source_path for d in load_vault(ROOT)}
    assert "entità note.md" in paths

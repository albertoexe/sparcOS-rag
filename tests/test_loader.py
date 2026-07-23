from pathlib import Path
from sparcos_rag.loader import load_vault, DEFAULT_DENYLIST, _is_denied

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


def _write(root: Path, rel: str, text: str = "---\nx: 1\n---\nbody here"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_denylist_excludes_dirs_and_filenames(tmp_path):
    _write(tmp_path, "archives/wiki/entities/foo.md")
    _write(tmp_path, "archives/processed/old.md")
    _write(tmp_path, "archives/templates/tmpl.md")
    _write(tmp_path, "playground/proj/AGENTS.md")
    _write(tmp_path, "archives/wiki/index.md")
    _write(tmp_path, ".trash/gone.md")
    paths = {d.source_path for d in load_vault(tmp_path)}
    assert paths == {"archives/wiki/entities/foo.md"}


def test_denylist_keeps_planning_notes(tmp_path):
    rel = "playground/Second Brain Setup/00 Planning/roadmap.md"
    _write(tmp_path, rel)
    assert rel in {d.source_path for d in load_vault(tmp_path)}


def test_denylist_custom_dict_overrides_default(tmp_path):
    _write(tmp_path, "keep/a.md")
    _write(tmp_path, "drop/b.md")
    dl = {"dir_prefixes": ("drop/",), "filenames": ()}
    paths = {d.source_path for d in load_vault(tmp_path, denylist=dl)}
    assert paths == {"keep/a.md"}


def test_is_denied_matches_prefix_and_filename():
    assert _is_denied("archives/processed/x.md", DEFAULT_DENYLIST) is True
    assert _is_denied("a/b/AGENTS.md", DEFAULT_DENYLIST) is True
    assert _is_denied("archives/wiki/concepts/rag.md", DEFAULT_DENYLIST) is False


def test_strips_zero_width_and_bom(tmp_path):
    body = "he​llo﻿ wor‍ld"
    _write(tmp_path, "a.md", f"---\nx: 1\n---\n{body}")
    doc = next(iter(load_vault(tmp_path)))
    assert doc.body == "hello world"

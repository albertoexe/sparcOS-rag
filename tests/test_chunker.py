from sparcos_rag.loader import Document
from sparcos_rag.chunker import chunk_document


def _doc(body):
    return Document(source_path="n.md", frontmatter={}, body=body, content_hash="x")


def test_splits_by_heading():
    body = "# Uno\nAlpha.\n## Due\nBeta."
    chunks = chunk_document(_doc(body))
    assert [c.heading for c in chunks] == ["Uno", "Due"]
    assert chunks[0].content.strip().endswith("Alpha.")
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_long_section_is_split():
    body = "# Big\n" + ("parola " * 600)
    chunks = chunk_document(_doc(body), max_chars=1500)
    assert len(chunks) >= 2
    assert all(len(c.content) <= 1500 for c in chunks)

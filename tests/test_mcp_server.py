import asyncio
from sparcos_rag.indexer import StatusReport
from sparcos_rag.retriever import FusedHit
from sparcos_rag.mcp_server import format_hits, status_payload, mcp


def test_format_hits_shape():
    hits = [FusedHit("a.md", "H", "testo", 0.123456)]
    out = format_hits(hits)
    assert out == [
        {"source_path": "a.md", "heading": "H", "content": "testo", "score": 0.1235}
    ]


def test_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"search", "stats", "status"} <= names


def test_status_payload_counts_and_flag():
    report = StatusReport(indexed=["a.md"], stale=["b.md"], new=[], removed=[])
    out = status_payload(report, last_at=None)
    assert out["indexed"] == 1 and out["stale"] == 1
    assert out["new"] == 0 and out["removed"] == 0
    assert out["is_stale"] is True
    assert out["last_indexed_at"] is None

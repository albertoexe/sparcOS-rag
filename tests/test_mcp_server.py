import asyncio
from sparcos_rag.retriever import FusedHit
from sparcos_rag.mcp_server import format_hits, mcp


def test_format_hits_shape():
    hits = [FusedHit("a.md", "H", "testo", 0.123456)]
    out = format_hits(hits)
    assert out == [
        {"source_path": "a.md", "heading": "H", "content": "testo", "score": 0.1235}
    ]


def test_tools_registered():
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    assert {"search", "stats"} <= names

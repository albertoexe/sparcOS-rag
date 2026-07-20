"""MCP server exposing sparcOS RAG retrieval as LLM-free tools.

The tools return raw retrieved chunks. Synthesis is left to the MCP client
(Claudian / Claude Code), which is already authenticated — so no API key lives
in the engine. This is the "thin primitives" pattern: retrieval is a stable
tool, the agent is the orchestrator.
"""
import psycopg
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from sparcos_rag.config import load
from sparcos_rag.embedder import Embedder
from sparcos_rag.store import Store
from sparcos_rag.retriever import hybrid_search

load_dotenv()

mcp = FastMCP("sparcos-rag")

_state: dict = {}


def _deps() -> dict:
    """Lazily build Store + Embedder from config (once per process)."""
    if "store" not in _state:
        cfg = load()
        conn = psycopg.connect(cfg.database_url, autocommit=True)
        _state["store"] = Store(conn)
        _state["embedder"] = Embedder(cfg.embed_model, cfg.ollama_host)
        _state["top_k"] = cfg.top_k
    return _state


def format_hits(hits) -> list[dict]:
    """Shape retrieved chunks into a stable, JSON-friendly structure."""
    return [
        {
            "source_path": h.source_path,
            "heading": h.heading,
            "content": h.content,
            "score": round(h.score, 4),
        }
        for h in hits
    ]


@mcp.tool()
def search(query: str, top_k: int = 10) -> list[dict]:
    """Cerca nel second brain di Alberto via hybrid retrieval (vettoriale + full-text + RRF).

    Restituisce i chunk più rilevanti, ciascuno con la fonte (`source_path`), la sezione
    (`heading`), il testo (`content`) e lo score. NON genera una risposta: sintetizza tu dai
    chunk restituiti e **cita sempre `source_path`**. Se i chunk non contengono la risposta,
    di' esattamente "Non presente nel cervello."
    """
    d = _deps()
    hits = hybrid_search(query, d["embedder"], d["store"], top_k=top_k)
    return format_hits(hits)


@mcp.tool()
def stats() -> dict:
    """Stato dell'indice: numero di note, numero di chunk, modello di embedding usato."""
    d = _deps()
    row = d["store"].conn.execute(
        "SELECT count(DISTINCT source_path), count(*), max(model_name) FROM chunks"
    ).fetchone()
    return {"notes": row[0], "chunks": row[1], "model": row[2]}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

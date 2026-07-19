# sparcOS RAG

Hybrid-retrieval engine for the sparcOS line. Indexes an Obsidian vault into
Postgres and answers questions via vector + full-text search fused with
Reciprocal Rank Fusion, then Claude for synthesis.

> Design & rationale: `SecondBrain/playground/Second Brain Setup/00 Planning/2026-07-19-sparcos-rag-v1-design.md`

## Architecture

```
Loader -> Chunker -> Embedder (Ollama, local) -> Store (Postgres+pgvector)
Query: Embedder -> vector_search + fulltext_search -> RRF -> top-K -> Claude
```

The query is embedded by the SAME local model that built the index (enforced by
a model-name guard). Retrieval is pure Postgres (no model); Claude only writes
the final answer from the retrieved chunks.

## Setup

```bash
docker compose up -d                       # Postgres 16 + pgvector on :5433
ollama pull bge-m3                         # local embedder (1024 dim)
python -m venv .venv && ./.venv/Scripts/python -m pip install -e ".[dev]"
cp .env.example .env                       # fill ANTHROPIC_API_KEY, VAULT_PATH, ANSWER_MODEL
```

## Commands

```bash
sparcos-rag index                          # build/update the index (incremental)
sparcos-rag query "cos'è Budge It?"        # retrieve + answer (needs ANTHROPIC_API_KEY)
sparcos-rag evaluate --k 10                # hit@k over eval/questions.yaml
```

## Tests

```bash
pytest
```

Tests DROP/CREATE the `chunks` table, so they run against a **dedicated
`sparcos_rag_test` database** — never the real index DB (`sparcos_rag`). Override
with `TEST_DATABASE_URL` only if it points at a throwaway DB.

```bash
docker exec sparcos-rag-db psql -U sparcos -d sparcos_rag -c "CREATE DATABASE sparcos_rag_test;"
docker exec sparcos-rag-db psql -U sparcos -d sparcos_rag_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Notes

- Indexing on CPU with bge-m3 is ~1.5s/chunk (a full 677-note vault ≈ 2.5–3h).
  Query-time retrieval stays ~0.8s. GPU or a lighter embedder speeds up indexing.
- v2 (deferred): rerank, IDF, age-decay, distill-before-embed, MCP tool surface,
  local offline generator, SaaS packaging.

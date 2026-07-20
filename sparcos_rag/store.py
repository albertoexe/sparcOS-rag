from dataclasses import dataclass
from pgvector import Vector
from pgvector.psycopg import register_vector


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    source_path: str
    heading: str
    content: str
    score: float
    rank: int


@dataclass(frozen=True)
class StoredChunk:
    source_path: str
    heading: str
    chunk_index: int
    content: str
    embedding: list[float]
    model_name: str
    content_hash: str


class Store:
    def __init__(self, conn):
        self.conn = conn
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)

    def init_schema(self, dim: int) -> None:
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                source_path TEXT NOT NULL,
                heading TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dim}) NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
                model_name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT now()
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tsv ON chunks USING GIN (tsv)")

    def indexed_model(self) -> str | None:
        row = self.conn.execute("SELECT model_name FROM chunks LIMIT 1").fetchone()
        return row[0] if row else None

    def last_indexed_at(self):
        row = self.conn.execute("SELECT max(updated_at) FROM chunks").fetchone()
        return row[0] if row else None

    def existing_hashes(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT DISTINCT source_path, content_hash FROM chunks").fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_source(self, source_path: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE source_path=%s", (source_path,))

    def upsert_chunks(self, chunks: list[StoredChunk]) -> None:
        for c in chunks:
            self.conn.execute(
                """INSERT INTO chunks
                   (source_path, heading, chunk_index, content, embedding, model_name, content_hash)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                (c.source_path, c.heading, c.chunk_index, c.content,
                 Vector([float(x) for x in c.embedding]), c.model_name, c.content_hash))

    def vector_search(self, query_vec: list[float], k: int) -> list[SearchHit]:
        qv = Vector([float(x) for x in query_vec])
        rows = self.conn.execute(
            """SELECT id, source_path, heading, content,
                      1 - (embedding <=> %s) AS score
               FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
            (qv, qv, k)).fetchall()
        return [SearchHit(r[0], r[1], r[2], r[3], float(r[4]), i + 1)
                for i, r in enumerate(rows)]

    def fulltext_search(self, query_text: str, k: int) -> list[SearchHit]:
        rows = self.conn.execute(
            """SELECT id, source_path, heading, content,
                      ts_rank(tsv, plainto_tsquery('simple', %s)) AS score
               FROM chunks
               WHERE tsv @@ plainto_tsquery('simple', %s)
               ORDER BY score DESC LIMIT %s""",
            (query_text, query_text, k)).fetchall()
        return [SearchHit(r[0], r[1], r[2], r[3], float(r[4]), i + 1)
                for i, r in enumerate(rows)]

import os
import pytest
import psycopg
from sparcos_rag.store import Store

DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://sparcos:sparcos@localhost:5433/sparcos_rag",
)


@pytest.fixture
def store():
    conn = psycopg.connect(DSN, autocommit=True)
    s = Store(conn)
    conn.execute("DROP TABLE IF EXISTS chunks")
    s.init_schema(dim=3)
    yield s
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.close()

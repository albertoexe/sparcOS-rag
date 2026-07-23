from pathlib import Path
import pytest
from sparcos_rag.config import load


def test_load_reads_env():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "OLLAMA_HOST": "http://localhost:11434",
        "ANSWER_MODEL": "claude-x",
        "TOP_K": "10",
    }
    cfg = load(env)
    assert cfg.vault_path == Path("/tmp/vault")
    assert cfg.embed_dim == 1024
    assert cfg.embed_model == "bge-m3"


def test_load_missing_key_raises():
    with pytest.raises(KeyError):
        load({})


def test_denylist_defaults_when_env_absent():
    from sparcos_rag.loader import DEFAULT_DENYLIST
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "ANSWER_MODEL": "claude-x",
    }
    assert load(env).denylist == DEFAULT_DENYLIST


def test_denylist_env_parsed():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "ANSWER_MODEL": "claude-x",
        "INDEX_DENYLIST": "drop/,SKIP.md",
    }
    dl = load(env).denylist
    assert dl["dir_prefixes"] == ("drop/",)
    assert dl["filenames"] == ("SKIP.md",)

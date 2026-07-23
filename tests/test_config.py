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


def test_rerank_defaults_off():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "ANSWER_MODEL": "claude-x",
    }
    cfg = load(env)
    assert cfg.rerank_enabled is False
    assert cfg.rerank_model_path is None
    assert cfg.rerank_candidate_k == 20


def test_rerank_reads_env():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "DATABASE_URL": "postgresql://u:p@localhost/db",
        "VAULT_PATH": "/tmp/vault",
        "EMBED_MODEL": "bge-m3",
        "EMBED_DIM": "1024",
        "ANSWER_MODEL": "claude-x",
        "RERANK_ENABLED": "true",
        "RERANK_MODEL_PATH": "/models/qwen3-reranker.gguf",
        "RERANK_CANDIDATE_K": "30",
    }
    cfg = load(env)
    assert cfg.rerank_enabled is True
    assert cfg.rerank_model_path == "/models/qwen3-reranker.gguf"
    assert cfg.rerank_candidate_k == 30


def test_load_missing_key_raises():
    with pytest.raises(KeyError):
        load({})

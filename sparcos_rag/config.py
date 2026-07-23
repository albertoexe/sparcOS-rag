from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
import os


@dataclass(frozen=True)
class Config:
    vault_path: Path
    database_url: str
    embed_model: str
    embed_dim: int
    ollama_host: str
    anthropic_api_key: str
    answer_model: str
    top_k: int
    rerank_enabled: bool
    rerank_model_path: str | None
    rerank_candidate_k: int


def _as_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def load(env: Mapping[str, str] | None = None) -> Config:
    e = env if env is not None else os.environ
    return Config(
        vault_path=Path(e["VAULT_PATH"]),
        database_url=e["DATABASE_URL"],
        embed_model=e["EMBED_MODEL"],
        embed_dim=int(e["EMBED_DIM"]),
        ollama_host=e.get("OLLAMA_HOST", "http://localhost:11434"),
        anthropic_api_key=e["ANTHROPIC_API_KEY"],
        answer_model=e["ANSWER_MODEL"],
        top_k=int(e.get("TOP_K", "10")),
        rerank_enabled=_as_bool(e.get("RERANK_ENABLED", "false")),
        rerank_model_path=e.get("RERANK_MODEL_PATH") or None,
        rerank_candidate_k=int(e.get("RERANK_CANDIDATE_K", "20")),
    )

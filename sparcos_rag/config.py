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
    denylist: dict


def _parse_denylist(raw: str) -> dict:
    from sparcos_rag.loader import DEFAULT_DENYLIST
    if not raw.strip():
        return DEFAULT_DENYLIST
    dirs, files = [], []
    for entry in raw.split(","):
        e = entry.strip()
        if not e:
            continue
        (dirs if e.endswith("/") else files).append(e)
    return {"dir_prefixes": tuple(dirs), "filenames": tuple(files)}


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
        denylist=_parse_denylist(e.get("INDEX_DENYLIST", "")),
    )

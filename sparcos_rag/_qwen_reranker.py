"""Qwen3-Reranker-0.6B backend via llama-cpp-python (GGUF, no torch).

SCAFFOLD — not yet validated end-to-end against a real GGUF.

Optional dependency: ``pip install llama-cpp-python``.
Model: a GGUF build of Qwen3-Reranker-0.6B (Hugging Face). Point
``RERANK_MODEL_PATH`` at the ``.gguf`` file. This adapter is imported lazily
by ``reranker.Reranker`` so the rest of the system runs without it installed.

Contract: ``score(query, documents) -> list[float]`` (higher = more relevant).

Qwen3-Reranker judges relevance as a yes/no next-token decision. We build the
official instruct prompt per (query, document) pair and read the probability
mass on the "yes" token from the logits. The exact template/tokens must be
confirmed against the chosen GGUF before flipping RERANK_ENABLED on — see the
model card. Until then this path is behind the default-off config flag.
"""
from __future__ import annotations

_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_INSTRUCT = "Given a search query, retrieve relevant passages that answer the query"


def _format(query: str, document: str) -> str:
    return (
        f"{_PREFIX}<Instruct>: {_INSTRUCT}\n<Query>: {query}\n"
        f"<Document>: {document}{_SUFFIX}"
    )


class Qwen3RerankerBackend:
    def __init__(self, model_path: str | None, llama=None, n_ctx: int = 8192):
        if not model_path:
            raise ValueError(
                "RERANK_MODEL_PATH is empty. Set it to a Qwen3-Reranker GGUF file "
                "to enable reranking."
            )
        self.model_path = model_path
        if llama is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "llama-cpp-python is not installed. Run "
                    "`pip install llama-cpp-python` to use the reranker."
                ) from exc
            llama = Llama(model_path=model_path, n_ctx=n_ctx, logits_all=False, verbose=False)
        self._llama = llama

    def score(self, query: str, documents: list[str]) -> list[float]:
        # SCAFFOLD: compute P(yes) per (query, document) via the yes/no logit.
        # Left unimplemented until validated against a concrete GGUF so we never
        # ship a silently-wrong ranking. Enable by implementing the logit read
        # here and setting RERANK_ENABLED=true.
        raise NotImplementedError(
            "Qwen3RerankerBackend.score is scaffolded but not validated. "
            "Implement the yes-token logit read for your GGUF, or inject a "
            "tested backend into Reranker(backend=...)."
        )

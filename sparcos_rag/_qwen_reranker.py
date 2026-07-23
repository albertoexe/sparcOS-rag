"""Qwen3-Reranker-0.6B backend via llama-cpp-python (GGUF, no torch).

Optional dependency: ``pip install llama-cpp-python`` (CPU prebuilt wheels at
https://abetlen.github.io/llama-cpp-python/whl/cpu). Model: a GGUF build of
Qwen3-Reranker-0.6B (e.g. ``ggml-org/Qwen3-Reranker-0.6B``). Point
``RERANK_MODEL_PATH`` at the ``.gguf`` file. Imported lazily by
``reranker.Reranker`` so the rest of the system runs without it installed.

Contract: ``score(query, documents) -> list[float]`` (higher = more relevant).

How it works (validated 2026-07-23 against the q8_0 GGUF):
Qwen3-Reranker is a causal LM that judges relevance as a yes/no decision.
llama.cpp exposes this via RANK pooling: load with ``embedding=True`` and
``pooling_type=LLAMA_POOLING_TYPE_RANK``, feed the official instruct prompt for
each (query, document) pair, and ``embed()`` returns a single sigmoid score in
[0, 1]. Empirically a relevant pair scores ~0.999 and an unrelated one ~0.0,
so the input formatting below is load-bearing — do not "simplify" it.
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


def _scalar(value) -> float:
    """RANK pooling yields one score per sequence; unwrap any list nesting."""
    while isinstance(value, list):
        if not value:
            return 0.0
        value = value[0]
    return float(value)


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
                import llama_cpp
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "llama-cpp-python is not installed. Install the CPU wheel: "
                    "pip install llama-cpp-python --extra-index-url "
                    "https://abetlen.github.io/llama-cpp-python/whl/cpu"
                ) from exc
            llama = Llama(
                model_path=model_path,
                embedding=True,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_RANK,
                n_ctx=n_ctx,
                verbose=False,
            )
        self._llama = llama

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        prompts = [_format(query, d) for d in documents]
        raw = self._llama.embed(prompts)
        # embed(list) should return one result per prompt; if the shape is
        # unexpected, fall back to scoring each prompt individually.
        if isinstance(raw, list) and len(raw) == len(prompts):
            return [_scalar(r) for r in raw]
        return [_scalar(self._llama.embed(p)) for p in prompts]

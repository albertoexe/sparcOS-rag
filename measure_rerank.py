"""A/B measurement: hybrid retrieval with vs without the Qwen3 reranker.

Runs the 43-question neutral gold set through the same pipeline twice and
prints hit@1 / hit@5 / hit@10 / MRR. This is the "fatto quando" proof for P1.1.

Usage:
    ./.venv/Scripts/python.exe measure_rerank.py [questions.yaml]

Requires: Docker db + Ollama up, llama-cpp-python, and a Qwen3-Reranker GGUF
(RERANK_MODEL_PATH env, or the default qmd cache path below).
"""
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from sparcos_rag.config import load
from sparcos_rag.embedder import Embedder
from sparcos_rag.store import Store
from sparcos_rag.eval import run_eval, hit_at_k, mrr
from sparcos_rag.retriever import hybrid_search
from sparcos_rag.reranker import Reranker

import psycopg

DEFAULT_MODEL = str(
    Path.home() / ".cache/qmd/models/hf_ggml-org_qwen3-reranker-0.6b-q8_0.gguf"
)


def main():
    qfile = sys.argv[1] if len(sys.argv) > 1 else "eval/questions_neutral.yaml"
    model_path = os.environ.get("RERANK_MODEL_PATH") or DEFAULT_MODEL
    pool = int(os.environ.get("RERANK_CANDIDATE_K", "20"))

    load_dotenv()
    cfg = load()
    conn = psycopg.connect(cfg.database_url, autocommit=True)
    store = Store(conn)
    store.init_schema(cfg.embed_dim)
    embedder = Embedder(cfg.embed_model, cfg.ollama_host)

    with open(qfile, encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    print(f"gold set: {qfile}  (n={len(questions)})  embed={cfg.embed_model}  pool={pool}")
    print("running baseline (no reranker)...", flush=True)
    base = run_eval(questions, embedder, store, k=10)

    print("running reranked (Qwen3-Reranker-0.6B)...", flush=True)
    reranker = Reranker(model_path=model_path)
    data = []
    for i, item in enumerate(questions, start=1):
        hits = hybrid_search(
            item["q"], embedder, store, top_k=10,
            reranker=reranker, rerank_candidate_k=pool,
        )
        data.append(([h.source_path for h in hits], item["expect"]))
        print(f"  reranked {i}/{len(questions)}", flush=True)
    rer = {
        "hit_at_k": hit_at_k(data, 10),
        "hit@1": hit_at_k(data, 1),
        "hit@5": hit_at_k(data, 5),
        "mrr": mrr(data),
        "n": len(questions),
    }

    def row(name, r):
        return (
            f"{name:<12} hit@1={r['hit@1']:.3f}  hit@5={r['hit@5']:.3f}  "
            f"hit@10={r['hit_at_k']:.3f}  MRR={r['mrr']:.3f}"
        )

    print("\n=== RESULT ===")
    print(row("baseline", base))
    print(row("reranked", rer))
    print(
        f"\ndelta        hit@1={rer['hit@1']-base['hit@1']:+.3f}  "
        f"hit@5={rer['hit@5']-base['hit@5']:+.3f}  "
        f"hit@10={rer['hit_at_k']-base['hit_at_k']:+.3f}  "
        f"MRR={rer['mrr']-base['mrr']:+.3f}"
    )
    sys.stdout.flush()
    # Avoid llama-cpp native GC segfault on Windows interpreter teardown.
    os._exit(0)


if __name__ == "__main__":
    main()

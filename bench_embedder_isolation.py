"""Isolate the EMBEDDER: sparcOS vector-only (bge-m3) vs qmd vsearch (EmbeddingGemma-300M).

Strips away reranker / query-expansion / BM25 / RRF. Only the vector space + chunking differ.
Run from sparcOS-rag repo root with its venv python.
"""
import json
import subprocess
import sys
import time
import shutil

import yaml
from dotenv import load_dotenv

from sparcos_rag.config import load
from sparcos_rag.embedder import Embedder
from sparcos_rag.store import Store
import psycopg

load_dotenv()
K = 10
QMD = shutil.which("qmd") or r"C:\Users\AlbertoDeCol\AppData\Roaming\npm\qmd.cmd"


def norm(p):
    p = (p or "").strip()
    if p.startswith("qmd://"):
        p = p[len("qmd://"):]
        p = p.split("/", 1)[1] if "/" in p else p
    return p.replace("\\", "/").replace(" ", "-").lower()


def dedup(paths):
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def sparcos_vec(q, embedder, store):
    t = time.perf_counter()
    qvec = embedder.embed([q])[0]
    hits = store.vector_search(qvec, 30)  # over-fetch, dedup to files
    dt = time.perf_counter() - t
    return dedup([norm(h.source_path) for h in hits])[:K], dt


def qmd_vec(q):
    t = time.perf_counter()
    out = subprocess.run([QMD, "vsearch", q, "-n", str(K), "--json"],
                         capture_output=True, encoding="utf-8", errors="replace", shell=False)
    dt = time.perf_counter() - t
    try:
        data = json.loads(out.stdout)
        return dedup([norm(r["file"]) for r in data])[:K], dt
    except Exception:
        return [], dt


def metrics(rows):
    n = len(rows)
    h = lambda k: sum(1 for r, e in rows if e in r[:k]) / n
    mrr = sum((1.0 / (r.index(e) + 1)) for r, e in rows if e in r) / n
    return h(1), h(5), h(10), mrr


def main():
    cfg = load()
    conn = psycopg.connect(cfg.database_url, autocommit=True)
    store = Store(conn); store.init_schema(cfg.embed_dim)
    embedder = Embedder(cfg.embed_model, cfg.ollama_host)
    qfile = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.yaml"
    with open(qfile, encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    sp_rows, qq_rows = [], []
    print(f"{'#':<3}{'question':<50}{'bge-m3':<9}{'Gemma-300M':<11}")
    for i, item in enumerate(questions, 1):
        q, exp = item["q"], norm(item["expect"])
        sp, _ = sparcos_vec(q, embedder, store)
        qq, _ = qmd_vec(q)
        sp_rows.append((sp, exp)); qq_rows.append((qq, exp))
        rk = lambda r: str(r.index(exp) + 1) if exp in r else "-"
        print(f"{i:<3}{q[:48]:<50}{rk(sp):<9}{rk(qq):<11}")

    for name, rows in [("sparcOS vector-only (bge-m3)", sp_rows),
                       ("qmd vsearch (EmbeddingGemma-300M)", qq_rows)]:
        h1, h5, h10, mrr = metrics(rows)
        print(f"{name:<36} hit@1={h1:.2f} hit@5={h5:.2f} hit@10={h10:.2f} MRR={mrr:.3f}")


if __name__ == "__main__":
    main()

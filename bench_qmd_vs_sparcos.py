"""Smoke benchmark: qmd (tobi) vs sparcOS RAG on the SAME eval set + SAME scoring.

Run with the sparcOS venv python from the repo root:
    ./.venv/Scripts/python.exe bench_qmd_vs_sparcos.py

Requires: Postgres (docker) + Ollama up for sparcOS; qmd installed + embedded for qmd.
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
from sparcos_rag.retriever import hybrid_search
import psycopg

load_dotenv()

K = 10
QMD = shutil.which("qmd") or r"C:\Users\AlbertoDeCol\AppData\Roaming\npm\qmd.cmd"


def norm(p: str) -> str:
    p = (p or "").strip()
    if p.startswith("qmd://"):
        p = p[len("qmd://"):]
        p = p.split("/", 1)[1] if "/" in p else p  # drop collection name
    p = p.replace("\\", "/").replace(" ", "-").lower()
    return p


def sparcos_paths(q, embedder, store):
    t = time.perf_counter()
    hits = hybrid_search(q, embedder, store, top_k=K)
    dt = time.perf_counter() - t
    return [norm(h.source_path) for h in hits], dt


def qmd_paths(q, mode="query"):
    t = time.perf_counter()
    out = subprocess.run([QMD, mode, q, "-n", str(K), "--json"],
                         capture_output=True, encoding="utf-8", errors="replace", shell=False)
    dt = time.perf_counter() - t
    try:
        data = json.loads(out.stdout)
        paths = [norm(r["file"]) for r in data]
    except Exception:
        paths = []
    return paths, dt


def metrics(rows):
    # rows: list of (ranked_paths, expected_norm)
    n = len(rows)
    def hit_at(k):
        return sum(1 for r, e in rows if e in r[:k]) / n
    def mrr():
        s = 0.0
        for r, e in rows:
            if e in r:
                s += 1.0 / (r.index(e) + 1)
        return s / n
    return {"hit@1": hit_at(1), "hit@5": hit_at(5), "hit@10": hit_at(10), "mrr": mrr()}


def main():
    cfg = load()
    conn = psycopg.connect(cfg.database_url, autocommit=True)
    store = Store(conn)
    store.init_schema(cfg.embed_dim)
    embedder = Embedder(cfg.embed_model, cfg.ollama_host)

    qfile = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.yaml"
    with open(qfile, encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    sp_rows, qq_rows, qs_rows = [], [], []
    sp_lat, qq_lat, qs_lat = [], [], []

    print(f"{'#':<3}{'question':<52}{'sparcOS':<9}{'qmd-query':<11}{'qmd-bm25':<10}")
    for i, item in enumerate(questions, 1):
        q, exp = item["q"], norm(item["expect"])
        sp, spd = sparcos_paths(q, embedder, store)
        qq, qqd = qmd_paths(q, "query")
        qs, qsd = qmd_paths(q, "search")
        sp_rows.append((sp, exp)); qq_rows.append((qq, exp)); qs_rows.append((qs, exp))
        sp_lat.append(spd); qq_lat.append(qqd); qs_lat.append(qsd)

        def rank(r):
            return str(r.index(exp) + 1) if exp in r else "-"
        print(f"{i:<3}{q[:50]:<52}{rank(sp):<9}{rank(qq):<11}{rank(qs):<10}")

    def avg(x):
        return sum(x) / len(x)

    print("\n=== METRICS (rank of expected doc; '-' = not in top-10) ===")
    for name, rows, lat in [("sparcOS (hybrid)", sp_rows, sp_lat),
                            ("qmd query (hybrid+rerank)", qq_rows, qq_lat),
                            ("qmd search (BM25 only)", qs_rows, qs_lat)]:
        m = metrics(rows)
        print(f"{name:<28} hit@1={m['hit@1']:.2f} hit@5={m['hit@5']:.2f} "
              f"hit@10={m['hit@10']:.2f} MRR={m['mrr']:.3f} "
              f"lat_avg={avg(lat):.2f}s")

    with open("bench_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "sparcos": {"rows": sp_rows, "lat": sp_lat, "metrics": metrics(sp_rows)},
            "qmd_query": {"rows": qq_rows, "lat": qq_lat, "metrics": metrics(qq_rows)},
            "qmd_search": {"rows": qs_rows, "lat": qs_lat, "metrics": metrics(qs_rows)},
        }, f, indent=2, ensure_ascii=False)
    print("\nwrote bench_results.json")


if __name__ == "__main__":
    main()

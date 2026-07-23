import click
import psycopg
from dotenv import load_dotenv
from sparcos_rag.config import load
from sparcos_rag.embedder import Embedder
from sparcos_rag.store import Store
from sparcos_rag.indexer import index_vault, status_vault, StatusReport
from sparcos_rag.retriever import hybrid_search
from sparcos_rag.answerer import answer

load_dotenv()


def _deps(cfg):
    conn = psycopg.connect(cfg.database_url, autocommit=True)
    store = Store(conn)
    store.init_schema(cfg.embed_dim)
    embedder = Embedder(cfg.embed_model, cfg.ollama_host)
    return store, embedder


@click.group()
def cli():
    pass


@cli.command()
def index():
    cfg = load()
    store, embedder = _deps(cfg)
    stats = index_vault(cfg.vault_path, embedder, store, denylist=cfg.denylist)
    click.echo(f"indexed={stats['indexed']} skipped={stats['skipped']} deleted={stats['deleted']}")


def render_status(report: StatusReport, last_at, verbose: bool) -> str:
    lines = [
        f"indicizzate={len(report.indexed)} stale={len(report.stale)} "
        f"nuove={len(report.new)} rimosse={len(report.removed)}"
    ]
    if last_at is not None:
        lines.append(f"ultimo index: {last_at:%Y-%m-%d %H:%M}")
    if verbose:
        for label, paths in (("stale", report.stale), ("nuove", report.new),
                             ("rimosse", report.removed)):
            for p in paths:
                lines.append(f"  {label}: {p}")
    verdict = "STALE -> serve reindex" if report.is_stale() else "pulito"
    lines.append(verdict)
    return "\n".join(lines)


@cli.command()
@click.option("--verbose", is_flag=True, help="elenca gli slug per categoria")
def status(verbose):
    cfg = load()
    store, _ = _deps(cfg)
    report = status_vault(cfg.vault_path, store, denylist=cfg.denylist)
    click.echo(render_status(report, store.last_indexed_at(), verbose))
    raise SystemExit(1 if report.is_stale() else 0)


@cli.command()
@click.argument("text")
def query(text):
    import anthropic
    cfg = load()
    store, embedder = _deps(cfg)
    hits = hybrid_search(text, embedder, store, top_k=cfg.top_k)
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    ans = answer(text, hits, client, model=cfg.answer_model)
    click.echo(ans.text)
    if ans.citations:
        click.echo("\nFonti: " + ", ".join(ans.citations))


@cli.command()
@click.option("--k", default=10)
def evaluate(k):
    import yaml
    from sparcos_rag.eval import run_eval
    cfg = load()
    store, embedder = _deps(cfg)
    with open("eval/questions.yaml", encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    result = run_eval(questions, embedder, store, k)
    click.echo(f"model={cfg.embed_model} hit@{k}={result['hit_at_k']:.2f} n={result['n']}")

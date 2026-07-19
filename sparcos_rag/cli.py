import click
import psycopg
from dotenv import load_dotenv
from sparcos_rag.config import load
from sparcos_rag.embedder import Embedder
from sparcos_rag.store import Store
from sparcos_rag.indexer import index_vault
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
    stats = index_vault(cfg.vault_path, embedder, store)
    click.echo(f"indexed={stats['indexed']} skipped={stats['skipped']} deleted={stats['deleted']}")


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

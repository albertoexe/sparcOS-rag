from click.testing import CliRunner
from sparcos_rag.cli import cli, render_status
from sparcos_rag.indexer import StatusReport


def test_cli_has_index_and_query():
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "index" in r.output and "query" in r.output


def test_cli_has_status():
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "status" in r.output


def test_render_status_clean_summary():
    report = StatusReport(indexed=["a.md", "b.md"], stale=[], new=[], removed=[])
    out = render_status(report, last_at=None, verbose=False)
    assert "indicizzate=2" in out
    assert "stale=0" in out and "nuove=0" in out and "rimosse=0" in out


def test_render_status_verbose_lists_changed_slugs():
    report = StatusReport(indexed=[], stale=["note-a.md"], new=["note-b.md"], removed=["ghost.md"])
    out = render_status(report, last_at=None, verbose=True)
    assert "note-a.md" in out and "note-b.md" in out and "ghost.md" in out

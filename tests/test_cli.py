from click.testing import CliRunner
from sparcos_rag.cli import cli


def test_cli_has_index_and_query():
    r = CliRunner().invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "index" in r.output and "query" in r.output

from pathlib import Path

from typer.testing import CliRunner

from scTenifold.__main__ import app

runner = CliRunner()


def test_config():
    result = runner.invoke(app, ["config", "-t", 1, "-p", "./net_config.yml"])
    assert Path("./net_config.yml").is_file()
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "-t", 2, "-p", "./knk_config.yml"])
    assert Path("./knk_config.yml").is_file()
    assert result.exit_code == 0
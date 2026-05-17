"""``xct`` / ``xct-merge`` CLI behaviour when the ``[xct]`` extra is absent.

Covers the ``ImportError`` -> ``typer.BadParameter`` branch in the
``build_xct`` / ``build_xct_merge`` commands of ``scTenifold/__main__.py``:
without scTenifoldXct the CLI must fail with a non-zero exit code and a
message pointing at ``pip install scTenifoldpy[xct]``. Unreachable in the
xct integration job, so it runs in the default torch-free matrix.
"""
import importlib.util

import pytest
from typer.testing import CliRunner

from scTenifold.__main__ import app

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("scTenifoldXct") is not None,
    reason="covers the xct-absent CLI branch; scTenifoldXct is installed",
)

runner = CliRunner()


def _result_text(result):
    """Collect every stream a Click error message might land on.

    typer renders errors in a Rich panel that word-wraps at the terminal
    width, so whitespace is collapsed to keep substring checks
    independent of where the wrap lands.
    """
    parts = [result.output, str(result.exception)]
    try:  # Click < 8.2 mixes stderr into output; >= 8.2 keeps it separate
        parts.append(result.stderr)
    except (ValueError, AttributeError):
        pass
    return " ".join(" ".join(p for p in parts if p).split())


def test_cli_xct_without_extra():
    result = runner.invoke(app, ["xct", "dummy.h5ad"])
    assert result.exit_code != 0
    assert "scTenifoldpy[xct]" in _result_text(result)


def test_cli_xct_merge_without_extra():
    result = runner.invoke(
        app, ["xct-merge", "dummy.h5ad", "condition", "WT", "KO"]
    )
    assert result.exit_code != 0
    assert "scTenifoldpy[xct]" in _result_text(result)

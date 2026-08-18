"""Tests for the ``sctenifold-ui`` CLI entry point.

Skipped automatically when the optional ``[ui]`` extra is absent, so the
default CI matrix stays green without fastapi/uvicorn. uvicorn.run is
mocked throughout so these tests never actually bind a port.
"""
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from scTenifold.webapp import cli


def test_no_browser_skips_webbrowser_open():
    with patch("uvicorn.run") as mock_run, patch("webbrowser.open") as mock_open:
        cli.main(["--no-browser", "--port", "8123"])

    mock_open.assert_not_called()
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123


def test_opens_browser_by_default():
    with patch("uvicorn.run") as mock_run, patch("webbrowser.open") as mock_open, \
         patch("threading.Thread") as mock_thread:
        cli.main(["--port", "8124"])

    mock_run.assert_called_once()
    mock_thread.assert_called_once()
    mock_open.assert_not_called()  # the (mocked) thread was never actually started/run


def test_main_module_delegates_to_cli_main():
    with patch.object(cli, "main") as mock_main:
        import runpy

        runpy.run_module("scTenifold.webapp.__main__", run_name="__main__")

    mock_main.assert_called_once_with()

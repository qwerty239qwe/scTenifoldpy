"""``sctenifold-ui`` console entry point: launch the local web UI in a browser."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from typing import Optional


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(prog="sctenifold-ui", description="Run the local scTenifoldpy web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "the local UI needs the 'ui' extra: pip install \"scTenifoldpy[ui]\""
        ) from exc

    from .main import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        def _open_browser():
            time.sleep(1.0)  # give uvicorn a moment to start listening
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    print(f"scTenifoldpy UI: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

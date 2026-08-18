"""Local web UI for scTenifoldpy.

Install with the ``ui`` extra and launch:

    pip install "scTenifoldpy[ui]"
    sctenifold-ui

Serves a single-page app (static/) backed by a small FastAPI JSON API
(main.py) that runs scTenifoldNet / scTenifoldKnk (scTenifold.core) as
background jobs.
"""

from .main import create_app

__all__ = ["create_app"]

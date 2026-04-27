"""Inference plugin entry point."""
from __future__ import annotations

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from .api import make_router

__all__ = ["register"]

PLUGIN_DIR = Path(__file__).resolve().parent


def section_context(ctx, request) -> dict:
    """No render-time data needed — section.html is fully populated by JS
    polling /api/inference/events."""
    return {}


def register(app, ctx) -> None:
    app.include_router(make_router(ctx))
    static_dir = PLUGIN_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static/inference", StaticFiles(directory=str(static_dir)), name="static-inference")


# No render_paths — the inference plugin doesn't add mediamtx paths
# (an annotated stream from the external producer arrives via mediamtx
# publish, not via our config).

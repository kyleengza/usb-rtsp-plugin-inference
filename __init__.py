"""Inference plugin entry point."""
from __future__ import annotations

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from .api import make_router
from . import jobs as jobs_mod
from . import models as models_mod
from . import render as render_mod

__all__ = ["register", "render_paths", "section_context"]

PLUGIN_DIR = Path(__file__).resolve().parent


def register(app, ctx) -> None:
    app.include_router(make_router(ctx))
    static_dir = PLUGIN_DIR / "static"
    if static_dir.is_dir():
        app.mount(
            "/static/inference",
            StaticFiles(directory=str(static_dir)),
            name="static-inference",
        )


def render_paths(ctx) -> dict:
    """Called by core/renderer at startup. Slice 1: returns {} until the
    Hailo + CPU backends land in subsequent slices."""
    return render_mod.build_paths(ctx)


def section_context(ctx, request) -> dict:
    """Dashboard render-time context. The card itself is populated by JS
    polling /api/inference/state; we pre-render the job list server-side
    too so the empty state shows up without a flash."""
    return {
        "inference_jobs": [jobs_mod.job_to_public_dict(j) for j in jobs_mod.list_jobs(ctx)],
        "inference_backends": {
            "hailo": models_mod.has_backend("hailo"),
            "cpu": models_mod.has_backend("cpu"),
        },
    }

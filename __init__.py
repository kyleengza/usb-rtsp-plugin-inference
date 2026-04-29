"""Inference plugin entry point."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from .api import make_router
from . import jobs as jobs_mod
from . import models as models_mod
from . import plugin_config
from . import render as render_mod

__all__ = ["register", "render_paths", "section_context", "live_paths_state"]

PLUGIN_DIR = Path(__file__).resolve().parent


def live_paths_state() -> dict:
    """Snapshot of mediamtx /v3/paths/list keyed by path name. Best-effort
    — returns {} if mediamtx is unreachable. Same pattern the relay
    plugin uses; consumed by section_context (server-side first paint)
    and api.state (JS poll)."""
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:9997/v3/paths/list", timeout=2,
        ) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError):
        return {}
    out: dict = {}
    for p in data.get("items", []):
        out[p.get("name")] = p
    return out


def _live_for_job(job_name: str, live: dict) -> dict | None:
    p = live.get(job_name)
    if not p:
        return None
    tracks = p.get("tracks") or []
    return {
        "ready": bool(p.get("ready") or p.get("sourceReady")),
        "bytes_received": p.get("bytesReceived", 0),
        "tracks": ", ".join(tracks),
        "readers": len(p.get("readers") or []),
    }


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
    """Dashboard render-time context. Pre-renders the job list with
    mediamtx live state so first paint shows correct status pills
    without a JS roundtrip; subsequent polls keep them fresh."""
    live = live_paths_state()
    enriched = []
    for j in jobs_mod.list_jobs(ctx):
        item = jobs_mod.job_to_public_dict(j)
        item["_live"] = _live_for_job(j.name, live)
        enriched.append(item)
    return {
        "inference_jobs": enriched,
        "inference_backends": {
            "hailo": models_mod.has_backend("hailo"),
            "cpu": models_mod.has_backend("cpu"),
        },
        "inference_coco_labels": models_mod.all_labels("coco"),
        "inference_clips_root": str(plugin_config.clips_root(ctx)),
    }

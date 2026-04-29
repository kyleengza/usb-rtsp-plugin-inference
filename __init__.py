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


def _live_for_job(job_name: str, live: dict, upstream_url: str = "") -> dict | None:
    p = live.get(job_name)
    if not p:
        return None
    tracks = p.get("tracks") or []
    # If the job's upstream is a local mediamtx path, surface whether
    # it's currently configured. When a USB cam or relay source is
    # disabled, the renderer drops it from mediamtx.yml and it
    # disappears from /v3/paths/list — so the inference job has
    # nothing to read from. Tell the dashboard so the status pill
    # can be honest about it.
    upstream_missing = False
    if upstream_url:
        prefix = "rtsp://127.0.0.1:8554/"
        if upstream_url.startswith(prefix):
            src_name = upstream_url[len(prefix):].rstrip("/")
            if src_name and src_name not in live:
                upstream_missing = True
    return {
        "ready": bool(p.get("ready") or p.get("sourceReady")),
        "bytes_received": p.get("bytesReceived", 0),
        "tracks": ", ".join(tracks),
        "readers": len(p.get("readers") or []),
        "upstream_missing": upstream_missing,
    }


def list_inference_sources(ctx) -> list[dict]:
    """Every mediamtx path that's a *source* (not itself an inference
    output) plus whether inference is currently turned on for it.
    Lets the settings page show one-click toggles for cam0 / mypc /
    relay sources without making the user hand-build job specs."""
    jobs_list = jobs_mod.list_jobs(ctx)
    inference_path_names = {j.name for j in jobs_list}
    # Map upstream URL → existing job (so toggle reflects ON when a
    # job exists, even if its name doesn't follow the <src>-ai pattern).
    by_upstream = {j.upstream: j for j in jobs_list}
    out: list[dict] = []
    for name, p in live_paths_state().items():
        if name in inference_path_names:
            continue  # it's an inference output, not a candidate source
        upstream = f"rtsp://127.0.0.1:8554/{name}"
        existing = by_upstream.get(upstream)
        out.append({
            "source": name,
            "upstream": upstream,
            "ready": bool(p.get("ready") or p.get("sourceReady")),
            "has_inference": existing is not None,
            "job_name": existing.name if existing else f"{name}-ai",
            "backend": existing.backend if existing else None,
            "model": existing.model if existing else None,
        })
    out.sort(key=lambda x: x["source"])
    return out


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
        item["_live"] = _live_for_job(j.name, live, j.upstream)
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

"""Build mediamtx path entries for each enabled inference job.

Each enabled job becomes a `source: publisher` path whose runOnDemand
spawns the in-plugin Python worker. The worker reads frames from
upstream, runs inference (slice 3+), and publishes annotated frames
to rtsp://127.0.0.1:$RTSP_PORT/$MTX_PATH.
"""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from . import jobs as jobs_mod
from . import models as models_mod

PLUGIN_DIR = Path(__file__).resolve().parent
WORKER_PATH = PLUGIN_DIR / "worker.py"


def _resolve_model_path(backend: str, model_name: str) -> str:
    m = models_mod.find_model(backend, model_name)
    if not m:
        return ""
    if backend == "hailo":
        return str(m.hef)  # type: ignore[union-attr]
    return str(m.onnx)  # type: ignore[union-attr]


def _build_runondemand(j: jobs_mod.Job) -> str:
    model_path = _resolve_model_path(j.backend, j.model)
    parts = [
        "/usr/bin/python3",
        str(WORKER_PATH),
        "--upstream", j.upstream,
        "--output", "rtsp://127.0.0.1:$RTSP_PORT/$MTX_PATH",
        "--backend", j.backend,
        "--model", j.model,
        "--model-path", model_path,
        "--job-name", j.name,
        "--threshold", f"{j.threshold:.3f}",
        "--inference-queue", str(j.inference_queue),
        "--track-occlusion-s", f"{j.track_occlusion_s:.2f}",
    ]
    if j.classes:
        parts.extend(["--classes", ",".join(j.classes)])
    if j.clips.enabled:
        parts.append("--clips-enabled")
        parts.extend(["--clip-post-roll-s", f"{j.clips.post_roll_s:.2f}"])
        parts.extend(["--clip-trigger", j.clips.trigger])
        parts.extend(["--clip-retention", str(j.clips.retention_count)])
        if j.clips.trigger_classes:
            parts.extend(["--clip-trigger-classes", ",".join(j.clips.trigger_classes)])
    return " ".join(shlex.quote(p) for p in parts)


def build_paths(ctx) -> dict[str, Any]:
    paths: dict[str, Any] = {}
    for j in jobs_mod.list_jobs(ctx):
        if not j.enabled:
            continue
        if not models_mod.find_model(j.backend, j.model):
            # Backend not available on this host — skip silently; the job
            # remains in YAML but produces no path. UI surfaces this via
            # the backend-availability badge.
            continue
        paths[j.name] = {
            "source": "publisher",
            "runOnDemand": _build_runondemand(j),
            # Hailo failures are usually transient (busy device, model
            # reload); CPU failures are usually persistent (bad ONNX,
            # missing class). Restart accordingly.
            "runOnDemandRestart": j.backend == "hailo",
            "runOnDemandStartTimeout": "20s",
            "runOnDemandCloseAfter": "10s",
        }
    return paths

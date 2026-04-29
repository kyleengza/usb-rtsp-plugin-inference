"""Inference plugin REST endpoints (mounted at /api/inference)."""
from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse

from core.helpers import REPO_DIR, systemctl

from . import jobs as jobs_mod
from . import models as models_mod
from . import events as events_mod
from . import clips as clips_mod


def _rerender_and_restart() -> dict[str, Any]:
    """Re-emit mediamtx.yml and bounce the mediamtx service so it picks
    up new/changed paths. Mirrors the relay plugin's save flow."""
    p = subprocess.run(
        ["python3", "-m", "core.renderer"],
        cwd=str(REPO_DIR),
        capture_output=True, text=True, timeout=15,
    )
    if p.returncode != 0:
        return {"render": "failed", "render_err": (p.stdout + p.stderr).strip()}
    code, _ = systemctl("restart", "usb-rtsp")
    return {"render": "ok", "restart": "ok" if code == 0 else "failed"}


class ClipsIn(BaseModel):
    enabled: bool = True
    pre_roll_s: int = Field(default=5, ge=0, le=600)
    post_roll_s: int = Field(default=10, ge=0, le=600)
    trigger: str = "track_enter"
    trigger_classes: list[str] = []
    retention_count: int = Field(default=100, ge=1, le=10000)


class JobIn(BaseModel):
    name: str
    upstream: str
    enabled: bool = True
    backend: str = "hailo"
    model: str = "yolov8s"
    classes: list[str] = []
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    inference_queue: int = Field(default=5, ge=0, le=60)
    track_occlusion_s: float = Field(default=2.0, ge=0.0, le=60.0)
    clips: ClipsIn = ClipsIn()


def _to_job(j: JobIn) -> jobs_mod.Job:
    return jobs_mod.Job(
        name=j.name,
        upstream=j.upstream,
        enabled=j.enabled,
        backend=j.backend,
        model=j.model,
        classes=list(j.classes),
        threshold=j.threshold,
        inference_queue=j.inference_queue,
        track_occlusion_s=j.track_occlusion_s,
        clips=jobs_mod.ClipsConfig(
            enabled=j.clips.enabled,
            pre_roll_s=j.clips.pre_roll_s,
            post_roll_s=j.clips.post_roll_s,
            trigger=j.clips.trigger,
            trigger_classes=list(j.clips.trigger_classes),
            retention_count=j.clips.retention_count,
        ),
    )


def make_router(ctx) -> APIRouter:
    r = APIRouter(prefix="/api/inference", tags=["inference"])

    @r.get("/state")
    def state() -> dict[str, Any]:
        # Local import to avoid a circular at module load — __init__.py
        # imports api.make_router.
        from . import live_paths_state, _live_for_job
        live = live_paths_state()
        out_jobs = []
        for j in jobs_mod.list_jobs(ctx):
            item = jobs_mod.job_to_public_dict(j)
            item["_live"] = _live_for_job(j.name, live)
            out_jobs.append(item)
        return {
            "jobs": out_jobs,
            "backends": {
                "hailo": models_mod.has_backend("hailo"),
                "cpu": models_mod.has_backend("cpu"),
            },
        }

    @r.get("/models")
    def list_models() -> dict[str, Any]:
        return {
            "hailo": [
                {"name": m.name, "fps_target": m.fps_target, "labels_count": len(m.labels)}
                for m in models_mod.hailo_models()
            ],
            "cpu": [
                {"name": m.name, "fps_target": m.fps_target, "labels_count": len(m.labels)}
                for m in models_mod.cpu_models()
            ],
            "labels_coco": models_mod.all_labels("coco"),
        }

    @r.get("/jobs")
    def get_jobs() -> list[dict[str, Any]]:
        return [jobs_mod.job_to_public_dict(j) for j in jobs_mod.list_jobs(ctx)]

    @r.get("/jobs/{name}")
    def get_one(name: str) -> dict[str, Any]:
        j = jobs_mod.get_job(ctx, name)
        if not j:
            raise HTTPException(404, f"no such job: {name}")
        return jobs_mod.job_to_public_dict(j)

    @r.post("/jobs", status_code=201)
    def create(payload: JobIn) -> dict[str, Any]:
        try:
            saved = jobs_mod.add_job(ctx, _to_job(payload))
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        return {"job": jobs_mod.job_to_public_dict(saved), **_rerender_and_restart()}

    @r.put("/jobs/{name}")
    def update(name: str, payload: JobIn) -> dict[str, Any]:
        try:
            saved = jobs_mod.update_job(ctx, name, _to_job(payload))
        except KeyError:
            raise HTTPException(404, f"no such job: {name}")
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        return {"job": jobs_mod.job_to_public_dict(saved), **_rerender_and_restart()}

    @r.delete("/jobs/{name}")
    def remove(name: str) -> dict[str, Any]:
        if not jobs_mod.delete_job(ctx, name):
            raise HTTPException(404, f"no such job: {name}")
        return {"ok": True, **_rerender_and_restart()}

    @r.get("/jobs/{name}/events")
    def job_events(name: str, n: int = 100) -> dict[str, Any]:
        if not jobs_mod.get_job(ctx, name):
            raise HTTPException(404, f"no such job: {name}")
        n = max(1, min(int(n), 500))
        return {"events": events_mod.read_recent(name, n=n)}

    @r.get("/jobs/{name}/clips")
    def list_clips(name: str) -> dict[str, Any]:
        if not jobs_mod.get_job(ctx, name):
            raise HTTPException(404, f"no such job: {name}")
        return {
            "clips": clips_mod.list_clips(name),
            "total_size_bytes": clips_mod.clips_total_size(name),
        }

    @r.get("/clips/{name}/{file_name}")
    def download_clip(name: str, file_name: str):
        p = clips_mod.clip_path(name, file_name)
        if not p:
            raise HTTPException(404, "no such clip")
        return FileResponse(str(p), media_type="video/mp4", filename=file_name)

    @r.delete("/clips/{name}/{file_name}")
    def remove_clip(name: str, file_name: str) -> dict[str, Any]:
        if not clips_mod.delete_clip(name, file_name):
            raise HTTPException(404, "no such clip")
        return {"ok": True}

    return r

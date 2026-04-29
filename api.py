"""Inference plugin REST endpoints (mounted at /api/inference)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import jobs as jobs_mod
from . import models as models_mod


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
        return {
            "jobs": [jobs_mod.job_to_public_dict(j) for j in jobs_mod.list_jobs(ctx)],
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
        return jobs_mod.job_to_public_dict(saved)

    @r.put("/jobs/{name}")
    def update(name: str, payload: JobIn) -> dict[str, Any]:
        try:
            saved = jobs_mod.update_job(ctx, name, _to_job(payload))
        except KeyError:
            raise HTTPException(404, f"no such job: {name}")
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        return jobs_mod.job_to_public_dict(saved)

    @r.delete("/jobs/{name}")
    def remove(name: str) -> dict[str, Any]:
        if not jobs_mod.delete_job(ctx, name):
            raise HTTPException(404, f"no such job: {name}")
        return {"ok": True}

    return r

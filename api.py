"""Inference plugin REST endpoints (mounted at /api/inference)."""
from __future__ import annotations

import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse

from core.helpers import REPO_DIR, api_get, api_post, systemctl

from . import jobs as jobs_mod
from . import models as models_mod
from . import events as events_mod
from . import clips as clips_mod
from . import plugin_config


def _rerender_yaml() -> tuple[bool, str]:
    """Re-emit mediamtx.yml so a future service restart loads the
    current config. Doesn't touch the running mediamtx — that's the
    job of the per-path apply below."""
    p = subprocess.run(
        ["python3", "-m", "core.renderer"],
        cwd=str(REPO_DIR),
        capture_output=True, text=True, timeout=15,
    )
    if p.returncode != 0:
        return False, (p.stdout + p.stderr).strip()
    return True, ""


def _api_delete(path: str, timeout: float = 3.0) -> int:
    """mediamtx HTTP DELETE — core.helpers only exposes GET/POST."""
    import urllib.request, urllib.error
    from core.helpers import MEDIAMTX_API
    req = urllib.request.Request(f"{MEDIAMTX_API}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def _apply_job_to_mediamtx(ctx, name: str) -> dict[str, Any]:
    """Push a single inference job's path config into the running
    mediamtx via /v3/config/paths/replace/{name}. Replaces our previous
    "rerender + systemctl restart" approach which killed every session
    across every path. Now only the affected path bounces — RTSP / HLS /
    WebRTC clients on other paths (cam0, relay sources) keep streaming.

    Returns {"render": ..., "applied": ...} for the API response so the
    JS toast can show what happened."""
    ok, err = _rerender_yaml()
    if not ok:
        return {"render": "failed", "render_err": err}

    from . import render as render_mod
    paths = render_mod.build_paths(ctx)
    cfg = paths.get(name)
    if cfg is None:
        # Job exists in jobs.yml but produces no path (disabled, model
        # missing, etc.) — make sure mediamtx forgets any prior copy.
        code = _api_delete(f"/v3/config/paths/delete/{name}")
        return {"render": "ok", "applied": "removed" if code == 200 else
                ("not_present" if code == 404 else f"delete_status_{code}")}

    code, _ = api_post(f"/v3/config/paths/replace/{name}", body=cfg)
    if 200 <= code < 300:
        return {"render": "ok", "applied": "patched"}
    return {"render": "ok", "applied": f"replace_status_{code}"}


def _remove_job_from_mediamtx(name: str) -> dict[str, Any]:
    """Companion for job deletion: drops the path from running mediamtx
    and rewrites mediamtx.yml so a future restart matches."""
    ok, err = _rerender_yaml()
    if not ok:
        return {"render": "failed", "render_err": err}
    code = _api_delete(f"/v3/config/paths/delete/{name}")
    return {"render": "ok", "applied": "removed" if code == 200 else
            ("not_present" if code == 404 else f"delete_status_{code}")}


class ClipsIn(BaseModel):
    enabled: bool = True
    pre_roll_s: int = Field(default=5, ge=0, le=600)
    post_roll_s: int = Field(default=10, ge=0, le=600)
    trigger: str = "track_enter"
    trigger_classes: list[str] = []
    retention_count: int = Field(default=100, ge=1, le=10000)


class ClipsToggleIn(BaseModel):
    enabled: bool


class PluginConfigIn(BaseModel):
    clips_root: str


class JobIn(BaseModel):
    name: str
    upstream: str
    enabled: bool = True
    backend: str = "hailo"
    model: str = "yolov8s"
    classes: list[str] = []
    threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    match_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    cpu_input_size: int = Field(default=640)
    max_inference_fps: int = Field(default=0, ge=0, le=60)
    inference_queue: int = Field(default=5, ge=0, le=60)
    track_occlusion_s: float = Field(default=2.0, ge=0.0, le=60.0)
    min_hits: int = Field(default=3, ge=1, le=30)
    always_on: bool = False
    clips: ClipsIn = ClipsIn()


class AlwaysOnIn(BaseModel):
    enabled: bool


class SourceToggleIn(BaseModel):
    enabled: bool


def _to_job(j: JobIn) -> jobs_mod.Job:
    return jobs_mod.Job(
        name=j.name,
        upstream=j.upstream,
        enabled=j.enabled,
        backend=j.backend,
        model=j.model,
        classes=list(j.classes),
        threshold=j.threshold,
        match_threshold=j.match_threshold,
        cpu_input_size=j.cpu_input_size,
        max_inference_fps=j.max_inference_fps,
        inference_queue=j.inference_queue,
        track_occlusion_s=j.track_occlusion_s,
        min_hits=j.min_hits,
        always_on=j.always_on,
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
        from . import live_paths_state, _live_for_job
        from pathlib import Path as _P
        import json as _json
        live = live_paths_state()
        stats_dir = _P.home() / ".cache" / "usb-rtsp" / "inference-stats"
        out_jobs = []
        for j in jobs_mod.list_jobs(ctx):
            item = jobs_mod.job_to_public_dict(j)
            item["_live"] = _live_for_job(j.name, live, j.upstream)
            # Per-worker stats file (FPS, inference latency). Only present
            # while the worker is alive AND the writeback has run at least
            # once — fresh-spawn jobs may not have a file for ~10s.
            stats_path = stats_dir / f"{j.name}.json"
            if stats_path.exists():
                try:
                    raw = _json.loads(stats_path.read_text())
                    # Stale if older than 30s — worker exited but file lingered.
                    import time as _t
                    if (_t.time() - float(raw.get("ts", 0))) < 30:
                        item["_stats"] = {
                            "fps": raw.get("fps"),
                            "inference_ms_avg": raw.get("inference_ms_avg"),
                            "dets_per_min": raw.get("dets_per_min"),
                            "uptime_s": raw.get("uptime_s"),
                        }
                except (OSError, ValueError):
                    pass
            out_jobs.append(item)
        return {
            "jobs": out_jobs,
            "backends": {
                "hailo": models_mod.has_backend("hailo"),
                "cpu": models_mod.has_backend("cpu"),
            },
            "clips_root": str(plugin_config.clips_root(ctx)),
        }

    @r.get("/config")
    def get_config() -> dict[str, Any]:
        return {"clips_root": str(plugin_config.clips_root(ctx))}

    @r.get("/sources")
    def get_sources() -> dict[str, Any]:
        """List every mediamtx path that *could* be inferred on, plus
        whether inference is currently enabled for it. Used by the
        settings-page one-click toggle."""
        from . import list_inference_sources
        # Pick a sensible default backend + model for sources that
        # don't yet have a job — Hailo if available, CPU otherwise.
        h = models_mod.hailo_models()
        c = models_mod.cpu_models()
        default_backend = "hailo" if h else ("cpu" if c else "hailo")
        default_model = (h[0].name if h else (c[0].name if c else "yolov8s"))
        return {
            "sources": list_inference_sources(ctx),
            "default_backend": default_backend,
            "default_model": default_model,
        }

    @r.patch("/sources/{source_name}")
    def toggle_source(source_name: str, payload: SourceToggleIn) -> dict[str, Any]:
        """One-click create/delete of an inference job for a given
        mediamtx source path. Job name defaults to ``<source>-ai``;
        upstream is rtsp://127.0.0.1:8554/<source>; backend+model
        come from the registry default."""
        if not jobs_mod.NAME_RE.match(source_name):
            raise HTTPException(400, f"invalid source name: {source_name!r}")
        upstream = f"rtsp://127.0.0.1:8554/{source_name}"
        existing = next(
            (j for j in jobs_mod.list_jobs(ctx) if j.upstream == upstream),
            None,
        )
        if payload.enabled:
            if existing:
                return {"job": jobs_mod.job_to_public_dict(existing),
                        "applied": "already_present"}
            h = models_mod.hailo_models()
            c = models_mod.cpu_models()
            backend = "hailo" if h else "cpu"
            model = h[0].name if h else (c[0].name if c else "")
            if not model:
                raise HTTPException(400, "no inference models available on this host")
            new_job = jobs_mod.Job(
                name=f"{source_name}-ai",
                upstream=upstream,
                enabled=True,
                backend=backend,
                model=model,
            )
            try:
                saved = jobs_mod.add_job(ctx, new_job)
            except jobs_mod.ValidationError as e:
                raise HTTPException(400, str(e))
            return {"job": jobs_mod.job_to_public_dict(saved),
                    **_apply_job_to_mediamtx(ctx, saved.name)}
        # disable
        if not existing:
            return {"applied": "not_present"}
        jobs_mod.delete_job(ctx, existing.name)
        return {"applied": "removed",
                **_remove_job_from_mediamtx(existing.name)}

    @r.put("/config")
    def put_config(payload: PluginConfigIn) -> dict[str, Any]:
        try:
            new_root = plugin_config.set_clips_root(ctx, payload.clips_root)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # clips_root affects every job's runOnDemand command (the
        # --clips-root arg). Re-render YAML and patch every active
        # path so workers spawned next pick up the new path. Existing
        # clips at the old root stay where they are.
        ok, err = _rerender_yaml()
        if not ok:
            return {"clips_root": str(new_root), "render": "failed", "render_err": err}
        from . import render as render_mod
        paths = render_mod.build_paths(ctx)
        applied = []
        for name, cfg in paths.items():
            code, _ = api_post(f"/v3/config/paths/replace/{name}", body=cfg)
            applied.append({"name": name, "status": code})
        return {"clips_root": str(new_root), "render": "ok", "applied": applied}

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
        return {"job": jobs_mod.job_to_public_dict(saved),
                **_apply_job_to_mediamtx(ctx, saved.name)}

    @r.put("/jobs/{name}")
    def update(name: str, payload: JobIn) -> dict[str, Any]:
        try:
            saved = jobs_mod.update_job(ctx, name, _to_job(payload))
        except KeyError:
            raise HTTPException(404, f"no such job: {name}")
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        return {"job": jobs_mod.job_to_public_dict(saved),
                **_apply_job_to_mediamtx(ctx, saved.name)}

    @r.delete("/jobs/{name}")
    def remove(name: str) -> dict[str, Any]:
        if not jobs_mod.delete_job(ctx, name):
            raise HTTPException(404, f"no such job: {name}")
        return {"ok": True, **_remove_job_from_mediamtx(name)}

    def _set_enabled(name: str, want: bool) -> dict[str, Any]:
        j = jobs_mod.get_job(ctx, name)
        if not j:
            raise HTTPException(404, f"no such job: {name}")
        if j.enabled == want:
            return {"job": jobs_mod.job_to_public_dict(j), "applied": "unchanged"}
        j.enabled = want
        try:
            saved = jobs_mod.update_job(ctx, name, j)
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        # Disabling: drop the path from mediamtx so its worker exits.
        # Enabling: push the new path config in.
        return {"job": jobs_mod.job_to_public_dict(saved),
                **(_apply_job_to_mediamtx(ctx, name) if want
                   else _remove_job_from_mediamtx(name))}

    @r.post("/jobs/{name}/enable")
    def enable_job(name: str) -> dict[str, Any]:
        return _set_enabled(name, True)

    @r.post("/jobs/{name}/disable")
    def disable_job(name: str) -> dict[str, Any]:
        return _set_enabled(name, False)

    @r.post("/jobs/{name}/kick")
    def kick_readers(name: str) -> dict[str, Any]:
        """Kick every reader on this job's path — RTSP, WebRTC, HLS
        muxer, RTMP conn — so the worker exits as soon as the panel
        intends, not whenever ICE / HLS-segment polling eventually
        idles out.

        Source of truth: the path's own ``readers`` list (each entry
        has ``type`` + ``id``). Iterating per-session-list missed
        HLS muxers and could miss in-flight WebRTC sessions that
        haven't registered in /v3/webrtcsessions/list yet."""
        if not jobs_mod.get_job(ctx, name):
            raise HTTPException(404, f"no such job: {name}")
        kicked: list[dict[str, Any]] = []
        path = api_get(f"/v3/paths/get/{name}") or {}
        for r in path.get("readers", []) or []:
            rtype = r.get("type", "")
            rid = r.get("id", "")
            # Map mediamtx reader type → kick endpoint. hlsmuxers kick
            # by path name (one muxer per path), the rest kick by id.
            if rtype == "rtspSession":
                code, _ = api_post(f"/v3/rtspsessions/kick/{rid}")
            elif rtype == "webrtcSession":
                code, _ = api_post(f"/v3/webrtcsessions/kick/{rid}")
            elif rtype == "hlsMuxer":
                code, _ = api_post(f"/v3/hlsmuxers/kick/{name}")
                rid = name
            elif rtype == "rtmpConn":
                code, _ = api_post(f"/v3/rtmpconns/kick/{rid}")
            elif rtype == "srtConn":
                code, _ = api_post(f"/v3/srtconns/kick/{rid}")
            else:
                # Unknown reader type — log it so we can extend later,
                # but don't fail the whole kick.
                kicked.append({"type": rtype, "id": rid, "status": "unknown-type"})
                continue
            kicked.append({"type": rtype, "id": rid, "status": code})
        # Also chase any session entries that haven't been promoted to
        # the path readers list yet (transient setup states).
        for kind in ("rtspsessions", "webrtcsessions"):
            data = api_get(f"/v3/{kind}/list") or {}
            for s in data.get("items", []) or []:
                if s.get("path") != name:
                    continue
                sid = s.get("id")
                if not sid:
                    continue
                # Skip if already kicked above.
                if any(k.get("id") == sid for k in kicked):
                    continue
                code, _ = api_post(f"/v3/{kind}/kick/{sid}")
                kicked.append({"type": kind, "id": sid, "status": code, "stage": "transient"})
        return {"kicked": kicked, "count": len(kicked)}

    @r.patch("/jobs/{name}/always-on")
    def toggle_always_on(name: str, payload: AlwaysOnIn) -> dict[str, Any]:
        """Quick on/off for background-inference mode. Updates always_on
        in jobs.yml and re-applies just this path's config (runOnInit
        for always-on, runOnDemand otherwise). Only the affected worker
        bounces; every other path's clients keep streaming."""
        j = jobs_mod.get_job(ctx, name)
        if not j:
            raise HTTPException(404, f"no such job: {name}")
        j.always_on = bool(payload.enabled)
        try:
            saved = jobs_mod.update_job(ctx, name, j)
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        return {"job": jobs_mod.job_to_public_dict(saved),
                **_apply_job_to_mediamtx(ctx, name)}

    @r.patch("/jobs/{name}/clips")
    def toggle_clips(name: str, payload: ClipsToggleIn) -> dict[str, Any]:
        """Quick on/off for the per-job clip recorder. Updates
        clips.enabled in jobs.yml and re-applies just THIS path's
        config to mediamtx — only the affected worker bounces; every
        other path's clients keep streaming."""
        j = jobs_mod.get_job(ctx, name)
        if not j:
            raise HTTPException(404, f"no such job: {name}")
        j.clips.enabled = bool(payload.enabled)
        try:
            saved = jobs_mod.update_job(ctx, name, j)
        except jobs_mod.ValidationError as e:
            raise HTTPException(400, str(e))
        return {"job": jobs_mod.job_to_public_dict(saved),
                **_apply_job_to_mediamtx(ctx, name)}

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
        root = plugin_config.clips_root(ctx)
        return {
            "clips": clips_mod.list_clips(root, name),
            "total_size_bytes": clips_mod.clips_total_size(root, name),
        }

    @r.get("/clips/{name}/{file_name}")
    def download_clip(name: str, file_name: str):
        root = plugin_config.clips_root(ctx)
        p = clips_mod.clip_path(root, name, file_name)
        if not p:
            raise HTTPException(404, "no such clip")
        return FileResponse(str(p), media_type="video/mp4", filename=file_name)

    @r.delete("/clips/{name}/{file_name}")
    def remove_clip(name: str, file_name: str) -> dict[str, Any]:
        root = plugin_config.clips_root(ctx)
        if not clips_mod.delete_clip(root, name, file_name):
            raise HTTPException(404, "no such clip")
        return {"ok": True}

    return r

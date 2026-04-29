"""Inference job CRUD against ~/.config/usb-rtsp/inference/jobs.yml."""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from . import models

NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
VALID_BACKENDS = ("hailo", "cpu")
VALID_TRIGGERS = ("track_enter", "any_detection", "class_filter")

_LOCK = threading.RLock()


@dataclass
class ClipsConfig:
    enabled: bool = True
    pre_roll_s: int = 5
    post_roll_s: int = 10
    trigger: str = "track_enter"
    trigger_classes: list[str] = field(default_factory=list)
    retention_count: int = 100


@dataclass
class Job:
    name: str
    upstream: str
    enabled: bool = True
    backend: str = "hailo"
    model: str = "yolov8s"
    classes: list[str] = field(default_factory=list)
    threshold: float = 0.4
    # Lower confidence floor used for matching detections to *existing*
    # tracks. Detections below `threshold` but above `match_threshold`
    # can keep a confirmed track alive (defeats brief confidence dips).
    # 0.0 = auto-derive as max(0.05, threshold * 0.5) in the worker.
    match_threshold: float = 0.0
    # CPU backend only: input resolution. 640 = native training size
    # (best accuracy, slowest). 416 ≈ 2× faster, 320 ≈ 4× faster.
    # Hailo backend ignores this — its HEFs are baked at 640×640.
    # Bench (yolo11n on Pi 5 CPU, threads=3):
    #   640: ~6 fps    416: ~15 fps    320: ~24 fps
    cpu_input_size: int = 640
    # Cap inference rate so the worker doesn't burn CPU faster than
    # needed. Frames between scheduled inferences are pulled from the
    # latest-frame reader and republished as-is (last annotation
    # carried forward). 0 = unlimited (worker runs as fast as backend
    # allows). Typical surveillance use is 5-10 fps; full-rate is
    # only useful for action recognition / fast-moving content.
    max_inference_fps: int = 0
    inference_queue: int = 5
    track_occlusion_s: float = 2.0
    # Frames a new track must match before it gets drawn ("banding"). 1 =
    # show immediately (jitters on noisy single-frame detections); 3 ≈
    # 100ms warm-up at 30fps; higher = even more confirmation, more lag
    # before the box first appears.
    min_hits: int = 3
    # When True, mediamtx renders the path with runOnInit (worker
    # always running). Default False (runOnDemand) keeps CPU/Hailo
    # idle until a viewer subscribes.
    always_on: bool = False
    clips: ClipsConfig = field(default_factory=ClipsConfig)


def _config_path_from(config_dir: Path) -> Path:
    return Path(config_dir) / "jobs.yml"


def _ctx_config_dir(ctx) -> Path:
    """The dashboard/API ctx wraps Plugin (.plugin.config_dir); the
    render-time ctx exposes .config_dir directly. Handle either."""
    plugin = getattr(ctx, "plugin", None)
    if plugin is not None and hasattr(plugin, "config_dir"):
        return Path(plugin.config_dir)
    return Path(ctx.config_dir)


def _load_jobs(path: Path) -> list[Job]:
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        return []
    out: list[Job] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        clips_raw = entry.get("clips") or {}
        clips = ClipsConfig(
            enabled=bool(clips_raw.get("enabled", True)),
            pre_roll_s=int(clips_raw.get("pre_roll_s", 5)),
            post_roll_s=int(clips_raw.get("post_roll_s", 10)),
            trigger=str(clips_raw.get("trigger", "track_enter")),
            trigger_classes=[str(x) for x in (clips_raw.get("trigger_classes") or [])],
            retention_count=int(clips_raw.get("retention_count", 100)),
        )
        out.append(Job(
            name=str(entry.get("name", "")),
            upstream=str(entry.get("upstream", "")),
            enabled=bool(entry.get("enabled", True)),
            backend=str(entry.get("backend", "hailo")),
            model=str(entry.get("model", "yolov8s")),
            classes=[str(x) for x in (entry.get("classes") or [])],
            threshold=float(entry.get("threshold", 0.4)),
            match_threshold=float(entry.get("match_threshold", 0.0)),
            cpu_input_size=int(entry.get("cpu_input_size", 640)),
            max_inference_fps=int(entry.get("max_inference_fps", 0)),
            inference_queue=int(entry.get("inference_queue", 5)),
            track_occlusion_s=float(entry.get("track_occlusion_s", 2.0)),
            min_hits=int(entry.get("min_hits", 3)),
            always_on=bool(entry.get("always_on", False)),
            clips=clips,
        ))
    return out


def _save_jobs(path: Path, jobs: list[Job]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [asdict(j) for j in jobs]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(serialised, sort_keys=False))
    tmp.replace(path)


class ValidationError(ValueError):
    """Raised when a job dict fails schema/value checks."""


def _validate(j: Job, existing_names: set[str], *, allow_existing: str | None = None) -> None:
    if not NAME_RE.match(j.name):
        raise ValidationError(f"name must match {NAME_RE.pattern}")
    if j.name in existing_names and j.name != allow_existing:
        raise ValidationError(f"duplicate job name: {j.name}")
    if j.backend not in VALID_BACKENDS:
        raise ValidationError(f"backend must be one of {VALID_BACKENDS}")
    if not j.upstream.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        raise ValidationError("upstream must be rtsp/rtmp/http(s)")
    self_loop = f"rtsp://127.0.0.1:8554/{j.name}"
    if j.upstream.rstrip("/") == self_loop.rstrip("/"):
        raise ValidationError("refusing self-loop (upstream cannot be this job's own output)")
    if not (0.0 <= j.threshold <= 1.0):
        raise ValidationError("threshold must be in [0, 1]")
    if not (0.0 <= j.match_threshold <= 1.0):
        raise ValidationError("match_threshold must be in [0, 1]")
    if j.match_threshold > 0 and j.match_threshold > j.threshold:
        raise ValidationError("match_threshold must be ≤ threshold (or 0 for auto)")
    if j.cpu_input_size not in (320, 416, 640):
        raise ValidationError("cpu_input_size must be 320, 416, or 640")
    if not (0 <= j.max_inference_fps <= 60):
        raise ValidationError("max_inference_fps must be in [0, 60]")
    if j.inference_queue < 0 or j.inference_queue > 60:
        raise ValidationError("inference_queue must be in [0, 60]")
    if j.track_occlusion_s < 0 or j.track_occlusion_s > 60:
        raise ValidationError("track_occlusion_s must be in [0, 60]")
    if j.min_hits < 1 or j.min_hits > 30:
        raise ValidationError("min_hits must be in [1, 30]")
    if j.clips.trigger not in VALID_TRIGGERS:
        raise ValidationError(f"clips.trigger must be one of {VALID_TRIGGERS}")
    if j.clips.pre_roll_s < 0 or j.clips.post_roll_s < 0:
        raise ValidationError("clip pre/post roll must be >= 0")
    if j.clips.retention_count < 1:
        raise ValidationError("retention_count must be >= 1")
    if not models.find_model(j.backend, j.model):
        raise ValidationError(f"model {j.model!r} not available for backend {j.backend!r}")


def list_jobs(ctx) -> list[Job]:
    with _LOCK:
        return _load_jobs(_config_path_from(_ctx_config_dir(ctx)))


def get_job(ctx, name: str) -> Job | None:
    for j in list_jobs(ctx):
        if j.name == name:
            return j
    return None


def add_job(ctx, j: Job) -> Job:
    with _LOCK:
        path = _config_path_from(_ctx_config_dir(ctx))
        jobs = _load_jobs(path)
        _validate(j, {x.name for x in jobs})
        jobs.append(j)
        _save_jobs(path, jobs)
    return j


def update_job(ctx, name: str, j: Job) -> Job:
    with _LOCK:
        path = _config_path_from(_ctx_config_dir(ctx))
        jobs = _load_jobs(path)
        if not any(x.name == name for x in jobs):
            raise KeyError(name)
        _validate(j, {x.name for x in jobs}, allow_existing=name)
        jobs = [j if x.name == name else x for x in jobs]
        _save_jobs(path, jobs)
    return j


def delete_job(ctx, name: str) -> bool:
    with _LOCK:
        path = _config_path_from(_ctx_config_dir(ctx))
        jobs = _load_jobs(path)
        new = [x for x in jobs if x.name != name]
        if len(new) == len(jobs):
            return False
        _save_jobs(path, new)
    return True


def job_to_public_dict(j: Job) -> dict[str, Any]:
    """Serialise a Job for the API/UI (mirrors the YAML shape)."""
    return asdict(j)

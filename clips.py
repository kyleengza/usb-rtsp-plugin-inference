"""Per-job clip recorder.

When a tracked-ID enter event fires for an allowed class, spawn a new
ffmpeg subprocess that writes an mp4 to disk; tee each subsequent frame
to it. When the triggering track has been absent for ``post_roll_s``
seconds, close the ffmpeg and finalise the file.

v1 limitation: post-roll only — no pre-roll. Pre-roll requires a frame
ring buffer (raw frames are too big at 1280x720 BGR; would need JPEG
or downscale ring). Documented in the plan; fine for a first cut.

Retention: after each clip finalises, prune the oldest until at most
``retention_count`` clips remain in the job's directory.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np  # type: ignore

CLIPS_ROOT = Path.home() / ".cache" / "usb-rtsp" / "inference-clips"


@dataclass
class _ActiveClip:
    track_id: int
    label: str
    started_at: float
    last_seen_at: float
    file_path: Path
    proc: subprocess.Popen


class ClipRecorder:
    def __init__(
        self,
        job_name: str,
        width: int,
        height: int,
        fps: int,
        post_roll_s: float = 10.0,
        retention_count: int = 100,
        trigger: str = "track_enter",
        trigger_classes: set[str] | None = None,
    ) -> None:
        self.job_name = job_name
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.post_roll_s = float(post_roll_s)
        self.retention_count = max(1, int(retention_count))
        self.trigger = trigger
        self.trigger_classes = set(trigger_classes) if trigger_classes else None
        self.dir = CLIPS_ROOT / job_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._active: list[_ActiveClip] = []

    # --- public API ----------------------------------------------------

    def should_trigger(self, event_kind: str, label: str) -> bool:
        """Whether an inbound track event should start a new clip."""
        if event_kind != "enter":
            return False
        if self.trigger == "any_detection":
            return True
        if self.trigger == "class_filter":
            return self.trigger_classes is None or label in self.trigger_classes
        # track_enter (default): trigger on any class the user permitted via
        # trigger_classes; if empty, all classes count.
        if self.trigger_classes:
            return label in self.trigger_classes
        return True

    def maybe_start(self, event_kind: str, track_id: int, label: str, ts: float) -> None:
        if not self.should_trigger(event_kind, label):
            return
        # Already an active clip for this track? Skip.
        if any(c.track_id == track_id for c in self._active):
            return
        path = self._build_path(ts, track_id, label)
        proc = self._spawn_ffmpeg(path)
        self._active.append(_ActiveClip(
            track_id=track_id, label=label,
            started_at=ts, last_seen_at=ts,
            file_path=path, proc=proc,
        ))

    def write_frame(self, frame: np.ndarray, current_track_ids: set[int], ts: float) -> None:
        """Tee the current frame to every active clip; refresh
        last_seen_at for clips whose track is still present; close any
        whose track has been gone for ``post_roll_s``."""
        if not self._active:
            return
        bytes_ = frame.tobytes()
        still_active: list[_ActiveClip] = []
        for clip in self._active:
            try:
                if clip.proc.stdin:
                    clip.proc.stdin.write(bytes_)
            except (BrokenPipeError, IOError):
                self._finalise(clip)
                continue
            if clip.track_id in current_track_ids:
                clip.last_seen_at = ts
                still_active.append(clip)
            else:
                if (ts - clip.last_seen_at) > self.post_roll_s:
                    self._finalise(clip)
                else:
                    still_active.append(clip)
        self._active = still_active

    def close_all(self) -> None:
        for clip in self._active:
            self._finalise(clip)
        self._active = []

    # --- internals -----------------------------------------------------

    def _build_path(self, ts: float, track_id: int, label: str) -> Path:
        # ISO8601 with seconds resolution, colons replaced for cross-FS safety.
        iso = time.strftime("%Y%m%dT%H%M%S", time.localtime(ts))
        safe_label = "".join(c if c.isalnum() or c == "_" else "_" for c in label)
        return self.dir / f"{iso}__t{track_id}__{safe_label}.mp4"

    def _spawn_ffmpeg(self, path: Path) -> subprocess.Popen:
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps * 2),
            "-an",
            "-f", "mp4",
            "-movflags", "+faststart",
            str(path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _finalise(self, clip: _ActiveClip) -> None:
        try:
            if clip.proc.stdin:
                clip.proc.stdin.close()
            clip.proc.wait(timeout=4)
        except Exception:
            try:
                clip.proc.kill()
            except Exception:
                pass
        self._enforce_retention()

    def _enforce_retention(self) -> None:
        try:
            files = sorted(
                (p for p in self.dir.glob("*.mp4") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
            excess = len(files) - self.retention_count
            for old in files[:max(0, excess)]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except Exception:
            pass


def list_clips(job_name: str) -> list[dict]:
    """Used by the API to render the clips list."""
    job_dir = CLIPS_ROOT / job_name
    if not job_dir.is_dir():
        return []
    out = []
    for p in sorted(job_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            stat = p.stat()
        except OSError:
            continue
        out.append({
            "name": p.name,
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return out


def clip_path(job_name: str, file_name: str) -> Path | None:
    """Resolve a clip path safely (no traversal). Returns None if not found."""
    if "/" in file_name or "\\" in file_name or ".." in file_name:
        return None
    p = (CLIPS_ROOT / job_name / file_name).resolve()
    root = (CLIPS_ROOT / job_name).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    return p if p.is_file() else None


def delete_clip(job_name: str, file_name: str) -> bool:
    p = clip_path(job_name, file_name)
    if not p:
        return False
    try:
        p.unlink()
        return True
    except OSError:
        return False

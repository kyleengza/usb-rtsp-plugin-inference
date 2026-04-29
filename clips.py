"""Per-job clip recorder.

Single-clip-at-a-time. While ANY qualifying trigger event is active, one
ffmpeg subprocess writes BGR frames to a timestamped mp4. The clip
extends as long as a tracked object is still in frame; it closes
``post_roll_s`` seconds after the frame goes empty. The next trigger
opens a fresh clip.

This avoids the per-track concurrency the previous design had — when
the tracker re-births a noisy track every couple of seconds, we used
to spawn an ffmpeg per re-birth, multiplying the encode load (and on
this Pi: tripping the PSU). One writer keeps load bounded and the
filename simple: ``YYYYMMDDTHHMMSS.mp4``.

v1 limitation: no pre-roll. A 5s pre-roll at 1280x720 BGR is ~250MB
of RAM; would need a JPEG/downscale ring buffer. Documented and
deferred.

Retention: after each clip finalises, prune the oldest mp4 in the
job's directory until at most ``retention_count`` remain.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # type: ignore

CLIPS_ROOT = Path.home() / ".cache" / "usb-rtsp" / "inference-clips"


@dataclass
class _ActiveClip:
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
        self._current: _ActiveClip | None = None

    # --- public API ----------------------------------------------------

    def should_trigger(self, event_kind: str, label: str) -> bool:
        """Whether an inbound track event should start (or extend) a clip."""
        if event_kind != "enter":
            return False
        if self.trigger == "any_detection":
            return True
        if self.trigger == "class_filter":
            return self.trigger_classes is None or label in self.trigger_classes
        # track_enter (default): if trigger_classes set, restrict; else allow all.
        if self.trigger_classes:
            return label in self.trigger_classes
        return True

    def on_trigger(self, ts: float) -> None:
        """Open a clip if none active; otherwise extend the current one."""
        if self._current is None:
            self._current = self._spawn(ts)
        else:
            self._current.last_seen_at = ts

    def write_frame(self, frame: np.ndarray, current_track_ids: set[int], ts: float) -> None:
        """Tee the current frame to the active clip (if any). Any tracked
        entity present in ``current_track_ids`` keeps the clip open. After
        ``post_roll_s`` seconds of empty frames, finalise."""
        if self._current is None:
            return
        try:
            if self._current.proc.stdin:
                self._current.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, IOError):
            self._finalise(self._current)
            self._current = None
            return
        if current_track_ids:
            # Any tracked object in this frame extends the recording.
            self._current.last_seen_at = ts
        elif (ts - self._current.last_seen_at) > self.post_roll_s:
            self._finalise(self._current)
            self._current = None

    def close_all(self) -> None:
        if self._current is not None:
            self._finalise(self._current)
            self._current = None

    # --- internals -----------------------------------------------------

    def _spawn(self, ts: float) -> _ActiveClip:
        path = self._build_path(ts)
        proc = self._spawn_ffmpeg(path)
        return _ActiveClip(
            started_at=ts, last_seen_at=ts, file_path=path, proc=proc,
        )

    def _build_path(self, ts: float) -> Path:
        iso = time.strftime("%Y%m%dT%H%M%S", time.localtime(ts))
        # Append a short -N suffix only when needed to avoid filename collisions
        # within the same second (rare, but possible if a clip closes + reopens
        # immediately).
        candidate = self.dir / f"{iso}.mp4"
        if not candidate.exists():
            return candidate
        for i in range(1, 100):
            c = self.dir / f"{iso}-{i}.mp4"
            if not c.exists():
                return c
        return candidate  # fallback

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


def clips_total_size(job_name: str) -> int:
    job_dir = CLIPS_ROOT / job_name
    if not job_dir.is_dir():
        return 0
    total = 0
    for p in job_dir.glob("*.mp4"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


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

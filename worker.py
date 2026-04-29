#!/usr/bin/env python3
"""Inference worker subprocess.

Spawned by mediamtx as the runOnDemand command for each enabled
inference job. Reads frames from --upstream, runs inference via the
backend named by --backend (hailo|cpu), draws annotations, pipes
annotated BGR frames to ffmpeg which re-encodes and publishes to
--output as RTSP.

Slice 3: real Hailo inference. CPU backend lands in slice 4.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Tier-2 telemetry: enabling HAILO_MONITOR=1 *before* hailo_platform is
# imported makes the SDK write per-stream stats into /tmp/.hailo_monitor/,
# which hailortcli monitor (and our panel-side helper) can read for
# utilisation %, FPS, queue depths. Cheap; no observable runtime cost.
os.environ.setdefault("HAILO_MONITOR", "1")

import cv2  # type: ignore
import numpy as np  # type: ignore

# When invoked as a script (mediamtx spawn), import sibling modules
# directly without dragging in the plugin's fastapi-flavoured __init__.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def log(msg: str) -> None:
    print(f"[inference-worker] {msg}", file=sys.stderr, flush=True)


def open_upstream(url: str, timeout_s: float = 20.0) -> "cv2.VideoCapture":
    deadline = time.monotonic() + timeout_s
    last_err = ""
    while time.monotonic() < deadline:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
            last_err = "opened but read returned false"
        else:
            last_err = "not opened"
        time.sleep(0.5)
    raise RuntimeError(f"upstream open timeout after {timeout_s}s ({last_err})")


def spawn_ffmpeg(out_url: str, w: int, h: int, fps: int) -> subprocess.Popen:
    """Encode-and-publish ffmpeg.

    -use_wallclock_as_timestamps + -fflags +genpts: timestamp frames by
    when they arrive, not by an assumed input rate. The CPU backend
    drives 1-3 fps for yolo11s while the camera is 30 fps — we can't
    pretend the input is 30 fps or PTS go off the rails.

    -force_key_frames every 1 s of wall-clock so WebRTC viewers
    actually see keyframes regardless of how slow inference runs. With
    the previous -g 60 derived from source fps, a 1.5 fps inference
    pipeline would key-frame every 40 s, well past WebRTC's
    handshake/I-frame timeout — visible to the user as "no preview"
    even though packets flow."""
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "warning",
        "-use_wallclock_as_timestamps", "1",
        "-fflags", "+genpts",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-force_key_frames", "expr:gte(t,n_forced*1)",
        "-an",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        out_url,
    ]
    log(f"spawn ffmpeg → {out_url} ({w}x{h}, wallclock pts, key/1s)")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ---------------------------------------------------------------------------
# Backend wiring


def _load_labels(backend: str, model_name: str) -> list[str]:
    """Resolve the COCO label list from the registry."""
    import models  # type: ignore  # imported via THIS_DIR on sys.path
    return models.all_labels("coco")


def _resolve_class_ids(labels: list[str], names_csv: str) -> set[int] | None:
    if not names_csv.strip():
        return None
    wanted = {n.strip().lower() for n in names_csv.split(",") if n.strip()}
    out = set()
    for idx, name in enumerate(labels):
        if name.lower() in wanted:
            out.add(idx)
    return out


def _match_threshold(register_threshold: float, override: float = 0.0) -> float:
    """The low floor passed to backends. Existing tracks can be matched
    (and so survive a brief confidence dip) by any detection above this
    value; new tracks still need ``register_threshold`` to spawn.
    ``override`` (>0) lets the user pin an explicit floor; clamped at
    register_threshold so it can't exceed the spawn cost. ``override=0``
    falls back to the auto rule: half the user threshold, 0.05 floor."""
    if override > 0:
        return min(override, register_threshold)
    return max(0.05, register_threshold * 0.5)


def make_backend(args, labels: list[str]):
    match_thr = _match_threshold(args.threshold, getattr(args, "match_threshold", 0.0))
    if args.backend == "hailo":
        from backend_hailo import HailoBackend  # type: ignore
        allow = _resolve_class_ids(labels, args.classes)
        return HailoBackend(
            hef_path=Path(args.model_path),
            labels=labels,
            threshold=match_thr,
            allow_class_ids=allow,
        )
    if args.backend == "cpu":
        from backend_cpu import CpuBackend  # type: ignore
        allow = _resolve_class_ids(labels, args.classes)
        sz = int(getattr(args, "cpu_input_size", 640))
        return CpuBackend(
            onnx_path=Path(args.model_path),
            labels=labels,
            threshold=match_thr,
            allow_class_ids=allow,
            input_size=(sz, sz),
        )
    raise ValueError(f"unknown backend: {args.backend}")


# ---------------------------------------------------------------------------
# Annotation drawing

# Stable colour-per-class palette so the same class always paints the
# same hue across frames (12 hues cycle for 80 COCO classes).
_PALETTE = [
    (66, 135, 245), (245, 87, 66), (66, 245, 138), (245, 197, 66),
    (171, 66, 245), (66, 245, 230), (245, 66, 158), (140, 245, 66),
    (245, 174, 66), (66, 89, 245), (66, 245, 90), (245, 235, 66),
]


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def annotate(frame: np.ndarray, detections, fps: float, det_count_window: int) -> None:
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        col = _color_for_class(det.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        tid = getattr(det, "track_id", 0)
        # Prefer the per-track EMA so the displayed confidence doesn't
        # flicker frame-to-frame; fall back to raw score if untracked.
        shown = getattr(det, "smoothed_score", 0.0) or det.score
        label = f"#{tid} {det.label} {shown:.2f}" if tid else f"{det.label} {shown:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # Stats badge bottom-left.
    h = frame.shape[0]
    badge = f"{fps:.1f} fps · {det_count_window} dets/sec"
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (10, h - th - 14), (16 + tw, h - 8), (0, 0, 0), -1)
    cv2.putText(frame, badge, (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=("hailo", "cpu"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--job-name", default="")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--match-threshold", type=float, default=0.0,
                    help="lower threshold for matching existing tracks; "
                         "0 = auto = max(0.05, threshold * 0.5)")
    ap.add_argument("--cpu-input-size", type=int, default=640,
                    help="CPU backend only: ONNX input HxW (320/416/640)")
    ap.add_argument("--classes", default="")
    ap.add_argument("--inference-queue", type=int, default=5)
    ap.add_argument("--track-occlusion-s", type=float, default=2.0)
    ap.add_argument("--min-hits", type=int, default=3)
    # Clip recording
    ap.add_argument("--clips-enabled", action="store_true")
    ap.add_argument("--clips-root", default="")  # absolute path, plugin-config-resolved
    ap.add_argument("--clip-post-roll-s", type=float, default=10.0)
    ap.add_argument("--clip-trigger", default="track_enter")
    ap.add_argument("--clip-trigger-classes", default="")
    ap.add_argument("--clip-retention", type=int, default=100)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    job_name = args.job_name or "(unnamed)"
    log(f"start: job={job_name} backend={args.backend} model={args.model} upstream={args.upstream}")

    labels = _load_labels(args.backend, args.model)
    backend = make_backend(args, labels)
    if backend is None:
        log(f"backend {args.backend!r} not implemented yet — exiting")
        return 0

    from tracker import IoUTracker  # type: ignore
    from events import JobEventLog  # type: ignore
    from clips import ClipRecorder  # type: ignore
    tracker = IoUTracker(
        iou_threshold=0.3,
        ttl_s=args.track_occlusion_s,
        min_hits=args.min_hits,
        register_threshold=args.threshold,
    )
    event_log = JobEventLog(args.job_name) if args.job_name else None

    cap = open_upstream(args.upstream)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_int = max(1, int(round(src_fps)))

    ff = spawn_ffmpeg(args.output, w, h, fps_int)

    clipper = None
    if args.clips_enabled and args.job_name:
        trig_classes = {c.strip() for c in args.clip_trigger_classes.split(",") if c.strip()}
        clipper = ClipRecorder(
            job_name=args.job_name,
            width=w, height=h, fps=fps_int,
            root=args.clips_root or None,
            post_roll_s=args.clip_post_roll_s,
            retention_count=args.clip_retention,
            trigger=args.clip_trigger,
            trigger_classes=trig_classes or None,
        )

    stop = {"flag": False}
    def _handler(*_): stop["flag"] = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    frames_pushed = 0
    inference_ms_sum = 0.0
    inference_count = 0
    dets_in_window = 0
    last_det_count = 0
    last_log = time.monotonic()
    started_at = time.monotonic()
    fps_observed = float(fps_int)
    stats_dir = Path.home() / ".cache" / "usb-rtsp" / "inference-stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / f"{args.job_name or 'unnamed'}.json"
    try:
        while not stop["flag"]:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            ts = time.time()
            inf_t0 = time.perf_counter()
            try:
                detections = backend.detect(frame)
            except Exception as e:
                log(f"inference error (continuing): {e}")
                detections = []
            inference_ms_sum += (time.perf_counter() - inf_t0) * 1000.0
            inference_count += 1
            dets_in_window += len(detections)
            tracked, track_events = tracker.step(detections, ts_s=ts)
            if event_log:
                for ev in track_events:
                    event_log.emit({
                        "ts": ev.ts,
                        "kind": ev.kind,
                        "track_id": ev.track_id,
                        "label": ev.label,
                        "score": round(ev.score, 4),
                        "box": [round(v, 2) for v in ev.box],
                    })
            if clipper:
                for ev in track_events:
                    if clipper.should_trigger(ev.kind, ev.label):
                        clipper.on_trigger(ev.ts)
            annotate(frame, tracked, fps_observed, len(tracked))
            if clipper:
                cur_ids = {getattr(d, "track_id", 0) for d in tracked}
                clipper.write_frame(frame, cur_ids, ts)
            try:
                ff.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError) as e:
                log(f"ffmpeg pipe closed: {e}")
                break
            frames_pushed += 1
            last_det_count = len(tracked)
            now = time.monotonic()
            if now - last_log > 10:
                window_s = now - last_log
                fps_observed = frames_pushed / window_s
                avg_inf_ms = (inference_ms_sum / inference_count) if inference_count else 0.0
                dets_per_min = (dets_in_window / window_s) * 60.0
                log(f"alive · {frames_pushed} frames in {window_s:.1f}s "
                    f"({fps_observed:.1f} fps) · {avg_inf_ms:.1f}ms inf · "
                    f"{last_det_count} dets last frame")
                # Persist stats so the panel can render them per-job.
                try:
                    tmp = stats_path.with_suffix(stats_path.suffix + ".tmp")
                    tmp.write_text(json.dumps({
                        "job": args.job_name,
                        "backend": args.backend,
                        "model": args.model,
                        "ts": time.time(),
                        "uptime_s": round(now - started_at, 1),
                        "fps": round(fps_observed, 2),
                        "inference_ms_avg": round(avg_inf_ms, 2),
                        "dets_per_min": round(dets_per_min, 1),
                    }, separators=(",", ":")))
                    tmp.replace(stats_path)
                except OSError:
                    pass
                frames_pushed = 0
                inference_ms_sum = 0.0
                inference_count = 0
                dets_in_window = 0
                last_log = now
    finally:
        log("shutting down")
        if clipper:
            try:
                clipper.close_all()
            except Exception:
                pass
        try:
            backend.close()
        except Exception:
            pass
        try:
            if ff.stdin: ff.stdin.close()
            ff.wait(timeout=3)
        except Exception:
            ff.kill()
        cap.release()
    log("exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

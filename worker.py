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
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

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
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(max(1, int(fps))),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-g", str(max(1, int(fps) * 2)),
        "-an",
        "-f", "rtsp", "-rtsp_transport", "tcp",
        out_url,
    ]
    log(f"spawn ffmpeg → {out_url} ({w}x{h}@{int(fps)})")
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


def make_backend(args, labels: list[str]):
    if args.backend == "hailo":
        from backend_hailo import HailoBackend  # type: ignore
        allow = _resolve_class_ids(labels, args.classes)
        return HailoBackend(
            hef_path=Path(args.model_path),
            labels=labels,
            threshold=args.threshold,
            allow_class_ids=allow,
        )
    if args.backend == "cpu":
        # Slice 4 will replace this with backend_cpu.CpuBackend.
        return None
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
        label = f"{det.label} {det.score:.2f}"
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
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--classes", default="")
    ap.add_argument("--inference-queue", type=int, default=5)
    ap.add_argument("--track-occlusion-s", type=float, default=2.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log(f"start: backend={args.backend} model={args.model} upstream={args.upstream}")

    labels = _load_labels(args.backend, args.model)
    backend = make_backend(args, labels)
    if backend is None:
        log(f"backend {args.backend!r} not implemented yet — exiting")
        return 0

    cap = open_upstream(args.upstream)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_int = max(1, int(round(src_fps)))

    ff = spawn_ffmpeg(args.output, w, h, fps_int)

    stop = {"flag": False}
    def _handler(*_): stop["flag"] = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    frames_pushed = 0
    last_det_count = 0
    last_log = time.monotonic()
    last_fps_calc = time.monotonic()
    fps_observed = float(fps_int)
    try:
        while not stop["flag"]:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            try:
                detections = backend.detect(frame)
            except Exception as e:  # don't crash the worker on an inference glitch
                log(f"inference error (continuing): {e}")
                detections = []
            annotate(frame, detections, fps_observed, len(detections))
            try:
                ff.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError) as e:
                log(f"ffmpeg pipe closed: {e}")
                break
            frames_pushed += 1
            last_det_count = len(detections)
            now = time.monotonic()
            if now - last_log > 10:
                fps_observed = frames_pushed / (now - last_log)
                log(f"alive · {frames_pushed} frames in {now - last_log:.1f}s "
                    f"({fps_observed:.1f} fps) · {last_det_count} dets last frame")
                frames_pushed = 0
                last_log = now
    finally:
        log("shutting down")
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

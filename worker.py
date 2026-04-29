#!/usr/bin/env python3
"""Inference worker subprocess.

Slice 2: frame plumbing only — reads frames from --upstream via opencv,
draws a fixed test annotation, pipes annotated BGR frames to ffmpeg
which re-encodes and publishes to --output as RTSP. No real inference
yet; that lands in slices 3 (Hailo) and 4 (CPU).

mediamtx invokes this as the `runOnDemand` command for every enabled
inference job. The plugin's render.py builds the argv with the job's
config baked in.
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


def log(msg: str) -> None:
    print(f"[inference-worker] {msg}", file=sys.stderr, flush=True)


def open_upstream(url: str, timeout_s: float = 15.0) -> "cv2.VideoCapture":
    """Open RTSP/RTMP/HTTP upstream with retry — when mediamtx spawns this
    worker via runOnDemand, the upstream path may itself be runOnDemand
    and not yet have its source ready. Retry for a bit before giving up."""
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
    """Spawn ffmpeg that ingests rawvideo on stdin and publishes RTSP.
    fps must be an int — ffmpeg's -r flag rejects 0/None."""
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


def annotate_test_pattern(frame, args) -> None:
    """Slice 2 placeholder annotation. Replaced by real inference output
    in slices 3 (Hailo) and 4 (CPU)."""
    text = f"inference · {args.backend}/{args.model} · slice 2 (no model loaded)"
    h = frame.shape[0]
    # Background bar at the bottom-left.
    cv2.rectangle(frame, (10, h - 50), (10 + 11 * len(text), h - 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (60, 220, 220), 1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=("hailo", "cpu"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default="")  # absolute path to hef/onnx
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--classes", default="")  # comma-separated; empty = all
    ap.add_argument("--inference-queue", type=int, default=5)
    ap.add_argument("--track-occlusion-s", type=float, default=2.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log(f"start: upstream={args.upstream} output={args.output} "
        f"backend={args.backend} model={args.model}")

    cap = open_upstream(args.upstream)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_int = max(1, int(round(src_fps)))

    ff = spawn_ffmpeg(args.output, w, h, fps_int)

    stop = {"flag": False}
    def _handler(*_):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    frame_count = 0
    last_log = time.monotonic()
    try:
        while not stop["flag"]:
            ok, frame = cap.read()
            if not ok:
                # Source may be momentarily silent — retry without spamming.
                time.sleep(0.05)
                continue
            annotate_test_pattern(frame, args)
            try:
                ff.stdin.write(frame.tobytes())
            except (BrokenPipeError, IOError) as e:
                log(f"ffmpeg pipe closed: {e}")
                break
            frame_count += 1
            now = time.monotonic()
            if now - last_log > 10:
                log(f"alive · {frame_count} frames pushed in last 10s "
                    f"({frame_count / (now - last_log):.1f} fps)")
                frame_count = 0
                last_log = now
    finally:
        try:
            if ff.stdin:
                ff.stdin.close()
            ff.wait(timeout=3)
        except Exception:
            ff.kill()
        cap.release()
    log("exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

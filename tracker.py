"""Minimal IoU-based multi-object tracker.

No Kalman, no optical flow — just greedy IoU matching with per-track
TTL. Suitable for short-occlusion handling on a single camera. Returns
the same Detection objects the backends emit, with .track_id populated.

Drop-in usage:
    tracker = IoUTracker(iou_threshold=0.3, ttl_s=2.0, fps_hint=30)
    for frame in stream:
        dets = backend.detect(frame)
        tracked, events = tracker.step(dets, ts_s=time.monotonic())
        # tracked: list of Detection with track_id set
        # events: list of (kind, track_id, det) for kind in {"enter","leave"}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

try:
    # Worker runs us as a top-level script (sys.path includes plugin dir).
    from dets import Detection  # type: ignore
except ImportError:
    # API process imports us via the package; relative import works there.
    from .dets import Detection  # type: ignore


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    class_id: int
    label: str
    box: tuple[float, float, float, float]
    score: float                  # raw per-frame score
    smoothed_score: float         # EMA over the track's lifetime
    last_seen_ts: float
    first_seen_ts: float
    hits: int = 1
    # True once the track has been matched ``min_hits`` times — only
    # confirmed tracks are drawn and only confirmed tracks emit
    # enter / leave events.
    confirmed: bool = False


# EMA factor for smoothed confidence. 0.25 means each new frame
# contributes 25% — half-life of about 3 frames at 30 fps. Smooths out
# digit-flicker without lagging real changes.
SCORE_EMA_ALPHA = 0.25


@dataclass
class TrackEvent:
    kind: str       # "enter" | "leave"
    track_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]
    ts: float


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        ttl_s: float = 2.0,
        min_hits: int = 3,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.ttl_s = float(ttl_s)
        # Banding: a track must match at least ``min_hits`` frames before
        # the worker draws its box. Filters out single-frame false
        # positives that would otherwise blink on for one frame and
        # leave a confidence ghost behind.
        self.min_hits = max(1, int(min_hits))
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def step(
        self,
        detections: list[Detection],
        ts_s: float,
    ) -> tuple[list[Detection], list[TrackEvent]]:
        events: list[TrackEvent] = []
        # Greedy match: for each existing track, find best-IoU detection of
        # the same class. Mark used detections; unmatched detections become
        # new tracks. Then expire stale tracks.
        used_det = set()
        matches: dict[int, int] = {}  # track_id -> det_idx
        det_indices_by_class: dict[int, list[int]] = {}
        for i, d in enumerate(detections):
            det_indices_by_class.setdefault(d.class_id, []).append(i)

        for tid, t in list(self._tracks.items()):
            best_iou = self.iou_threshold
            best_idx = -1
            for i in det_indices_by_class.get(t.class_id, []):
                if i in used_det:
                    continue
                iou = _iou(t.box, detections[i].box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                d = detections[best_idx]
                t.box = d.box
                t.score = d.score
                t.smoothed_score = (
                    SCORE_EMA_ALPHA * d.score
                    + (1.0 - SCORE_EMA_ALPHA) * t.smoothed_score
                )
                t.last_seen_ts = ts_s
                t.hits += 1
                used_det.add(best_idx)
                matches[tid] = best_idx

        # Unmatched detections → new (tentative) tracks. Don't emit
        # an "enter" event yet — wait until the track is confirmed
        # via min_hits, otherwise a one-frame false positive would
        # spam events + clip triggers.
        for i, d in enumerate(detections):
            if i in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(
                track_id=tid, class_id=d.class_id, label=d.label,
                box=d.box, score=d.score, smoothed_score=d.score,
                first_seen_ts=ts_s, last_seen_ts=ts_s,
            )
            matches[tid] = i

        # Confirmation: a track that just hit min_hits this frame fires
        # its delayed "enter" event. Tracking this on _Track via the
        # ``confirmed`` flag avoids a per-step recompute.
        for tid, det_idx in matches.items():
            t = self._tracks.get(tid)
            if not t or t.confirmed:
                continue
            if t.hits >= self.min_hits:
                t.confirmed = True
                d = detections[det_idx]
                events.append(TrackEvent(
                    kind="enter", track_id=tid, label=d.label,
                    score=d.score, box=d.box, ts=ts_s,
                ))

        # Expire stale tracks. Only emit "leave" for tracks that were
        # actually shown to the user (confirmed); tentative-then-dropped
        # tracks vanish silently.
        for tid in list(self._tracks.keys()):
            t = self._tracks[tid]
            if (ts_s - t.last_seen_ts) > self.ttl_s:
                if t.confirmed:
                    events.append(TrackEvent(
                        kind="leave", track_id=tid, label=t.label,
                        score=t.score, box=t.box, ts=ts_s,
                    ))
                del self._tracks[tid]

        # Emit only confirmed tracks that matched this frame. Tentative
        # tracks (hits < min_hits) are tracked internally but invisible
        # — that's the "banding" filter. Unmatched-but-still-alive
        # tracks are also invisible: when an object disappears, the
        # box disappears immediately, no ghosts. ttl_s still controls
        # how long the tracker holds the slot for re-id continuity if
        # the object pops back into view at a similar position.
        tracked: list[Detection] = []
        for tid, det_idx in matches.items():
            t = self._tracks.get(tid)
            if not t or not t.confirmed:
                continue
            d = detections[det_idx]
            tracked.append(Detection(
                class_id=d.class_id, label=d.label, score=d.score, box=d.box,
                track_id=tid,
                smoothed_score=t.smoothed_score,
            ))
        return tracked, events

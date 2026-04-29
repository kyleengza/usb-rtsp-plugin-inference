"""Shared Detection dataclass — single source of truth so backends and
the tracker don't drift."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source-frame px
    track_id: int = 0  # 0 = untracked / pre-tracker
    # Per-track EMA of score, populated by the tracker. Annotation should
    # prefer this over `score` so the displayed confidence stops flickering
    # frame-to-frame. 0.0 means "not tracked yet, use score".
    smoothed_score: float = 0.0

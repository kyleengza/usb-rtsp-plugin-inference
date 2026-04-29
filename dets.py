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

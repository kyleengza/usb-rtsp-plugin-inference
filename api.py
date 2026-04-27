"""Inference plugin REST endpoints (mounted at /api/inference)."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

API_KEY_FILE_NAME = "api-key"
RING_SIZE = 50

# Module-level ring buffer — survives across requests but not across
# restarts. That's the contract; persistence is out of scope.
_events: deque = deque(maxlen=RING_SIZE)


class Detection(BaseModel):
    label: str
    conf: float = Field(ge=0.0, le=1.0)
    bbox: list[float] | None = None  # [x1, y1, x2, y2] normalised


class InferenceEvent(BaseModel):
    path: str
    ts: float
    frame_id: int | None = None
    detections: list[Detection] = []


def _read_api_key(config_dir: Path) -> str | None:
    p = config_dir / API_KEY_FILE_NAME
    if not p.exists():
        return None
    val = p.read_text().strip()
    return val or None


def make_router(ctx) -> APIRouter:
    cfg_dir = ctx.plugin.config_dir
    cfg_dir.mkdir(parents=True, exist_ok=True)
    auth_lib = ctx.auth
    router = APIRouter(prefix="/api/inference", tags=["inference"])

    def _check_producer_auth(request: Request) -> None:
        """A producer can authenticate either with a panel cookie (same as
        a logged-in human) or with X-API-Key. If panel auth is OFF, both
        are optional."""
        if not auth_lib.panel_enabled():
            return
        # logged-in user takes precedence
        if getattr(request.state, "user", None):
            return
        api_key = _read_api_key(cfg_dir)
        provided = request.headers.get("x-api-key")
        if api_key and provided and provided == api_key:
            return
        raise HTTPException(401, "auth required (cookie or X-API-Key)")

    @router.post("/events")
    async def post_event(event: InferenceEvent, request: Request) -> JSONResponse:
        _check_producer_auth(request)
        record = {
            "path": event.path,
            "ts": event.ts,
            "received_at": time.time(),
            "frame_id": event.frame_id,
            "detections": [d.model_dump() for d in event.detections],
        }
        _events.append(record)
        return JSONResponse({"stored": True, "buffer_size": len(_events)})

    @router.get("/events")
    def list_events(limit: int = 50) -> JSONResponse:
        n = max(1, min(int(limit), RING_SIZE))
        items = list(_events)[-n:]
        # newest first
        items.reverse()
        # rough by-label counts in the last 60 s
        now = time.time()
        recent = [e for e in items if now - e["received_at"] < 60]
        counts: dict[str, int] = {}
        for e in recent:
            for d in e["detections"]:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
        return JSONResponse({
            "items": items,
            "buffer_size": len(_events),
            "label_counts_60s": counts,
        })

    @router.delete("/events")
    def clear_events(request: Request) -> JSONResponse:
        _check_producer_auth(request)
        _events.clear()
        return JSONResponse({"cleared": True})

    return router

# Inference plugin — external producer contract

This plugin **does not run inference**. It receives results pushed by an
external project (Hailo-8, GPU box, etc.) and surfaces them in the panel.

The producer publishes via two channels:

## 1. Annotated video stream (optional)

If you want the panel to show a live preview with bounding boxes drawn,
the external project re-encodes the source frame plus overlay and
**publishes** the result back into our mediamtx as an RTSP path.

URL the external project should publish to:

```
rtsp://stream:<password>@<usb-rtsp-host>:8554/<source>-ai
```

(omit credentials if stream auth is disabled)

The `-ai` suffix is convention, not enforced — anything goes. mediamtx
accepts publishers without a pre-configured path. The new path appears
under `/api/paths/list` and on the panel like a regular cam.

## 2. Detection events (required)

For every frame (or every Nth frame) the producer POSTs a JSON event
to the panel:

```
POST http://<usb-rtsp-host>:8080/api/inference/events
Content-Type: application/json
X-API-Key: <key>            # if a key is configured (otherwise omit)

{
  "path": "cam0",                                 # source path name
  "ts": 1714234567.123,                           # unix epoch seconds, float
  "frame_id": 18342,                              # optional — for ordering
  "detections": [
    {"label": "person", "conf": 0.91,
     "bbox": [0.12, 0.34, 0.45, 0.78]},           # x1,y1,x2,y2 normalised 0-1
    {"label": "car",    "conf": 0.74,
     "bbox": [0.50, 0.50, 0.90, 0.95]}
  ]
}
```

We keep the most recent **50** events in an in-memory ring buffer (no
disk, no DB). Restart the panel and the buffer empties.

If panel auth is on, the producer must include the same cookie a
browser would (`usb-rtsp-auth=...`) **or** an `X-API-Key` header
matching the value in `~/.config/usb-rtsp/inference/api-key`. Easiest:
generate the key once, paste into the producer's config.

## What the panel does with events

- Latest 50 events render in a table on the dashboard.
- Per-source aggregation: count of detections per label in the last 60 s.
- (Future) bbox overlay on the WebRTC preview when an annotated `<source>-ai`
  path exists.

## What the panel does NOT do

- No persistence — install a real time-series store if you want history.
- No inference itself — detections are entirely the producer's call.
- No control plane — the panel can't ask the producer to start/stop or
  change models. Build that orchestration separately.

# usb-rtsp-plugin-inference

Real-time object detection + tracking + clip recording for any
[`usb-rtsp`](https://github.com/kyleengza/usb-rtsp) mediamtx path.
Subscribes to a USB camera or relay source, runs YOLO inference,
and republishes annotated frames as a new mediamtx path you can
watch over RTSP / HLS / WebRTC.

## Install

This is an **out-of-tree plugin** — installs from this repo into the
host's user-plugin dir, opt-in by default.

```sh
cd ~/usb-rtsp
./install.sh --add-plugin https://github.com/kyleengza/usb-rtsp-plugin-inference
```

Then enable in the panel: **Settings → Plugins → inference → enable**.

The first time you enable it, the panel-side dependencies are checked.
You'll need:

- **Python 3.11+** (system) — already present on a usb-rtsp host.
- **`python3-onnxruntime`** (apt) — required for the CPU backend.
  `sudo apt install python3-onnxruntime`
- **`hailort` + `hailo-tappas-core`** (apt: `hailo-all`) — required
  only if you want the Hailo backend. Already installed when the
  host went through `pi-bringup` with the AI HAT detected.
- **`ffmpeg`, `python3-opencv`, `python3-numpy`, `python3-yaml`** —
  on every usb-rtsp host already.

For Hailo + CPU model files, run the [companion bringup
script](https://github.com/kyleengza/pi-bringup) once:

```sh
cd ~/pi-bringup && ./scripts/pi-bringup-inference.sh
```

That fetches `yolov8m_h8.hef` (Hailo-8) + `yolo11n.onnx` / `yolo11s.onnx`
(CPU) into `~/.cache/usb-rtsp/inference-models/`.

## What it gives you

A new mediamtx path per inference job — `<source>-ai` by default.
That path is just like any other mediamtx path: WebRTC + HLS + RTSP
URLs, on-demand spawn (worker only runs while a viewer is subscribed,
unless you flip background mode on).

Frame pipeline:

```
RTSP source ──► cv2.VideoCapture ──► tracker ──► annotate ──► ffmpeg ──► RTSP republish
                       │                  │           │
                       └──► backend.detect ┘           └──► clips/<job>/*.mp4
                            (Hailo or CPU)            └──► events JSONL
```

## Backends

Picked per-job on the dashboard card via a left/right pill toggle.

### Hailo (default when HAT present)

Uses the `hailo_platform` Python bindings against any `.hef` in the
registry. Model HEFs are baked at 640×640 and have NMS bundled, so
output decoding is just reading the per-class bbox list.

| Model        | Params | Pi 5 + Hailo-8 |
|--------------|--------|----------------|
| `yolov8s`    | 11 M   | ~30 fps native |
| `yolov8m`    | 26 M   | ~23 fps native |
| `yolov6n`    | 4.7 M  | ~30 fps native |

Heavier models eat more of the chip's 26 TOPS — concurrent jobs
share the device. The dashboard's Hailo card surfaces live NNC
utilisation when at least one inference job is active.

### CPU (onnxruntime)

Uses `onnxruntime` with the CPU provider (3 threads, leaves 1 core
for ffmpeg). Selectable input size per job:

| Input | yolo11n  | yolo11s |
|-------|----------|---------|
| 640²  | ~6 fps   | ~2 fps  |
| 416²  | ~14 fps  | ~5 fps  |
| 320²  | ~24 fps  | ~8 fps  |

The "Max inference fps" knob caps the rate so the worker doesn't burn
CPU faster than your use case needs (5 fps is plenty for surveillance;
halving the cap halves CPU usage proportionally).

## Tracking + banding

Single-object IoU tracker with two-stage threshold and warm-up
confirmation:

- **Threshold (initial)** — score required to *spawn* a new track.
  Higher = fewer false-positive tracks.
- **Match threshold (lower)** — score required to *keep* an existing
  track alive. Lower = better tolerance to brief confidence dips
  (e.g. partial occlusion). `0` = auto = ½ threshold.
- **Min hits** — track must be matched in this many consecutive
  frames before the box is *displayed*. Filters one-frame false
  positives. Default 3 (~100 ms warm-up at 30 fps).
- **Track occlusion (s)** — seconds of no match before the slot is
  released. Higher = better re-id continuity if the object
  briefly leaves frame.

Confidence scores shown on boxes are EMA-smoothed per track so the
displayed value doesn't flicker frame-to-frame.

## Clip recording

Toggle from the card header pill or the Settings ▾ form.

When on: the worker keeps a single rolling ffmpeg encoder. It opens
on the first qualifying detection event, keeps recording while
*any* tracked object is in frame, and finalises `post_roll_s`
seconds after the last leaves. Filenames are pure timestamps:
`YYYYMMDDTHHMMSS.mp4`. Newest-N retention per job.

Trigger options:

- `track_enter` (default) — clip opens when a new tracked ID enters
  with a class in `trigger_classes` (or any allowed class if empty).
- `class_filter` — same as track_enter but explicit class list.
- `any_detection` — opens on any detection above threshold.

Clips browser is on the dashboard card (▶ to play inline,
download / delete buttons per row). Total disk usage shown above
the list. Storage root is configurable plugin-wide
(Settings → Inference → "Clips root") — point at a network mount
to keep clips off the SD card.

## Background mode (always-on)

Default is on-demand: worker spawns when a viewer connects, exits
~2 s after the last disconnects. Background mode (header pill)
flips the path to `runOnInit` so the worker keeps running for
event/clip recording even with no viewer. Hailo or CPU stays warm.

## Source toggle (settings page)

`/settings → Inference targets` lists every available mediamtx
path with a switch:

```
[✓] cam0  → cam0-ai     [hailo · yolov8s]
[ ] mypc  → mypc-ai     off
[✓] front → front-ai    source disabled/missing  (dimmed row)
```

Flipping a switch on creates an inference job with sensible defaults
(Hailo + first installed HEF; falls back to first CPU model). Off
deletes the job. Orphan rows (job exists but its source path is gone)
stay visible-but-dimmed so you can clean them up.

## REST API

Mounted at `/api/inference`:

- `GET /state` — all jobs + per-job live state (mediamtx readers,
  upstream missing flag, perf stats).
- `GET /sources` — list of mediamtx paths + has-inference flag.
- `PATCH /sources/{name}` body `{enabled: bool}` — one-click toggle.
- `GET /jobs`, `GET /jobs/{name}`, `POST /jobs`, `PUT /jobs/{name}`,
  `DELETE /jobs/{name}` — full CRUD.
- `POST /jobs/{name}/{enable,disable}` — quick on/off (used by the
  dashboard card slider).
- `PATCH /jobs/{name}/clips` — clip-recording toggle.
- `PATCH /jobs/{name}/always-on` — background-mode toggle.
- `POST /jobs/{name}/kick` — kick every reader on the path
  (used by the preview-fold cleanup).
- `GET /jobs/{name}/events?n=100` — recent track-enter / track-leave
  events.
- `GET /jobs/{name}/clips` — list recorded clips with sizes +
  durations.
- `GET /clips/{name}/{file}` — download.
- `DELETE /clips/{name}/{file}` — remove.
- `GET /models` — installed model list.
- `GET /config` / `PUT /config` — plugin-wide config (clips_root).

All save endpoints use mediamtx's per-path config API
(`/v3/config/paths/replace/{name}` / `delete/{name}`) — only the
affected path bounces, never the whole service.

## Files & paths

```
~/.local/share/usb-rtsp/plugins/inference/   ← clone target (this repo)
~/.config/usb-rtsp/inference/jobs.yml        ← per-job config
~/.config/usb-rtsp/inference/plugin.yml      ← plugin-wide (clips_root)
~/.cache/usb-rtsp/inference-models/          ← .hef + .onnx
~/.cache/usb-rtsp/inference-clips/<job>/     ← recorded mp4s
~/.cache/usb-rtsp/inference-events/<job>.jsonl  ← detection events
~/.cache/usb-rtsp/inference-stats/<job>.json    ← live FPS / latency
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ICE failed` on preview | Check `usb-rtsp` has STUN servers + LAN IP advertised; mediamtx 1.18 schema in use |
| `source disabled/missing` on card | The upstream path is not in mediamtx — re-enable the USB cam / relay source |
| CPU worker only 1-2 fps | Drop CPU input size to 416 or 320 in Settings ▾; check threads=3 in `backend_cpu.py` |
| Hailo `UNSUPPORTED_OPCODE` on power read | HAT firmware doesn't expose chip-power; only dev kit does — not a bug |
| Clip plays for a few seconds then folds | Earlier bug fixed in `8b63003`; re-pull |
| `[object Object]` toast | Pydantic 422 with list-of-dicts detail — already wrapped in `fmtErrDetail` |
| PSU brownout under sustained inference + clips | Move PSU to Pi USB-C, charge HAT separately; see `pi-bringup` watchdog `RESPECT_PI_RAIL` |

## License

Same as the parent project.
